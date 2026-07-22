"""ADR-027 safety-tier ordering and asymmetric relevance floors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pharmarag.generate.context import assemble

pytestmark = pytest.mark.deterministic


@dataclass
class S:
    chunk_id: str
    calibrated_score: float
    payload: dict[str, Any]


def _mk(cid: str, score: float, loinc: str, parent: str, content_type: str = "prose") -> S:
    return S(
        cid,
        score,
        {
            "loinc_section_code": loinc,
            "parent_chunk_id": parent,
            "section_path": loinc,
            "ingredient_name": "d",
            "display_text": f"text {cid}",
            "effective_time": None,
            "source_url": "u",
            "content_type": content_type,
        },
    )


def test_boxed_warning_outranks_higher_scoring_dosing() -> None:
    """Relevance ordering would bury a contraindication. Ours does not."""
    kept, _ = assemble(
        [_mk("dose", 0.91, "34068-7", "p1"), _mk("boxed", 0.44, "34066-1", "p2")], {}
    )
    assert kept[0].chunk_id == "boxed"


def test_tier1_still_subject_to_its_own_floor() -> None:
    """'Never dropped' means never dropped FOR LENGTH, not exempt from relevance."""
    kept, dropped = assemble(
        [_mk("ok", 0.80, "34068-7", "p1"), _mk("irrelevant_boxed", 0.10, "34066-1", "p2")], {}
    )
    assert "irrelevant_boxed" in dropped
    assert all(k.chunk_id != "irrelevant_boxed" for k in kept)


def test_parent_deduplication() -> None:
    """Four children of one parent return that parent once (Challenge #6)."""
    kept, _ = assemble([_mk(f"c{i}", 0.9 - i * 0.01, "34068-7", "same") for i in range(4)], {})
    assert len(kept) == 1


def test_table_rows_are_never_deduplicated_by_parent() -> None:
    """ADR-008: a table's rows share one parent, and that parent is NOT their superset.

    Deduping them would keep the top-scoring row only, replace its text with the
    section intro, and silently discard every other row of the table — which is how
    a documented interaction row can be retrieved and still never reach the model.
    """
    rows = [_mk(f"t{i}", 0.9 - i * 0.01, "34073-7", "same-table", "table_row") for i in range(4)]
    kept, _ = assemble(rows, {"same-table": "TABLE SECTION INTRO ONLY"})
    assert len(kept) == 4
    assert [k.text for k in kept] == [f"text t{i}" for i in range(4)]


def test_prose_still_collapses_into_its_parent() -> None:
    """The table-row exemption must not disable parent retrieval for prose."""
    kept, _ = assemble([_mk("c0", 0.9, "34068-7", "p")], {"p": "FULL PARENT TEXT"})
    assert [k.text for k in kept] == ["FULL PARENT TEXT"]
