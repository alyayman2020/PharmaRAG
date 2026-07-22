"""Selection invariants (ADR-003).

The bug this guards against: ``load_expanded`` silently returned the 49-drug
Track A slice because the freeze it reads never existed. These tests pin the
corpus at exactly ``CORPUS_SIZE`` unique, single-ingredient concepts and prove
the machinery that enforces it.
"""

from __future__ import annotations

import sqlite3

import pytest

from pharmarag.config import DB_MAIN
from pharmarag.ingest.selection import (
    BACKUP_INGREDIENTS,
    CORPUS_SIZE,
    load_expanded,
    save_expanded,
    selection_path,
    track_a_slice,
)

pytestmark = pytest.mark.deterministic

# Combination-product separators — a single ingredient concept has none (ADR-003).
_COMBO_SEPARATORS = ("/", ",", " and ", " with ", ";", "+")


def _is_single_ingredient(name: str) -> bool:
    n = name.strip().lower()
    return bool(n) and not any(sep in n for sep in _COMBO_SEPARATORS)


def test_corpus_size_is_1000() -> None:
    assert CORPUS_SIZE == 1000


def test_track_a_slice_is_unique_and_single_ingredient() -> None:
    slice_ = track_a_slice()
    assert len(slice_) == len(set(slice_))
    assert all(_is_single_ingredient(n) for n in slice_)


def test_backup_ingredients_unique_and_single_ingredient() -> None:
    assert len(BACKUP_INGREDIENTS) == len(set(BACKUP_INGREDIENTS))
    assert all(_is_single_ingredient(n) for n in BACKUP_INGREDIENTS)


def test_save_expanded_enforces_exact_size(tmp_path) -> None:
    good = [f"drug{i}" for i in range(CORPUS_SIZE)]
    p = save_expanded(good, path=tmp_path / "sel.json")
    assert p.is_file()
    assert load_expanded(path=p) == good


def test_save_expanded_rejects_wrong_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly"):
        save_expanded(["a", "b"], path=tmp_path / "sel.json")


def test_save_expanded_rejects_duplicates(tmp_path) -> None:
    dupes = [f"drug{i}" for i in range(CORPUS_SIZE - 1)] + ["drug0"]
    with pytest.raises(ValueError, match="duplicates"):
        save_expanded(dupes, path=tmp_path / "sel.json")


@pytest.mark.skipif(
    not selection_path().is_file(),
    reason="corpus_selection.json not built yet — run scripts/build_corpus_1000.py",
)
def test_frozen_selection_is_exactly_1000_unique() -> None:
    drugs = load_expanded()
    assert len(drugs) == CORPUS_SIZE
    normalized = [d.strip().lower() for d in drugs]
    assert len(set(normalized)) == CORPUS_SIZE, "duplicates after normalization"
    assert all(_is_single_ingredient(d) for d in drugs)


@pytest.mark.skipif(
    not (selection_path().is_file() and DB_MAIN.is_file()),
    reason="needs both the freeze and the built corpus DB",
)
def test_frozen_selection_all_resolve_into_corpus() -> None:
    """Every frozen ingredient has retrievable chunks in the corpus (no silent drops).

    Checks the ``chunks`` table, not ``documents``: that is the exact invariant the
    freeze is built from (a drug is frozen only if it has >=1 retrievable chunk),
    and it is robust to the ``documents`` set_id projection — a combination-product
    SPL (e.g. sodium nitrite + thiosulfate antidote kit) maps two names to one
    set_id, so only one name can own that single ``documents`` row even though both
    have their own chunks. Retrievability is what actually matters for the corpus.
    """
    conn = sqlite3.connect(DB_MAIN)
    try:
        in_corpus = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT lower(ingredient_name) FROM chunks WHERE retrievable=1"
            )
        }
    finally:
        conn.close()
    missing = [d for d in load_expanded() if d.strip().lower() not in in_corpus]
    assert not missing, (
        f"{len(missing)} frozen drugs have no retrievable chunks: {missing[:10]}. "
        "If a build is in progress, wait for it to finish and re-run."
    )
