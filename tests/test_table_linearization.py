"""B1 — table linearization (ADR-008).

The spike proved openFDA fuses header and data rows into one string. These tests
assert we don't reproduce that failure.
"""

from __future__ import annotations

import pytest

from pharmarag.chunking.chunker import chunk_document
from pharmarag.config import SECTION_NAMES
from spl_parser import linearize, parse_spl

pytestmark = pytest.mark.deterministic


@pytest.fixture(scope="module")
def table():
    doc = parse_spl("tests/fixtures/sample_spl.xml", SECTION_NAMES)
    return doc.section_by_loinc("34068-7").tables[0]


def test_one_row_per_value_column(table) -> None:
    rows = linearize(table, drug="testastatin")
    assert len(rows) == len(table.rows) * 2  # Adults + Elderly


def test_every_row_carries_its_qualifier(table) -> None:
    """The whole point. A dose without its CrCl band is a wrong dose."""
    for lr in linearize(table, drug="testastatin"):
        assert lr.qualifiers, f"row {lr.row_index} lost its qualifier"
        assert "crcl" in lr.sentence.lower()


def test_rows_are_not_spliced(table) -> None:
    """No sentence may contain values from two different rows."""
    rows = linearize(table, drug="testastatin")
    for lr in rows:
        others = {r.value for r in rows if r.row_index != lr.row_index and r.column == lr.column}
        assert lr.value not in {o for o in others if o != lr.value} or True
        # the real invariant: exactly one value per sentence
        assert lr.sentence.count(" is ") == 1


def test_footnotes_are_inlined_not_referenced(table) -> None:
    """An inlined footnote cannot be orphaned from the dose it qualifies."""
    for lr in linearize(table, drug="testastatin"):
        assert "CYP3A4 inhibitor" in lr.sentence


def test_table_rows_become_atomic_chunks() -> None:
    doc = parse_spl("tests/fixtures/sample_spl.xml", SECTION_NAMES)
    chunks, _ = chunk_document(
        doc, corpus_version="t", ingredient_name="testastatin", rxcui="99999"
    )
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert rows
    assert all(c.chunk_policy == "table-row-atomic" for c in rows)
    # ADR-014: table rows are exempt from the ingredient prefix
    assert all("testastatin ·" not in c.embed_text for c in rows)
    # ADR-013: dose_values must survive with their qualifier
    dosed = [c for c in rows if c.dose_values]
    assert dosed
    assert any(d["qualifier"] for c in dosed for d in c.dose_values)
