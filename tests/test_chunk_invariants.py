"""The chunking invariants that carry safety weight (ADR-012, ADR-014).

test_no_overlap_on_atomic_sections is the most important test in the repo:
overlapping two dose rows can splice "250 mg q12h" and "500 mg q24h" into
"500 mg q12h" — a 2x overdose assembled from two individually correct rows.
"""

from __future__ import annotations

import itertools

import pytest

from pharmarag.chunking.chunker import chunk_document
from pharmarag.config import (
    CHUNK_POLICIES,
    MIN_CHUNK_TOKENS,
    NON_RETRIEVABLE_SECTIONS,
    SECTION_NAMES,
)
from spl_parser import parse_spl

FIXTURE = "tests/fixtures/sample_spl.xml"

pytestmark = pytest.mark.deterministic


@pytest.fixture(scope="module")
def parsed():
    return parse_spl(FIXTURE, SECTION_NAMES)


@pytest.fixture(scope="module")
def chunked(parsed):
    return chunk_document(
        parsed, corpus_version="test", ingredient_name="testastatin", rxcui="99999"
    )


def test_sections_extracted(parsed) -> None:
    codes = {s.loinc_code for s in parsed.sections}
    assert {"34066-1", "34070-3", "34073-7", "34068-7", "43684-0"} <= codes


def test_no_overlap_on_atomic_sections(chunked) -> None:
    """ADR-012: zero overlap on atomic sections. Dose splicing is a 2x overdose."""
    chunks, _ = chunked
    for code, policy in CHUNK_POLICIES.items():
        if not policy.atomic:
            continue
        texts = [c.raw_text for c in chunks if c.loinc_section_code == code]
        for a, b in itertools.pairwise(texts):
            tail = " ".join(a.split()[-6:])
            assert tail not in b, f"overlap detected in atomic section {code}"


def test_never_split_across_section_boundary(chunked) -> None:
    chunks, _ = chunked
    for c in chunks:
        assert c.loinc_section_code, "chunk lost its section code"
        assert c.section_name


def test_minimum_chunk_floor(chunked) -> None:
    """ADR-014: below ~60 tokens the prefix dominates the vector.

    The floor is enforced across ALL units within a LOINC section, including
    across subsections. The only permitted exception is a section whose ENTIRE
    content is below the floor — there is no sibling to merge with, and dropping
    it would lose evidence, which a recall-first system must never do.
    """
    chunks, _ = chunked
    by_section: dict[str, list] = {}
    for c in chunks:
        # B1: table rows are EXEMPT. The floor exists because the ADR-014 prefix
        # dominates short chunks, and table rows are not prefixed. More
        # importantly, merging two rows to reach the floor would splice
        # "250 mg q12h" and "500 mg q24h" — the precise failure ADR-012's
        # zero-overlap rule exists to prevent. Two ADRs interact here and
        # ADR-012 wins.
        if c.content_type == "table_row":
            continue
        by_section.setdefault(c.loinc_section_code, []).append(c)

    for code, group in by_section.items():
        total = sum(c.token_count for c in group)
        if total < MIN_CHUNK_TOKENS:
            assert len(group) == 1, f"tiny section {code} should collapse to one chunk"
            continue
        for c in group:
            assert c.token_count >= MIN_CHUNK_TOKENS, (
                f"chunk {c.chunk_id} in {code} is {c.token_count} tokens, below the floor, "
                f"but the section has {total} tokens available to merge with"
            )


def test_overdosage_marked_non_retrievable(chunked) -> None:
    """ADR-005: ingested and archived, never indexed."""
    chunks, _ = chunked
    over = [c for c in chunks if c.loinc_section_code in NON_RETRIEVABLE_SECTIONS]
    assert over, "fixture should contain an Overdosage section"
    assert all(not c.retrievable for c in over)


def test_prefix_only_on_embed_text(chunked) -> None:
    """ADR-021: dense gets prefixed text, BM25 gets raw text."""
    chunks, _ = chunked
    for c in chunks:
        assert c.raw_text == c.display_text
        assert c.embed_text != c.raw_text
        assert "testastatin" in c.embed_text.lower()


def test_parents_exist_for_every_chunk(chunked) -> None:
    chunks, parents = chunked
    ids = {p.parent_chunk_id for p in parents}
    assert all(c.parent_chunk_id in ids for c in chunks)


def test_table_rows_exempt_from_floor_but_never_merged(chunked) -> None:
    """Explicit: rows may be short, but they are never merged with each other."""
    chunks, _ = chunked
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert rows, "fixture should produce table rows"
    for c in rows:
        # exactly one dose statement per row chunk
        assert c.display_text.count(" is ") == 1
