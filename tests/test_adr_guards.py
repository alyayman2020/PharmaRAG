"""Structural guards. These fail loudly if an ADR is violated in code."""

from __future__ import annotations

import pytest

from pharmarag.config import (
    NON_RETRIEVABLE_SECTIONS,
    PAYLOAD_INDEXES,
    RETRIEVABLE_SECTIONS,
)
from pharmarag.db import init_db, verify_audit_immutability

pytestmark = pytest.mark.deterministic


def test_adr005_sections_are_disjoint() -> None:
    assert not (RETRIEVABLE_SECTIONS & NON_RETRIEVABLE_SECTIONS)
    assert "34088-5" in NON_RETRIEVABLE_SECTIONS


def test_adr018_effective_time_is_not_a_payload_index() -> None:
    """Staleness is SURFACED, never filtered.

    Filtering on effective_time would remove the only evidence for a drug and
    turn a currency problem into a false negative — the worst failure direction
    for a recall-first system.
    """
    assert "effective_time" not in PAYLOAD_INDEXES


def test_adr018_retrievable_is_not_a_payload_index() -> None:
    """Enforced by ABSENCE, not by a WHERE clause a developer can forget."""
    assert "retrievable" not in PAYLOAD_INDEXES


def test_adr018_rxcui_is_indexed() -> None:
    assert "rxcui" in PAYLOAD_INDEXES


def test_adr047_audit_log_is_append_only() -> None:
    init_db()
    assert verify_audit_immutability()


def test_adr005_upsert_refuses_non_retrievable_chunks() -> None:
    qdrant = pytest.importorskip("qdrant_client")  # noqa: F841
    from pharmarag.index.upsert import NonRetrievableChunkError, upsert_chunks

    with pytest.raises(NonRetrievableChunkError):
        upsert_chunks(None, [{"chunk_id": "c1", "retrievable": False}], [[0.0]], [None])  # type: ignore[arg-type]
