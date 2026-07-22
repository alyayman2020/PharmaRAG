"""Hybrid retrieval with entity pre-filtering (ADR-021, ADR-022, ADR-023).

Two properties that are easy to lose:

  * PRE-filtering, not post-filtering. Fetch top-50 globally then filter to one
    RxCUI and you may be left with three chunks, or zero. Pre-filtering restricts
    the search space first, so top-k means what it says.

  * RRF, not weighted score blending. Cosine is bounded, BM25 is unbounded and
    shifts per query, so a fixed alpha gets dominated by whichever branch has
    larger raw magnitudes. RRF fuses on RANKS and sidesteps this entirely.

RRF here is a CANDIDATE GENERATOR, not a final ranker — the cross-encoder
supplies ordering. That is why the branch limits are tuned for recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from pharmarag.config import (
    COLLECTION,
    DENSE_VECTOR,
    PREFETCH_DENSE_PROSE,
    PREFETCH_DENSE_TABLE,
    PREFETCH_SPARSE_PROSE,
    PREFETCH_SPARSE_TABLE,
    SPARSE_VECTOR,
)


class EmptyCandidateSetError(Exception):
    """ADR-023: an empty filtered set is a HARD REFUSAL trigger.

    Never fall back to unfiltered search — that is exactly how parametric
    knowledge leaks into a system that promised corpus-only grounding.
    """


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    score: float
    payload: dict[str, Any]

    @property
    def parent_chunk_id(self) -> str:
        return str(self.payload.get("parent_chunk_id", ""))


def _rxcui_filter(rxcuis: list[str]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="rxcui", match=models.MatchAny(any=rxcuis))]
    )


def hybrid_search(
    client: QdrantClient,
    *,
    dense_vector: list[float],
    sparse_vector: Any,
    rxcuis: list[str],
    section_codes: list[str] | None = None,
    limit: int = 70,
) -> list[Candidate]:
    """Four prefetch branches fused with RRF, all inside the RxCUI partition."""
    if not rxcuis:
        raise EmptyCandidateSetError(
            "no RxCUIs resolved — refusing rather than searching unfiltered"
        )

    base = _rxcui_filter(rxcuis)
    sparse_q = models.SparseVector(
        indices=list(sparse_vector.indices), values=list(sparse_vector.values)
    )

    def with_type(content_type: str) -> models.Filter:
        return models.Filter(
            must=[
                *(base.must or []),
                models.FieldCondition(
                    key="content_type", match=models.MatchValue(value=content_type)
                ),
            ]
        )

    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using=DENSE_VECTOR,
            filter=with_type("prose"),
            limit=PREFETCH_DENSE_PROSE,
        ),
        models.Prefetch(
            query=sparse_q,
            using=SPARSE_VECTOR,
            filter=with_type("prose"),
            limit=PREFETCH_SPARSE_PROSE,
        ),
        models.Prefetch(
            query=dense_vector,
            using=DENSE_VECTOR,
            filter=with_type("table_row"),
            limit=PREFETCH_DENSE_TABLE,
        ),
        models.Prefetch(
            query=sparse_q,
            using=SPARSE_VECTOR,
            filter=with_type("table_row"),
            limit=PREFETCH_SPARSE_TABLE,
        ),
    ]

    # Section-scoped branch on detected intent (e.g. contraindication queries).
    if section_codes:
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=15,
                filter=models.Filter(
                    must=[
                        *(base.must or []),
                        models.FieldCondition(
                            key="loinc_section_code", match=models.MatchAny(any=section_codes)
                        ),
                    ]
                ),
            )
        )

    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    candidates = [
        Candidate(
            chunk_id=str((p.payload or {}).get("chunk_id", p.id)),
            score=float(p.score or 0.0),
            payload=dict(p.payload or {}),
        )
        for p in result.points
    ]
    if not candidates:
        raise EmptyCandidateSetError(
            f"no chunks matched rxcui in {rxcuis} — the corpus lacks evidence for this drug"
        )
    return candidates


def retrieve_for_pair(
    client: QdrantClient,
    *,
    dense_vector: list[float],
    sparse_vector: Any,
    rxcui_a: str,
    rxcui_b: str,
    limit: int = 40,
) -> list[Candidate]:
    """Primitive the agent loops over for compound regimens (ADR-023, ADR-033).

    Pairwise coverage is NECESSARY BUT NOT SUFFICIENT — additive risks (the
    NSAID + ACE-inhibitor + diuretic "triple whammy", QT burden, serotonin
    syndrome) are class-count properties, not pairwise ones. The class-burden
    check lands at milestone B8.
    """
    return hybrid_search(
        client,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        rxcuis=[rxcui_a, rxcui_b],
        section_codes=["34073-7"],
        limit=limit,
    )
