"""Writing chunks into Qdrant, with the ADR-005 guard enforced in code."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from pharmarag.config import COLLECTION, DENSE_VECTOR, SPARSE_VECTOR


class NonRetrievableChunkError(RuntimeError):
    """Raised if a chunk marked non-retrievable reaches the vector store.

    ADR-005 excluded Overdosage so that a guardrail bypass has nothing to
    retrieve. If this ever raises, defense in depth has a hole.
    """


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "chunk_id",
        "set_id",
        "parent_chunk_id",
        "doc_version",
        "effective_time",
        "corpus_version",
        "content_sha256",
        "chunk_policy",
        "token_count",
        "section_path",
        "source_url",
        "display_text",
        "raw_text",
        "rxcui",
        "rxcui_all",
        "ingredient_name",
        "brand_names",
        "pharm_class_epc",
        "loinc_section_code",
        "section_name",
        "content_type",
        "table_id",
        "footnote_text",
        "units_present",
        "population_tags",
        "dose_values",
        "application_type",
        "is_canonical",
        "is_variant",
        "conflict_of",
    )
    return {k: chunk.get(k) for k in keep}


def upsert_chunks(
    client: QdrantClient,
    chunks: Sequence[dict[str, Any]],
    dense: Sequence[Sequence[float]],
    sparse: Sequence[Any],
    *,
    batch_size: int = 128,
) -> int:
    """Write chunks. Refuses non-retrievable chunks loudly."""
    offenders = [c["chunk_id"] for c in chunks if not c.get("retrievable", True)]
    if offenders:
        raise NonRetrievableChunkError(
            f"ADR-005 violation — {len(offenders)} non-retrievable chunk(s) "
            f"reached the vector store: {offenders[:5]}"
        )

    written = 0
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        points = [
            models.PointStruct(
                id=_point_id(c["chunk_id"]),
                vector={
                    DENSE_VECTOR: list(d),
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(s.indices), values=list(s.values)
                    ),
                },
                payload=_payload(c),
            )
            for c, d, s in zip(chunks[start:end], dense[start:end], sparse[start:end], strict=True)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        written += len(points)
    return written
