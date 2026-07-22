"""Qdrant collection management (ADR-017, ADR-018).

Single collection with named dense + sparse vectors. Splitting prose and table
rows into separate collections would force two round trips and app-side fusion,
reintroducing the score-scale problem RRF exists to avoid.

Runs against a native Qdrant server (scripts/start_qdrant.ps1 — a single
qdrant.exe, no Docker needed on Windows) when QDRANT_URL is set. Embedded local
mode survives only for tests: it loads every point into Python memory at client
construction, which stopped being viable when the corpus reached 1000 drugs.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from pharmarag.config import (
    COLLECTION,
    DENSE_VECTOR,
    EMBED_DIM,
    PAYLOAD_INDEXES,
    QDRANT_PATH,
    SPARSE_VECTOR,
    settings,
)

# ADR-018: `retrievable` is deliberately absent. Non-retrievable chunks are never
# written at all — the guardrail is enforced by ABSENCE, not by a WHERE clause a
# developer can forget in one code path.
_KEYWORD_FIELDS = frozenset(PAYLOAD_INDEXES) - {"is_canonical"}


def get_client(path: str | None = None) -> QdrantClient:
    """Server client when QDRANT_URL is set, embedded local mode otherwise.

    Embedded mode deserializes the whole collection into Python memory at
    construction — at the 1000-drug corpus size that is 6+ GB of RAM and minutes
    of silent load. It remains only as the zero-setup path for tests, which pass
    an explicit ``path`` and therefore always bypass the URL.
    """
    if path is None and settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(path or QDRANT_PATH))


def create_collection(client: QdrantClient, *, recreate: bool = False) -> None:
    exists = client.collection_exists(COLLECTION)
    if exists and not recreate:
        return
    if exists:
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
    )

    # Unindexed payload fields fall back to full scan on filter, which defeats
    # the entire entity-first design (ADR-003).
    for field in _KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="is_canonical",
        field_schema=models.PayloadSchemaType.BOOL,
    )


def collection_stats(client: QdrantClient) -> dict[str, Any]:
    if not client.collection_exists(COLLECTION):
        return {"exists": False, "points": 0}
    info = client.get_collection(COLLECTION)
    return {"exists": True, "points": info.points_count}
