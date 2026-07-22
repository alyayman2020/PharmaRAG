"""FastAPI service with SSE stage streaming (ADR-048).

The API is the artifact for technical reviewers; the Streamlit UI is the artifact
for everyone else. The ADR-029 output schema becomes the response model, so the
safety contract shows up as browsable OpenAPI docs for free.

Streaming the pipeline stages turns a 12-second wait into the demo — the
reviewer WATCHES the safety layers fire instead of reading about them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pharmarag.config import DATA, settings
from pharmarag.entity.gazetteer import Gazetteer
from pharmarag.entity.lasa import load as load_lasa
from pharmarag.entity.resolve import Resolver
from pharmarag.generate.schema import DISCLAIMER
from pharmarag.pipeline import Stage, answer_question

app = FastAPI(
    title="PharmaRAG",
    version="0.1.0",
    description=(
        "Educational demonstration. NOT a medical device, NOT a clinical decision "
        "support tool, and not placed on the market for a clinical purpose. "
        "Every answer is grounded in FDA Structured Product Labeling with "
        "verifiable citations, and the system refuses when the corpus lacks evidence."
    ),
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    use_llm_guard: bool = True


@lru_cache(maxsize=1)
def _resolver() -> Resolver:
    gaz_path = DATA / "gazetteer.json"
    gaz = Gazetteer.load(gaz_path) if gaz_path.exists() else Gazetteer()
    return Resolver(gaz, load_lasa(DATA / "lasa_table.json"))


@lru_cache(maxsize=1)
def _qdrant() -> Any:
    """One local QdrantClient per process — the on-disk store is single-instance
    (portalocker), so a second client on the same path raises."""
    from pharmarag.index.store import get_client

    return get_client()


@lru_cache(maxsize=1)
def _runtime() -> tuple[Any, Any, Any]:
    from fastembed import SparseTextEmbedding

    from pharmarag.rerank.reranker import Reranker

    return _qdrant(), Reranker(), SparseTextEmbedding(model_name="Qdrant/bm25")


def _run(question: str, use_llm_guard: bool, on_stage: Any = None) -> Any:
    from pharmarag.embed.client import embed_query

    client, reranker, bm25 = _runtime()
    return answer_question(
        question,
        resolver=_resolver(),
        client=client,
        reranker=reranker,
        embed_dense=embed_query,
        embed_sparse=lambda q: next(iter(bm25.embed([q]))),
        use_llm_guard=use_llm_guard,
        on_stage=on_stage,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    from pharmarag.index.store import collection_stats

    return {
        "status": "ok",
        "device": settings.resolve_device(),
        "corpus_version": settings.corpus_version,
        "calibrator_version": settings.calibrator_version,
        "index": collection_stats(_qdrant()),
        "disclaimer": DISCLAIMER,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    result = _run(req.question, req.use_llm_guard)
    return {
        "query_id": result.query_id,
        "stages": [
            {"name": s.name, "detail": s.detail, "ms": round(s.ms, 1)} for s in result.stages
        ],
        **result.payload,
    }


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    async def gen() -> AsyncIterator[dict[str, str]]:
        emitted: list[Stage] = []
        result = _run(req.question, req.use_llm_guard, on_stage=emitted.append)
        for s in emitted:
            yield {
                "event": "stage",
                "data": json.dumps({"name": s.name, "detail": s.detail, "ms": round(s.ms, 1)}),
            }
        yield {"event": "answer", "data": json.dumps(result.payload, default=str)}

    return EventSourceResponse(gen())


@app.get("/audit/{query_id}")
def audit(query_id: str) -> dict[str, Any]:
    from pharmarag.db import session

    with session() as conn:
        row = conn.execute("SELECT * FROM audit_log WHERE query_id=?", (query_id,)).fetchone()
    return dict(row) if row else {"error": "not found"}
