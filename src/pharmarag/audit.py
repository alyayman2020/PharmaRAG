"""Audit log writer (ADR-047).

Design standard: a reader must be able to reconstruct WHY the system said what
it said, from this record alone.

Two properties worth understanding:
  * `context_assembled_chunk_ids` is DISTINCT from `retrieved_chunk_ids`. The 8k
    cap drops chunks; only the assembled set could have influenced the output.
  * `raw_query` is NULL when K1 flags first-person clinical framing. "I have CKD
    and take metformin" is special-category health data, and the guard that
    detects it already exists — reuse it rather than logging PHI.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from typing import Any

from pharmarag.config import settings
from pharmarag.db import session

_FIRST_PERSON_CLINICAL = re.compile(
    r"\b(i|my|me|we|our)\b[^.?!]{0,60}\b(have|has|take|taking|am on|is on|was "
    r"prescribed|suffer|diagnosed|allergic)\b",
    re.IGNORECASE,
)


def is_personal_clinical(query: str) -> bool:
    return bool(_FIRST_PERSON_CLINICAL.search(query))


def new_query_id() -> str:
    return f"q-{uuid.uuid4().hex[:16]}"


def _j(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def write_audit(record: dict[str, Any]) -> str:
    query_id = record.get("query_id") or new_query_id()
    raw_query = record.get("raw_query", "")
    redacted = is_personal_clinical(str(raw_query))

    row = {
        "query_id": query_id,
        "parent_query_id": record.get("parent_query_id"),
        "session_id": record.get("session_id"),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "raw_query": None if redacted else raw_query,
        "normalized_query": record.get("normalized_query", ""),
        "redacted": int(redacted),
        "guard_verdict": record.get("guard_verdict"),
        "guard_model_version": record.get("guard_model_version"),
        "resolved_rxcuis": _j(record.get("resolved_rxcuis")),
        "resolution_tier": record.get("resolution_tier"),
        "resolution_confidence": record.get("resolution_confidence"),
        "substitutions_surfaced": _j(record.get("substitutions_surfaced")),
        "expansion_applied": _j(record.get("expansion_applied")),
        "expansion_overflow": int(bool(record.get("expansion_overflow"))),
        "retrieved_chunk_ids": _j(record.get("retrieved_chunk_ids")),
        "chunk_sha256": _j(record.get("chunk_sha256")),
        "reranker_scores": _j(record.get("reranker_scores")),
        "calibrated_scores": _j(record.get("calibrated_scores")),
        "context_assembled_chunk_ids": _j(record.get("context_assembled_chunk_ids")),
        "prompt_template_version": record.get("prompt_template_version", "unknown"),
        "prompt_hash": record.get("prompt_hash"),
        "synthesis_model_version": record.get("synthesis_model_version"),
        "evaluator_model_version": record.get("evaluator_model_version"),
        "guardrail_results": _j(record.get("guardrail_results")),
        # Per-claim K2 verdicts (ADR-039), so a reader can see which claim failed and
        # why — JSON-encoded like the other structured columns.
        "evaluator_verdict": _j(record.get("evaluator_verdict")),
        "retry_count": record.get("retry_count", 0),
        "rejection_reasons": _j(record.get("rejection_reasons")),
        "structured_output": _j(record.get("structured_output")),
        "final_action": record.get("final_action"),
        "reason_code": record.get("reason_code"),
        "disclaimer_shown": int(record.get("disclaimer_shown", True)),
        "corpus_version": settings.corpus_version,
        "graph_version": settings.graph_version,
        "calibrator_version": settings.calibrator_version,
        "latency_ms_by_stage": _j(record.get("latency_ms_by_stage")),
        "cost_usd": record.get("cost_usd"),
    }

    cols = ",".join(row)
    marks = ",".join("?" * len(row))
    with session() as conn:
        conn.execute(f"INSERT INTO audit_log ({cols}) VALUES ({marks})", tuple(row.values()))
    return query_id
