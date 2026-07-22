"""Milestone B18 — exact-match caching (ADR-050).

SEMANTIC CACHING IS REJECTED, on safety grounds. "Metformin in CKD stage 3" and
"...stage 4" are neighbours in embedding space and clinically different. A
semantic hit returns a confidently wrong, cited, cached answer — the worst
possible artifact this system could produce.

Exact match on the CANONICAL RESOLVED query captures most of the real win
anyway: repeated demo queries are identical, not merely similar.

Two rules:
  * Version fields are MANDATORY in the key, or a corpus refresh silently serves
    stale answers and Challenge #10 walks back in through the cache.
  * The K1 guard verdict is NEVER cached. Caching a SAFE verdict means one
    successful jailbreak is cached as permanently safe — the cache becomes a
    persistence mechanism for the bypass.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from pharmarag.config import PROMPT_TEMPLATE_VERSION, settings
from pharmarag.db import session

_DDL = """
CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    final_action TEXT,
    created_at  TEXT NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
"""


def ensure_table() -> None:
    with session() as conn:
        conn.executescript(_DDL)


def cache_key(
    *,
    intent: str,
    rxcuis: list[str],
    population_tags: list[str] | None = None,
) -> str:
    payload = {
        "intent": intent,
        "rxcuis": sorted(set(rxcuis)),
        "population_tags": sorted(set(population_tags or [])),
        "corpus_version": settings.corpus_version,
        "graph_version": settings.graph_version,
        "calibrator_version": settings.calibrator_version,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def get(key: str) -> dict[str, Any] | None:
    ensure_table()
    with session() as conn:
        row = conn.execute("SELECT payload FROM answer_cache WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE answer_cache SET hits = hits + 1 WHERE cache_key=?", (key,))
    data: dict[str, Any] = json.loads(row["payload"])
    data["_cache_hit"] = True
    return data


def put(key: str, payload: dict[str, Any]) -> None:
    """Refusals are cached too — repeated out-of-scope probing costs nothing
    after the first hit."""
    ensure_table()
    with session() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO answer_cache "
            "(cache_key, payload, final_action, created_at, hits) VALUES (?,?,?,?,0)",
            (
                key,
                json.dumps(payload, default=str),
                payload.get("answer_type"),
                dt.datetime.now(dt.UTC).isoformat(),
            ),
        )


def stats() -> dict[str, Any]:
    ensure_table()
    with session() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(hits),0) h FROM answer_cache"
        ).fetchone()
    return {"entries": row["n"], "hits": row["h"]}


def clear() -> None:
    ensure_table()
    with session() as conn:
        conn.execute("DELETE FROM answer_cache")
