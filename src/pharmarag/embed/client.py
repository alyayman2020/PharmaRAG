"""OpenAI embeddings with a SHA-keyed cache (ADR-016).

Re-running the pipeline on unchanged chunks costs $0. Across ~5 chunking
experiments that is the difference between ~$0.55 and ~$0.15.

Track A uses the synchronous endpoint: at 50 drugs the whole run is fractions of
a cent and sync is far easier to debug. The Batch API (50% off) is worth wiring
at milestone B3 when the corpus scales to 1,500 — the cost only becomes material
at that size. Marked here so the trade-off is explicit rather than forgotten.
"""

from __future__ import annotations

import array
import datetime as dt
import hashlib
from collections.abc import Sequence

from pharmarag.config import EMBED_DIM, EMBED_MODEL
from pharmarag.db import session

PRICE_PER_1M_TOKENS = 0.02  # standard tier; batch is 0.01


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER: 999 on
# builds before 3.32, 32766 after). At 1,500 drugs `shas` is ~200k long, so the
# IN (...) lookup must be batched. 900 stays under the 999 floor even with the two
# fixed params (model, dim), so it is safe on every SQLite version.
_CACHE_GET_BATCH = 900


def _cache_get(shas: Sequence[str]) -> dict[str, list[float]]:
    if not shas:
        return {}
    out: dict[str, list[float]] = {}
    with session() as conn:
        for start in range(0, len(shas), _CACHE_GET_BATCH):
            batch = shas[start : start + _CACHE_GET_BATCH]
            marks = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT content_sha256, vector FROM embedding_cache "
                f"WHERE model=? AND dim=? AND content_sha256 IN ({marks})",
                (EMBED_MODEL, EMBED_DIM, *batch),
            ).fetchall()
            for r in rows:
                arr = array.array("f")
                arr.frombytes(r["vector"])
                out[r["content_sha256"]] = list(arr)
    return out


def _cache_put(items: dict[str, list[float]]) -> None:
    now = dt.datetime.now(dt.UTC).isoformat()
    with session() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache "
            "(content_sha256, model, dim, vector, created_at) VALUES (?,?,?,?,?)",
            [
                (sha, EMBED_MODEL, EMBED_DIM, array.array("f", vec).tobytes(), now)
                for sha, vec in items.items()
            ],
        )


def embed_texts(
    texts: Sequence[str], *, use_cache: bool = True, batch: int = 128, progress: bool = True
) -> list[list[float]]:
    """Embed with cache hits skipped. Returns vectors in input order.

    Resumable by design: each batch is written to the cache the moment it succeeds
    (ADR-016). A stalled connection, a crash, or a Ctrl-C loses at most one batch —
    a re-run picks up from the cache instead of re-embedding (and re-paying for)
    everything. ``batch`` is kept modest so the response stays small enough to pass
    cleanly through a TLS-inspecting proxy.
    """
    shas = [content_hash(t) for t in texts]
    cached = _cache_get(shas) if use_cache else {}
    missing_idx = [i for i, s in enumerate(shas) if s not in cached]

    if missing_idx:
        from pharmarag.http import openai_client

        client = openai_client()
        total = len(missing_idx)
        done = 0
        for start in range(0, total, batch):
            idxs = missing_idx[start : start + batch]
            resp = client.embeddings.create(model=EMBED_MODEL, input=[texts[i] for i in idxs])
            fresh = {shas[i]: item.embedding for i, item in zip(idxs, resp.data, strict=True)}
            if use_cache:
                _cache_put(fresh)  # incremental — this is what makes it resumable
            cached.update(fresh)
            done += len(idxs)
            if progress and (done == total or done % (batch * 20) == 0):
                print(f"[embed] {done:,}/{total:,} new vectors cached", flush=True)

    return [cached[s] for s in shas]


def embed_query(text: str) -> list[float]:
    return embed_texts([text], use_cache=False)[0]


def estimate_cost(total_tokens: int, *, batch: bool = False) -> float:
    rate = PRICE_PER_1M_TOKENS * (0.5 if batch else 1.0)
    return total_tokens / 1_000_000 * rate
