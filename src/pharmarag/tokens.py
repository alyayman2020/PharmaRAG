"""Token counting. Uses tiktoken when available, else a calibrated heuristic.

The heuristic keeps chunking deterministic in environments without tiktoken
(CI containers, offline dev). Chunk boundaries shift slightly between the two,
so `chunk_policy` and `token_count` are recorded per chunk and the tokenizer
backend is logged to MLflow — never compare runs across backends.
"""

from __future__ import annotations

import functools

_BACKEND = "heuristic"

try:
    import tiktoken

    _ENC: tiktoken.Encoding | None = tiktoken.get_encoding("cl100k_base")
    _BACKEND = "tiktoken:cl100k_base"
except Exception:
    _ENC = None


def backend() -> str:
    return _BACKEND


@functools.lru_cache(maxsize=100_000)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # ~4 chars/token for English clinical prose, +1 per number-ish run.
    return max(1, len(text) // 4)
