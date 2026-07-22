"""Context assembly (ADR-027).

Safety-tier ordering, NOT relevance ordering. Our failure mode is not "the model
missed the most relevant chunk" — it is "the model missed the contraindication".
Relevance ordering places a high-scoring dosing chunk above a moderately-scoring
boxed warning. Exactly backwards.

Asymmetric relevance floors resolve the conflict between "tier 1 never dropped"
and the ADR-026 thresholds: tier-1 chunks are never dropped FOR LENGTH, but they
are not exempt from relevance. A boxed warning scoring 0.05 is about something
else entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pharmarag.config import (
    FLOOR_TIER_1,
    FLOOR_TIER_OTHER,
    MAX_CONTEXT_TOKENS,
    SAFETY_TIER,
    TOP_PARENTS,
)
from pharmarag.tokens import count_tokens


@dataclass(slots=True)
class ContextBlock:
    chunk_id: str
    parent_chunk_id: str
    tier: int
    score: float
    section_path: str
    section_name: str
    ingredient_name: str
    effective_time: str | None
    source_url: str
    text: str
    token_count: int


def assemble(
    scored: list[Any],
    parent_lookup: dict[str, str],
    *,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    top_parents: int = TOP_PARENTS,
) -> tuple[list[ContextBlock], list[str]]:
    """Returns (assembled blocks, dropped chunk ids).

    Parent deduplication is mandatory: four children from one parent return that
    parent ONCE. Direct protection for Challenge #6 and the context budget.
    """
    dropped: list[str] = []
    seen_parents: set[str] = set()
    blocks: list[ContextBlock] = []

    for s in scored:
        loinc = str(s.payload.get("loinc_section_code", ""))
        tier = SAFETY_TIER.get(loinc, 4)
        floor = FLOOR_TIER_1 if tier == 1 else FLOOR_TIER_OTHER
        if s.calibrated_score < floor:
            dropped.append(s.chunk_id)
            continue

        pid = str(s.payload.get("parent_chunk_id", ""))
        # ADR-008: table rows are ATOMIC, self-describing sentences, and a whole
        # table's rows share one parent whose text is the section intro — not a
        # superset of the rows. Deduping rows by parent replaces the top-scoring
        # row with that stub and silently discards every other row of the table,
        # which is how a rank-2 interaction row never reached the model. Rows
        # therefore keep their own text and never dedup; prose children still
        # collapse into their parent once.
        if str(s.payload.get("content_type", "")) == "table_row":
            text = str(s.payload.get("display_text", ""))
        else:
            if pid and pid in seen_parents:
                continue
            seen_parents.add(pid)
            text = parent_lookup.get(pid) or str(s.payload.get("display_text", ""))
        blocks.append(
            ContextBlock(
                chunk_id=s.chunk_id,
                parent_chunk_id=pid,
                tier=tier,
                score=s.calibrated_score,
                section_path=str(s.payload.get("section_path", "")),
                section_name=str(s.payload.get("section_name", "")),
                ingredient_name=str(s.payload.get("ingredient_name", "")),
                effective_time=s.payload.get("effective_time"),
                source_url=str(s.payload.get("source_url", "")),
                text=text,
                token_count=count_tokens(text),
            )
        )

    # Tier first, relevance within tier.
    blocks.sort(key=lambda b: (b.tier, -b.score))

    kept: list[ContextBlock] = []
    total = 0
    # Drop from the BACK (tier 4 upward) so tier 1 survives the cap.
    for b in blocks[:top_parents]:
        if total + b.token_count > max_tokens and b.tier > 1:
            dropped.append(b.chunk_id)
            continue
        kept.append(b)
        total += b.token_count
    dropped.extend(b.chunk_id for b in blocks[top_parents:])
    return kept, dropped


def render(blocks: list[ContextBlock]) -> str:
    parts: list[str] = []
    for i, b in enumerate(blocks, 1):
        date = f", effective {b.effective_time}" if b.effective_time else ""
        parts.append(
            f"[SOURCE {i}] id={b.chunk_id}\n"
            f"Drug: {b.ingredient_name} | Section: {b.section_path or b.section_name}{date}\n"
            f"---\n{b.text}\n"
        )
    return "\n".join(parts)
