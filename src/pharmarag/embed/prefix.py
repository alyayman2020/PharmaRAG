"""ADR-014 deterministic contextual prefixing.

Short atomic chunks ("Concomitant use increases the risk of myopathy") are
near-empty in isolation. Prefixing is free, reproducible, and puts the drug name
inside every vector — partial compensation for the Challenge #5 risk that
ADR-002 accepted.

Two rules that are easy to get wrong:
  * Table rows are EXEMPT. Their linearization already names the drug; prefixing
    would name it twice and skew the vector.
  * Only `embed_text` is prefixed. BM25 indexes `raw_text` (ADR-021), because a
    corpus-wide IDF gives the drug name real weight and every chunk of that drug
    would get a uniform lexical boost inside the entity-filtered partition.
"""

from __future__ import annotations


def build_embed_text(
    *,
    ingredient_name: str,
    section_name: str,
    body: str,
    subsection: str | None = None,
    content_type: str = "prose",
) -> str:
    if content_type in {"table_row", "table_json"}:
        return f"{section_name}: {body}" if section_name else body
    head = f"{ingredient_name} · {section_name}" if ingredient_name else section_name
    if subsection:
        head = f"{head} › {subsection}"
    return f"{head}: {body}" if head else body
