"""Candidate generation — makes labeling ~3-4 min/item instead of ~10.

The model PROPOSES a question against a specific chunk. You review, edit, and
sign off. The model never decides what is correct; it only saves you typing.

Sampling is stratified so the 250 items exercise every path, rather than 250
easy dosing questions from whichever drugs happen to sort first.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from eval.schema import Category
from pharmarag.config import MODEL_EVALUATOR, settings
from pharmarag.db import session

# Which LOINC sections feed which category.
SECTION_FOR_CATEGORY: dict[Category, tuple[str, ...]] = {
    Category.DDI: ("34073-7",),
    Category.CONTRAINDICATION: ("34070-3", "34066-1"),
    Category.DOSING: ("34068-7",),
    Category.RENAL_HEPATIC: ("34068-7", "43684-0"),
    Category.LASA: ("34068-7", "34073-7"),
    Category.COMPOUND: ("34073-7",),
}

PROMPT = """You draft evaluation questions for a drug-information retrieval system.

Given a SOURCE passage from an FDA drug label, write ONE question that this \
passage answers completely and specifically.

Rules:
- The question must be answerable from THIS passage alone.
- Name the drug explicitly.
- If the passage states a dose that depends on a condition (renal function band, \
hepatic class, age, indication), the question MUST name that condition.
- Ask what a working pharmacist would ask. No meta-questions about the document.
- One sentence. No preamble.

Return JSON: {"question": "...", "difficulty": "easy|medium|hard", "why": "..."}"""


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    drug: str
    section: str
    section_path: str
    text: str
    content_type: str
    question: str = ""
    difficulty: str = "medium"
    why: str = ""


def sample_chunks(
    category: Category,
    n: int = 1,
    *,
    seed: int | None = None,
    prefer_tables: bool = False,
) -> list[Candidate]:
    """Stratified sample from the indexed corpus."""
    sections = SECTION_FOR_CATEGORY.get(category, ("34068-7",))
    marks = ",".join("?" * len(sections))
    order = "ORDER BY RANDOM()"
    type_filter = "AND content_type = 'table_row'" if prefer_tables else ""

    with session() as conn:
        rows = conn.execute(
            f"SELECT chunk_id, ingredient_name, section_name, section_path, "
            f"display_text, content_type FROM chunks "
            f"WHERE retrievable=1 AND loinc_section_code IN ({marks}) {type_filter} "
            f"AND LENGTH(display_text) > 120 {order} LIMIT ?",
            (*sections, n * 3),
        ).fetchall()

    rng = random.Random(seed)
    picked = rng.sample(list(rows), min(n, len(rows))) if rows else []
    return [
        Candidate(
            chunk_id=r["chunk_id"],
            drug=r["ingredient_name"],
            section=r["section_name"],
            section_path=r["section_path"] or "",
            text=r["display_text"],
            content_type=r["content_type"],
        )
        for r in picked
    ]


def propose_question(candidate: Candidate) -> Candidate:
    """Ask the model to draft a question. ~$0.0005 per call on mini."""
    if not settings.openai_api_key:
        candidate.question = ""
        candidate.why = "no API key — write the question manually"
        return candidate

    from pharmarag.http import openai_client

    try:
        resp = openai_client().chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": f"DRUG: {candidate.drug}\nSECTION: {candidate.section_path}\n\n"
                    f"SOURCE:\n{candidate.text[:2000]}",
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=200,
        )
        data: dict[str, Any] = json.loads(resp.choices[0].message.content or "{}")
        candidate.question = str(data.get("question", ""))
        candidate.difficulty = str(data.get("difficulty", "medium"))
        candidate.why = str(data.get("why", ""))
    except Exception as exc:
        candidate.why = f"generation failed: {exc}"
    return candidate


def next_candidate(category: Category, *, prefer_tables: bool = False) -> Candidate | None:
    picks = sample_chunks(category, 1, prefer_tables=prefer_tables)
    return propose_question(picks[0]) if picks else None
