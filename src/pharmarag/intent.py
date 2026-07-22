"""Deterministic query intent (ADR-050 cache key component).

Regex, not `MODEL_ROUTE`. The intent is part of the cache key, so it must be
free, instant, and stable: an LLM router would cost a call on every lookup —
including hits, defeating the cache — and any drift in its output would
silently fragment the key space, turning hits into misses and re-answering
questions that were already answered.

Order is significant. The patterns are checked most-specific first because a
question like "is atorvastatin contraindicated in pregnancy" is a
contraindication question that happens to mention a population, and the
narrower section is the one worth scoping to.

`SECTION_CODES` maps each intent to the LOINC sections that answer it. Nothing
consumes it yet — `hybrid_search` accepts `section_codes` and `pipeline.py` does
not pass it — but the mapping belongs with the classifier rather than at the
call site.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    CONTRAINDICATION = "contraindication"
    INTERACTION = "interaction"
    DOSING = "dosing"
    POPULATION = "population"
    WARNING = "warning"
    PHARMACOLOGY = "pharmacology"
    GENERAL = "general"


# Evaluated in order; first match wins.
_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.CONTRAINDICATION,
        re.compile(
            r"\b(contraindicat\w*|should not (be )?(take|use)|must not (take|use)|"
            r"who should not|not recommended for)\b",
            re.I,
        ),
    ),
    (
        Intent.INTERACTION,
        re.compile(
            r"\b(interact\w*|combin\w*|together with|concomitant\w*|co-?administ\w*|"
            r"taken with|along with)\b",
            re.I,
        ),
    ),
    (
        Intent.DOSING,
        re.compile(
            r"\b(dos\w*|mg\b|mcg\b|how much|how many|titrat\w*|administ\w*|"
            r"frequency|every \d+ hours?)\b",
            re.I,
        ),
    ),
    (
        Intent.WARNING,
        re.compile(
            r"\b(warning|precaution|boxed|black box|adverse|side effects?|risks?|" r"toxicit\w*)\b",
            re.I,
        ),
    ),
    (
        Intent.POPULATION,
        re.compile(
            r"\b(renal|kidney|hepatic|liver|pregnan\w*|lactat\w*|breast-?feed\w*|"
            r"pediatric|paediatric|children|geriatric|elderly|impairment)\b",
            re.I,
        ),
    ),
    (
        Intent.PHARMACOLOGY,
        re.compile(
            r"\b(mechanism|pharmacokinetic\w*|pharmacodynamic\w*|metabolis\w*|"
            r"half-?life|cyp\d\w*|clearance|absorption)\b",
            re.I,
        ),
    ),
)

SECTION_CODES: dict[Intent, tuple[str, ...]] = {
    Intent.CONTRAINDICATION: ("34070-3",),
    Intent.INTERACTION: ("34073-7",),
    Intent.DOSING: ("34068-7",),
    Intent.POPULATION: ("43684-0",),
    Intent.WARNING: ("34066-1", "43685-7", "34071-1"),
    Intent.PHARMACOLOGY: ("34090-1",),
    Intent.GENERAL: (),
}


def classify(question: str) -> Intent:
    for intent, pattern in _PATTERNS:
        if pattern.search(question):
            return intent
    return Intent.GENERAL


def section_codes(question: str) -> list[str]:
    return list(SECTION_CODES[classify(question)])
