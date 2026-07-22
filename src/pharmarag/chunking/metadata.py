"""Deterministic metadata extraction (ADR-013). No LLM, no cost, no latency.

`dose_values` is the substrate for the K3 guardrail (ADR-040). Every dose the
model states must be traceable to one of these or to raw chunk text — otherwise
the answer is blocked.
"""

from __future__ import annotations

import re
from typing import Final

# --------------------------------------------------------------------------- units
# ADR-040: normalize to a canonical base BEFORE comparing magnitudes, or a
# correct 1 g -> 1000 mg conversion gets falsely blocked as a 1000x error.
UNIT_CANONICAL: Final[dict[str, tuple[str, float]]] = {
    "mcg": ("mg", 0.001),
    "µg": ("mg", 0.001),
    "ug": ("mg", 0.001),
    "mg": ("mg", 1.0),
    "g": ("mg", 1000.0),
    "gm": ("mg", 1000.0),
    "gram": ("mg", 1000.0),
    "kg": ("mg", 1_000_000.0),
    "ng": ("mg", 0.000001),
    "ml": ("ml", 1.0),
    "l": ("ml", 1000.0),
    "liter": ("ml", 1000.0),
    "unit": ("unit", 1.0),
    "units": ("unit", 1.0),
    "iu": ("unit", 1.0),
    "meq": ("meq", 1.0),
    "mmol": ("mmol", 1.0),
}

_UNIT_ALT = "|".join(sorted((re.escape(u) for u in UNIT_CANONICAL), key=len, reverse=True))
_UNIT_RE = re.compile(rf"\b({_UNIT_ALT})\b", re.IGNORECASE)

# Rate units are kept verbatim — they are not simple scalars.
_RATE_RE = re.compile(r"\b(ml/min|l/min|mg/kg|mcg/kg|mg/m2|ml/min/1\.73\s*m2)\b", re.IGNORECASE)

# `(?!\s*/)` rejects rate expressions: "50 mL/min" is renal function, not a dose.
# The trailing context is a LOOKAHEAD, not a consuming group — a greedy group
# here swallows the next dose in the sentence and silently drops it.
_DOSE_RE = re.compile(
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_ALT})\b(?!\s*/)" r"(?=(?P<rest>[^.;]{0,60}))",
    re.IGNORECASE,
)

_FREQ_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(r"\bonce\s+(?:a\s+)?dai?ly\b|\bevery\s+24\s+hours?\b|\bq24h?\b|\bqd\b", re.I),
        "q24h",
    ),
    (
        re.compile(r"\btwice\s+(?:a\s+)?dai?ly\b|\bevery\s+12\s+hours?\b|\bq12h?\b|\bbid\b", re.I),
        "q12h",
    ),
    (
        re.compile(
            r"\bthree\s+times\s+(?:a\s+)?dai?ly\b|\bevery\s+8\s+hours?\b|\bq8h?\b|\btid\b", re.I
        ),
        "q8h",
    ),
    (
        re.compile(
            r"\bfour\s+times\s+(?:a\s+)?dai?ly\b|\bevery\s+6\s+hours?\b|\bq6h?\b|\bqid\b", re.I
        ),
        "q6h",
    ),
    (re.compile(r"\bevery\s+other\s+day\b|\bq48h?\b", re.I), "q48h"),
    (re.compile(r"\bweekly\b|\bevery\s+week\b", re.I), "weekly"),
    (re.compile(r"\bper\s+day\b|\bdai?ly\b", re.I), "daily"),
]

# The ADR-013 `qualifier` — a dose stripped of this is a wrong dose wearing a
# right dose's clothes.
# Tightly bounded. A loose `[^,.;]{0,45}` runs past the qualifier into the rest
# of the sentence, and the over-captured text then satisfies the K3 presence
# check on common words like "dose" — silently disabling the qualifier rule.
_QUALIFIER_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:CrCl|creatinine clearance|GFR|eGFR)\s*(?:of\s*)?"
        r"(?:greater than|less than|below|above|at least|[<>≥≤])?\s*"
        r"\d+(?:\s*(?:to|-|–)\s*\d+)?\s*(?:mL/min(?:/1\.73\s*m2)?)?",
        re.I,
    ),
    re.compile(r"Child[- ]Pugh\s*(?:class\s*)?[ABC]?", re.I),
    re.compile(r"\b(?:ages?|aged|patients?)\s+\d+\s*(?:to|-|–)\s*\d+\s*(?:years?|months?)", re.I),
    re.compile(r"\b(?:severe|moderate|mild)\s+(?:renal|hepatic)\s+impairment\b", re.I),
    re.compile(r"\b(?:renal|hepatic)\s+impairment\b", re.I),
    re.compile(r"\b(?:pediatric|paediatric|geriatric|elderly|neonates?|neonatal)\b", re.I),
    re.compile(r"\b(?:maximum|not\s+to\s+exceed|do\s+not\s+exceed)\b", re.I),
]

# Words too common to prove a qualifier survived into the answer.
_QUALIFIER_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "dose",
        "doses",
        "dosage",
        "recommended",
        "patient",
        "patients",
        "with",
        "the",
        "and",
        "for",
        "class",
        "than",
        "least",
        "once",
        "twice",
        "daily",
    }
)

POPULATION_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "renal": re.compile(
        r"\brenal\b|\bkidney\b|\bCrCl\b|\bcreatinine clearance\b|\bGFR\b|\bnephro\w+|\bdialysis\b|\bCKD\b",
        re.I,
    ),
    "hepatic": re.compile(
        r"\bhepatic\b|\bliver\b|\bChild[- ]Pugh\b|\bcirrhosis\b|\bhepato\w+", re.I
    ),
    "pediatric": re.compile(
        r"\bpediatric\b|\bpaediatric\b|\bchildren\b|\binfants?\b|\bneonat\w+|\badolescents?\b", re.I
    ),
    "geriatric": re.compile(r"\bgeriatric\b|\belderly\b|\bolder adults?\b|\b65 years\b", re.I),
    "pregnancy": re.compile(r"\bpregnan\w+|\bteratogen\w+|\bfetal\b|\bfoetal\b", re.I),
    "lactation": re.compile(
        r"\blactation\b|\bbreast[- ]?feed\w*|\bnursing mothers?\b|\bhuman milk\b", re.I
    ),
}


def normalize_dose(value: float, unit: str) -> tuple[float, str] | None:
    """Convert to a canonical base unit. Returns None for unknown units."""
    entry = UNIT_CANONICAL.get(unit.lower())
    if entry is None:
        return None
    base, factor = entry
    return value * factor, base


def detect_units(text: str) -> list[str]:
    found = {m.group(1).lower() for m in _UNIT_RE.finditer(text)}
    found |= {m.group(1).lower() for m in _RATE_RE.finditer(text)}
    return sorted(found)


def detect_population_tags(text: str) -> list[str]:
    return sorted(tag for tag, rx in POPULATION_PATTERNS.items() if rx.search(text))


def _frequency(fragment: str) -> str:
    for rx, label in _FREQ_PATTERNS:
        if rx.search(fragment):
            return label
    return ""


def _qualifier(sentence: str) -> str:
    hits = [m.group(0).strip() for rx in _QUALIFIER_PATTERNS if (m := rx.search(sentence))]
    return "; ".join(dict.fromkeys(hits))


def extract_dose_values(text: str) -> list[dict[str, object]]:
    """Pull structured dose records out of free text.

    Precision-biased: a missed dose degrades K3 to its substring fallback, which
    is merely conservative. A wrong dose record would be far more expensive.
    """
    out: list[dict[str, object]] = []
    for sentence in re.split(r"(?<=[.;])\s+", text):
        qualifier = _qualifier(sentence)
        for m in _DOSE_RE.finditer(sentence):
            raw_unit = m.group("unit").lower()
            if raw_unit not in UNIT_CANONICAL:
                continue
            try:
                value = float(m.group("value").replace(",", "."))
            except ValueError:
                continue
            norm = normalize_dose(value, raw_unit)
            tail = m.group("rest") or ""
            out.append(
                {
                    "value": value,
                    "unit": raw_unit,
                    "normalized_value": norm[0] if norm else None,
                    "normalized_unit": norm[1] if norm else None,
                    "frequency": _frequency(tail) or _frequency(sentence),
                    "qualifier": qualifier,
                }
            )
    return out
