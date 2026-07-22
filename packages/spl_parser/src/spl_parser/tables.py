"""Milestone B1 — table row linearization (ADR-008).

The spike proved this is necessary. openFDA returned levofloxacin's renal dosing
table as one flat string:

    "...Type of Infection Dose Every 24 hours Duration (days)
       Nosocomial Pneumonia (1.1) 750 mg 7 to 14 ..."

Header row and first data row fused. An LLM reading that can splice a dose from
one row onto a duration from another. This module exists so that never happens.

Dual representation, per ADR-008:
  * `as_json()`      — structured record for exact lookup, K3 verification, and
                       rendering the original table in the UI
  * `linearize()`    — one natural-language sentence per (row x value-column),
                       generated deterministically. THIS is what gets embedded.

Why linearization beats a markdown table for embedding: pipes and dashes dilute
the semantics, and — more importantly — a row in a markdown table does not carry
its own units. Header inheritance forces every linearized row to restate the
column it belongs to, so a row physically cannot lose the CrCl band that
qualifies it. That converts Challenge #3 (table destruction) and Challenge #9
(unit/magnitude confusion) into the same solved problem, structurally rather
than by prompt instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spl_parser.models import Table

# Column headers that describe the ROW (a condition/qualifier) rather than a
# measured value. These become the sentence's subordinate clause.
_QUALIFIER_HEADER = re.compile(
    r"renal|creatinine|crcl|gfr|hepatic|child[- ]pugh|impair|function|"
    r"age|weight|population|patient|indication|type of|infection|severity|"
    r"class|group|status|category|condition",
    re.IGNORECASE,
)

# Headers naming something measured — the sentence's predicate.
_VALUE_HEADER = re.compile(
    r"dose|dosage|amount|mg|mcg|gram|unit|frequency|interval|schedule|"
    r"regimen|duration|days|weeks|rate|infusion|maximum|max|adjust|reduce",
    re.IGNORECASE,
)

_EMPTY_CELL = re.compile(r"^[\s\-–—*†‡\u00a0]*$")

MAX_SENTENCE_WORDS = 40


@dataclass(slots=True)
class LinearizedRow:
    """One (row x value-column) pair, ready to embed."""

    table_id: str
    row_index: int
    column: str
    sentence: str
    qualifiers: dict[str, str]
    value: str
    caption: str | None = None
    footnotes: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        """Structured half of the dual representation (ADR-008)."""
        return {
            "table_id": self.table_id,
            "row_index": self.row_index,
            "column": self.column,
            "value": self.value,
            "qualifiers": self.qualifiers,
            "caption": self.caption,
            "footnotes": self.footnotes,
        }


def _is_empty(cell: str) -> bool:
    return bool(_EMPTY_CELL.match(cell or ""))


def _flatten_headers(headers: list[list[str]]) -> list[str]:
    """Collapse a multi-row header into one label per column.

    SPL renal dosing tables routinely use two header rows:

        | Renal Function | Recommended Dose        |   <- colspan=2
        |                | Adults      | Elderly   |

    The parser already expanded colspan and carried rowspan down, so both rows
    have equal width. Joining vertically and de-duplicating gives
    "Renal Function", "Recommended Dose Adults", "Recommended Dose Elderly" —
    which is what makes each linearized row self-describing.
    """
    if not headers:
        return []
    width = max(len(h) for h in headers)
    out: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in headers:
            cell = row[col] if col < len(row) else ""
            cell = (cell or "").strip()
            if cell and cell not in parts:
                parts.append(cell)
        out.append(" ".join(parts).strip())
    return out


def _classify(headers: list[str]) -> tuple[list[int], list[int]]:
    """Split columns into qualifier columns and value columns."""
    qualifier_idx: list[int] = []
    value_idx: list[int] = []
    for i, h in enumerate(headers):
        if _VALUE_HEADER.search(h):
            value_idx.append(i)
        elif _QUALIFIER_HEADER.search(h) or i == 0:
            qualifier_idx.append(i)
        else:
            value_idx.append(i)

    # Degenerate tables: no recognizable value column. Treat column 0 as the
    # qualifier and everything else as values rather than emitting nothing —
    # dropping a dosing table silently is the worst possible failure here.
    if not value_idx:
        qualifier_idx = [0] if headers else []
        value_idx = list(range(1, len(headers)))
    if not qualifier_idx and len(headers) > 1:
        qualifier_idx = [0]
        value_idx = [i for i in value_idx if i != 0]
    return qualifier_idx, value_idx


def _sentence(
    drug: str, caption: str | None, qualifiers: dict[str, str], column: str, value: str
) -> str:
    """Deterministic natural-language rendering. No LLM, no cost, reproducible.

    Stacked headers ("Recommended Dose" over "Adults") produce a flattened label
    like "Recommended Dose Adults". Rendering that literally gives "the
    recommended dose adults of X is ...", which embeds worse than natural prose.
    Split the trailing sub-header into an "in <group>" clause instead.
    """
    clauses = [f"{k.lower()} {v}" if k else v for k, v in qualifiers.items() if v]
    subject = f"For {', '.join(clauses)}, " if clauses else ""

    label = column.strip() or "value"
    modifier = ""
    parts = label.split()
    # A short trailing token that isn't part of the measured quantity is a group.
    if len(parts) > 1 and not _VALUE_HEADER.search(parts[-1]):
        modifier = f" in {parts[-1].lower()}"
        label = " ".join(parts[:-1])

    head = f" of {drug}" if drug else ""
    sentence = f"{subject}the {label.lower()}{head}{modifier} is {value}."
    sentence = re.sub(r"\s{2,}", " ", sentence)
    if caption:
        sentence = f"{sentence} (From: {caption})"

    words = sentence.split()
    if len(words) > MAX_SENTENCE_WORDS:
        sentence = " ".join(words[:MAX_SENTENCE_WORDS]) + "…"
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def linearize(table: Table, *, drug: str = "") -> list[LinearizedRow]:
    """Turn a parsed Table into one LinearizedRow per (row x value-column).

    Footnotes are INLINED, not referenced. A referenced footnote requires a
    second fetch at generation time; an inlined one cannot be orphaned. That
    matters because dosing footnotes carry exactly the qualifiers that make a
    dose safe — "in patients also receiving a CYP3A4 inhibitor, reduce by 50%".
    """
    headers = _flatten_headers(table.headers)
    if not headers and table.rows:
        headers = [f"column {i + 1}" for i in range(len(table.rows[0]))]
    if not headers:
        return []

    qualifier_idx, value_idx = _classify(headers)
    out: list[LinearizedRow] = []

    for r, row in enumerate(table.rows):
        if all(_is_empty(c) for c in row):
            continue

        qualifiers = {
            headers[i]: row[i].strip()
            for i in qualifier_idx
            if i < len(row) and not _is_empty(row[i])
        }
        for i in value_idx:
            if i >= len(row) or _is_empty(row[i]):
                continue
            value = row[i].strip()
            sentence = _sentence(drug, table.caption, qualifiers, headers[i], value)
            if table.footnotes:
                sentence = f"{sentence} Note: {' '.join(table.footnotes)}"
            out.append(
                LinearizedRow(
                    table_id=table.table_id,
                    row_index=r,
                    column=headers[i],
                    sentence=sentence,
                    qualifiers=qualifiers,
                    value=value,
                    caption=table.caption,
                    footnotes=list(table.footnotes),
                )
            )
    return out


def linearize_all(tables: list[Table], *, drug: str = "") -> list[LinearizedRow]:
    out: list[LinearizedRow] = []
    for t in tables:
        out.extend(linearize(t, drug=drug))
    return out
