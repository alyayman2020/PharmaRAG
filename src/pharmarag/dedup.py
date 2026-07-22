"""Milestone B2 — canonical label selection and variant conflict diffs (ADR-009).

The naive version — index 2 variant labels per drug — would triple the index
with near-duplicates, because ANDA labels must substantially match the reference
listed drug. That mitigates Challenge #8 by WORSENING Challenges #1 and #6.

Instead: index the canonical label only, diff the variants, and index only the
SUBSTANTIVE divergent spans as `is_variant` conflict evidence. Index growth of
roughly +5% instead of +200%, and the diff itself becomes an empirical finding:
"of N ingredients, M showed substantive labeling divergence on dosing
thresholds" is a real README line, not a hand-wave.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spl_parser.models import SPLDocument

# Differences that are noise, not clinical divergence.
_TRIVIAL = re.compile(
    r"^\s*$|^\d{4,}-\d+-\d+$|manufactur|distribut|revised:|rev\.|"
    r"^ndc\b|trademark|®|™|marketed by|licensed",
    re.IGNORECASE,
)
MIN_DIVERGENCE_CHARS = 40


@dataclass(slots=True)
class LabelCandidate:
    set_id: str
    doc: SPLDocument
    application_type: str | None
    coverage_score: int = 0
    effective_time: str | None = None


@dataclass(slots=True)
class ConflictSpan:
    canonical_set_id: str
    variant_set_id: str
    loinc_section_code: str
    section_name: str
    canonical_text: str
    variant_text: str

    def as_chunk_text(self) -> str:
        return (
            f"[LABEL DIVERGENCE — {self.section_name}] "
            f"Another manufacturer's label for this ingredient states: "
            f"{self.variant_text}"
        )


@dataclass(slots=True)
class DedupResult:
    canonical: LabelCandidate
    variants: list[LabelCandidate] = field(default_factory=list)
    conflicts: list[ConflictSpan] = field(default_factory=list)


def _rank(candidate: LabelCandidate, target_sections: frozenset[str]) -> tuple[int, int, str]:
    """ADR-009 cascade: NDA/BLA first, then section coverage, then recency."""
    app = (candidate.application_type or "").upper()
    priority = 0 if app.startswith(("NDA", "BLA")) else 1
    coverage = sum(1 for present in candidate.doc.coverage(target_sections).values() if present)
    return (
        priority,
        -coverage,
        "9999"
        if not candidate.effective_time
        else str(9999 - int(candidate.effective_time.replace("-", "")[:4])),
    )


def _is_trivial(text: str) -> bool:
    return bool(_TRIVIAL.search(text)) or len(text.strip()) < MIN_DIVERGENCE_CHARS


def diff_sections(
    canonical: LabelCandidate,
    variant: LabelCandidate,
    target_sections: frozenset[str],
) -> list[ConflictSpan]:
    """Substantive divergences only. Trivial diffs (NDC codes, manufacturer
    names, whitespace) are discarded before anything is indexed."""
    out: list[ConflictSpan] = []
    for code in sorted(target_sections):
        c_sec = canonical.doc.section_by_loinc(code)
        v_sec = variant.doc.section_by_loinc(code)
        if c_sec is None or v_sec is None:
            continue
        c_text, v_text = c_sec.text.strip(), v_sec.text.strip()
        if not c_text or not v_text or c_text == v_text:
            continue

        matcher = difflib.SequenceMatcher(None, c_text, v_text, autojunk=False)
        if matcher.quick_ratio() > 0.98:
            continue
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            v_span = v_text[j1:j2].strip()
            if _is_trivial(v_span):
                continue
            out.append(
                ConflictSpan(
                    canonical_set_id=canonical.set_id,
                    variant_set_id=variant.set_id,
                    loinc_section_code=code,
                    section_name=c_sec.section_name,
                    canonical_text=c_text[i1:i2].strip()[:400],
                    variant_text=v_span[:400],
                )
            )
    return out


def select_canonical(
    candidates: list[LabelCandidate],
    target_sections: frozenset[str],
    *,
    max_variants: int = 2,
) -> DedupResult:
    if not candidates:
        raise ValueError("no label candidates")
    ordered = sorted(candidates, key=lambda c: _rank(c, target_sections))
    canonical, rest = ordered[0], ordered[1 : max_variants + 1]
    conflicts: list[ConflictSpan] = []
    for variant in rest:
        conflicts.extend(diff_sections(canonical, variant, target_sections))
    return DedupResult(canonical=canonical, variants=rest, conflicts=conflicts)


def conflict_report(results: list[DedupResult]) -> dict[str, Any]:
    """The empirical finding for the README."""
    by_section: dict[str, int] = {}
    with_conflict = 0
    for r in results:
        if r.conflicts:
            with_conflict += 1
        for c in r.conflicts:
            by_section[c.section_name] = by_section.get(c.section_name, 0) + 1
    return {
        "ingredients_examined": len(results),
        "ingredients_with_substantive_divergence": with_conflict,
        "divergence_rate": round(with_conflict / len(results), 3) if results else 0.0,
        "divergences_by_section": dict(sorted(by_section.items(), key=lambda x: -x[1])),
    }
