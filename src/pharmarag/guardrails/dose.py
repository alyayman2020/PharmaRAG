"""K3 — deterministic dose checks (ADR-040). $0, no latency, no LLM.

Five checks. The magnitude check normalizes units FIRST: without that, a correct
"1 g" -> "1000 mg" conversion is falsely blocked as a 1000x error, and a
systematic false-refusal poisons the false-refusal metric.

The qualifier check is the quiet killer. "250 mg every 12 hours" is correct for
CrCl 30-50 and wrong for normal renal function. A dose stripped of its qualifier
is a wrong dose wearing a right dose's clothes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pharmarag.chunking.metadata import (
    _QUALIFIER_STOPWORDS,
    extract_dose_values,
)


@dataclass(slots=True)
class DoseResult:
    passed: bool
    failures: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.failures)


def _qualifier_survived(qualifier: str, answer_text: str) -> bool:
    """Did the qualifier make it into the answer?

    Requires a distinctive token — a number or a non-stopword clinical term.
    """
    answer = answer_text.lower()
    tokens = [
        t.strip("(),.;:")
        for t in qualifier.lower().split()
        if len(t) > 2 and t.strip("(),.;:") not in _QUALIFIER_STOPWORDS
    ]
    distinctive = [t for t in tokens if any(ch.isdigit() for ch in t) or len(t) > 5]
    if not distinctive:
        distinctive = tokens
    return bool(distinctive) and all(t in answer for t in distinctive[:3])


def _source_doses(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for c in chunks:
        dv = c.get("dose_values")
        if isinstance(dv, list) and dv:
            out.extend(dv)
        else:
            out.extend(extract_dose_values(str(c.get("display_text", ""))))
    return out


def check_doses(answer_text: str, cited_chunks: list[dict[str, object]]) -> DoseResult:
    answer_doses = extract_dose_values(answer_text)
    if not answer_doses:
        return DoseResult(passed=True)

    source_doses = _source_doses(cited_chunks)
    source_text = " ".join(str(c.get("display_text", "")) for c in cited_chunks).lower()
    failures: list[str] = []

    for ad in answer_doses:
        a_norm = ad.get("normalized_value")
        a_unit = ad.get("normalized_unit")
        surface = f"{ad['value']:g} {ad['unit']}".lower()

        # 1 · Value presence — normalized match, or verbatim in the source text.
        matched = [
            sd
            for sd in source_doses
            if sd.get("normalized_value") == a_norm and sd.get("normalized_unit") == a_unit
        ]
        if not matched and surface not in source_text:
            failures.append(f"dose {surface} appears in no cited chunk")
            continue
        if not matched:
            continue  # substring fallback satisfied; nothing structured to compare

        # 2 · Unit match after normalization (ADR-040).
        if not any(sd.get("normalized_unit") == a_unit for sd in matched):
            failures.append(f"unit mismatch for {surface}")

        # 3 · Magnitude — only fires when normalization does NOT reconcile them.
        for sd in source_doses:
            nv = sd.get("normalized_value")
            if sd.get("normalized_unit") != a_unit or not isinstance(nv, int | float) or not nv:
                continue
            sv = float(nv)
            av = float(a_norm) if isinstance(a_norm, int | float) else 0.0
            if sv and av and (0.9 < (av / sv) / 1000 < 1.1 or 0.9 < (sv / av) / 1000 < 1.1):
                failures.append(f"1000x magnitude discrepancy near {surface}")
                break

        # 4 · Frequency — the per-dose vs per-day check.
        a_freq = ad.get("frequency")
        if a_freq:
            freqs = {str(f) for sd in matched if (f := sd.get("frequency"))}
            if freqs and str(a_freq) not in freqs:
                failures.append(
                    f"frequency {a_freq} not supported for {surface} (source: {sorted(freqs)})"
                )

        # 5 · Qualifier — a qualified source dose must keep its qualifier.
        # Match on DISTINCTIVE tokens (numbers, clinical terms). Matching on any
        # long token lets "dose" or "recommended" satisfy the check and silently
        # disable the rule.
        quals = [str(sd.get("qualifier", "")) for sd in matched if sd.get("qualifier")]
        if quals and not any(_qualifier_survived(q, answer_text) for q in quals):
            failures.append(f"dose {surface} stated without its qualifier ({quals[0]!r})")

    return DoseResult(passed=not failures, failures=failures)


def check_units_sanity(answer_text: str, cited_chunks: list[dict[str, object]]) -> DoseResult:
    """Standalone unit check for answers with no parseable dose structure."""
    from pharmarag.chunking.metadata import detect_units

    a_units = set(detect_units(answer_text))
    if not a_units:
        return DoseResult(passed=True)
    s_units: set[str] = set()
    for c in cited_chunks:
        up = c.get("units_present")
        s_units |= (
            set(up) if isinstance(up, list) else set(detect_units(str(c.get("display_text", ""))))
        )
    scale = {"mcg", "µg", "ug", "mg", "g", "kg", "ng"}
    missing = (a_units & scale) - s_units
    if missing and (s_units & scale):
        return DoseResult(
            False, [f"answer uses {sorted(missing)}; sources use {sorted(s_units & scale)}"]
        )
    return DoseResult(passed=True)
