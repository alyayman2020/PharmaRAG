"""K4 — deterministic LASA display gate (ADR-041). Block, no retry.

Even with perfect retrieval a small model can drift on generation, writing
"hydroxyzine" when every retrieved chunk says "hydralazine". This is the last of
three independent LASA layers:

    resolution (ADR-020) -> retrieval filter (ADR-018) -> display gate (here)

Uses the SAME matching rules as ADR-036, or "codeine" validates against
"hydrocodone" and the gate leaks exactly the errors it exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pharmarag.entity.gazetteer import Gazetteer


@dataclass(slots=True)
class LasaResult:
    passed: bool
    unauthorized: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.unauthorized:
            return ""
        return (
            f"answer names {self.unauthorized} — not resolved from the query and "
            "not present in any cited chunk"
        )


def check_drug_names(
    answer_text: str,
    resolved_rxcuis: set[str],
    cited_chunks: list[dict[str, object]],
    gazetteer: Gazetteer,
) -> LasaResult:
    """Every drug named in the answer must be authorized.

    Authorized means: resolved from this query, OR mentioned in a cited chunk.
    A legitimate answer often names drugs absent from the query — "atorvastatin
    interacts with clarithromycin" — which the second condition covers.

    Scoped to gazetteer-resolvable tokens only: class names like
    "CYP3A4 inhibitors" are not drug names and must not trip the gate.
    """
    answer_mentions = gazetteer.find_mentions(answer_text)
    if not answer_mentions:
        return LasaResult(passed=True)

    source_text = " ".join(str(c.get("display_text", "")) for c in cited_chunks)
    source_names = {m.normalized for m in gazetteer.find_mentions(source_text)}

    unauthorized = [
        m.normalized
        for m in answer_mentions
        if (m.rxcui not in resolved_rxcuis) and (m.normalized not in source_names)
    ]
    return LasaResult(passed=not unauthorized, unauthorized=sorted(set(unauthorized)))
