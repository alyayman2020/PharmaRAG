"""Tiered entity resolution (ADR-020).

Fuzzy matching is actively dangerous here. "hydralazine -> hydroxyzine" is not a
ranking error, it is an antihypertensive answered with an antihistamine. The
standard RAG move — fuzzy match, take the top hit, proceed — is the single most
harmful thing this system could do.

Tier 3 is calibrated abstention applied at the ENTITY level: refuse to retrieve
for the wrong drug rather than detect the error afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pharmarag.entity.gazetteer import Gazetteer, Mention, normalize
from pharmarag.entity.lasa import jaro_winkler


class ResolutionType(str, Enum):
    DRUG = "DRUG"
    CLASS = "CLASS"
    POPULATION_ONLY = "POPULATION_ONLY"  # scope refusal — corpus is drug-organized
    AMBIGUOUS = "AMBIGUOUS"  # Tier-3 disambiguation interrupt
    NONE = "NONE"


class Tier(str, Enum):
    EXACT = "TIER_1_EXACT"
    APPROXIMATE = "TIER_2_APPROXIMATE"
    AMBIGUOUS = "TIER_3_AMBIGUOUS"


@dataclass(slots=True)
class Resolution:
    type: ResolutionType
    rxcuis: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    tier: Tier | None = None
    confidence: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)
    substitutions: list[dict[str, str]] = field(default_factory=list)
    message: str = ""


TIER2_MIN_SCORE = 0.92
TIER2_MIN_MARGIN = 0.06

_POPULATION_WORDS = {
    "pregnancy",
    "pregnant",
    "lactation",
    "breastfeeding",
    "pediatric",
    "paediatric",
    "children",
    "geriatric",
    "elderly",
    "renal",
    "hepatic",
}


class Resolver:
    def __init__(self, gazetteer: Gazetteer, lasa: dict[str, list[str]] | None = None) -> None:
        self.gazetteer = gazetteer
        self.lasa = lasa or {}

    def resolve(self, query: str) -> Resolution:
        mentions = self.gazetteer.find_mentions(query)
        if mentions:
            rxcuis = [m.rxcui for m in mentions if m.rxcui]

            # A brand alias resolves to an identity whose name differs from the
            # matched text ("lipitor" -> "atorvastatin"). Report the INGREDIENT as
            # the resolved name — retrieval is filtered on it, so displaying the
            # brand would describe a partition the system did not search — and
            # surface the swap as a substitution (ADR-020), never silently.
            def canonical(m: Mention) -> str:
                return (self.gazetteer.canonical_name(m.rxcui) if m.rxcui else None) or m.normalized

            names = [canonical(m) for m in mentions]
            subs = [
                {"from": m.text, "to": canonical(m)}
                for m in mentions
                if canonical(m) != m.normalized
            ]
            return Resolution(
                type=ResolutionType.DRUG,
                rxcuis=list(dict.fromkeys(rxcuis)),
                names=list(dict.fromkeys(names)),
                tier=Tier.EXACT,
                confidence=1.0,
                substitutions=subs,
            )

        approx = self._approximate(query)
        if approx is not None:
            return approx

        tokens = {t.strip(".,?!") for t in normalize(query).split()}
        if tokens & _POPULATION_WORDS:
            return Resolution(
                type=ResolutionType.POPULATION_ONLY,
                message=(
                    "This corpus is organized by drug. A sweep across every label is a "
                    "report, not a retrieval — name a specific drug or drug class."
                ),
            )

        return Resolution(
            type=ResolutionType.NONE,
            message="No drug or drug class recognized. Which medication did you mean?",
        )

    def _approximate(self, query: str) -> Resolution | None:
        """Tier 2/3. Only reached when no exact match exists."""
        scored: list[tuple[str, float]] = []
        for token in normalize(query).split():
            if len(token) < 5:
                continue
            for name in self.gazetteer._map:
                s = jaro_winkler(token, name)
                if s >= 0.80:
                    scored.append((name, s))
        if not scored:
            return None

        scored.sort(key=lambda x: -x[1])
        best, best_score = scored[0]
        runner_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = best_score - runner_score
        is_lasa = bool(set(self.lasa.get(best, [])) & {n for n, _ in scored[1:3]})

        if best_score >= TIER2_MIN_SCORE and margin >= TIER2_MIN_MARGIN and not is_lasa:
            rx = self.gazetteer.rxcui(best)
            return Resolution(
                type=ResolutionType.DRUG,
                rxcuis=[rx] if rx else [],
                names=[best],
                tier=Tier.APPROXIMATE,
                confidence=best_score,
                # ADR-020: surfaced in the answer, not merely logged.
                substitutions=[{"from": query.strip(), "to": best}],
            )

        return Resolution(
            type=ResolutionType.AMBIGUOUS,
            tier=Tier.AMBIGUOUS,
            confidence=best_score,
            candidates=scored[:4],
            message="Did you mean one of these? Confirm before I retrieve anything.",
        )
