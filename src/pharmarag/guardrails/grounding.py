"""K2 — grounding verification (ADR-039). Milestone B11.

ADR-028 verifies citation INTEGRITY: the ID resolves and the chunk was in the
assembled context. This module verifies citation VALIDITY: does that chunk
actually support the claim?

The distinction matters and is worth stating in the README. A model can cite a
real chunk for a claim the chunk does not support. Conflating "the citation
resolves" with "the claim is grounded" is how a project overclaims.

Two layers, cheapest first:
  1. Deterministic numeric pre-check — $0, catches the highest-consequence
     failures before spending an LLM call.
  2. Per-claim entailment on gpt-5.4-mini, seeing claim + cited chunk only
     (~2k tokens, not the full 12k context). That input shrinkage is precisely
     what makes the ADR-030 model tiering affordable.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pharmarag.config import MODEL_EVALUATOR, SAFETY_TIER, settings

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# Claims per answer are few (schema keeps them discrete), so this ceiling exists
# to bound burst concurrency against the API, not to throttle a large fan-out.
_MAX_PARALLEL_CLAIMS = 8

# Label cross-references the model echoes verbatim: "[see Warnings (5.2)]",
# "section 7.1", "(2.3)". Their digits are navigation, not clinical claims.
_XREF = re.compile(
    r"\[see[^\]]*\]|\bsee\s+[A-Z][A-Za-z ]*\([\d.]+\)|\bsections?\s*[\d.]+|\(\s*\d+(?:\.\d+)*\s*\)",
    re.IGNORECASE,
)

# A number carrying a dose unit — the fabrication that must always hard-refuse.
_NUMBER_WITH_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:mg|mcg|µg|ug|g|kg|ng|ml|l|unit|units|iu|meq|mmol|%)\b",
    re.IGNORECASE,
)


class Entailment(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(slots=True)
class ClaimVerdict:
    index: int
    verdict: Entailment
    reason: str = ""
    safety_tier: int = 4
    action: str = "keep"  # keep | retry | drop | refuse


@dataclass(slots=True)
class GroundingResult:
    passed: bool
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    must_refuse: bool = False
    droppable: list[int] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def reason(self) -> str:
        bad = [v for v in self.verdicts if v.verdict is not Entailment.SUPPORTED]
        return "; ".join(f"claim {v.index}: {v.verdict.value} ({v.reason})" for v in bad)


SYSTEM = """You verify whether a CLAIM is supported by its SOURCE text.

Reply with exactly one word:
  SUPPORTED            — the source states this, including all numbers, units, \
frequencies, and qualifying conditions.
  PARTIALLY_SUPPORTED  — the source is related but the claim adds, omits, or \
alters a material detail (a dose, a unit, a frequency, or the condition under \
which it applies).
  UNSUPPORTED          — the source does not state this.

A dose stated without the condition that qualifies it (renal function band, \
hepatic class, age range) is PARTIALLY_SUPPORTED at best. Never infer beyond \
the source text."""


def _numeric_precheck(claim: str, source: str) -> tuple[str, bool] | None:
    """Numbers in the claim must appear in the source. Free, no latency.

    Returns (reason, dose_bearing) or None. `dose_bearing` separates the two very
    different failures this check catches:

      * A number carrying a UNIT ("10 mg") that is absent from the source is a
        fabricated dose — the highest-consequence error in the system. Hard refusal.
      * A bare numeral is usually incidental — a cross-reference "(7.1)", a list
        index, a count the model restated. Blocking a whole correct answer over one
        of those is a false refusal, so it degrades to PARTIALLY_SUPPORTED and takes
        the ADR-039 tier route (tier 1 refuses; the rest drop and disclose).

    Cross-references are stripped before extraction — "[see Drug Interactions (7)]"
    is label boilerplate the model echoes, not a factual numeric claim.
    """
    stripped = _XREF.sub(" ", claim)
    claim_nums = {n.replace(",", ".") for n in _NUMBER.findall(stripped)}
    if not claim_nums:
        return None
    source_nums = {n.replace(",", ".") for n in _NUMBER.findall(source)}
    missing = claim_nums - source_nums
    if not missing:
        return None
    dose_bearing = {m.group(1).replace(",", ".") for m in _NUMBER_WITH_UNIT.finditer(stripped)}
    unit_backed = sorted(missing & dose_bearing)
    if unit_backed:
        return f"dose value(s) not in source: {unit_backed}", True
    return f"numbers not in source: {sorted(missing)}", False


def _classify(claim: str, source: str) -> tuple[Entailment, str, float]:
    from pharmarag.http import openai_client

    client = openai_client()
    # gpt-5.4-mini is a reasoning model: it spends completion tokens on internal
    # reasoning BEFORE any visible text, so a tiny cap starves the verdict and the
    # call 400s — which fails closed and blocks every answer. The cap is a ceiling,
    # not a target: with minimal effort the call bills only what it uses.
    resp = client.chat.completions.create(
        model=MODEL_EVALUATOR,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"SOURCE:\n{source}\n\nCLAIM:\n{claim}"},
        ],
        max_completion_tokens=50000,
        # "low", not "none": entailment is subtle (a dose missing its renal
        # qualifier is not a surface mismatch) — the evaluator must actually
        # reason. Higher tiers add seconds per claim for marginal gain.
        reasoning_effort="low",
    )
    raw = (resp.choices[0].message.content or "").strip().upper()
    # Tolerate "SUPPORTED." / "verdict: SUPPORTED" — degrading a supported claim
    # to PARTIALLY_SUPPORTED over punctuation silently drops it from the answer.
    # Longest-first: SUPPORTED is a substring of the other two labels, so an
    # unordered scan would read UNSUPPORTED as SUPPORTED — a fail-open.
    members = sorted(Entailment.__members__, key=len, reverse=True)
    label = next((m for m in members if m in raw), "")
    verdict = Entailment(label) if label else Entailment.PARTIALLY_SUPPORTED
    usage = resp.usage
    cost = (
        getattr(usage, "prompt_tokens", 0) / 1e6 * 0.75
        + getattr(usage, "completion_tokens", 0) / 1e6 * 4.50
    )
    return verdict, "llm entailment", cost


def verify_grounding(
    claims: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    *,
    use_llm: bool = True,
) -> GroundingResult:
    """Per-claim entailment with tier-routed consequences (ADR-039).

    ADR-039's routing table — PARTIALLY_SUPPORTED had no defined behaviour in
    the original K2 spec, which meant it fell through as PASSING. A silent
    permissive default in the grounding gate:

        UNSUPPORTED                          -> retry, then refuse
        PARTIALLY_SUPPORTED + safety tier 1  -> retry, then refuse
        PARTIALLY_SUPPORTED + other          -> retry, then DROP + DISCLOSE

    Disclosure is what makes dropping acceptable. A silently stripped answer
    reads as complete and hides the failure.
    """
    total_cost = 0.0

    def route(verdict: Entailment, tier: int) -> str:
        if verdict is Entailment.SUPPORTED:
            return "keep"
        if verdict is Entailment.UNSUPPORTED or tier == 1:
            return "refuse"
        return "drop"

    def evaluate(i: int, claim: dict[str, Any]) -> tuple[ClaimVerdict, float]:
        text = str(claim.get("text", ""))
        ids = [str(c) for c in (claim.get("citation_ids") or [])]
        source = "\n".join(
            str(chunk_lookup[c].get("display_text", "")) for c in ids if c in chunk_lookup
        )

        tier = min(
            (
                SAFETY_TIER.get(str(chunk_lookup[c].get("loinc_section_code", "")), 4)
                for c in ids
                if c in chunk_lookup
            ),
            default=4,
        )

        if not source:
            return ClaimVerdict(
                i, Entailment.UNSUPPORTED, "no cited source text", tier, "refuse"
            ), 0.0

        numeric_issue = _numeric_precheck(text, source)
        if numeric_issue is not None:
            reason, dose_bearing = numeric_issue
            if dose_bearing:
                # A fabricated dose is never survivable, at any tier.
                return ClaimVerdict(i, Entailment.UNSUPPORTED, reason, tier, "refuse"), 0.0
            return (
                ClaimVerdict(
                    i,
                    Entailment.PARTIALLY_SUPPORTED,
                    reason,
                    tier,
                    route(Entailment.PARTIALLY_SUPPORTED, tier),
                ),
                0.0,
            )

        if not use_llm or not settings.openai_api_key:
            return ClaimVerdict(
                i, Entailment.SUPPORTED, "numeric pre-check only", tier, "keep"
            ), 0.0

        try:
            verdict, reason, cost = _classify(text, source)
        except Exception as exc:
            # Fail CLOSED. An unavailable evaluator is not a pass.
            return ClaimVerdict(
                i, Entailment.UNSUPPORTED, f"evaluator unavailable: {exc}", tier, "refuse"
            ), 0.0

        return ClaimVerdict(i, verdict, reason, tier, route(verdict, tier)), cost

    # Claims are independent, so the per-claim evaluator calls run concurrently.
    # Sequentially this stage cost one round trip per claim and dominated latency;
    # the verdicts and their routing are unchanged, only the wall clock is.
    if len(claims) > 1 and use_llm and settings.openai_api_key:
        with ThreadPoolExecutor(max_workers=min(len(claims), _MAX_PARALLEL_CLAIMS)) as pool:
            results = list(pool.map(lambda ic: evaluate(*ic), list(enumerate(claims))))
    else:
        results = [evaluate(i, c) for i, c in enumerate(claims)]

    verdicts = [v for v, _ in results]
    total_cost = sum(c for _, c in results)

    must_refuse = any(v.action == "refuse" for v in verdicts)
    droppable = [v.index for v in verdicts if v.action == "drop"]
    return GroundingResult(
        passed=not must_refuse and not droppable,
        verdicts=verdicts,
        must_refuse=must_refuse,
        droppable=droppable,
        cost_usd=round(total_cost, 6),
    )


def apply_drops(payload: dict[str, Any], droppable: list[int]) -> dict[str, Any]:
    """Remove unverifiable non-safety claims AND disclose the removal."""
    if not droppable:
        return payload
    claims = payload.get("claims", [])
    payload["claims"] = [c for i, c in enumerate(claims) if i not in set(droppable)]
    payload["omitted_claims_disclosed"] = len(droppable)
    payload["omission_notice"] = (
        f"{len(droppable)} supporting statement(s) were omitted because they could "
        "not be fully verified against their cited sources."
    )
    return payload
