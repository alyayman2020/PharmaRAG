"""K1 — layered input guards (ADR-038).

Two layers because neither is sufficient alone. Regex alone is trivially
bypassed; an LLM alone is jailbreakable. Layered, a bypass must beat both — and
per ADR-005 even a successful bypass finds NO overdosage content in the index to
retrieve. Three independent failures required.

NEVER CACHE THE VERDICT. Caching a SAFE verdict means one successful jailbreak
is cached as permanently safe: the cache becomes a persistence mechanism for the
bypass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from pharmarag.config import MODEL_GUARD, settings


class Verdict(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"  # harm-seeking
    PERSONAL_ADVICE = "PERSONAL_ADVICE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(slots=True)
class GuardResult:
    verdict: Verdict
    reason: str = ""
    layer: str = ""
    reformulation: str = ""


_HARM_PATTERNS = [
    re.compile(r"\b(lethal|fatal|deadly)\s+(dose|amount|quantity)", re.I),
    re.compile(
        r"\bhow\s+(much|many)\b[^.?!]{0,40}\b(kill|die|death|overdose|end (my|his|her) life)", re.I
    ),
    re.compile(r"\boverdose\s+on\b", re.I),
    re.compile(r"\b(kill|poison)\s+(myself|himself|herself|someone|a person)", re.I),
    re.compile(r"\bundetectable\b[^.?!]{0,30}\b(poison|dose)", re.I),
]

_PERSONAL_PATTERNS = [
    re.compile(
        r"\b(i|my|we|our)\b[^.?!]{0,50}\b(should i take|can i take|how much should i|do i need)",
        re.I,
    ),
    re.compile(r"\bi\s+(have|has|am|was)\b[^.?!]{0,50}\b(how much|what dose|should i)", re.I),
    re.compile(r"\bshould i\s+(take|stop|start|increase|decrease)\b", re.I),
    re.compile(r"\bis it safe for me\b", re.I),
]

SYSTEM = """Classify a query for a drug-information system into exactly one label:

SAFE — an educational question about drugs, interactions, contraindications, or dosing.
UNSAFE — seeks information to cause harm (lethal doses, overdose facilitation, poisoning).
PERSONAL_ADVICE — asks what THIS person should personally do with a medication.
OUT_OF_SCOPE — not about drugs or pharmacology at all.

Distinguishing example:
  "What is the max metformin dose in CKD stage 3?"  -> SAFE (educational)
  "I have CKD, how much metformin should I take?"   -> PERSONAL_ADVICE

Reply with the single label and nothing else."""


def _deterministic(query: str) -> GuardResult | None:
    for rx in _HARM_PATTERNS:
        if rx.search(query):
            return GuardResult(Verdict.UNSAFE, f"matched harm pattern: {rx.pattern[:40]}", "regex")
    for rx in _PERSONAL_PATTERNS:
        if rx.search(query):
            return GuardResult(
                Verdict.PERSONAL_ADVICE,
                "first-person clinical framing",
                "regex",
                reformulation=_reformulate(query),
            )
    return None


def _reformulate(query: str) -> str:
    """ADR-038: refuse the frame, OFFER the reformulation, never auto-answer it.

    Auto-answering the general version while claiming to refuse the personal one
    is a refusal in name only. Making the user restate it keeps the boundary real
    while leaving the information fully available.
    """
    return (
        "I can't advise on your personal dosing. I can tell you what the FDA label "
        "says — ask the general form of the question directly, and talk to your "
        "prescriber about your own case."
    )


def check_input(query: str, *, use_llm: bool = True) -> GuardResult:
    det = _deterministic(query)
    if det is not None:
        return det
    if not use_llm or not settings.openai_api_key:
        return GuardResult(Verdict.SAFE, "regex layer only (no API key)", "regex")

    try:
        from pharmarag.http import openai_client

        client = openai_client()
        # gpt-5.4 is a reasoning model — a tiny completion cap starves the visible
        # verdict (tokens go to reasoning first). Ceiling is high, effort minimal:
        # the call bills only what it uses, and the verdict always arrives.
        resp = client.chat.completions.create(
            model=MODEL_GUARD,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
            max_completion_tokens=50000,
            # "low": the guard must reason about intent (paraphrased harm-seeking),
            # but it sits on every query's critical path — higher effort adds
            # seconds of latency for marginal classification gain.
            reasoning_effort="low",
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if not raw:
            # An empty guard response is an unavailable guard, not a SAFE verdict.
            return GuardResult(Verdict.OUT_OF_SCOPE, "guard returned empty output", "llm-error")
        # Longest-first: SAFE is a substring of UNSAFE — an unordered scan
        # would read a harm verdict as safe.
        members = sorted(Verdict.__members__, key=len, reverse=True)
        label = next((m for m in members if m in raw), "")
        verdict = Verdict(label) if label else Verdict.SAFE
    except Exception as exc:
        # Fail CLOSED on guard failure, never open.
        return GuardResult(Verdict.OUT_OF_SCOPE, f"guard unavailable: {exc}", "llm-error")

    return GuardResult(
        verdict,
        "llm classification",
        MODEL_GUARD,
        reformulation=_reformulate(query) if verdict is Verdict.PERSONAL_ADVICE else "",
    )
