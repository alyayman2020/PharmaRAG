"""Structured output schema (ADR-029).

Two consequences beyond parseability:
  1. Discrete claims with their own citations make grounding checkable
     claim-by-claim without an LLM.
  2. It shrinks the Safety-Evaluator's input by ~6x (claims + cited chunks, not
     full context) — which is precisely what makes model tiering affordable.

Kept SHALLOW: strict-schema adherence degrades on small models with deep nesting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

DISCLAIMER = (
    "Educational demonstration only. Not medical advice and not a clinical "
    "decision support tool. Verify against current prescribing information and "
    "consult a qualified healthcare professional."
)


class ReasonCode(str, Enum):
    NO_EVIDENCE_IN_CORPUS = "NO_EVIDENCE_IN_CORPUS"
    BELOW_CONFIDENCE_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNSAFE_QUERY = "UNSAFE_QUERY"
    AMBIGUOUS_DRUG = "AMBIGUOUS_DRUG"
    POPULATION_ONLY_SWEEP = "POPULATION_ONLY_SWEEP"
    EXPANSION_TOO_BROAD = "EXPANSION_TOO_BROAD"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"


# JSON Schema handed to the OpenAI structured-output API.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # strict:True requires EVERY key in properties to appear here. Genuinely optional
    # fields are expressed as nullable unions instead, and the nulls are stripped after
    # parsing (see synthesize) so consumers still see absence, as the refusal path does.
    "required": [
        "answer_type",
        "claims",
        "summary",
        "ddi_severity",
        "ddi_mechanism",
        "ddi_management",
    ],
    "properties": {
        "answer_type": {"type": "string", "enum": ["answer", "refusal"]},
        "summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citation_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "citation_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "ddi_severity": {"type": ["string", "null"]},
        "ddi_mechanism": {"type": ["string", "null"]},
        "ddi_management": {"type": ["string", "null"]},
    },
}

# Nullable-only because strict mode forbids omitting them; absent is the real contract.
OPTIONAL_KEYS: tuple[str, ...] = ("ddi_severity", "ddi_mechanism", "ddi_management")


def refusal(
    reason: ReasonCode,
    message: str,
    *,
    confidence: float = 0.0,
    what_would_help: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "answer_type": "refusal",
        "summary": message,
        "claims": [],
        "refusal": {
            "reason_code": reason.value,
            "confidence": round(confidence, 3),
            "what_would_help": what_would_help,
        },
        "disclaimer": DISCLAIMER,
        **extra,
    }
