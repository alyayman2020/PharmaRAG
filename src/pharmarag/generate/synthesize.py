"""Synthesis against assembled context (ADR-029, ADR-030).

Synthesis runs on nano — the only call that sees the full ~12k context, so token
volume x price actually bites. That is defensible ONLY because the schema
constrains the task to extraction and restatement, not open-ended clinical
reasoning.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pharmarag.config import MODEL_SYNTHESIS, PROMPT_TEMPLATE_VERSION
from pharmarag.generate.context import ContextBlock, render
from pharmarag.generate.schema import ANSWER_SCHEMA, DISCLAIMER, OPTIONAL_KEYS, ReasonCode

SYSTEM_PROMPT = """You answer questions about drug interactions, contraindications, \
and dosing using ONLY the numbered SOURCE blocks provided.

Absolute rules:
1. Use ONLY the SOURCE blocks. You have no other knowledge. If the sources do \
not contain the answer, set answer_type to "refusal".
2. Every claim MUST cite at least one source id, copied exactly from the \
"id=" field of the SOURCE block it came from.
3. Never state a dose without the condition that qualifies it (renal function \
band, hepatic class, age range). A dose without its qualifier is a wrong dose.
4. Copy numbers, units, and frequencies exactly as written in the source. Never \
convert, round, or infer.
5. Do not name any drug that does not appear in the sources.
6. Do not soften or omit contraindications or boxed warnings that appear in the \
sources.

Return JSON matching the provided schema. Keep each claim to one factual \
statement so it can be verified independently."""


def build_prompt(question: str, blocks: list[ContextBlock]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{render(blocks)}\n\nQUESTION: {question}"},
    ]


def prompt_hash(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def synthesize(question: str, blocks: list[ContextBlock]) -> dict[str, Any]:
    from pharmarag.http import openai_client

    messages = build_prompt(question, blocks)
    client = openai_client()

    resp = client.chat.completions.create(
        model=MODEL_SYNTHESIS,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pharmarag_answer", "strict": True, "schema": ANSWER_SCHEMA},
        },
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        data = {"answer_type": "refusal", "summary": "Malformed model output.", "claims": []}

    # Strict mode forces the model to emit these keys; drop the nulls so a non-DDI answer
    # looks like it always has — key absent rather than present-but-null.
    for key in OPTIONAL_KEYS:
        if data.get(key) is None:
            data.pop(key, None)

    # A model-generated refusal (rule 1: sources lack the answer) carries no typed
    # reason — stamp it, or the UI badge and the eval buckets read as empty.
    if data.get("answer_type") == "refusal" and not data.get("refusal"):
        data["refusal"] = {
            "reason_code": ReasonCode.NO_EVIDENCE_IN_CORPUS.value,
            "confidence": 0.0,
            "what_would_help": "The retrieved label sections do not contain this information.",
        }

    usage = resp.usage
    data["disclaimer"] = DISCLAIMER
    data["_meta"] = {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(messages),
        "model": MODEL_SYNTHESIS,
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "cost_usd": round(
            getattr(usage, "prompt_tokens", 0) / 1e6 * 0.20
            + getattr(usage, "completion_tokens", 0) / 1e6 * 1.25,
            6,
        ),
    }
    return data
