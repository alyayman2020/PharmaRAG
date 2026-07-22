"""Milestone B12 — LangGraph orchestration (ADR-031, ADR-032, ADR-034).

ADR-031 audited where agency actually earns its keep and found three places:

  1. Compound-query decomposition — variable iteration over drug pairs
  2. Evaluator -> regenerate — bounded revision when the safety gate rejects
  3. Tier-3 disambiguation — SUSPEND/RESUME across user turns

Everything else is deterministic and stays that way. Calling the whole pipeline
"agentic" would be overclaiming, and a technical reviewer spots that in thirty
seconds. Under EU AI Act Article 14 human-oversight thinking, determinism is a
COMPLIANCE VIRTUE, not a limitation — lead with that.

Node functions are plain typed callables with no LangGraph types in their
signatures (ADR-032 guardrail). If LangGraph becomes friction, only this file
changes.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pharmarag.config import THRESHOLD_INCLUDE
from pharmarag.generate.schema import ReasonCode, refusal

MAX_RETRIES = 1  # ADR-034: one retry, then refuse. Never a third attempt.
MAX_PAIR_ITERATIONS = 20  # ADR-033 loop ceiling


def _merge(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    return {**(a or {}), **(b or {})}


class PharmaState(TypedDict, total=False):
    """Typed graph state. Checkpointed to SQLite so disambiguation can resume."""

    question: str
    query_id: str
    session_id: str
    parent_query_id: str | None

    guard_verdict: str
    resolution_type: str
    resolved_rxcuis: list[str]
    resolution_tier: str
    resolution_confidence: float
    substitutions: list[dict[str, str]]
    disambiguation_candidates: list[dict[str, Any]]
    user_choice: str | None

    expansion_overflow: bool
    expansion_message: str

    is_compound: bool
    pairs: list[tuple[str, str]]
    pair_cursor: int
    additive_risks: list[dict[str, Any]]
    combination_alerts: list[dict[str, Any]]
    regimen_capped: bool
    not_assessed_pairs: list[list[str]]

    candidates: list[Any]
    scored: list[Any]
    context_blocks: list[Any]
    retrieved_ids: list[str]
    assembled_ids: list[str]

    payload: dict[str, Any]
    retry_count: int
    rejection_reasons: list[str]
    guardrail_results: Annotated[dict[str, Any], _merge]
    evaluator_verdicts: list[dict[str, Any]]  # per-claim K2 verdicts (ADR-039)

    stages: list[dict[str, Any]]
    cost_usd: float
    terminal: bool


# --------------------------------------------------------------------------- routing
def route_after_guard(state: PharmaState) -> Literal["resolve", "end"]:
    return "end" if state.get("terminal") else "resolve"


def route_after_resolution(
    state: PharmaState,
) -> Literal["disambiguate", "decompose", "retrieve", "end"]:
    if state.get("terminal"):
        return "end"
    if state.get("resolution_type") == "AMBIGUOUS":
        return "disambiguate"
    if state.get("is_compound"):
        return "decompose"
    return "retrieve"


def route_after_pairs(state: PharmaState) -> Literal["decompose", "synthesize"]:
    """The compound-query loop — genuine variable iteration (ADR-033)."""
    cursor = int(state.get("pair_cursor", 0))
    pairs = state.get("pairs") or []
    if cursor >= len(pairs) or cursor >= MAX_PAIR_ITERATIONS:
        return "synthesize"
    return "decompose"


def route_after_guardrails(state: PharmaState) -> Literal["synthesize", "end"]:
    """The evaluator loop — bounded revision (ADR-034).

    Retry is ROUTED BY REJECTION REASON. Retrying synthesis on evidence that was
    insufficient to begin with just burns a call, so that case refuses
    immediately. And retry NEVER re-retrieves: changing the evidence base
    mid-run makes the failure untraceable and breaks the audit record.
    """
    results = state.get("guardrail_results") or {}
    if all(results.values()):
        return "end"
    reasons = " ".join(state.get("rejection_reasons") or []).lower()
    if "insufficient" in reasons or "no cited source" in reasons:
        return "end"
    if int(state.get("retry_count", 0)) >= MAX_RETRIES:
        return "end"
    return "synthesize"


# --------------------------------------------------------------------------- graph
def build_pipeline_graph(nodes: dict[str, Any], *, checkpointer: Any = None) -> Any:
    """Assemble the DAG. `nodes` maps names to plain callables.

    Injecting the callables keeps LangGraph confined to this file — the ADR-032
    guardrail. Every node has signature (PharmaState) -> dict[str, Any].
    """
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(PharmaState)

    for name in (
        "input_guard",
        "resolve",
        "disambiguate",
        "decompose",
        "retrieve",
        "rerank",
        "assemble",
        "synthesize",
        "guardrails",
        "audit",
    ):
        g.add_node(name, nodes[name])

    g.add_edge(START, "input_guard")
    g.add_conditional_edges(
        "input_guard", route_after_guard, {"resolve": "resolve", "end": "audit"}
    )
    g.add_conditional_edges(
        "resolve",
        route_after_resolution,
        {
            "disambiguate": "disambiguate",
            "decompose": "decompose",
            "retrieve": "retrieve",
            "end": "audit",
        },
    )
    # Suspend/resume: the interrupt lives inside the disambiguate node.
    g.add_edge("disambiguate", "resolve")

    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges(
        "rerank", route_after_pairs, {"decompose": "decompose", "synthesize": "assemble"}
    )
    g.add_edge("assemble", "synthesize")
    g.add_edge("synthesize", "guardrails")
    g.add_conditional_edges(
        "guardrails", route_after_guardrails, {"synthesize": "synthesize", "end": "audit"}
    )
    g.add_edge("audit", END)

    return g.compile(checkpointer=checkpointer)


def make_checkpointer(db_path: str) -> Any:
    """SqliteSaver on the ADR-019 store. One persistence layer, not two."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(db_path)


# --------------------------------------------------------------------------- interrupt
def disambiguation_interrupt(state: PharmaState) -> dict[str, Any]:
    """Tier-3 abstention as a suspend/resume interrupt (ADR-020, ADR-031).

    This is the third control-flow pattern — not a loop. The graph SUSPENDS,
    the user answers, and execution RESUMES with the original retrieval state
    intact. Without checkpointing, the follow-up either re-runs the whole
    pipeline or loses the original query context.

    The user's choice is a HUMAN OVERSIGHT EVENT and belongs in the audit log.
    """
    from langgraph.types import interrupt

    candidates = state.get("disambiguation_candidates") or []
    choice = interrupt(
        {
            "type": "disambiguation",
            "question": "Which drug did you mean?",
            "candidates": candidates,
            "why": (
                "These names are look-alike/sound-alike. I will not guess — "
                "retrieving for the wrong drug is a clinical error, not a "
                "ranking error."
            ),
        }
    )
    return {"user_choice": str(choice), "resolution_type": "DRUG"}


# --------------------------------------------------------------------------- helpers
def below_threshold_refusal(top_score: float) -> dict[str, Any]:
    return refusal(
        ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
        f"No retrieved passage exceeded the relevance threshold (best {top_score:.2f}).",
        confidence=top_score,
        what_would_help=(
            "Name the specific section — dosing, interactions, " "or contraindications."
        ),
    )


def guardrail_refusal(reasons: list[str]) -> dict[str, Any]:
    return refusal(
        ReasonCode.GUARDRAIL_BLOCKED,
        "A generated answer failed verification and was blocked.",
        what_would_help="; ".join(reasons),
    )


def threshold() -> float:
    return THRESHOLD_INCLUDE
