"""Node functions for the LangGraph pipeline (ADR-031, ADR-032, ADR-034).

Thin wrappers around the stages already in `pipeline.py` — the logic lives there
and is not duplicated here. This module only adapts those stages to the graph's
state contract.

ADR-032 guardrail: every node is `(PharmaState) -> dict[str, Any]`. No LangGraph
type appears in any signature below, so swapping frameworks touches `graph.py`
alone. Nodes need collaborators (resolver, reranker, embedders) that a bare
`(state)` signature cannot carry, so `make_nodes()` closes over them and returns
the plain callables the graph wants.

Three conventions hold the graph together:

  * A node that refuses sets `terminal` and `payload`, and every later node
    no-ops on `terminal`. The DAG has no terminal check between `rerank` and
    `assemble`, so short-circuiting has to be the nodes' own responsibility.
    An empty `guardrail_results` then routes to `audit` via `all({}) is True`.
  * `retry_count` increments in `synthesize`, not `guardrails`. Counting on
    failure would make `route_after_guardrails` see 1 >= MAX_RETRIES on the
    first failure and refuse without ever retrying (ADR-034 wants exactly one).
  * `guardrails` records the verdict but never writes the refusal payload —
    it cannot know whether the router will retry. `audit` converts a failed
    verdict into the refusal, and is the single exit.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from pharmarag.audit import new_query_id, write_audit
from pharmarag.config import (
    MODEL_EVALUATOR,
    MODEL_SYNTHESIS,
    PROMPT_TEMPLATE_VERSION,
    THRESHOLD_INCLUDE,
)
from pharmarag.entity.resolve import Resolution, ResolutionType
from pharmarag.generate.context import assemble
from pharmarag.generate.schema import DISCLAIMER, ReasonCode, refusal
from pharmarag.generate.synthesize import synthesize
from pharmarag.guardrails.citations import verify_citations
from pharmarag.guardrails.dose import check_doses
from pharmarag.guardrails.grounding import apply_drops, verify_grounding
from pharmarag.guardrails.input_guard import Verdict, check_input
from pharmarag.guardrails.lasa_gate import check_drug_names
from pharmarag.orchestrate.graph import (
    MAX_PAIR_ITERATIONS,
    PharmaState,
    below_threshold_refusal,
    guardrail_refusal,
)
from pharmarag.pipeline import _grounding_audit, _parents
from pharmarag.retrieve.search import EmptyCandidateSetError, hybrid_search

Node = Callable[[PharmaState], dict[str, Any]]

# A regimen, not an interaction pair. Two drugs is one pair and the ordinary
# single-retrieval path already answers it; pairwise decomposition only earns its
# keep from three drugs up (ADR-033).
COMPOUND_MIN_DRUGS = 3


def _stage(state: PharmaState, name: str, detail: str, t0: float) -> list[dict[str, Any]]:
    """Append one stage record. `stages` has no reducer, so it is rebuilt whole."""
    prior = list(state.get("stages") or [])
    prior.append({"name": name, "detail": detail, "ms": (time.perf_counter() - t0) * 1000})
    return prior


def _refuse(
    state: PharmaState, name: str, payload: dict[str, Any], detail: str, t0: float, **extra: Any
) -> dict[str, Any]:
    payload.setdefault("disclaimer", DISCLAIMER)
    return {
        "payload": payload,
        "terminal": True,
        "stages": _stage(state, name, detail, t0),
        **extra,
    }


def make_nodes(
    *,
    resolver: Any,
    client: Any,
    reranker: Any,
    embed_dense: Callable[[str], list[float]],
    embed_sparse: Callable[[str], Any],
    kg: Any = None,
    use_llm_guard: bool = True,
    use_llm_grounding: bool = True,
) -> dict[str, Node]:
    """Build the node map for `build_pipeline_graph`.

    `kg` is the property graph from `graph.build.load`. Without it there is no
    pairwise decomposition, so compound queries fall back to single retrieval.
    """

    def node_input_guard(state: PharmaState) -> dict[str, Any]:
        t = time.perf_counter()
        qid = state.get("query_id") or new_query_id()
        guard = check_input(state["question"], use_llm=use_llm_guard)
        common = {"query_id": qid, "guard_verdict": guard.verdict.value}

        if guard.verdict is Verdict.UNSAFE:
            return _refuse(
                state,
                "Input guard",
                refusal(
                    ReasonCode.UNSAFE_QUERY,
                    "I can't help with that. If you're struggling, please reach out to "
                    "someone you trust or a local crisis line.",
                ),
                f"{guard.verdict.value} ({guard.layer})",
                t,
                **common,
            )
        if guard.verdict is Verdict.PERSONAL_ADVICE:
            return _refuse(
                state,
                "Input guard",
                refusal(
                    ReasonCode.OUT_OF_SCOPE,
                    guard.reformulation,
                    what_would_help="Ask the general form: 'What does the label say about X in Y?'",
                ),
                f"{guard.verdict.value} ({guard.layer})",
                t,
                **common,
            )
        if guard.verdict is Verdict.OUT_OF_SCOPE:
            return _refuse(
                state,
                "Input guard",
                refusal(
                    ReasonCode.OUT_OF_SCOPE, "This system only answers drug-information questions."
                ),
                f"{guard.verdict.value} ({guard.layer})",
                t,
                **common,
            )

        return {
            **common,
            "stages": _stage(state, "Input guard", f"{guard.verdict.value} ({guard.layer})", t),
        }

    def node_resolve(state: PharmaState) -> dict[str, Any]:
        """Also the resume point: `disambiguate` routes back here with `user_choice`."""
        if state.get("terminal"):
            return {}
        t = time.perf_counter()

        choice = state.get("user_choice")
        # Resolve the chosen name, not the original question — the ambiguity is
        # already settled and re-resolving the question would just re-trigger it.
        res: Resolution = resolver.resolve(str(choice) if choice else state["question"])

        if res.type is ResolutionType.AMBIGUOUS and not choice:
            return {
                "resolution_type": res.type.value,
                "disambiguation_candidates": [
                    {"name": n, "score": round(s, 3)} for n, s in res.candidates
                ],
                "resolution_tier": res.tier.value if res.tier else None,
                "resolution_confidence": res.confidence,
                "stages": _stage(state, "Entity resolution", "AMBIGUOUS → interrupt", t),
            }
        if res.type is ResolutionType.POPULATION_ONLY:
            return _refuse(
                state,
                "Entity resolution",
                refusal(ReasonCode.POPULATION_ONLY_SWEEP, res.message),
                res.type.value,
                t,
                resolution_type=res.type.value,
            )
        if res.type is ResolutionType.NONE or not res.rxcuis:
            return _refuse(
                state,
                "Entity resolution",
                refusal(
                    ReasonCode.NO_EVIDENCE_IN_CORPUS,
                    res.message or "No recognized drug in this corpus.",
                ),
                res.type.value,
                t,
                resolution_type=res.type.value,
            )

        is_compound = kg is not None and len(res.rxcuis) >= COMPOUND_MIN_DRUGS
        return {
            "resolution_type": res.type.value,
            "resolved_rxcuis": res.rxcuis,
            "resolution_tier": res.tier.value if res.tier else None,
            "resolution_confidence": res.confidence,
            "substitutions": res.substitutions,
            "is_compound": is_compound,
            "stages": _stage(
                state, "Entity resolution", f"{res.type.value} → {res.names or res.rxcuis}", t
            ),
        }

    def node_decompose(state: PharmaState) -> dict[str, Any]:
        """One pair per visit — the genuine variable-iteration loop (ADR-033).

        Sets `resolved_rxcuis` to the current pair so `retrieve` stays unchanged,
        and advances the cursor so `route_after_pairs` can terminate.
        """
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        out: dict[str, Any] = {}
        pairs = [tuple(p) for p in (state.get("pairs") or [])]

        if not pairs:
            from pharmarag.graph.traverse import plan_regimen

            plan = plan_regimen(kg, list(state.get("resolved_rxcuis") or []))
            pairs = [tuple(p) for p in plan.pairs]
            out |= {
                "pairs": pairs,
                # asdict, not vars — these are slotted dataclasses with no __dict__.
                "additive_risks": [asdict(r) for r in plan.additive_risks],
                "combination_alerts": [asdict(a) for a in plan.combination_alerts],
                "regimen_capped": plan.capped,
                "not_assessed_pairs": [list(p) for p in pairs[MAX_PAIR_ITERATIONS:]],
            }

        cursor = int(state.get("pair_cursor", 0))
        if cursor >= len(pairs) or cursor >= MAX_PAIR_ITERATIONS:
            return out  # loop exhausted; router sends us to assemble
        out |= {
            "resolved_rxcuis": list(pairs[cursor]),
            "pair_cursor": cursor + 1,
            "stages": _stage(
                state, "Decompose", f"pair {cursor + 1}/{len(pairs)}: {'+'.join(pairs[cursor])}", t
            ),
        }
        return out

    def node_retrieve(state: PharmaState) -> dict[str, Any]:
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        question = state["question"]
        try:
            candidates = hybrid_search(
                client,
                dense_vector=embed_dense(question),
                sparse_vector=embed_sparse(question),
                rxcuis=list(state.get("resolved_rxcuis") or []),
            )
        except EmptyCandidateSetError as exc:
            # Mid-regimen an empty pair is not fatal — other pairs may still
            # carry evidence, so record nothing and let the loop continue.
            if state.get("is_compound"):
                return {
                    "candidates": [],
                    "stages": _stage(state, "Retrieval", "empty for this pair", t),
                }
            return _refuse(
                state,
                "Retrieval",
                refusal(
                    ReasonCode.NO_EVIDENCE_IN_CORPUS,
                    str(exc),
                    what_would_help="This drug may not be in the 1000-drug corpus.",
                ),
                "empty candidate set",
                t,
            )

        seen = list(state.get("retrieved_ids") or [])
        seen.extend(c.chunk_id for c in candidates)
        return {
            "candidates": candidates,
            "retrieved_ids": sorted(set(seen)),
            "stages": _stage(
                state, "Retrieval", f"{len(candidates)} candidates, 4 branches → RRF", t
            ),
        }

    def node_rerank(state: PharmaState) -> dict[str, Any]:
        """Scores every candidate — no skip path (ADR-025).

        The threshold gate only fires for simple queries. In a regimen the loop
        must finish before a global judgement is possible, so `assemble` gates.
        """
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        candidates = list(state.get("candidates") or [])
        fresh = reranker.score(state["question"], candidates) if candidates else []

        merged = list(state.get("scored") or [])
        known = {s.chunk_id for s in merged}
        merged.extend(s for s in fresh if s.chunk_id not in known)

        top = max((s.calibrated_score for s in merged), default=0.0)
        stages = _stage(state, "Reranking", f"{len(fresh)} scored, top={top:.2f}", t)

        if not state.get("is_compound") and (not merged or top < THRESHOLD_INCLUDE):
            return _refuse(
                state,
                "Reranking",
                below_threshold_refusal(top),
                f"{len(fresh)} scored, top={top:.2f}",
                t,
                scored=merged,
            )
        return {"scored": merged, "stages": stages}

    def node_assemble(state: PharmaState) -> dict[str, Any]:
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        scored = list(state.get("scored") or [])
        top = max((s.calibrated_score for s in scored), default=0.0)

        if not scored or top < THRESHOLD_INCLUDE:
            return _refuse(
                state,
                "Context assembly",
                below_threshold_refusal(top),
                f"nothing cleared the floor (best {top:.2f})",
                t,
            )

        blocks, dropped = assemble(
            scored, _parents([s.payload.get("parent_chunk_id", "") for s in scored])
        )
        if not blocks:
            return _refuse(
                state,
                "Context assembly",
                refusal(
                    ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
                    "Nothing cleared the relevance floor.",
                    confidence=top,
                ),
                "no blocks",
                t,
            )
        return {
            "context_blocks": blocks,
            "assembled_ids": sorted({b.chunk_id for b in blocks}),
            "stages": _stage(
                state,
                "Context assembly",
                f"{len(blocks)} parents (tier-ordered), {len(dropped)} dropped",
                t,
            ),
        }

    def node_synthesize(state: PharmaState) -> dict[str, Any]:
        """Counts the retry here, on re-entry — see the module docstring."""
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        retry = int(state.get("retry_count", 0))
        is_retry = bool(state.get("payload"))
        if is_retry:
            retry += 1

        data = synthesize(state["question"], list(state.get("context_blocks") or []))
        detail = f"{len(data.get('claims', []))} claims" + (f" (retry {retry})" if is_retry else "")
        return {
            "payload": data,
            "retry_count": retry,
            "stages": _stage(state, "Synthesis", detail, t),
        }

    def node_guardrails(state: PharmaState) -> dict[str, Any]:
        """Records the verdict only. `audit` writes the refusal — see docstring."""
        if state.get("terminal"):
            return {}
        t = time.perf_counter()
        data = dict(state.get("payload") or {})
        scored = list(state.get("scored") or [])
        assembled = set(state.get("assembled_ids") or [])
        retrieved = set(state.get("retrieved_ids") or [])
        cited = [dict(s.payload) for s in scored if s.chunk_id in assembled]
        answer_text = " ".join(
            [str(data.get("summary", ""))]
            + [str(c.get("text", "")) for c in data.get("claims", [])]
        )

        cit = verify_citations(data.get("claims", []), retrieved, assembled)
        dose = check_doses(answer_text, cited)
        lasa = check_drug_names(
            answer_text, set(state.get("resolved_rxcuis") or []), cited, resolver.gazetteer
        )
        lookup = {s.chunk_id: dict(s.payload) for s in scored}
        ground = verify_grounding(data.get("claims", []), lookup, use_llm=use_llm_grounding)

        results = {
            "citations": cit.passed,
            "dose": dose.passed,
            "lasa": lasa.passed,
            "grounding": not ground.must_refuse,
        }
        reasons = [r for r in (cit.reason, dose.reason, lasa.reason, ground.reason) if r]

        if all(results.values()) and ground.droppable:
            data = apply_drops(data, ground.droppable)

        return {
            "payload": data,
            "guardrail_results": results,
            "rejection_reasons": reasons,
            "cost_usd": float(state.get("cost_usd", 0.0)) + ground.cost_usd,
            "evaluator_verdicts": _grounding_audit(ground),
            "stages": _stage(
                state,
                "Guardrails",
                " · ".join(f"{k}={'✓' if v else '✗'}" for k, v in results.items()),
                t,
            ),
        }

    def node_audit(state: PharmaState) -> dict[str, Any]:
        """Single exit. Converts a failed guardrail verdict into the refusal."""
        t = time.perf_counter()
        payload = dict(state.get("payload") or {})
        results = state.get("guardrail_results") or {}
        reasons = list(state.get("rejection_reasons") or [])
        reason_code: str | None = payload.get("refusal", {}).get("reason_code")

        if results and not all(results.values()):
            payload = guardrail_refusal(reasons)
            reason_code = ReasonCode.GUARDRAIL_BLOCKED.value

        payload.setdefault("disclaimer", DISCLAIMER)
        # The disambiguation answer is a human-oversight event and must be logged
        # (ADR-031). audit_log has no column for it and write_audit drops unknown
        # keys silently, so it rides in structured_output rather than vanishing.
        if state.get("user_choice"):
            payload["_meta"] = {**payload.get("_meta", {}), "user_choice": state.get("user_choice")}
        stages = _stage(state, "Audit", payload.get("answer_type", "answer"), t)

        write_audit(
            {
                "query_id": state.get("query_id") or new_query_id(),
                "session_id": state.get("session_id"),
                "parent_query_id": state.get("parent_query_id"),
                "raw_query": state.get("question", ""),
                "normalized_query": str(state.get("question", "")).lower().strip(),
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "guard_verdict": state.get("guard_verdict"),
                "resolved_rxcuis": state.get("resolved_rxcuis"),
                "resolution_tier": state.get("resolution_tier"),
                "resolution_confidence": state.get("resolution_confidence"),
                "substitutions_surfaced": state.get("substitutions"),
                "expansion_overflow": state.get("expansion_overflow"),
                "retrieved_chunk_ids": state.get("retrieved_ids"),
                "context_assembled_chunk_ids": state.get("assembled_ids"),
                "guardrail_results": results,
                "rejection_reasons": reasons,
                "retry_count": state.get("retry_count", 0),
                "evaluator_verdict": state.get("evaluator_verdicts"),
                "evaluator_model_version": MODEL_EVALUATOR,
                "synthesis_model_version": MODEL_SYNTHESIS,
                "structured_output": payload,
                "final_action": payload.get("answer_type"),
                "reason_code": reason_code,
                "latency_ms_by_stage": {s["name"]: round(float(s["ms"]), 1) for s in stages},
                "cost_usd": float(state.get("cost_usd", 0.0))
                + float(payload.get("_meta", {}).get("cost_usd") or 0.0),
            }
        )
        return {"payload": payload, "terminal": True, "stages": stages}

    return {
        "input_guard": node_input_guard,
        "resolve": node_resolve,
        "disambiguate": _disambiguate,
        "decompose": node_decompose,
        "retrieve": node_retrieve,
        "rerank": node_rerank,
        "assemble": node_assemble,
        "synthesize": node_synthesize,
        "guardrails": node_guardrails,
        "audit": node_audit,
    }


def _disambiguate(state: PharmaState) -> dict[str, Any]:
    """Indirection so the interrupt is imported lazily (LangGraph stays optional)."""
    from pharmarag.orchestrate.graph import disambiguation_interrupt

    return disambiguation_interrupt(state)
