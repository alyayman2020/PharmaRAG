"""End-to-end Track A pipeline.

Deliberately a plain function with explicit stages, not a LangGraph graph.
LangGraph (ADR-032) lands at milestone B12 once the two bounded loops and the
suspend/resume interrupt exist. Wiring the framework before there is any
branching to orchestrate would be ceremony.

Stage order is load-bearing — see the docstring on each stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pharmarag import cache
from pharmarag.audit import new_query_id, write_audit
from pharmarag.chunking.metadata import detect_population_tags
from pharmarag.config import (
    MODEL_EVALUATOR,
    MODEL_SYNTHESIS,
    PROMPT_TEMPLATE_VERSION,
    THRESHOLD_INCLUDE,
)
from pharmarag.db import session
from pharmarag.entity.resolve import Resolution, ResolutionType, Resolver
from pharmarag.generate.context import assemble
from pharmarag.generate.schema import DISCLAIMER, ReasonCode, refusal
from pharmarag.generate.synthesize import synthesize
from pharmarag.guardrails.citations import verify_citations
from pharmarag.guardrails.dose import check_doses
from pharmarag.guardrails.grounding import GroundingResult, apply_drops, verify_grounding
from pharmarag.guardrails.input_guard import Verdict, check_input
from pharmarag.guardrails.lasa_gate import check_drug_names
from pharmarag.intent import classify as classify_intent
from pharmarag.retrieve.search import EmptyCandidateSetError, hybrid_search


@dataclass(slots=True)
class Stage:
    name: str
    detail: str
    ms: float


@dataclass(slots=True)
class Answer:
    payload: dict[str, Any]
    stages: list[Stage] = field(default_factory=list)
    query_id: str = ""
    context_ids: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    # Per-guardrail verdicts, surfaced so a UI can show WHY an answer is
    # trustworthy. Empty when the query never reached the guardrail stage.
    guardrails: dict[str, bool] = field(default_factory=dict)


def _grounding_audit(ground: GroundingResult) -> list[dict[str, Any]]:
    """Per-claim K2 verdicts for the audit record (ADR-047 reconstruction)."""
    return [
        {
            "index": v.index,
            "verdict": v.verdict.value,
            "safety_tier": v.safety_tier,
            "action": v.action,
            "reason": v.reason,
        }
        for v in ground.verdicts
    ]


def _parents(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    with session() as conn:
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT parent_chunk_id, text FROM parents WHERE parent_chunk_id IN ({marks})",
            tuple(ids),
        ).fetchall()
    return {r["parent_chunk_id"]: r["text"] for r in rows}


def answer_question(
    question: str,
    *,
    resolver: Resolver,
    client: Any,
    reranker: Any,
    embed_dense: Callable[[str], list[float]],
    embed_sparse: Callable[[str], Any],
    use_llm_guard: bool = True,
    on_stage: Callable[[Stage], None] | None = None,
) -> Answer:
    stages: list[Stage] = []
    qid = new_query_id()
    t_all = time.perf_counter()

    def stage(name: str, detail: str, t0: float) -> None:
        s = Stage(name, detail, (time.perf_counter() - t0) * 1000)
        stages.append(s)
        if on_stage:
            on_stage(s)

    def finish(
        payload: dict[str, Any], reason: str | None, *, cacheable: bool = False, **audit: Any
    ) -> Answer:
        payload.setdefault("disclaimer", DISCLAIMER)
        # `cacheable` is opt-in per call site, never a default. Only sites BELOW the
        # 2b lookup may pass it — `key` is bound there, and every K1/resolution exit
        # above returns before it exists, so a guard verdict cannot reach cache.put()
        # even by mistake. `_meta` is stripped: a replayed answer must not report a
        # fresh cost or latency it did not incur.
        if cacheable:
            cache.put(key, {k: v for k, v in payload.items() if k != "_meta"})
        write_audit(
            {
                "query_id": qid,
                "raw_query": question,
                "normalized_query": question.lower().strip(),
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "final_action": payload.get("answer_type"),
                "reason_code": reason,
                "structured_output": payload,
                "latency_ms_by_stage": {s.name: round(s.ms, 1) for s in stages},
                "synthesis_model_version": MODEL_SYNTHESIS,
                **audit,
            }
        )
        return Answer(
            payload,
            stages,
            qid,
            audit.get("context_assembled_chunk_ids", []),
            audit.get("retrieved_chunk_ids", []),
            audit.get("guardrail_results", {}),
        )

    # 1 · INPUT GUARD — first, always, never cached (ADR-038).
    t = time.perf_counter()
    guard = check_input(question, use_llm=use_llm_guard)
    stage("Input guard", f"{guard.verdict.value} ({guard.layer})", t)
    if guard.verdict is Verdict.UNSAFE:
        return finish(
            refusal(
                ReasonCode.UNSAFE_QUERY,
                "I can't help with that. If you're struggling, please reach out to "
                "someone you trust or a local crisis line.",
            ),
            ReasonCode.UNSAFE_QUERY.value,
            guard_verdict=guard.verdict.value,
        )
    if guard.verdict is Verdict.PERSONAL_ADVICE:
        return finish(
            refusal(
                ReasonCode.OUT_OF_SCOPE,
                guard.reformulation,
                what_would_help="Ask the general form: 'What does the label say about X in Y?'",
            ),
            ReasonCode.OUT_OF_SCOPE.value,
            guard_verdict=guard.verdict.value,
        )
    if guard.verdict is Verdict.OUT_OF_SCOPE:
        return finish(
            refusal(
                ReasonCode.OUT_OF_SCOPE, "This system only answers drug-information questions."
            ),
            ReasonCode.OUT_OF_SCOPE.value,
            guard_verdict=guard.verdict.value,
        )

    # 2 · ENTITY RESOLUTION — typed, with a Tier-3 abstention band (ADR-020).
    t = time.perf_counter()
    res: Resolution = resolver.resolve(question)
    stage("Entity resolution", f"{res.type.value} → {res.names or res.rxcuis}", t)

    if res.type is ResolutionType.AMBIGUOUS:
        payload = refusal(ReasonCode.AMBIGUOUS_DRUG, res.message, confidence=res.confidence)
        payload["candidates"] = [{"name": n, "score": round(s, 3)} for n, s in res.candidates]
        return finish(
            payload,
            ReasonCode.AMBIGUOUS_DRUG.value,
            resolution_tier=res.tier.value if res.tier else None,
        )
    if res.type is ResolutionType.POPULATION_ONLY:
        return finish(
            refusal(ReasonCode.POPULATION_ONLY_SWEEP, res.message),
            ReasonCode.POPULATION_ONLY_SWEEP.value,
        )
    if res.type is ResolutionType.NONE or not res.rxcuis:
        return finish(
            refusal(
                ReasonCode.NO_EVIDENCE_IN_CORPUS,
                res.message or "No recognized drug in this corpus.",
            ),
            ReasonCode.NO_EVIDENCE_IN_CORPUS.value,
        )

    # 2b · CACHE LOOKUP (ADR-050) — after resolution so the key is the CANONICAL
    # resolved query, before retrieval so a hit skips retrieval, reranking, and
    # synthesis entirely.
    #
    # Placement is the safety property, not an optimisation detail. K1 runs at
    # stage 1 and every path above returns before reaching here, so the guard is
    # re-evaluated on every single query and its verdict is never what gets
    # stored or replayed. A jailbreak that slips past K1 once is answered once;
    # it does not become a permanently cached SAFE verdict.
    #
    # Version fields are inside cache_key() itself and cannot be omitted from a
    # call site, so a corpus, graph, calibrator, or prompt-template bump changes
    # every key and stale answers become unreachable rather than merely unlikely.
    t = time.perf_counter()
    intent = classify_intent(question)
    tags = detect_population_tags(question)
    key = cache.cache_key(intent=intent.value, rxcuis=res.rxcuis, population_tags=tags)
    if (hit := cache.get(key)) is not None:
        stage("Cache", f"HIT ({intent.value}{'/' + ','.join(tags) if tags else ''})", t)
        return finish(
            hit,
            hit.get("refusal", {}).get("reason_code"),
            guard_verdict=guard.verdict.value,
            resolved_rxcuis=res.rxcuis,
            resolution_tier=res.tier.value if res.tier else None,
        )
    stage("Cache", f"MISS ({intent.value})", t)

    # 3 · RETRIEVAL — pre-filtered. Empty set is a HARD REFUSAL (ADR-023).
    t = time.perf_counter()
    try:
        candidates = hybrid_search(
            client,
            dense_vector=embed_dense(question),
            sparse_vector=embed_sparse(question),
            rxcuis=res.rxcuis,
        )
    except EmptyCandidateSetError as exc:
        stage("Retrieval", "empty candidate set", t)
        return finish(
            refusal(
                ReasonCode.NO_EVIDENCE_IN_CORPUS,
                str(exc),
                what_would_help="This drug may not be in the 1000-drug corpus.",
            ),
            ReasonCode.NO_EVIDENCE_IN_CORPUS.value,
            cacheable=True,
            resolved_rxcuis=res.rxcuis,
        )
    stage("Retrieval", f"{len(candidates)} candidates, 4 branches → RRF", t)

    # 4 · RERANK — scores EVERY candidate. No skip path (ADR-025).
    t = time.perf_counter()
    scored = reranker.score(question, candidates)
    top = scored[0].calibrated_score if scored else 0.0
    stage("Reranking", f"{len(scored)} scored, top={top:.2f}", t)

    if not scored or top < THRESHOLD_INCLUDE:
        return finish(
            refusal(
                ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
                f"No retrieved passage exceeded the relevance threshold (best {top:.2f}).",
                confidence=top,
                what_would_help="Try naming the specific section — dosing, interactions, contraindications.",
            ),
            ReasonCode.BELOW_CONFIDENCE_THRESHOLD.value,
            cacheable=True,
            resolved_rxcuis=res.rxcuis,
            retrieved_chunk_ids=[c.chunk_id for c in candidates],
            reranker_scores=[round(s.raw_score, 4) for s in scored[:10]],
        )

    # 5 · CONTEXT ASSEMBLY — safety-tier order, parent dedup, 8k cap (ADR-027).
    t = time.perf_counter()
    blocks, dropped = assemble(
        scored, _parents([s.payload.get("parent_chunk_id", "") for s in scored])
    )
    stage("Context assembly", f"{len(blocks)} parents (tier-ordered), {len(dropped)} dropped", t)
    if not blocks:
        return finish(
            refusal(
                ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
                "Nothing cleared the relevance floor.",
                confidence=top,
            ),
            ReasonCode.BELOW_CONFIDENCE_THRESHOLD.value,
            cacheable=True,
        )

    # 6 · SYNTHESIS
    t = time.perf_counter()
    data = synthesize(question, blocks)
    stage("Synthesis", f"{len(data.get('claims', []))} claims", t)

    # 7 · GUARDRAILS — fail closed, always (ADR-028/040/041).
    t = time.perf_counter()
    assembled_ids = {b.chunk_id for b in blocks}
    retrieved_ids = {c.chunk_id for c in candidates}
    # Guardrails must verify against the text the model actually SAW — the
    # rendered parent blocks — not the child-chunk excerpts that scored in
    # retrieval. A block's id is the child's, but its text is the parent's
    # (ADR-011); checking claims against the child excerpt systematically
    # blocks claims drawn from the rest of the parent the model was given.
    block_text = {b.chunk_id: b.text for b in blocks}
    cited = []
    for s in scored:
        if s.chunk_id in assembled_ids:
            p = dict(s.payload)
            p["display_text"] = block_text[s.chunk_id]
            cited.append(p)
    answer_text = " ".join(
        [str(data.get("summary", ""))] + [str(c.get("text", "")) for c in data.get("claims", [])]
    )

    cit = verify_citations(data.get("claims", []), retrieved_ids, assembled_ids)
    dose = check_doses(answer_text, cited)
    lasa = check_drug_names(answer_text, set(res.rxcuis), cited, resolver.gazetteer)

    # K2 grounding (ADR-039). Runs INSIDE this block, not after it: the fail-closed
    # check below is the only thing that turns a failed guardrail into a refusal, so a
    # grounding verdict computed after it would be audited but never enforced.
    # `passed` here is `not must_refuse` deliberately — droppable claims are a
    # drop-and-disclose outcome, not a blocked answer.
    lookup = {s.chunk_id: dict(s.payload) for s in scored}
    for cid, txt in block_text.items():
        if cid in lookup:
            lookup[cid]["display_text"] = txt
    ground = verify_grounding(data.get("claims", []), lookup, use_llm=True)

    results = {
        "citations": cit.passed,
        "dose": dose.passed,
        "lasa": lasa.passed,
        "grounding": not ground.must_refuse,
    }
    stage("Guardrails", " · ".join(f"{k}={'✓' if v else '✗'}" for k, v in results.items()), t)

    # NOT cacheable, deliberately. This refusal is derived from nondeterministic
    # synthesis and LLM grounding, so caching it would let one transient failure
    # permanently block a query that succeeds on retry — until a version bump.
    if not all(results.values()):
        reasons = [r for r in (cit.reason, dose.reason, lasa.reason, ground.reason) if r]
        return finish(
            refusal(
                ReasonCode.GUARDRAIL_BLOCKED,
                "A generated answer failed verification and was blocked.",
                what_would_help="; ".join(reasons),
            ),
            ReasonCode.GUARDRAIL_BLOCKED.value,
            guardrail_results=results,
            rejection_reasons=reasons,
            evaluator_verdict=_grounding_audit(ground),
            evaluator_model_version=MODEL_EVALUATOR,
            cost_usd=(data.get("_meta", {}).get("cost_usd") or 0.0) + ground.cost_usd,
            resolved_rxcuis=res.rxcuis,
            retrieved_chunk_ids=sorted(retrieved_ids),
            context_assembled_chunk_ids=sorted(assembled_ids),
        )

    # ADR-039: non-safety claims that could not be fully verified are removed and the
    # removal is disclosed in the payload. Silent stripping would read as a complete answer.
    if ground.droppable:
        data = apply_drops(data, ground.droppable)

    data["sources"] = [
        {
            "id": b.chunk_id,
            "drug": b.ingredient_name,
            "section": b.section_path,
            "effective_time": b.effective_time,
            "url": b.source_url,
            "score": round(b.score, 3),
            "text": b.text,
        }
        for b in blocks
    ]
    data["substitutions_surfaced"] = res.substitutions
    data["_meta"] = {**data.get("_meta", {}), "total_ms": (time.perf_counter() - t_all) * 1000}

    # A model-generated refusal must audit under its typed reason, not None —
    # an untyped refusal is the "" bucket that polluted the first scorecard.
    # It is also NOT cacheable, for the same reason guardrail blocks are not:
    # synthesis is nondeterministic, and caching one transient "sources don't
    # say" permanently blocks a query that succeeds on retry. The refusals
    # cached above (empty retrieval, below-threshold) are deterministic ones.
    is_model_refusal = data.get("answer_type") == "refusal"
    final_reason = data.get("refusal", {}).get("reason_code") if is_model_refusal else None

    return finish(
        data,
        final_reason,
        cacheable=not is_model_refusal,
        guard_verdict=guard.verdict.value,
        resolved_rxcuis=res.rxcuis,
        resolution_tier=res.tier.value if res.tier else None,
        resolution_confidence=res.confidence,
        substitutions_surfaced=res.substitutions,
        retrieved_chunk_ids=sorted(retrieved_ids),
        chunk_sha256=[c.get("content_sha256") for c in cited],
        reranker_scores=[round(s.raw_score, 4) for s in scored[:10]],
        calibrated_scores=[round(s.calibrated_score, 4) for s in scored[:10]],
        context_assembled_chunk_ids=sorted(assembled_ids),
        guardrail_results=results,
        evaluator_verdict=_grounding_audit(ground),
        evaluator_model_version=MODEL_EVALUATOR,
        prompt_hash=data.get("_meta", {}).get("prompt_hash"),
        cost_usd=(data.get("_meta", {}).get("cost_usd") or 0.0) + ground.cost_usd,
    )
