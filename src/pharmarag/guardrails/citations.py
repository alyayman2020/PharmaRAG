"""ADR-028 citation INTEGRITY verifier. Deterministic, no LLM, no latency.

Verifies that citations resolve. It does NOT verify grounding — the model can
cite a real chunk for a claim that chunk does not support. That is the
Safety-Evaluator's job (K2, milestone B11). Conflating the two is how a project
overclaims, and it is the first thing a sharp reviewer probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CitationResult:
    passed: bool
    unknown_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    uncited_claims: int = 0

    @property
    def reason(self) -> str:
        if self.unknown_ids:
            return f"citation IDs not in retrieved set: {self.unknown_ids}"
        if self.dropped_ids:
            return f"cited chunks were dropped at the context cap: {self.dropped_ids}"
        if self.uncited_claims:
            return f"{self.uncited_claims} claim(s) carried no citation"
        return ""


def verify_citations(
    claims: list[dict[str, object]],
    retrieved_ids: set[str],
    assembled_ids: set[str],
) -> CitationResult:
    """Two checks, both cheap.

    1. Every cited ID exists in the retrieved set.
    2. Every cited chunk was in the ASSEMBLED context, not merely the candidate
       pool — a chunk dropped at the 8k cap (ADR-027) cannot legitimately be cited.
    """
    unknown: list[str] = []
    dropped: list[str] = []
    uncited = 0

    for claim in claims:
        ids = claim.get("citation_ids") or []
        if not isinstance(ids, list) or not ids:
            uncited += 1
            continue
        for cid in ids:
            cid = str(cid)
            if cid not in retrieved_ids:
                unknown.append(cid)
            elif cid not in assembled_ids:
                dropped.append(cid)

    return CitationResult(
        passed=not unknown and not dropped and uncited == 0,
        unknown_ids=sorted(set(unknown)),
        dropped_ids=sorted(set(dropped)),
        uncited_claims=uncited,
    )
