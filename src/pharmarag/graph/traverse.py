"""Milestones B8 + B13 — graph traversal and additive-risk detection.

Two rules govern everything here:

  1. GRAPH EDGES ARE NEVER EVIDENCE (ADR-037). The graph decides what to LOOK
     AT; the corpus decides what to SAY. A system that cited
     `simvastatin --is_substrate_of--> CYP3A4` as evidence would be citing its
     own inference — the exact failure the project exists to avoid.

  2. EVERY EXPANSION HAS A CEILING (ADR-037). "What interacts with simvastatin?"
     expands to CYP3A4 substrate -> inhibitors of CYP3A4, which can be 100+
     drugs. A filter of 100 RxCUIs is barely a filter — it approaches the
     unfiltered search ADR-023 forbids, arriving through the front door.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import networkx as nx

from pharmarag.config import MAX_EXPANSION_RXCUIS
from pharmarag.graph.build import COMBINATION_ALERTS, RISK_CLASSES

_RISK_BY_ID = {rc.id: rc for rc in RISK_CLASSES}


# --------------------------------------------------------------------------- expansion
@dataclass(slots=True)
class Expansion:
    rxcuis: list[str]
    overflow: bool = False
    total_available: int = 0
    via: list[str] = field(default_factory=list)
    members_preview: list[tuple[str, str]] = field(default_factory=list)

    @property
    def message(self) -> str:
        if not self.overflow:
            return ""
        names = ", ".join(n for _, n in self.members_preview[:8])
        return (
            f"That expands to {self.total_available} drugs — too broad to retrieve "
            f"reliably. Members include: {names}. Name one or two specific drugs."
        )


def expand_class(
    g: nx.MultiDiGraph, class_id: str, *, cap: int = MAX_EXPANSION_RXCUIS
) -> Expansion:
    """class -> member drugs, bounded. Over the cap returns a productive redirect."""
    members = [
        (n, str(g.nodes[n].get("name", n)))
        for n in g.predecessors(class_id)
        if g.nodes[n].get("kind") == "Drug"
    ]
    if len(members) > cap:
        return Expansion(
            [],
            overflow=True,
            total_available=len(members),
            via=[class_id],
            members_preview=sorted(members, key=lambda x: x[1]),
        )
    return Expansion([rx for rx, _ in members], total_available=len(members), via=[class_id])


def multi_hop(
    g: nx.MultiDiGraph, rxcui: str, *, max_hops: int = 3, cap: int = MAX_EXPANSION_RXCUIS
) -> Expansion:
    """drug -> class -> sibling drugs, breadth-first with a hard ceiling.

    This is the committed multi-hop differentiator:
        ketoconazole -> CYP3A4 inhibitor -> CYP3A4 substrates -> simvastatin
    """
    seen_drugs: set[str] = {rxcui}
    frontier: set[str] = {rxcui}
    via: list[str] = []

    for _ in range(max_hops):
        classes = {
            c for d in frontier for c in g.successors(d) if g.nodes[c].get("kind") == "PharmClass"
        }
        via.extend(sorted(classes))
        next_drugs = {
            n for c in classes for n in g.predecessors(c) if g.nodes[n].get("kind") == "Drug"
        } - seen_drugs
        if not next_drugs:
            break
        if len(seen_drugs) + len(next_drugs) > cap:
            preview = sorted(
                ((n, str(g.nodes[n].get("name", n))) for n in next_drugs),
                key=lambda x: x[1],
            )
            return Expansion(
                sorted(seen_drugs),
                overflow=True,
                total_available=len(seen_drugs) + len(next_drugs),
                via=via,
                members_preview=preview,
            )
        seen_drugs |= next_drugs
        frontier = next_drugs

    return Expansion(sorted(seen_drugs), total_available=len(seen_drugs), via=via)


def explain_path(g: nx.MultiDiGraph, source: str, target: str) -> list[str] | None:
    """Shortest path for the UI's multi-hop visualization.

    Display only — never evidence.
    """
    try:
        path = nx.shortest_path(g.to_undirected(as_view=True), source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [str(g.nodes[n].get("name", n)) for n in path]


# --------------------------------------------------------------------------- additive risk
@dataclass(slots=True)
class AdditiveRisk:
    risk_class: str
    label: str
    members: list[str]
    member_count: int
    threshold: int
    note: str = ""
    provenance: str = "graph"

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_class": self.risk_class,
            "label": self.label,
            "members": self.members,
            "member_count": self.member_count,
            "threshold": self.threshold,
            "note": self.note,
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class CombinationAlert:
    alert_id: str
    label: str
    detail: str
    contributing: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "label": self.label,
            "detail": self.detail,
            "contributing": self.contributing,
        }


def class_burden(
    g: nx.MultiDiGraph,
    rxcuis: list[str],
) -> tuple[list[AdditiveRisk], list[CombinationAlert]]:
    """ADR-033 — the check pairwise decomposition structurally cannot perform.

    Pairwise C(n,2) coverage is NECESSARY BUT NOT SUFFICIENT. QT burden,
    serotonin syndrome, cumulative anticholinergic load, and the triple whammy
    are class-COUNT properties. Three drugs that each pair as "moderate" can
    combine into a recognized serious hazard, and no pairwise pass will say so.

    Deterministic, $0, no LLM.
    """
    by_risk: dict[str, list[str]] = {}
    provenance: dict[str, set[str]] = {}

    for rxcui in rxcuis:
        if rxcui not in g:
            continue
        name = str(g.nodes[rxcui].get("name", rxcui))
        for _, target, data in g.out_edges(rxcui, data=True):
            if data.get("edge") != "carries_risk":
                continue
            if g.nodes.get(target, {}).get("kind") != "RiskClass":
                continue
            by_risk.setdefault(target, []).append(name)
            provenance.setdefault(target, set()).add(str(data.get("extraction_method", "unknown")))

    risks: list[AdditiveRisk] = []
    for risk_id, members in by_risk.items():
        rc = _RISK_BY_ID.get(risk_id)
        threshold = rc.min_members if rc else 2
        unique = sorted(set(members))
        if len(unique) >= threshold:
            risks.append(
                AdditiveRisk(
                    risk_class=risk_id,
                    label=rc.label if rc else str(g.nodes.get(risk_id, {}).get("name", risk_id)),
                    members=unique,
                    member_count=len(unique),
                    threshold=threshold,
                    note=rc.note if rc else "",
                    provenance="+".join(sorted(provenance.get(risk_id, {"graph"}))),
                )
            )

    alerts: list[CombinationAlert] = []
    present = {rid: sorted(set(m)) for rid, m in by_risk.items()}
    for spec in COMBINATION_ALERTS:
        required = spec["requires"]
        if all(present.get(r) for r in required):
            alerts.append(
                CombinationAlert(
                    alert_id=str(spec["id"]),
                    label=str(spec["label"]),
                    detail=str(spec["detail"]),
                    contributing={r: present[r] for r in required},
                )
            )

    risks.sort(key=lambda r: -r.member_count)
    return risks, alerts


# --------------------------------------------------------------------------- compound
@dataclass(slots=True)
class RegimenPlan:
    """ADR-033 — pairwise decomposition plus the class-burden check."""

    pairs: list[tuple[str, str]]
    additive_risks: list[AdditiveRisk]
    combination_alerts: list[CombinationAlert]
    capped: bool = False
    cap: int = 6


def plan_regimen(g: nx.MultiDiGraph, rxcuis: list[str], *, cap: int = 6) -> RegimenPlan:
    """Build the retrieval plan for a multi-drug regimen.

    The cap is a context/latency constraint, NOT a clinical one — real
    polypharmacy patients take 10+. It belongs in the model card as a stated
    limitation rather than presented as a design feature.
    """
    unique = list(dict.fromkeys(rxcuis))
    capped = len(unique) > cap
    working = unique[:cap]
    risks, alerts = class_burden(g, working)
    return RegimenPlan(
        pairs=list(combinations(working, 2)),
        additive_risks=risks,
        combination_alerts=alerts,
        capped=capped,
        cap=cap,
    )
