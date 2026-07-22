"""Milestone B7 — RxClass harvesting (ADR-006, ADR-036).

ADR-006 rejected full GraphRAG because multi-pass LLM extraction over 1,500
labels cannot be bounded under 3 days and would eat a third of the budget.
RxClass hands you a curated, maintained ontology for free:

    drug -> established pharmacologic class -> mechanism of action -> enzyme

Cost: $0. Runtime: ~9,000 HTTP calls, roughly half a working day. That is the
whole Graph-RAG differentiator without the GraphRAG price tag.

NETWORK REQUIRED — not exercised offline. Verify against
https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxClass.getClassByRxNormDrugId.html
before a full run; NLM does move these paths.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pharmarag.config import settings
from pharmarag.http import get_json

RXCLASS_BY_DRUG = "{base}/REST/rxclass/class/byRxcui.json"
RXCLASS_MEMBERS = "{base}/REST/rxclass/classMembers.json"
RXCUI_BY_NAME = "{base}/REST/rxcui.json"

# MED-RT relationship sources we trust for deterministic edges.
RELA_SOURCES = ("ATC", "MEDRT", "DAILYMED", "RXNORM")

# Class types -> the edge they imply.
CLASS_TYPE_EDGE = {
    "EPC": "has_class",  # Established Pharmacologic Class
    "MOA": "acts_via",  # Mechanism of Action
    "PE": "has_effect",  # Physiologic Effect (drug -> PharmClass; the safety-path
    #   "carries_risk" edges to RiskClass nodes are added explicitly in build.py)
    "CHEM": "has_structure",
    "ATC1-4": "has_atc",
}


@dataclass(slots=True)
class ClassMembership:
    rxcui: str
    class_id: str
    class_name: str
    class_type: str
    rela: str = ""

    @property
    def edge(self) -> str:
        return CLASS_TYPE_EDGE.get(self.class_type, "has_class")


@dataclass(slots=True)
class HarvestResult:
    memberships: list[ClassMembership] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    resolved: dict[str, str] = field(default_factory=dict)  # name -> rxcui

    def summary(self) -> dict[str, int]:
        by_type: dict[str, int] = {}
        for m in self.memberships:
            by_type[m.class_type] = by_type.get(m.class_type, 0) + 1
        return {"drugs": len(self.resolved), "memberships": len(self.memberships), **by_type}


def resolve_rxcui(name: str) -> str | None:
    """Exact RxNorm lookup. Deliberately NOT approximate — a wrong RxCUI here
    poisons every downstream edge for that drug."""
    data = get_json(RXCUI_BY_NAME.format(base=settings.rxnav_base), {"name": name, "search": "1"})
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    return str(ids[0]) if ids else None


def classes_for_rxcui(rxcui: str) -> list[ClassMembership]:
    data = get_json(RXCLASS_BY_DRUG.format(base=settings.rxnav_base), {"rxcui": rxcui})
    items = (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []
    out: list[ClassMembership] = []
    for item in items:
        concept = item.get("rxclassMinConceptItem") or {}
        cid, cname, ctype = (
            concept.get("classId"),
            concept.get("className"),
            concept.get("classType"),
        )
        if not (cid and cname and ctype):
            continue
        out.append(
            ClassMembership(rxcui, str(cid), str(cname), str(ctype), str(item.get("rela") or ""))
        )
    return out


def members_of_class(class_id: str, rela_source: str = "MEDRT") -> list[tuple[str, str]]:
    """(rxcui, name) members of a class — the class -> member expansion for ADR-020."""
    data = get_json(
        RXCLASS_MEMBERS.format(base=settings.rxnav_base),
        {"classId": class_id, "relaSource": rela_source},
    )
    members = (data.get("drugMemberGroup") or {}).get("drugMember") or []
    out: list[tuple[str, str]] = []
    for m in members:
        c = m.get("minConcept") or {}
        if c.get("rxcui") and c.get("name"):
            out.append((str(c["rxcui"]), str(c["name"])))
    return out


def harvest(drug_names: list[str], *, delay: float = 0.06, progress: bool = True) -> HarvestResult:
    """Harvest class memberships for a drug list.

    `delay` throttles to stay well inside RxNav's published limit. At 1,500
    drugs and ~2 calls each this is roughly 5-10 minutes, not the 4-8 hours
    ADR-006 budgeted — the estimate was conservative.
    """
    result = HarvestResult()
    for i, name in enumerate(drug_names, 1):
        try:
            rxcui = resolve_rxcui(name)
            if not rxcui:
                result.errors[name] = "no RxCUI"
                continue
            result.resolved[name] = rxcui
            result.memberships.extend(classes_for_rxcui(rxcui))
        except Exception as exc:
            result.errors[name] = f"{type(exc).__name__}: {exc}"
        if progress and i % 10 == 0:
            print(
                f"[rxclass] {i}/{len(drug_names)} · "
                f"{len(result.memberships)} memberships · {len(result.errors)} errors",
                flush=True,
            )
        time.sleep(delay)
    return result


def save(result: HarvestResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "resolved": result.resolved,
                "errors": result.errors,
                "memberships": [
                    {
                        "rxcui": m.rxcui,
                        "class_id": m.class_id,
                        "class_name": m.class_name,
                        "class_type": m.class_type,
                        "rela": m.rela,
                    }
                    for m in result.memberships
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def load(path: Path) -> HarvestResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return HarvestResult(
        memberships=[ClassMembership(**m) for m in data["memberships"]],
        errors=data.get("errors", {}),
        resolved=data.get("resolved", {}),
    )
