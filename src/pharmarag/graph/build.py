"""Milestone B7 — deterministic property graph (ADR-035, ADR-036).

Nodes:  Drug(rxcui) · PharmClass(id) · RiskClass(id)
Edges:  has_class · acts_via · carries_risk · has_atc · interacts_with

Every edge carries `source` and `extraction_method`. An edge you cannot trace
back to a chunk or an ontology is an edge you cannot defend.

Extraction is PRECISION-BIASED — the opposite of the recall-first stance
elsewhere. A false edge silently widens the retrieval filter and pulls an
unrelated drug's label into evidence; a missed edge just means the vector path
handles that query alone. Wrong edges cost more than missing ones.

Serialized as JSON node-link, never pickle: pickle executes arbitrary code on
load and breaks across Python versions, which is a bad thing to hand a recruiter
who clones the repo. The artifact is a CACHE — the builder is the source of
truth, and if they disagree the builder wins.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from pharmarag.entity.gazetteer import Gazetteer
    from pharmarag.graph.rxclass import HarvestResult

# --------------------------------------------------------------------------- risk classes
# ADR-033: additive risks are CLASS-COUNT properties, not pairwise ones. The
# canonical case is the "triple whammy" — NSAID + ACE-inhibitor/ARB + diuretic.
# Each PAIR is typically labeled moderate; the TRIPLE is a recognized cause of
# acute kidney injury. C(3,2) coverage returns three moderate findings and never
# names the actual hazard.
#
# RxClass coverage for these is UNVERIFIED (plan.md verification item 4), so each
# risk class carries both an RxClass pattern and a curated fallback. Provenance
# is recorded per edge so the model card can state exactly which is which —
# never ship partial coverage on a safety feature silently.


@dataclass(frozen=True, slots=True)
class RiskClass:
    id: str
    label: str
    rxclass_pattern: str  # matched against RxClass className
    curated_members: frozenset[str]
    min_members: int = 2  # count at which the burden is flagged
    note: str = ""


RISK_CLASSES: tuple[RiskClass, ...] = (
    RiskClass(
        "qt_prolongation",
        "QT prolongation / torsades risk",
        r"qt.?prolong|torsade|antiarrhythmic",
        frozenset(
            {
                "amiodarone",
                "sotalol",
                "dofetilide",
                "quinidine",
                "procainamide",
                "citalopram",
                "escitalopram",
                "haloperidol",
                "ziprasidone",
                "thioridazine",
                "methadone",
                "ondansetron",
                "clarithromycin",
                "erythromycin",
                "levofloxacin",
                "moxifloxacin",
                "ciprofloxacin",
                "fluconazole",
                "hydroxyzine",
                "domperidone",
                "droperidol",
            }
        ),
        note="Each agent individually may be labeled moderate; the additive burden is the hazard.",
    ),
    RiskClass(
        "serotonergic",
        "Serotonin syndrome risk",
        r"serotonin|ssri|snri|maoi|triptan",
        frozenset(
            {
                "sertraline",
                "fluoxetine",
                "paroxetine",
                "citalopram",
                "escitalopram",
                "venlafaxine",
                "duloxetine",
                "tramadol",
                "linezolid",
                "sumatriptan",
                "rizatriptan",
                "trazodone",
                "buspirone",
                "lithium",
                "dextromethorphan",
                "phenelzine",
                "selegiline",
                "amitriptyline",
                "nortriptyline",
            }
        ),
    ),
    RiskClass(
        "anticholinergic",
        "Cumulative anticholinergic burden",
        r"anticholinergic|antimuscarinic|cholinergic antagonist",
        frozenset(
            {
                "diphenhydramine",
                "hydroxyzine",
                "amitriptyline",
                "nortriptyline",
                "oxybutynin",
                "tolterodine",
                "scopolamine",
                "benztropine",
                "chlorpheniramine",
                "paroxetine",
                "olanzapine",
                "quetiapine",
                "cyclobenzaprine",
                "promethazine",
            }
        ),
        min_members=3,
        note="Burden is dose- and count-dependent; 3+ agents is the usual flag threshold.",
    ),
    RiskClass(
        "nephrotoxic",
        "Additive nephrotoxicity",
        r"nephrotox|nsaid|cyclooxygenase|aminoglycoside",
        frozenset(
            {
                "ibuprofen",
                "naproxen",
                "diclofenac",
                "indomethacin",
                "ketorolac",
                "celecoxib",
                "meloxicam",
                "vancomycin",
                "gentamicin",
                "tobramycin",
                "amikacin",
                "amphotericin b",
                "cisplatin",
                "tacrolimus",
                "cyclosporine",
            }
        ),
    ),
    RiskClass(
        "raas_inhibitor",
        "RAAS inhibition",
        r"angiotensin|ace inhibitor|renin",
        frozenset(
            {
                "lisinopril",
                "enalapril",
                "ramipril",
                "captopril",
                "benazepril",
                "losartan",
                "valsartan",
                "irbesartan",
                "olmesartan",
                "candesartan",
                "telmisartan",
                "aliskiren",
                "sacubitril",
            }
        ),
    ),
    RiskClass(
        "diuretic",
        "Diuretic",
        r"diuretic|thiazide|loop.*diuretic",
        frozenset(
            {
                "furosemide",
                "bumetanide",
                "torsemide",
                "hydrochlorothiazide",
                "chlorthalidone",
                "indapamide",
                "spironolactone",
                "eplerenone",
                "amiloride",
                "triamterene",
                "metolazone",
            }
        ),
    ),
    RiskClass(
        "bleeding",
        "Additive bleeding risk",
        r"anticoagul|antiplatelet|thrombolytic|factor xa|thrombin inhibitor",
        frozenset(
            {
                "warfarin",
                "apixaban",
                "rivaroxaban",
                "edoxaban",
                "dabigatran",
                "enoxaparin",
                "heparin",
                "clopidogrel",
                "prasugrel",
                "ticagrelor",
                "aspirin",
                "ibuprofen",
                "naproxen",
                "diclofenac",
                "ketorolac",
            }
        ),
    ),
    RiskClass(
        "cns_depressant",
        "Additive CNS / respiratory depression",
        r"opioid|benzodiazepine|sedative|hypnotic|cns depress",
        frozenset(
            {
                "morphine",
                "oxycodone",
                "hydrocodone",
                "fentanyl",
                "methadone",
                "tramadol",
                "codeine",
                "alprazolam",
                "lorazepam",
                "diazepam",
                "clonazepam",
                "temazepam",
                "zolpidem",
                "eszopiclone",
                "phenobarbital",
                "baclofen",
                "cyclobenzaprine",
                "gabapentin",
                "pregabalin",
            }
        ),
    ),
)

# Named multi-drug syndromes that pairwise analysis structurally cannot see.
COMBINATION_ALERTS: tuple[dict[str, Any], ...] = (
    {
        "id": "triple_whammy",
        "label": "Triple whammy — acute kidney injury risk",
        "requires": ("nephrotoxic", "raas_inhibitor", "diuretic"),
        "detail": (
            "An NSAID combined with a RAAS inhibitor and a diuretic is a recognized "
            "cause of acute kidney injury. Each pair is commonly labeled moderate; "
            "the combination of all three is the hazard."
        ),
    },
)


# --------------------------------------------------------------------------- extraction
_INTERACTION_CUE = re.compile(
    r"concomitant|coadminist|co-administ|concurrent|combination with|together with|"
    r"avoid.*use|increases? (?:the )?(?:risk|exposure|concentration)|"
    r"reduces? (?:the )?(?:efficacy|concentration)|contraindicated with",
    re.IGNORECASE,
)


def extract_interaction_edges(
    chunks: list[dict[str, Any]],
    gazetteer: Gazetteer,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Mine `interacts_with` edges from §Drug Interactions chunks.

    Uses the gazetteer's word-boundary + longest-match + blocklist rules
    (ADR-036) — naive substring matching finds "codeine" inside "hydrocodone"
    and writes a false edge that silently widens every downstream filter.
    """
    edges: list[tuple[str, str, dict[str, Any]]] = []
    for chunk in chunks:
        if chunk.get("loinc_section_code") != "34073-7":
            continue
        text = str(chunk.get("display_text", ""))
        if not _INTERACTION_CUE.search(text):
            continue  # precision-biased: no cue, no edge
        subject = str(chunk.get("rxcui") or "")
        if not subject:
            continue
        for mention in gazetteer.find_mentions(text):
            if not mention.rxcui or mention.rxcui == subject:
                continue
            edges.append(
                (
                    subject,
                    mention.rxcui,
                    {
                        "edge": "interacts_with",
                        "source_chunk_id": chunk.get("chunk_id"),
                        "extraction_method": "regex_cue+gazetteer",
                        "evidence": text[:280],
                    },
                )
            )
    return edges


# --------------------------------------------------------------------------- build
def build_graph(
    harvest: HarvestResult | None = None,
    interaction_edges: list[tuple[str, str, dict[str, Any]]] | None = None,
    *,
    name_to_rxcui: dict[str, str] | None = None,
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    if harvest is not None:
        for name, rxcui in harvest.resolved.items():
            g.add_node(rxcui, kind="Drug", name=name)
        for m in harvest.memberships:
            g.add_node(m.class_id, kind="PharmClass", name=m.class_name, class_type=m.class_type)
            g.add_edge(
                m.rxcui,
                m.class_id,
                key=m.edge,
                edge=m.edge,
                extraction_method="rxclass",
                rela=m.rela,
                source="RxClass",
            )

    # Risk classes: RxClass pattern match where the harvest supplies it,
    # curated fallback otherwise. Provenance recorded per edge.
    names = dict(name_to_rxcui or (harvest.resolved if harvest else {}))
    lower_names = {n.lower(): rx for n, rx in names.items()}

    for rc in RISK_CLASSES:
        g.add_node(rc.id, kind="RiskClass", name=rc.label, min_members=rc.min_members, note=rc.note)
        matched_via_rxclass: set[str] = set()

        if harvest is not None:
            pattern = re.compile(rc.rxclass_pattern, re.IGNORECASE)
            for m in harvest.memberships:
                if pattern.search(m.class_name):
                    g.add_edge(
                        m.rxcui,
                        rc.id,
                        key="carries_risk",
                        edge="carries_risk",
                        extraction_method="rxclass_pattern",
                        source=m.class_name,
                    )
                    matched_via_rxclass.add(m.rxcui)

        for member in rc.curated_members:
            member_rxcui = lower_names.get(member)
            if member_rxcui and member_rxcui not in matched_via_rxclass:
                g.add_node(member_rxcui, kind="Drug", name=member)
                g.add_edge(
                    member_rxcui,
                    rc.id,
                    key="carries_risk",
                    edge="carries_risk",
                    extraction_method="curated",
                    source="PharmaRAG curated list",
                )

    for src, dst, attrs in interaction_edges or []:
        g.add_edge(src, dst, key="interacts_with", **attrs)

    return g


def graph_stats(g: nx.MultiDiGraph) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        k = str(data.get("kind", "?"))
        kinds[k] = kinds.get(k, 0) + 1
    edges: dict[str, int] = {}
    methods: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        e = str(data.get("edge", "?"))
        edges[e] = edges.get(e, 0) + 1
        m = str(data.get("extraction_method", "?"))
        methods[m] = methods.get(m, 0) + 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "by_kind": kinds,
        "by_edge": edges,
        "by_extraction_method": methods,
    }


def save(g: nx.MultiDiGraph, path: Path) -> None:
    """JSON node-link. Human-readable, diffable in git, no arbitrary code on load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="links")
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def load(path: Path) -> nx.MultiDiGraph:
    if not path.is_file():
        # Almost always GRAPH_VERSION left at its "none" default, giving none.json.
        # The raw FileNotFoundError names the missing path but not the cause, so say it.
        available = (
            sorted(p.stem for p in path.parent.glob("*.json")) if path.parent.is_dir() else []
        )
        hint = (
            f"available: {', '.join(available)} — set GRAPH_VERSION in .env"
            if available
            else "no graphs built yet — run: uv run python scripts/build_graph.py --offline"
        )
        raise FileNotFoundError(f"no graph at {path} ({hint})")
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
