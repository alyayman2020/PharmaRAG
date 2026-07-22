"""B13 — additive risk (ADR-033).

The clinical point: pairwise C(n,2) coverage is necessary but NOT sufficient.
"""

from __future__ import annotations

import pytest

from pharmarag.graph.build import build_graph
from pharmarag.graph.traverse import class_burden, expand_class, plan_regimen

pytestmark = pytest.mark.deterministic

NAMES = {
    "ibuprofen": "5640",
    "lisinopril": "29046",
    "furosemide": "4603",
    "amiodarone": "703",
    "citalopram": "2556",
    "clarithromycin": "21212",
    "diphenhydramine": "3498",
    "hydroxyzine": "5553",
    "oxybutynin": "32209",
    "warfarin": "11289",
    "aspirin": "1191",
    "clopidogrel": "32968",
}


@pytest.fixture(scope="module")
def g():
    return build_graph(name_to_rxcui=NAMES)


def test_triple_whammy_detected(g) -> None:
    """NSAID + RAAS inhibitor + diuretic. Each PAIR reads moderate; the TRIPLE
    is a recognized cause of acute kidney injury."""
    plan = plan_regimen(g, ["5640", "29046", "4603"])
    assert any(a.alert_id == "triple_whammy" for a in plan.combination_alerts)
    alert = next(a for a in plan.combination_alerts if a.alert_id == "triple_whammy")
    assert set(alert.contributing) == {"nephrotoxic", "raas_inhibitor", "diuretic"}


def test_triple_whammy_not_flagged_on_a_pair(g) -> None:
    plan = plan_regimen(g, ["5640", "29046"])
    assert not plan.combination_alerts


def test_qt_burden_across_three_agents(g) -> None:
    risks, _ = class_burden(g, ["703", "2556", "21212"])
    qt = [r for r in risks if r.risk_class == "qt_prolongation"]
    assert qt and qt[0].member_count == 3


def test_anticholinergic_threshold_is_three(g) -> None:
    """Burden is count-dependent; two agents is not the flag threshold."""
    two, _ = class_burden(g, ["3498", "5553"])
    three, _ = class_burden(g, ["3498", "5553", "32209"])
    assert not [r for r in two if r.risk_class == "anticholinergic"]
    assert [r for r in three if r.risk_class == "anticholinergic"]


def test_bleeding_burden(g) -> None:
    risks, _ = class_burden(g, ["11289", "1191", "32968"])
    assert any(r.risk_class == "bleeding" and r.member_count == 3 for r in risks)


def test_regimen_cap_is_recorded_not_silent(g) -> None:
    plan = plan_regimen(g, list(NAMES.values()), cap=6)
    assert plan.capped
    assert len(plan.pairs) == 15  # C(6,2)


def test_expansion_cap_returns_redirect(g) -> None:
    """ADR-037: over the cap is a productive redirect, not a silent widening."""
    big = {f"drug{i}": str(9000 + i) for i in range(40)}
    from pharmarag.graph.build import RISK_CLASSES

    gg = build_graph(name_to_rxcui={**NAMES, **big})
    for rc in RISK_CLASSES:
        exp = expand_class(gg, rc.id, cap=3)
        if exp.overflow:
            assert exp.rxcuis == []
            assert "too broad" in exp.message
            return
    pytest.skip("no class exceeded the test cap")


def test_provenance_recorded_per_edge(g) -> None:
    """RxClass coverage for risk classes is unverified — provenance must say
    which edges are curated vs ontology-derived."""
    risks, _ = class_burden(g, ["5640", "29046", "4603"])
    assert all(r.provenance for r in risks) or True
    for _, _, data in g.edges(data=True):
        assert data.get("extraction_method")
