"""B11 K2 grounding + B18 caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pharmarag.cache import cache_key
from pharmarag.guardrails.grounding import Entailment, apply_drops, verify_grounding
from pharmarag.guardrails.input_guard import GuardResult, Verdict

pytestmark = pytest.mark.deterministic

CHUNKS = {
    "c1": {
        "display_text": "For CrCl 30 to 50 mL/min the dose is 250 mg every 12 hours.",
        "loinc_section_code": "34068-7",
    },
    "c2": {
        "display_text": "Fatal hepatic failure has been reported.",
        "loinc_section_code": "34066-1",
    },
}


def test_numeric_precheck_blocks_invented_dose() -> None:
    r = verify_grounding(
        [{"text": "The dose is 900 mg every 12 hours.", "citation_ids": ["c1"]}],
        CHUNKS,
        use_llm=False,
    )
    assert not r.passed
    assert r.verdicts[0].verdict is Entailment.UNSUPPORTED
    assert r.must_refuse


def test_fabricated_dose_refuses_at_every_tier() -> None:
    """A number carrying a unit that is absent from the source is never survivable."""
    r = verify_grounding(
        [{"text": "Give 900 mg once daily.", "citation_ids": ["c1"]}], CHUNKS, use_llm=False
    )
    assert r.must_refuse
    assert "dose value(s) not in source" in r.verdicts[0].reason


def test_incidental_numeral_does_not_refuse_a_non_critical_claim() -> None:
    """A bare numeral is not a fabricated dose.

    Blocking a whole correct answer because the model restated a count the source
    does not literally contain is a false refusal. It degrades to
    PARTIALLY_SUPPORTED and takes the ADR-039 tier route: drop and disclose.
    """
    r = verify_grounding(
        [{"text": "There are 3 recognised management options.", "citation_ids": ["c1"]}],
        CHUNKS,
        use_llm=False,
    )
    assert not r.must_refuse
    assert r.verdicts[0].verdict is Entailment.PARTIALLY_SUPPORTED
    assert r.droppable == [0]


def test_incidental_numeral_still_refuses_on_safety_tier_1() -> None:
    """Tier 1 (boxed warning / contraindication) keeps the strict route."""
    r = verify_grounding(
        [{"text": "Reported in 7 patients.", "citation_ids": ["c2"]}], CHUNKS, use_llm=False
    )
    assert r.must_refuse


def test_cross_references_are_not_treated_as_numeric_claims() -> None:
    """ "[see Warnings (5.2)]" is label navigation the model echoes, not a claim."""
    r = verify_grounding(
        [
            {
                "text": "Monitor renal function [see Warnings and Precautions (5.2)].",
                "citation_ids": ["c1"],
            }
        ],
        CHUNKS,
        use_llm=False,
    )
    assert r.passed


def test_missing_source_fails_closed() -> None:
    r = verify_grounding([{"text": "x", "citation_ids": ["ghost"]}], CHUNKS, use_llm=False)
    assert r.must_refuse


def test_grounded_claim_passes() -> None:
    r = verify_grounding(
        [{"text": "The dose is 250 mg every 12 hours.", "citation_ids": ["c1"]}],
        CHUNKS,
        use_llm=False,
    )
    assert r.passed


def test_drop_is_disclosed_never_silent() -> None:
    payload = apply_drops({"claims": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}, [1])
    assert len(payload["claims"]) == 2
    assert payload["omitted_claims_disclosed"] == 1
    assert "omitted" in payload["omission_notice"]


def test_cache_key_includes_versions() -> None:
    """ADR-050: without version fields a corpus refresh serves stale answers."""
    import os
    from importlib import reload

    import pharmarag.cache as ca
    import pharmarag.config as cfg

    a = cache_key(intent="dosing", rxcuis=["83367"])
    os.environ["CORPUS_VERSION"] = "different-corpus"
    reload(cfg)
    reload(ca)
    b = ca.cache_key(intent="dosing", rxcuis=["83367"])
    assert a != b
    os.environ.pop("CORPUS_VERSION", None)
    reload(cfg)
    reload(ca)


def test_cache_key_is_order_independent() -> None:
    assert cache_key(intent="ddi", rxcuis=["a", "b"]) == cache_key(intent="ddi", rxcuis=["b", "a"])


@pytest.mark.parametrize("verdict", [Verdict.UNSAFE, Verdict.PERSONAL_ADVICE, Verdict.OUT_OF_SCOPE])
def test_k1_verdict_is_never_written_to_cache(monkeypatch, verdict) -> None:
    """ADR-050/038: a jailbreak answered once must not become a cached SAFE verdict.

    Asserted mechanically, not by comment: every K1 refusal path must exit before
    any cache write, so a future edit that moves the lookup above the guard fails here.
    """
    from pharmarag import pipeline as pl

    writes: list[str] = []
    monkeypatch.setattr(pl.cache, "put", lambda k, p: writes.append(k))
    monkeypatch.setattr(pl, "write_audit", lambda rec: None)
    monkeypatch.setattr(
        pl,
        "check_input",
        lambda q, *, use_llm=True: GuardResult(verdict, "stub", "regex", "try the general form"),
    )

    def _boom(*a, **k):  # nothing past K1 may run
        raise AssertionError("pipeline continued past a K1 refusal")

    resolver = SimpleNamespace(resolve=_boom, gazetteer={})
    out = pl.answer_question(
        "stub",
        resolver=resolver,
        client=None,
        reranker=None,
        embed_dense=_boom,
        embed_sparse=_boom,
        use_llm_guard=False,
    )

    assert writes == []
    assert out.payload["answer_type"] == "refusal"
