"""Streamlit UI (ADR-049).

One primary surface with progressive disclosure, not a five-tab nav. A reviewer
gives you ~90 seconds; a five-tab layout asks them to assemble the story
themselves and they won't.

The reading order is deliberate and is the argument the project is making:

    the answer  ->  what verified it  ->  the evidence it came from  ->  the record

A refusal occupies exactly the same position and weight as an answer, because in
this system refusing *is* an outcome, not an error. Presentation rules live in
`theme.py`; this file stays pipeline-in, evidence-out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Streamlit's source watcher calls get_module_paths() on every module in sys.modules,
# which trips transformers' lazy loader into importing ~100 image processors. Each one
# needs torchvision, which this app has no use for, so each logs a warning traceback on
# every rerun. Silencing the watcher's logger keeps auto-rerun on save without the flood.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages/spl_parser/src"))
sys.path.insert(0, str(ROOT / "app"))

import theme  # noqa: E402

from pharmarag.config import DATA, settings  # noqa: E402
from pharmarag.entity.gazetteer import Gazetteer  # noqa: E402
from pharmarag.entity.lasa import load as load_lasa  # noqa: E402
from pharmarag.entity.resolve import Resolver  # noqa: E402
from pharmarag.generate.schema import DISCLAIMER  # noqa: E402
from pharmarag.pipeline import answer_question  # noqa: E402

st.set_page_config(
    page_title="PharmaRAG — grounded clinical drug information",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

# Each example proves a specific decision, so a reviewer with no pharmacology
# background can demonstrate the architecture in a handful of clicks.
CAPABILITY: list[tuple[str, str]] = [
    (
        "Atorvastatin + cyclosporine interaction",
        "Hybrid retrieval inside the resolved drug partition, cited per claim (ADR-023)",
    ),
    (
        "Warfarin and amiodarone interaction",
        "Narrow-therapeutic-index stratum — table-row evidence with its qualifier (ADR-008)",
    ),
    (
        "Atorvastatin dose in renal impairment",
        "The qualifier rule: a dose never appears without its condition (ADR-013, K3)",
    ),
    (
        "Contraindications for atorvastatin",
        "Context assembled in safety-tier order, not relevance order (ADR-027)",
    ),
]

REFUSALS: list[tuple[str, str]] = [
    (
        "What is the dose of unobtainium?",
        "Hard refusal — the corpus has no evidence and the model never falls back to memory",
    ),
    (
        "What is a lethal dose of amitriptyline?",
        "K1 input guard — harm-seeking queries are refused before any retrieval",
    ),
    (
        "I have CKD, how much metformin should I take?",
        "K1 — personal medical advice is out of scope; a reformulation is offered",
    ),
    (
        "What is contraindicated in pregnancy?",
        "Population-only sweep — the corpus is organised by drug (ADR-020)",
    ),
]


@st.cache_resource(show_spinner=False)
def _resolver() -> Resolver:
    gp = DATA / "gazetteer.json"
    return Resolver(
        Gazetteer.load(gp) if gp.exists() else Gazetteer(), load_lasa(DATA / "lasa_table.json")
    )


@st.cache_resource(show_spinner=False)
def _runtime():
    from fastembed import SparseTextEmbedding

    from pharmarag.index.store import get_client
    from pharmarag.rerank.reranker import Reranker

    return get_client(), Reranker(), SparseTextEmbedding(model_name="Qdrant/bm25")


@st.cache_data(show_spinner=False, ttl=300)
def _index_stats() -> dict:
    try:
        from pharmarag.index.store import collection_stats, get_client

        return dict(collection_stats(get_client()))
    except Exception:
        return {}


def _set_q(text: str) -> None:
    st.session_state["q"] = text


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="pr-h">System</div>', unsafe_allow_html=True)
    stats = _index_stats()
    a, b = st.columns(2)
    a.metric("Corpus", "1,000", help="Canonical drug ingredients, frozen selection")
    b.metric(
        "Passages",
        f"{stats.get('points', 0):,}" if stats.get("points") else "—",
        help="Indexed, retrievable chunks in the vector store",
    )
    c, d = st.columns(2)
    c.metric("Device", settings.resolve_device().upper())
    d.metric("Calibration", settings.calibrator_version)
    st.caption(f"Corpus version `{settings.corpus_version}`")

    st.divider()
    st.markdown('<div class="pr-h">Query options</div>', unsafe_allow_html=True)
    use_guard = st.toggle(
        "LLM input guard",
        value=True,
        help="Layer 2 of K1. The regex layer always runs; this adds the model "
        "classifier on the highest-consequence tier (~$0.001/query).",
    )

    st.divider()
    st.markdown('<div class="pr-h">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Input guard** — harm, personal-advice, scope\n"
        "2. **Entity resolution** — identity before retrieval\n"
        "3. **Hybrid search** — dense + sparse, inside the drug partition\n"
        "4. **Rerank** — scores every candidate, feeds abstention\n"
        "5. **Context** — ordered by clinical consequence\n"
        "6. **Synthesis** — strict JSON, one citation per claim\n"
        "7. **Guardrails** — citations · dose · identity · grounding\n"
        "8. **Audit** — append-only record of every decision"
    )
    st.divider()
    st.caption(
        "Abstention thresholds are **uncalibrated** until the Platt calibrator is "
        "refit; confidence is reported honestly as such. Overdosage content is "
        "deliberately excluded from retrieval."
    )

# ------------------------------------------------------------------ masthead
st.markdown(theme.hero(), unsafe_allow_html=True)
st.markdown(theme.disclosure(), unsafe_allow_html=True)

# ------------------------------------------------------------------- prompts
left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="pr-h">Grounded answers</div>', unsafe_allow_html=True)
    for text, why in CAPABILITY:
        st.button(
            text,
            help=why,
            use_container_width=True,
            on_click=_set_q,
            args=(text,),
            key=f"cap-{text}",
        )
with right:
    st.markdown('<div class="pr-h">Refusals — by design</div>', unsafe_allow_html=True)
    for text, why in REFUSALS:
        st.button(
            text,
            help=why,
            use_container_width=True,
            on_click=_set_q,
            args=(text,),
            key=f"ref-{text}",
        )

st.markdown("")
question = st.text_input(
    "Ask a question",
    value=st.session_state.get("q", ""),
    placeholder="e.g. What is the atorvastatin dose in renal impairment?",
    label_visibility="collapsed",
)

# --------------------------------------------------------------------- answer
if question:
    stages: list = []
    with st.status("Running the pipeline…", expanded=True) as status:

        def on_stage(s) -> None:
            stages.append(s)
            status.write(f"**{s.name}** — {s.detail}  ·  {s.ms:,.0f} ms")

        try:
            from pharmarag.embed.client import embed_query

            client, reranker, bm25 = _runtime()
            result = answer_question(
                question,
                resolver=_resolver(),
                client=client,
                reranker=reranker,
                embed_dense=embed_query,
                embed_sparse=lambda q: next(iter(bm25.embed([q]))),
                use_llm_guard=use_guard,
                on_stage=on_stage,
            )
            status.update(
                label=f"Pipeline complete · {sum(s.ms for s in stages):,.0f} ms",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            status.update(label="Pipeline error", state="error")
            st.error(f"**{type(exc).__name__}** — {exc}")
            st.stop()

    payload = result.payload
    meta = payload.get("_meta", {}) or {}
    is_refusal = payload.get("answer_type") == "refusal"

    for sub in payload.get("substitutions_surfaced", []) or []:
        st.markdown(
            f'<div class="pr-sub">Interpreted <b>{theme.e(sub.get("from"))}</b> as '
            f'<b>{theme.e(sub.get("to"))}</b> — substitution surfaced, never silent.</div>',
            unsafe_allow_html=True,
        )

    if is_refusal:
        ref = payload.get("refusal", {}) or {}
        st.markdown(
            theme.refusal_block(
                str(ref.get("reason_code", "")),
                str(payload.get("summary", "")),
                str(ref.get("what_would_help", "")),
            ),
            unsafe_allow_html=True,
        )
        if payload.get("candidates"):
            st.markdown('<div class="pr-h">Did you mean</div>', unsafe_allow_html=True)
            cols = st.columns(min(4, len(payload["candidates"])))
            for i, cand in enumerate(payload["candidates"]):
                cols[i % len(cols)].button(
                    f"{cand['name']}  ·  {cand['score']}",
                    use_container_width=True,
                    key=f"cand-{cand['name']}",
                    on_click=_set_q,
                    args=(f"{cand['name']} {question}",),
                )
    else:
        st.markdown(
            theme.answer_block(str(payload.get("summary", "")), payload.get("claims", []) or []),
            unsafe_allow_html=True,
        )
        st.markdown(theme.facets(payload), unsafe_allow_html=True)
        if result.guardrails:
            st.markdown('<div class="pr-h">Verification</div>', unsafe_allow_html=True)
            st.markdown(theme.guardrail_row(result.guardrails), unsafe_allow_html=True)
        if payload.get("omission_notice"):
            st.info(payload["omission_notice"], icon="⚠️")

    sources = payload.get("sources", []) or []
    if sources:
        st.markdown('<div class="pr-h">Evidence</div>', unsafe_allow_html=True)
        st.caption(
            f"{len(sources)} passages assembled in safety-tier order — critical "
            "sections lead regardless of relevance score."
        )
        for src in sources:
            st.markdown(theme.source_card(src), unsafe_allow_html=True)

    with st.expander("Audit record — how this answer can be reconstructed"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{meta.get('total_ms', sum(s.ms for s in stages)):,.0f} ms")
        m2.metric("Cost", f"${meta.get('cost_usd', 0) or 0:.5f}")
        m3.metric("Retrieved", len(result.retrieved_ids))
        m4.metric("In context", len(result.context_ids))
        st.markdown(
            f'<span class="pr-srcid">query_id {theme.e(result.query_id)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="pr-h" style="margin-top:1rem">Stages</div>', unsafe_allow_html=True
        )
        st.markdown(theme.stage_rows(stages), unsafe_allow_html=True)
        st.caption(
            "Every query — answered or refused — writes an append-only row capturing "
            "the evidence, model and prompt versions, guardrail verdicts, latency and "
            "cost. First-person clinical queries are redacted before logging."
        )

st.markdown(
    f'<div class="pr-foot">{theme.e(DISCLAIMER)}<br>'
    "Corpus: U.S. FDA Structured Product Labeling via DailyMed (public domain). "
    "Identity and class: RxNorm / RxClass, U.S. National Library of Medicine."
    "</div>",
    unsafe_allow_html=True,
)
