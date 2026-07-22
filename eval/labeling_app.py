"""Milestone C1 — the labeling app. Run this; it unlocks ~20 hours of your time.

Two modes:
  GOLDEN       — 250 question-level items (~3-4 min each with generated drafts)
  CALIBRATION  — 800 candidate-level relevance judgments (~10 s each)

The rubric is enforced on save, not suggested. An item that fails validation
cannot be written, because a dataset you cannot defend is worse than a smaller
one you can.

    uv run streamlit run eval/labeling_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "packages/spl_parser/src", ROOT):
    sys.path.insert(0, str(p))

from eval.candidates import next_candidate  # noqa: E402
from eval.schema import (  # noqa: E402
    RUBRIC,
    CalibrationJudgment,
    Category,
    Difficulty,
    GoldenItem,
    progress,
    read_jsonl,
    save_golden,
    save_judgment,
)
from pharmarag.config import settings  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "data" / "golden.jsonl"
CALIB_PATH = ROOT / "eval" / "data" / "calibration.jsonl"

st.set_page_config(page_title="PharmaRAG · Labeling", page_icon="🏷️", layout="wide")

with st.sidebar:
    st.header("Labeling")
    reviewer = st.text_input("Reviewer", value="aly")
    mode = st.radio("Mode", ["Golden items", "Calibration judgments"])
    st.divider()
    prog = progress(GOLDEN_PATH)
    st.metric("Golden items", f"{prog['labeled']} / {prog['target']}")
    st.progress(min(1.0, prog["labeled"] / prog["target"]))
    st.caption(f"must-refuse labeled: {prog['must_refuse_labeled']} / 45")
    st.metric("Calibration judgments", f"{len(read_jsonl(CALIB_PATH))} / 800")
    st.divider()
    st.caption("**Remaining by category**")
    for cat, n in prog["remaining"].items():
        if n:
            st.caption(f"· {cat}: {n}")

# ---------------------------------------------------------------- golden mode
if mode == "Golden items":
    st.title("🏷️ Golden dataset")

    remaining = {k: v for k, v in prog["remaining"].items() if v}
    default_cat = next(iter(remaining), Category.DDI.value)
    cat = st.selectbox(
        "Category",
        [c.value for c in Category],
        index=[c.value for c in Category].index(default_cat),
        help="The sidebar shows what's still needed. Work the deficit, not your preference.",
    )
    category = Category(cat)
    st.info(RUBRIC.get(cat, RUBRIC["_principle"]), icon="📋")

    must_refuse_cat = category in {Category.OUT_OF_CORPUS, Category.UNSAFE}

    if "cand" not in st.session_state:
        st.session_state.cand = None
        st.session_state.t0 = time.time()

    c1, c2 = st.columns([1, 3])
    with c1:
        prefer_tables = st.checkbox(
            "Prefer table rows",
            value=category is Category.RENAL_HEPATIC,
            help="Exercises the B1 linearization + K3 qualifier path",
        )
        if st.button("Draw a passage", use_container_width=True, disabled=must_refuse_cat):
            with st.spinner("sampling + drafting…"):
                st.session_state.cand = next_candidate(category, prefer_tables=prefer_tables)
                st.session_state.t0 = time.time()
    with c2:
        if must_refuse_cat:
            st.warning(
                "Must-refuse items are written from scratch — there is no source "
                "passage, because the point is that the corpus cannot answer them.",
                icon="⚠️",
            )

    cand = st.session_state.cand
    if cand and not must_refuse_cat:
        st.caption(f"**{cand.drug}** · {cand.section_path or cand.section} · `{cand.chunk_id}`")
        st.code(cand.text[:1800], language=None)

    with st.form("golden", clear_on_submit=True):
        question = st.text_area(
            "Question",
            value="" if must_refuse_cat else (cand.question if cand else ""),
            height=80,
            help="Edit freely. The draft is a time-saver, not an authority.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            difficulty = st.selectbox("Difficulty", [d.value for d in Difficulty], index=1)
            must_refuse = st.checkbox("Must refuse", value=must_refuse_cat)
        with col_b:
            reason_code = st.selectbox(
                "Expected reason code (must-refuse only)",
                [
                    "",
                    "NO_EVIDENCE_IN_CORPUS",
                    "UNSAFE_QUERY",
                    "OUT_OF_SCOPE",
                    "AMBIGUOUS_DRUG",
                    "POPULATION_ONLY_SWEEP",
                    "BELOW_CONFIDENCE_THRESHOLD",
                ],
            )

        gold_ids = st.text_input(
            "Gold citation IDs (comma-separated) — the SUFFICIENT set",
            value="" if must_refuse_cat else (cand.chunk_id if cand else ""),
        )
        supporting = st.text_input(
            "Supporting chunk IDs (relevant but not required)",
            help="Recording these keeps the calibration labels honest — without them, "
            "every un-cited candidate looks irrelevant and the calibrator "
            "becomes over-confident that things are NOT relevant.",
        )
        spans = st.text_area(
            "Gold evidence span(s)",
            height=60,
            help="Paste the exact sentence(s) that answer the question.",
        )

        pairs = risks = ""
        if category is Category.COMPOUND:
            pairs = st.text_input("Expected flagged pairs (a|b, c|d)")
            risks = st.text_input("Expected additive risk classes (comma-separated)")

        notes = st.text_area(
            "Notes",
            height=60,
            help="For LASA items: which drug you MEAN vs which might be retrieved.",
        )

        if st.form_submit_button("Save item", type="primary", use_container_width=True):
            item = GoldenItem(
                question=question.strip(),
                category=category,
                difficulty=Difficulty(difficulty),
                gold_citation_ids=[s.strip() for s in gold_ids.split(",") if s.strip()],
                supporting_chunk_ids=[s.strip() for s in supporting.split(",") if s.strip()],
                gold_evidence_spans=[s for s in spans.split("\n") if s.strip()],
                must_refuse=must_refuse,
                expected_reason_code=reason_code or None,
                expected_pairs=[p.split("|") for p in pairs.split(",") if "|" in p],
                expected_additive_risks=[r.strip() for r in risks.split(",") if r.strip()],
                notes=notes.strip(),
                corpus_version=settings.corpus_version,
                label_duration_s=round(time.time() - st.session_state.get("t0", time.time()), 1),
            )
            errs = save_golden(item, reviewer, GOLDEN_PATH)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.success(f"Saved {item.item_id}")
                st.session_state.cand = None
                st.rerun()

# ---------------------------------------------------------------- calibration
else:
    st.title("🎯 Calibration judgments")
    st.info(
        "**Would this chunk ALONE let a pharmacist answer this question correctly?**\n\n"
        "Yes → relevant. Partially, or it needs another chunk → NOT relevant. "
        "Being about the right drug is not enough. Keep it mechanical — if you "
        "deliberate, the calibrator learns your mood instead of relevance.",
        icon="🎯",
    )

    golden = read_jsonl(GOLDEN_PATH)
    answerable = [g for g in golden if not g.get("must_refuse")]
    if not answerable:
        st.warning("Label some answerable golden items first — calibration needs queries to run.")
        st.stop()

    labeled_pairs = {(j["query_id"], j["chunk_id"]) for j in read_jsonl(CALIB_PATH)}
    target_item = st.selectbox(
        "Question",
        answerable,
        format_func=lambda g: f"[{g['category'][:12]}] {g['question'][:70]}",
    )

    if st.button("Retrieve top-20 candidates", type="primary"):
        with st.spinner("retrieving + reranking…"):
            try:
                from fastembed import SparseTextEmbedding

                from pharmarag.embed.client import embed_query
                from pharmarag.index.store import get_client
                from pharmarag.rerank.reranker import Reranker
                from pharmarag.retrieve.search import hybrid_search

                bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
                cands = hybrid_search(
                    get_client(),
                    dense_vector=embed_query(target_item["question"]),
                    sparse_vector=next(iter(bm25.embed([target_item["question"]]))),
                    rxcuis=target_item.get("resolved_rxcuis") or [],
                    limit=70,
                )
                st.session_state.scored = Reranker().score(target_item["question"], cands)[:20]
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                st.caption(
                    "Tip: golden items need `resolved_rxcuis`, or run this after "
                    "wiring the resolver into the labeling flow."
                )

    for rank, s in enumerate(st.session_state.get("scored", []), 1):
        key = (target_item["item_id"], s.chunk_id)
        if key in labeled_pairs:
            continue
        with st.container(border=True):
            st.caption(
                f"#{rank} · `{s.chunk_id}` · raw {s.raw_score:.3f} · "
                f"{s.payload.get('ingredient_name')} · {s.payload.get('section_path')}"
            )
            st.write(str(s.payload.get("display_text", ""))[:700])
            a, b = st.columns(2)
            for label, is_rel, col in (("✅ Relevant", True, a), ("❌ Not relevant", False, b)):
                if col.button(label, key=f"{label}{s.chunk_id}", use_container_width=True):
                    save_judgment(
                        CalibrationJudgment(
                            query_id=target_item["item_id"],
                            chunk_id=s.chunk_id,
                            raw_score=float(s.raw_score),
                            is_relevant=is_rel,
                            category=target_item["category"],
                            rank=rank,
                        ),
                        reviewer,
                        CALIB_PATH,
                    )
                    st.rerun()
