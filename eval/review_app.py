"""Fast review — silver to gold at ~10 seconds per item.

This is the compromise that keeps your differentiator intact. Full labeling was
~15 hours; reviewing pre-generated items is ~45 minutes for 250. You are not
writing questions, only answering: is this a fair test?

Keyboard-free, three buttons. Accept promotes to gold. Reject drops it. Edit
opens the fields, and an edited item is gold too — you changed it, so you own it.

    uv run streamlit run eval/review_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "packages/spl_parser/src", ROOT):
    sys.path.insert(0, str(p))

from eval.schema import append_jsonl, read_jsonl  # noqa: E402
from pharmarag.db import session  # noqa: E402

SILVER = ROOT / "eval" / "data" / "silver.jsonl"
GOLDEN = ROOT / "eval" / "data" / "golden.jsonl"
REJECTED = ROOT / "eval" / "data" / "rejected.jsonl"

st.set_page_config(page_title="PharmaRAG · Review", page_icon="✅", layout="wide")

silver = read_jsonl(SILVER)
gold = read_jsonl(GOLDEN)
rejected = read_jsonl(REJECTED)
seen = {i["item_id"] for i in gold} | {i["item_id"] for i in rejected}
queue = [i for i in silver if i["item_id"] not in seen]

with st.sidebar:
    st.header("Review")
    reviewer = st.text_input("Reviewer", value="aly")
    st.metric("Promoted to gold", len(gold))
    st.metric("Rejected", len(rejected))
    st.metric("Remaining", len(queue))
    if silver:
        st.progress(len(seen) / len(silver))
    st.divider()
    st.caption(
        "**Accept if the item is a FAIR TEST.** You are not checking whether the "
        "system answers it — only whether a pharmacist would call it a reasonable "
        "question with the right evidence attached."
    )
    st.caption("~10 s/item. 250 items ≈ 45 min.")

st.title("✅ Silver → Gold")

if not queue:
    st.success(f"Queue empty. {len(gold)} gold items, {len(rejected)} rejected.")
    if gold:
        by_cat: dict[str, int] = {}
        for g in gold:
            by_cat[g["category"]] = by_cat.get(g["category"], 0) + 1
        st.json(by_cat)
    st.stop()

item = queue[0]
st.caption(f"`{item['item_id']}` · {len(queue)} left")

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader(item["question"])
    tags = f"**{item['category']}** · {item['difficulty']}"
    if item.get("must_refuse"):
        tags += f" · ⛔ must refuse → `{item.get('expected_reason_code')}`"
    st.markdown(tags)
    if item.get("notes"):
        st.info(item["notes"], icon="📝")
with c2:
    ids = [i for i in (item.get("gold_citation_ids") or []) if i != "*"]
    if ids:
        with session() as conn:
            marks = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT chunk_id, ingredient_name, section_path, display_text "
                f"FROM chunks WHERE chunk_id IN ({marks})",
                tuple(ids),
            ).fetchall()
        for r in rows:
            st.caption(f"**{r['ingredient_name']}** · {r['section_path']}")
            st.code(r["display_text"][:900], language=None)
    elif item.get("must_refuse"):
        st.warning("No evidence by design — the corpus must not answer this.", icon="⛔")

st.divider()
a, b, c = st.columns(3)


def _finish(dest: Path, **extra) -> None:
    rec = {**item, **extra, "reviewed_by": reviewer}
    append_jsonl(dest, rec)
    st.rerun()


if a.button("✅ Accept — fair test", type="primary", use_container_width=True):
    _finish(GOLDEN, validated_by_human=True, provenance="llm_generated_human_reviewed")

if b.button("✏️ Edit first", use_container_width=True):
    st.session_state.editing = True

if c.button("❌ Reject", use_container_width=True):
    _finish(REJECTED, validated_by_human=True, rejection="not a fair test")

if st.session_state.get("editing"):
    with st.form("edit"):
        q = st.text_area("Question", value=item["question"], height=80)
        notes = st.text_area("Notes", value=item.get("notes", ""), height=60)
        diff = st.selectbox(
            "Difficulty",
            ["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(item["difficulty"]),
        )
        if st.form_submit_button("Save as gold", type="primary"):
            st.session_state.editing = False
            _finish(
                GOLDEN,
                question=q,
                notes=notes,
                difficulty=diff,
                validated_by_human=True,
                provenance="llm_generated_human_edited",
            )
