"""Automated calibration judgments (ADR-026, ADR-044).

The calibrator needs (raw_score, is_relevant) pairs at the CANDIDATE level. This
produces them without you judging 800 chunks by hand.

Two signals, combined:

  1. DISTANT SUPERVISION — a candidate is relevant if it IS the gold chunk, or
     shares its parent. Free, deterministic, high precision, LOW RECALL: it
     marks genuinely-relevant siblings as irrelevant.

  2. LLM JUDGE — the same rubric a human would apply. ~$0.40 for 800 judgments.
     Higher recall, noisier.

Why both: distant supervision alone trains a calibrator that is systematically
over-confident that things are NOT relevant, which produces OVER-ABSTENTION —
precisely the failure a recall-first system must avoid. The LLM judge corrects
that, and disagreements between the two are surfaced as a review queue.

    uv run python eval/autolabel.py --queries 40 --topk 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "packages/spl_parser/src", ROOT):
    sys.path.insert(0, str(p))

from eval.schema import CalibrationJudgment, append_jsonl, read_jsonl  # noqa: E402
from pharmarag.config import DATA, MODEL_EVALUATOR  # noqa: E402

SILVER = ROOT / "eval" / "data" / "silver.jsonl"
CALIB = ROOT / "eval" / "data" / "calibration.jsonl"
DISAGREE = ROOT / "eval" / "data" / "calibration_disagreements.jsonl"

JUDGE = """Would this PASSAGE alone let a pharmacist answer the QUESTION correctly?

YES  — the passage contains the complete answer.
NO   — it is merely related, about the right drug but the wrong fact, or needs \
another passage to be complete.

Being about the right drug is not enough. Reply with one word: YES or NO."""


def judge(question: str, passage: str) -> bool | None:
    from pharmarag.http import openai_client

    try:
        r = openai_client().chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": JUDGE},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nPASSAGE:\n{passage[:1500]}"},
            ],
            max_completion_tokens=4,
        )
        return (r.choices[0].message.content or "").strip().upper().startswith("YES")
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--no-llm", action="store_true", help="distant supervision only, $0")
    args = ap.parse_args()

    from fastembed import SparseTextEmbedding

    from pharmarag.embed.client import embed_query
    from pharmarag.entity.gazetteer import Gazetteer
    from pharmarag.entity.lasa import load as load_lasa
    from pharmarag.entity.resolve import Resolver
    from pharmarag.index.store import get_client
    from pharmarag.rerank.reranker import Reranker
    from pharmarag.retrieve.search import EmptyCandidateSetError, hybrid_search

    items = [
        i
        for i in read_jsonl(SILVER)
        if not i.get("must_refuse")
        and i.get("gold_citation_ids")
        and "*" not in i["gold_citation_ids"]
    ][: args.queries]
    if not items:
        print("[autolabel] no answerable items — run eval/autogen.py first", file=sys.stderr)
        return 2

    gaz = DATA / "gazetteer.json"
    resolver = Resolver(
        Gazetteer.load(gaz) if gaz.exists() else Gazetteer(), load_lasa(DATA / "lasa_table.json")
    )
    client, reranker = get_client(), Reranker()
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

    done = {(j["query_id"], j["chunk_id"]) for j in read_jsonl(CALIB)}
    written = agree = disagree = 0

    for n, item in enumerate(items, 1):
        res = resolver.resolve(item["question"])
        if not res.rxcuis:
            continue
        try:
            cands = hybrid_search(
                client,
                dense_vector=embed_query(item["question"]),
                sparse_vector=next(iter(bm25.embed([item["question"]]))),
                rxcuis=res.rxcuis,
                limit=70,
            )
        except EmptyCandidateSetError:
            continue

        scored = reranker.score(item["question"], cands)[: args.topk]
        gold = set(item["gold_citation_ids"])
        gold_parents = {c.payload.get("parent_chunk_id") for c in cands if c.chunk_id in gold}

        for rank, s in enumerate(scored, 1):
            if (item["item_id"], s.chunk_id) in done:
                continue
            distant = s.chunk_id in gold or s.payload.get("parent_chunk_id") in gold_parents
            llm = (
                None
                if args.no_llm
                else judge(item["question"], str(s.payload.get("display_text", "")))
            )
            # Distant supervision is high-precision: a gold hit is relevant, full
            # stop. Elsewhere trust the judge, which has the better recall.
            relevant = True if distant else (bool(llm) if llm is not None else False)
            if llm is not None and llm != distant:
                disagree += 1
                append_jsonl(
                    DISAGREE,
                    {
                        "query_id": item["item_id"],
                        "chunk_id": s.chunk_id,
                        "question": item["question"],
                        "distant": distant,
                        "llm": llm,
                        "rank": rank,
                        "raw_score": float(s.raw_score),
                        "passage": str(s.payload.get("display_text", ""))[:400],
                    },
                )
            else:
                agree += 1

            append_jsonl(
                CALIB,
                CalibrationJudgment(
                    query_id=item["item_id"],
                    chunk_id=s.chunk_id,
                    raw_score=float(s.raw_score),
                    is_relevant=relevant,
                    category=item["category"],
                    rank=rank,
                    reviewed_by="auto",
                    reviewed_at="",
                ).to_dict(),
            )
            written += 1

        if n % 5 == 0:
            print(f"[autolabel] {n}/{len(items)} queries · {written} judgments", flush=True)

    total = agree + disagree
    print(f"\n[autolabel] {written} judgments written")
    if total:
        print(f"[autolabel] distant/LLM agreement: {agree}/{total} ({100*agree/total:.0f}%)")
        print(f"[autolabel] {disagree} disagreements -> {DISAGREE}")
        print("[autolabel] Reviewing ONLY the disagreements is the highest-value")
        print("[autolabel] human pass here — they are where the labels are uncertain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
