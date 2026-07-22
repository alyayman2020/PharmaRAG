"""Milestone C6 — the harness.

uv run python eval/run_eval.py                 # deterministic only, ~$0
uv run python eval/run_eval.py --llm-guard     # full pipeline, costs money
uv run python eval/run_eval.py --calibrate     # fit Platt from judgments
uv run python eval/run_eval.py --coverage      # DDInter corpus coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "packages/spl_parser/src", ROOT):
    sys.path.insert(0, str(p))

from eval.calibrate import abstention_curve, fit  # noqa: E402
from eval.metrics import score  # noqa: E402
from eval.schema import read_jsonl  # noqa: E402
from pharmarag.config import DATA  # noqa: E402

GOLDEN = ROOT / "eval" / "data" / "golden.jsonl"
SILVER = ROOT / "eval" / "data" / "silver.jsonl"
CALIB = ROOT / "eval" / "data" / "calibration.jsonl"
OUT = ROOT / "eval" / "data"


def run_pipeline(items: list[dict], *, use_llm_guard: bool) -> list[dict]:
    from fastembed import SparseTextEmbedding

    from pharmarag.embed.client import embed_query
    from pharmarag.entity.gazetteer import Gazetteer
    from pharmarag.entity.lasa import load as load_lasa
    from pharmarag.entity.resolve import Resolver
    from pharmarag.index.store import get_client
    from pharmarag.pipeline import answer_question
    from pharmarag.rerank.reranker import Reranker

    gaz = DATA / "gazetteer.json"
    resolver = Resolver(
        Gazetteer.load(gaz) if gaz.exists() else Gazetteer(), load_lasa(DATA / "lasa_table.json")
    )
    client, reranker = get_client(), Reranker()
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

    out: list[dict] = []
    for i, item in enumerate(items, 1):
        try:
            ans = answer_question(
                item["question"],
                resolver=resolver,
                client=client,
                reranker=reranker,
                embed_dense=embed_query,
                embed_sparse=lambda q: next(iter(bm25.embed([q]))),
                use_llm_guard=use_llm_guard,
            )
            payload = dict(ans.payload)
            payload["context_assembled_chunk_ids"] = ans.context_ids
            # ADR-045's dose/LASA escape gates are computed from this field.
            # Without it they read an empty dict on every item and pass
            # vacuously — a green gate that can never fire is worse than none.
            payload["guardrail_results"] = ans.guardrails
            out.append(payload)
        except Exception as exc:
            out.append(
                {
                    "answer_type": "refusal",
                    "refusal": {"reason_code": f"ERROR_{type(exc).__name__}"},
                    "claims": [],
                }
            )
        if i % 25 == 0:
            print(f"[eval] {i}/{len(items)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-guard", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.calibrate:
        judgments = read_jsonl(CALIB)
        print(f"[calib] {len(judgments)} judgments")
        if len(judgments) < 100:
            print("[calib] too few — aim for ~800 (40 queries x top-20)", file=sys.stderr)
            return 2
        cal = fit(judgments)
        cal.save(DATA / "calibrator.json")
        (OUT / "abstention_curve.json").write_text(
            json.dumps(abstention_curve(judgments, cal), indent=1), encoding="utf-8"
        )
        print(f"[calib] ECE (out-of-fold) = {cal.reliability['ece']}")
        print(f"[calib] wrote {DATA / 'calibrator.json'}")
        print("[calib] set CALIBRATOR_VERSION=v1 in .env — abstention is now calibrated")
        return 0

    if args.coverage:
        from eval.ddinter import coverage_report

        print(json.dumps(coverage_report(), indent=2))
        return 0

    # Gold first; fall back to silver so you can score before reviewing.
    items = read_jsonl(GOLDEN) or read_jsonl(SILVER)
    items = items[: args.limit]
    if not items:
        print("[eval] no dataset — run: uv run python eval/autogen.py --all", file=sys.stderr)
        return 2

    n_gold = sum(1 for i in items if i.get("validated_by_human", True))
    n_silver = len(items) - n_gold
    print(
        f"[eval] {len(items)} items ({n_gold} gold, {n_silver} silver) · "
        f"llm_guard={args.llm_guard}"
    )
    if n_silver:
        print("[eval] ⚠️  SILVER items are LLM-generated and NOT pharmacist-validated.")
        print("[eval]     Any metric below is provisional. Run eval/review_app.py to promote.")
    results = run_pipeline(items, use_llm_guard=args.llm_guard)
    sc = score(items, results)

    (OUT / "results.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in results), encoding="utf-8"
    )
    (OUT / "scorecard.json").write_text(json.dumps(asdict(sc), indent=2), encoding="utf-8")

    print("\nSCORECARD")
    print(f"  {'dataset_tier':32s} {n_gold} gold / {n_silver} silver")
    for k, v in asdict(sc).items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.3f}")
    print(f"\n  false refusals by reason: {sc.false_refusal_by_reason}")

    gates = sc.ci_gates()
    if gates:
        print("\nCI GATES FAILED (ADR-045 — escape rate must be 0):")
        for g in gates:
            print(f"  ✗ {g}")
        return 1
    print("\nCI GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
