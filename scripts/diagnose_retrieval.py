"""Diagnose a high false-refusal rate. Run this BEFORE reviewing any silver items.

Reviewing 229 items built on a broken corpus wastes the 45 minutes, and a
calibrator fitted on that data is garbage-in.

Traces the funnel stage by stage so the failure has one location, not five
candidates:

    corpus health -> gazetteer coverage -> resolution -> filter -> retrieval
    -> rerank -> threshold

    uv run python scripts/diagnose_retrieval.py
    uv run python scripts/diagnose_retrieval.py --sample 40 --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "packages/spl_parser/src", ROOT):
    sys.path.insert(0, str(p))

from pharmarag.config import DATA, THRESHOLD_INCLUDE  # noqa: E402
from pharmarag.db import session  # noqa: E402

SILVER = ROOT / "eval" / "data" / "silver.jsonl"
GOLDEN = ROOT / "eval" / "data" / "golden.jsonl"

BAR = "─" * 72


def corpus_health() -> dict[str, object]:
    print(BAR)
    print("1 · CORPUS HEALTH")
    with session() as conn:
        docs = conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) n FROM chunks WHERE retrievable=1").fetchone()["n"]
        per_doc = [
            r["n"]
            for r in conn.execute(
                "SELECT d.set_id, COUNT(c.chunk_id) n FROM documents d "
                "LEFT JOIN chunks c ON c.set_id=d.set_id AND c.retrievable=1 "
                "GROUP BY d.set_id"
            )
        ]
        null_rxcui = conn.execute(
            "SELECT COUNT(*) n FROM documents WHERE rxcui IS NULL"
        ).fetchone()["n"]
        by_type = dict(
            conn.execute(
                "SELECT content_type, COUNT(*) n FROM chunks WHERE retrievable=1 "
                "GROUP BY content_type"
            ).fetchall()
            and [
                (r["content_type"], r["n"])
                for r in conn.execute(
                    "SELECT content_type, COUNT(*) n FROM chunks WHERE retrievable=1 "
                    "GROUP BY content_type"
                )
            ]
        )

    empty = sum(1 for n in per_doc if n == 0)
    thin = sum(1 for n in per_doc if 0 < n < 50)
    healthy = sum(1 for n in per_doc if n >= 50)
    med = statistics.median(per_doc) if per_doc else 0

    print(f"  documents                {docs:>8,}")
    print(f"  retrievable chunks       {chunks:>8,}")
    print(f"  median chunks/document   {med:>8.0f}")
    print(f"  documents with 0 chunks  {empty:>8,}   <- parse or fetch failed")
    print(f"  thin documents (<50)     {thin:>8,}   <- minimal labels, not full PI")
    print(f"  healthy documents (50+)  {healthy:>8,}")
    print(f"  documents with NULL rxcui{null_rxcui:>8,}")
    print(f"  by content_type          {by_type}")

    if med < 50:
        print()
        print("  ⚠️  Median well below the ~310 chunks/drug measured on curated")
        print("      labels. Most documents are minimal entries (OTC, kits,")
        print("      discontinued) rather than full prescribing information.")
        print("      build_snapshot takes hits[0] with no quality check — that is")
        print("      the bug. See --fix output below.")
    return {
        "docs": docs,
        "chunks": chunks,
        "median": med,
        "empty": empty,
        "thin": thin,
        "healthy": healthy,
    }


def gazetteer_health() -> set[str]:
    print(BAR)
    print("2 · GAZETTEER vs CORPUS")
    path = DATA / "gazetteer.json"
    if not path.is_file():
        print("  ✗ no gazetteer.json — run scripts/build_entities.py")
        return set()
    gaz: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    with session() as conn:
        names = {
            r["n"]
            for r in conn.execute(
                "SELECT DISTINCT LOWER(ingredient_name) n FROM documents "
                "WHERE ingredient_name IS NOT NULL"
            )
        }
        chunk_rx = {
            r["r"]
            for r in conn.execute(
                "SELECT DISTINCT rxcui r FROM chunks WHERE retrievable=1 AND rxcui IS NOT NULL"
            )
        }

    values = set(gaz.values())
    overlap = values & chunk_rx
    print(f"  gazetteer entries        {len(gaz):>8,}")
    print(f"  distinct ingredients     {len(names):>8,}")
    print(f"  distinct chunk.rxcui     {len(chunk_rx):>8,}")
    print(f"  gazetteer∩chunk rxcui    {len(overlap):>8,}   <- MUST be high")
    if chunk_rx and len(overlap) / max(len(chunk_rx), 1) < 0.8:
        print()
        print("  ⚠️  Gazetteer values do not match chunk.rxcui. Entity-first")
        print("      filtering will return nothing. Rebuild entities AFTER the index.")
    return chunk_rx


def trace(sample: int, verbose: bool) -> None:
    print(BAR)
    print(f"3 · PIPELINE FUNNEL  (n={sample})")

    from fastembed import SparseTextEmbedding

    from pharmarag.embed.client import embed_query
    from pharmarag.entity.gazetteer import Gazetteer
    from pharmarag.entity.lasa import load as load_lasa
    from pharmarag.entity.resolve import Resolver
    from pharmarag.index.store import get_client
    from pharmarag.rerank.reranker import Reranker
    from pharmarag.retrieve.search import EmptyCandidateSetError, hybrid_search

    items = [
        json.loads(line)
        for line in (GOLDEN if GOLDEN.is_file() else SILVER)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    items = [
        i
        for i in items
        if not i.get("must_refuse")
        and i.get("gold_citation_ids")
        and "*" not in i["gold_citation_ids"]
    ][:sample]
    if not items:
        print("  ✗ no answerable items — run eval/autogen.py first")
        return

    gaz_path = DATA / "gazetteer.json"
    resolver = Resolver(
        Gazetteer.load(gaz_path) if gaz_path.is_file() else Gazetteer(),
        load_lasa(DATA / "lasa_table.json"),
    )
    client, reranker = get_client(), Reranker()
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

    stages = Counter()
    res_types = Counter()
    gold_ranks: list[int] = []
    gold_scores: list[float] = []

    for item in items:
        q = item["question"]
        gold = set(item["gold_citation_ids"])

        res = resolver.resolve(q)
        res_types[res.type.value] += 1
        if not res.rxcuis:
            stages["A · resolution found no drug"] += 1
            if verbose:
                print(f"    [no-drug] {q[:70]}")
            continue

        try:
            cands = hybrid_search(
                client,
                dense_vector=embed_query(q),
                sparse_vector=next(iter(bm25.embed([q]))),
                rxcuis=res.rxcuis,
                limit=70,
            )
        except EmptyCandidateSetError:
            stages["B · filter returned zero candidates"] += 1
            if verbose:
                print(f"    [empty-filter] rxcuis={res.rxcuis[:3]} · {q[:55]}")
            continue

        ids = [c.chunk_id for c in cands]
        if not (gold & set(ids)):
            stages["C · gold chunk not retrieved"] += 1
            if verbose:
                print(f"    [gold-missing] {len(cands)} cands · {q[:55]}")
            continue

        scored = reranker.score(q, cands)
        rank = next((i for i, s in enumerate(scored, 1) if s.chunk_id in gold), None)
        score = next((s.calibrated_score for s in scored if s.chunk_id in gold), 0.0)
        gold_ranks.append(rank or 999)
        gold_scores.append(score)

        if scored and scored[0].calibrated_score < THRESHOLD_INCLUDE:
            stages["D · top score below THRESHOLD_INCLUDE"] += 1
        elif score < THRESHOLD_INCLUDE:
            stages["E · gold retrieved but below threshold"] += 1
        else:
            stages["F · reached synthesis ✓"] += 1

    print(f"  resolution types: {dict(res_types)}")
    print()
    for stage, n in sorted(stages.items()):
        bar = "█" * int(40 * n / max(len(items), 1))
        print(f"  {stage:42s} {n:>3}  {bar}")
    if gold_ranks:
        print()
        print(
            f"  gold chunk rank    median {statistics.median(gold_ranks):.0f} "
            f"· best {min(gold_ranks)}"
        )
        print(
            f"  gold chunk score   median {statistics.median(gold_scores):.3f} "
            f"· threshold {THRESHOLD_INCLUDE}"
        )

    print()
    print(BAR)
    print("VERDICT")
    top = stages.most_common(1)[0][0] if stages else "unknown"
    if top.startswith("A"):
        print("  Entity resolution. The gazetteer is missing drugs or its values")
        print("  do not match chunk.rxcui. Rebuild entities AFTER the index.")
    elif top.startswith("B"):
        print("  The RxCUI pre-filter. Resolution yields identifiers that match no")
        print("  chunk payload. Check gazetteer values vs chunks.rxcui in section 2.")
    elif top.startswith("C"):
        print("  Retrieval. The gold chunk exists but never enters the top 70.")
        print("  Suspect embedding/index drift — did you re-index after changing")
        print("  chunk_id? Stale point ids retain old payloads.")
    elif top.startswith(("D", "E")):
        print(f"  Thresholding. Retrieval works; scores sit under {THRESHOLD_INCLUDE}.")
        print("  bge raw scores are NOT probabilities. Lower THRESHOLD_INCLUDE to")
        print("  ~0.35 and record it, or fit the calibrator on GOOD retrieval data.")
    else:
        print("  Pipeline reaches synthesis. Failures are downstream (guardrails).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-trace", action="store_true")
    args = ap.parse_args()

    health = corpus_health()
    gazetteer_health()
    if not args.skip_trace:
        trace(args.sample, args.verbose)

    print(BAR)
    print("SUGGESTED FIX ORDER")
    if health["median"] < 50:
        print("  1. Re-fetch the corpus with a label-quality filter (see below).")
        print("  2. Rebuild index -> rebuild entities -> rebuild graph, IN THAT ORDER.")
        print("  3. Re-run autogen + autolabel. The current silver set was generated")
        print("     from thin labels and is not worth reviewing.")
    else:
        print("  Corpus looks healthy. Fix whatever the funnel verdict names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
