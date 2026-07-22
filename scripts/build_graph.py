"""Milestone B7/B8 — build the deterministic property graph.

    uv run python scripts/build_graph.py --offline    # curated risk classes only, $0, seconds
    uv run python scripts/build_graph.py              # full RxClass harvest, ~10 min

NETWORK: the full run makes ~2 RxNav calls per drug. If you are behind a
TLS-inspecting proxy (Avast, Zscaler), set PHARMARAG_CA_BUNDLE first:

    $env:PHARMARAG_CA_BUNDLE = "certs/ca-bundle-avast.pem"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages/spl_parser/src"))

from pharmarag.config import DATA, settings  # noqa: E402
from pharmarag.db import session  # noqa: E402
from pharmarag.entity.gazetteer import Gazetteer  # noqa: E402
from pharmarag.graph.build import (  # noqa: E402
    build_graph,
    extract_interaction_edges,
    graph_stats,
    save,
)
from pharmarag.graph.traverse import class_burden, plan_regimen  # noqa: E402
from pharmarag.ingest.selection import track_a_slice  # noqa: E402


def load_names() -> dict[str, str]:
    from pharmarag.ingest.selection import load_expanded, selection_path

    try:
        with session() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ingredient_name, rxcui FROM documents "
                "WHERE ingredient_name IS NOT NULL"
            ).fetchall()
        if rows:
            names = {r["ingredient_name"]: (r["rxcui"] or r["ingredient_name"]) for r in rows}
            # Restrict to the frozen canonical corpus so leftover rows from earlier
            # builds never add a node the retrievable corpus does not contain.
            if selection_path().is_file():
                allow = {n.strip().lower() for n in load_expanded()}
                names = {n: rx for n, rx in names.items() if n.strip().lower() in allow}
            return names
    except Exception:
        pass
    print("[graph] no corpus in DB — using the Track A selection list")
    return {n: n for n in track_a_slice()}


def load_interaction_chunks() -> list[dict[str, object]]:
    try:
        with session() as conn:
            rows = conn.execute(
                "SELECT chunk_id, rxcui, display_text, loinc_section_code FROM chunks "
                "WHERE loinc_section_code='34073-7' AND retrievable=1"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip RxNav; curated risk classes only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = load_names()
    print(f"[graph] {len(names)} drugs")

    harvest = None
    if not args.offline:
        from pharmarag.graph.rxclass import harvest as run_harvest
        from pharmarag.graph.rxclass import save as save_harvest

        print("[graph] harvesting RxClass (this is the ~10 min step)…")
        harvest = run_harvest(sorted(names))
        save_harvest(harvest, DATA / "rxclass_harvest.json")
        print(f"[graph] harvest: {harvest.summary()}")
        if harvest.errors:
            print(f"[graph] {len(harvest.errors)} unresolved: " f"{list(harvest.errors)[:6]}")
        names = {n: rx for n, rx in harvest.resolved.items()} or names

    gaz_path = DATA / "gazetteer.json"
    interaction_edges = []
    if gaz_path.exists():
        chunks = load_interaction_chunks()
        if chunks:
            interaction_edges = extract_interaction_edges(chunks, Gazetteer.load(gaz_path))
            print(
                f"[graph] {len(interaction_edges)} interaction edges from "
                f"{len(chunks)} §Drug Interactions chunks"
            )

    g = build_graph(harvest, interaction_edges, name_to_rxcui=names)
    stats = graph_stats(g)
    print(f"[graph] {json.dumps(stats, indent=2)}")

    version = f"graph-{settings.corpus_version}"
    out = Path(args.out) if args.out else DATA / "graph" / f"{version}.json"
    save(g, out)
    print(f"[graph] wrote {out}")
    print(f"[graph] set GRAPH_VERSION={version} in .env")

    # Smoke-test the safety feature the graph exists for.
    demo = [rx for n, rx in names.items() if n in {"ibuprofen", "lisinopril", "furosemide"}]
    if len(demo) == 3:
        plan = plan_regimen(g, demo)
        print("\n[graph] ADR-033 smoke test — triple whammy:")
        for a in plan.combination_alerts:
            print(f"  *** {a.label}")
            print(f"      {a.detail}")
    else:
        risks, _ = class_burden(g, list(names.values())[:6])
        print(f"\n[graph] class-burden smoke test: {[r.risk_class for r in risks]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
