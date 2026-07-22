"""Milestone A5: gazetteer + LASA table from the indexed corpus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from pharmarag.config import DATA
from pharmarag.db import session
from pharmarag.entity import brands as brand_mod
from pharmarag.entity.gazetteer import Gazetteer
from pharmarag.entity.lasa import build_lasa_table, save
from pharmarag.ingest.selection import load_expanded, selection_path, track_a_slice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--brands",
        action="store_true",
        help="re-harvest brand-name aliases from RxNav (network, $0, ~10 min)",
    )
    args = ap.parse_args()
    entries: dict[str, str] = {}
    try:
        with session() as conn:
            for r in conn.execute(
                "SELECT DISTINCT ingredient_name, rxcui FROM documents WHERE ingredient_name IS NOT NULL"
            ):
                entries[r["ingredient_name"]] = r["rxcui"] or r["ingredient_name"]
    except Exception:
        pass

    # Restrict to the frozen canonical corpus so leftover rows from earlier builds
    # never leak a drug into the gazetteer that Qdrant cannot actually retrieve.
    if selection_path().is_file():
        allow = {n.strip().lower() for n in load_expanded()}
        before = len(entries)
        entries = {n: rx for n, rx in entries.items() if n.strip().lower() in allow}
        print(f"[entities] restricted to frozen corpus: {len(entries)}/{before} drugs")

    if not entries:
        print("[entities] no documents in DB — falling back to the Track A selection list")
        entries = {n: n for n in track_a_slice()}

    # Brand aliases (ADR-020): "Lipitor" must resolve to atorvastatin, and the
    # substitution is surfaced to the user rather than applied silently.
    if args.brands:
        print(f"[brands] harvesting from RxNav for {len(entries)} ingredients (network, $0)")
        mapping, ambiguous = brand_mod.harvest(sorted(entries))
        brand_mod.save(mapping)
        print(f"[brands] {len(mapping)} unambiguous aliases -> {brand_mod.brands_path()}")
        print(
            f"[brands] {len(ambiguous)} combination-product brands excluded "
            f"(map to >1 corpus ingredient)"
        )
        for b, ings in list(ambiguous.items())[:5]:
            print(f"[brands]   excluded {b} -> {ings}")
    brand_aliases = brand_mod.load()

    g = Gazetteer(entries)
    if brand_aliases:
        # Aliases are added AFTER ingredients so a brand can never overwrite a
        # real ingredient name that happens to collide with it.
        g.add_many({b: entries[i] for b, i in brand_aliases.items() if i in entries})
        print(f"[entities] + {len(brand_aliases)} brand aliases folded in")
    g.save(DATA / "gazetteer.json")
    print(f"[entities] gazetteer: {len(g)} names -> {DATA / 'gazetteer.json'}")

    table = build_lasa_table(sorted(entries), threshold=0.88)
    save(table, DATA / "lasa_table.json")
    print(f"[entities] LASA table: {len(table)} names with confusable neighbours")
    for k, v in list(table.items())[:10]:
        print(f"[entities]   {k:18s} <-> {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
