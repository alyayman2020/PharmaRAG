"""Milestone B3 — scale the corpus 50 -> 1,500 by stratified selection (ADR-003).

The Track A slice is 50 hand-picked drugs (``selection.track_a_slice``). Track B
needs ~1,500 ingredient-level concepts, and "top-1,500 by prescription volume"
would over-represent cardiovascular and CNS drugs and miss whole therapeutic
areas. ADR-003 commits to *stratified* selection instead: sample across the ATC
anatomical main groups so every therapeutic area — and therefore every safety
mechanism — has something to bite on.

The strata are RxClass's ATC level-1 classes. For each we pull the member
ingredients, drop combination products (ADR-003 is ingredient-level, one
canonical label per ingredient), then fill the quota round-robin across classes
so no single group dominates. The curated Track A slice is always included so the
five hand-built safety strata and the LASA pairs survive the scale-up.

    uv run python -m pharmarag.ingest.select --n 1500 --strategy stratified
    uv run python -m pharmarag.ingest.select --n 1500 --strategy stratified --ingest
    uv run python -m pharmarag.ingest.select --n 40 --strategy stratified --limit 40 --ingest --snapshot dailymed-smoke

NETWORK REQUIRED for the stratified strategy (RxClass ATC walk, ~14 calls) and
for --ingest (DailyMed SPL fetch). --strategy curated is offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pharmarag.config import DATA
from pharmarag.ingest.selection import track_a_slice

SELECTION_DIR = DATA / "selection"

# ATC level-1 anatomical main groups — the strata (ADR-003). members_of_class on
# each of these single-letter classIds returns every ingredient RxNorm files under
# that therapeutic area.
ATC1_SEEDS: dict[str, str] = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood-forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Anti-infectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}


def _is_ingredient(name: str) -> bool:
    """Keep single ingredient concepts, drop combination products (ADR-003).

    RxClass returns combinations like ``abacavir / lamivudine`` and packs like
    ``amoxicillin, clavulanate``. Those are not ingredient-level concepts and
    would each duplicate labeling already covered by their components, so the
    canonical-label rule excludes them here rather than downstream.
    """
    n = name.strip().lower()
    if not n:
        return False
    return not any(sep in n for sep in ("/", ",", " and ", " with ", ";", "+"))


def _class_members(class_id: str) -> list[str]:
    """Sorted, de-duplicated, ingredient-only member names for one ATC class."""
    from pharmarag.graph.rxclass import members_of_class

    members = members_of_class(class_id, rela_source="ATC")
    return sorted({name.lower() for _rxcui, name in members if _is_ingredient(name)})


def stratified_selection(
    n: int = 1500,
    *,
    delay: float = 0.1,
    progress: bool = True,
) -> dict[str, object]:
    """Select up to ``n`` ingredients, stratified across the ATC-1 groups.

    Returns a provenance dict (not just a list) so the selection is auditable:
    which strata contributed, and how many from each.
    """
    core = track_a_slice()
    picked: dict[str, None] = dict.fromkeys(core)  # curated core, guaranteed
    provenance: dict[str, str] = {name: "track_a_core" for name in core}

    buckets: dict[str, list[str]] = {}
    for cid, label in ATC1_SEEDS.items():
        try:
            names = _class_members(cid)
        except Exception as exc:
            if progress:
                print(f"[select] ATC {cid} ({label}) failed: {exc}", file=sys.stderr)
            names = []
        buckets[cid] = names
        if progress:
            print(f"[select] ATC {cid} {label:42s} {len(names):4d} ingredients", flush=True)
        time.sleep(delay)

    # Round-robin fill: take the i-th name from each stratum in turn so the quota
    # is spread evenly rather than exhausting one therapeutic area first.
    i = 0
    while len(picked) < n:
        advanced = False
        for cid, names in buckets.items():
            if len(picked) >= n:
                break
            if i < len(names):
                advanced = True
                name = names[i]
                if name not in picked:
                    picked[name] = None
                    provenance[name] = f"ATC:{cid}"
        if not advanced:  # every stratum exhausted
            break
        i += 1

    selected = list(picked)[:n]
    by_stratum: dict[str, int] = {}
    for name in selected:
        by_stratum[provenance[name]] = by_stratum.get(provenance[name], 0) + 1
    return {
        "strategy": "stratified",
        "n_requested": n,
        "n_selected": len(selected),
        "by_stratum": dict(sorted(by_stratum.items())),
        "atc_seeds": ATC1_SEEDS,
        "drugs": selected,
    }


def curated_selection() -> dict[str, object]:
    """Offline fallback: the 50-drug Track A slice, no network."""
    drugs = track_a_slice()
    return {
        "strategy": "curated",
        "n_requested": len(drugs),
        "n_selected": len(drugs),
        "by_stratum": {"track_a_core": len(drugs)},
        "drugs": drugs,
    }


def select(n: int, strategy: str) -> dict[str, object]:
    if strategy == "curated":
        return curated_selection()
    if strategy == "stratified":
        return stratified_selection(n)
    raise ValueError(f"unknown strategy {strategy!r} (use 'stratified' or 'curated')")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stratified drug selection + optional ingest (ADR-003)."
    )
    ap.add_argument("--n", type=int, default=1500, help="target corpus size (hard cap, ADR-003)")
    ap.add_argument("--strategy", choices=("stratified", "curated"), default="stratified")
    ap.add_argument(
        "--snapshot",
        default=None,
        help="snapshot dir under data/snapshots/ (default: $CORPUS_VERSION, else today)",
    )
    ap.add_argument(
        "--out", default=None, help="selection JSON path (default: data/selection/<snapshot>.json)"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-select even if a cached selection exists"
    )
    ap.add_argument(
        "--ingest",
        action="store_true",
        help="after selecting, fetch + archive one SPL per drug from DailyMed",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of drugs actually ingested (for smoke tests)",
    )
    ap.add_argument(
        "--delay", type=float, default=0.1, help="throttle between DailyMed fetches (s)"
    )
    args = ap.parse_args()

    from pharmarag.config import settings
    from pharmarag.ingest.dailymed import snapshot_id

    snap = args.snapshot or (
        settings.corpus_version if settings.corpus_version != "dev" else snapshot_id()
    )
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else SELECTION_DIR / f"{snap}.json"

    if out_path.is_file() and not args.force:
        result = json.loads(out_path.read_text(encoding="utf-8"))
        print(
            f"[select] using cached selection {out_path} "
            f"({result['n_selected']} drugs; pass --force to re-select)"
        )
    else:
        result = select(args.n, args.strategy)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[select] {result['strategy']}: {result['n_selected']} drugs -> {out_path}")
        print(f"[select] by stratum: {json.dumps(result['by_stratum'])}")

    if not args.ingest:
        print(
            "[select] selection only. Re-run with --ingest to fetch SPLs, "
            f"or: uv run python scripts/build_index.py --snapshot {snap}"
        )
        return 0

    from pharmarag.ingest.dailymed import build_snapshot

    drugs = list(result["drugs"])
    if args.limit is not None:
        drugs = drugs[: args.limit]
    print(f"[select] ingesting {len(drugs)} drugs into snapshot {snap} …")
    manifest = build_snapshot(drugs, snapshot=snap, delay=args.delay)
    print(f"[select] snapshot ready: {manifest['count']} SPLs archived under {snap}")
    print(f"[select] next: uv run python scripts/build_index.py --snapshot {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
