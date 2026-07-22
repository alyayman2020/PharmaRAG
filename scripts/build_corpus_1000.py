"""Corpus-sizing bridge (ADR-003) — freeze EXACTLY 1000 canonical ingredients.

This is the module the old ``selection.py`` docstring called ``expand_corpus.py``.
It never existed, which is why ``load_expanded`` silently fell back to the 49-drug
Track A slice. This replaces it and closes the loop:

    stratified candidates  ->  DailyMed fetch (quality-gated, resumable)
                           ->  parse discovery (which labels yield real chunks)
                           ->  freeze first 1000 parseable  ->  corpus_selection.json

Deterministic candidate order (so the freeze is reproducible and re-runs reuse
already-archived work):

    1. Track A safety slice        (the five hand-built strata + LASA pairs)
    2. existing stratified selection (data/selection/*.json — the 1,500 already fetched)
    3. fresh stratified ATC expansion (headroom beyond 1,500; NETWORK)
    4. ranked backup ingredients     (documented substitutes for failures)

Usage:
    uv run python scripts/build_corpus_1000.py --snapshot dailymed-2026-07-20
    uv run python scripts/build_corpus_1000.py --snapshot ... --freeze-only  # no network
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/spl_parser/src"))

from pharmarag.config import DATA, DB_MAIN, SNAPSHOTS
from pharmarag.ingest.selection import (
    BACKUP_INGREDIENTS,
    CORPUS_SIZE,
    save_expanded,
    track_a_slice,
)


def _dedup(names: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for n in names:
        k = n.strip().lower()
        if k:
            seen.setdefault(k, None)
    return list(seen)


def candidate_pool(pool_size: int, *, offline: bool) -> list[str]:
    """Deterministic ordered candidate list (see module docstring)."""
    pool: list[str] = list(track_a_slice())

    # 2 · whatever stratified selection was already run and (partly) fetched.
    for sel in sorted((DATA / "selection").glob("*.json")):
        try:
            pool += list(json.loads(sel.read_text(encoding="utf-8")).get("drugs", []))
        except Exception:
            continue

    # 3 · fresh ATC expansion for headroom beyond what has been fetched. The pool
    #     is PERSISTED so later --freeze-only runs see the identical candidate
    #     order without re-walking RxClass.
    if not offline:
        from pharmarag.ingest.select import stratified_selection

        try:
            pool += list(stratified_selection(pool_size, progress=True)["drugs"])
        except Exception as exc:
            print(
                f"[corpus] stratified expansion failed ({exc}); "
                "continuing with existing selection + backups",
                file=sys.stderr,
            )

    # 4 · ranked backups.
    pool += list(BACKUP_INGREDIENTS)
    deduped = _dedup(pool)

    if not offline:
        pool_path = DATA / "selection" / "candidate-pool.json"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(
            json.dumps({"source": "build_corpus_1000 candidate_pool", "drugs": deduped}, indent=1),
            encoding="utf-8",
        )
        print(f"[corpus] candidate pool persisted -> {pool_path}")
    return deduped


def parseable_drugs() -> set[str]:
    """Ingredient names with >=1 retrievable chunk already persisted."""
    conn = sqlite3.connect(DB_MAIN)
    try:
        rows = conn.execute(
            "SELECT DISTINCT lower(ingredient_name) FROM chunks WHERE retrievable=1"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def plan_freeze(pool: list[str], *, snapshot: str) -> tuple[list[str], list[str], set[str]]:
    """Plan the freeze: (canonical names, names skipped as shared-SPL, claimed setids).

    Uniqueness is enforced at the DOCUMENT (set_id) level, not just the name
    level: DailyMed search sometimes returns the same label for two names (salt
    forms like levothyroxine/levothyroxine sodium, or a combination product
    matching several ingredients). Two names sharing one set_id are ONE corpus
    document — the first name in pool order claims it, later claimants are
    skipped. This guarantees N names <-> N distinct SPL documents.
    """
    from pharmarag.ingest.dailymed import load_manifest

    good = parseable_drugs()
    docs = load_manifest(snapshot)["documents"]
    name_to_setid = {
        str(d["drug"]).strip().lower(): str(d["setid"])
        for d in docs
        if isinstance(d, dict) and "sha256" in d
    }

    canonical: list[str] = []
    claimed: set[str] = set()
    skipped_shared: list[str] = []
    for d in pool:
        if len(canonical) >= CORPUS_SIZE:
            break
        if d not in good:
            continue
        setid = name_to_setid.get(d)
        if setid is None:
            continue
        if setid in claimed:
            skipped_shared.append(d)
            continue
        claimed.add(setid)
        canonical.append(d)
    return canonical, skipped_shared, claimed


def refetch_shared(names: list[str], *, snapshot: str, delay: float) -> None:
    """Give shared-SPL names their OWN document.

    Drops their manifest rows, then re-fetches with every already-claimed setid
    excluded, so the multi-candidate quality gate lands on a distinct label
    (e.g. metformin gets a metformin label instead of the glyburide/metformin
    combination that another name already claimed). NETWORK.
    """
    from pharmarag.ingest.dailymed import build_snapshot

    mp = SNAPSHOTS / snapshot / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    wanted = {n.strip().lower() for n in names}
    keep = [d for d in m["documents"] if str(d.get("drug", "")).strip().lower() not in wanted]
    claimed = {str(d["setid"]) for d in keep if isinstance(d, dict) and "sha256" in d}
    m["documents"] = keep
    mp.write_text(json.dumps(m, indent=2), encoding="utf-8")

    print(
        f"[corpus] re-fetching {len(names)} shared-SPL names with their old "
        f"setids excluded ({len(claimed)} claimed) …"
    )
    build_snapshot(
        names,
        snapshot=snapshot,
        delay=delay,
        resume=True,
        quality_filter=True,
        candidates_per_drug=5,
        exclude_setids=claimed,
    )


def freeze(pool: list[str], *, snapshot: str) -> list[str]:
    """Freeze exactly CORPUS_SIZE canonical names (one per distinct SPL document)."""
    canonical, skipped_shared, claimed = plan_freeze(pool, snapshot=snapshot)

    if skipped_shared:
        print(
            f"[corpus] {len(skipped_shared)} names skipped — their SPL was already "
            "claimed by an earlier name (salt forms / combination labels):"
        )
        for name in skipped_shared[:10]:
            print(f"[corpus]   {name}")

    if len(canonical) < CORPUS_SIZE:
        print(
            f"[corpus] only {len(canonical)} unique parseable documents available "
            f"(< {CORPUS_SIZE}). Fetch more candidates (raise --buffer/--pool-size) "
            "or run without --freeze-only.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    path = save_expanded(canonical, meta={"snapshot": snapshot, "source": "build_corpus_1000"})
    print(
        f"[corpus] froze {len(canonical)} canonical ingredients "
        f"({len(claimed)} distinct SPL documents) -> {path}"
    )
    return canonical


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze exactly 1000 canonical ingredients (ADR-003).")
    ap.add_argument(
        "--snapshot",
        default="dailymed-2026-07-20",
        help="snapshot dir under data/snapshots/ to fetch into and index",
    )
    ap.add_argument("--target", type=int, default=CORPUS_SIZE, help="exact corpus size to freeze")
    ap.add_argument(
        "--buffer",
        type=int,
        default=90,
        help="extra good labels to fetch beyond target (covers parse drops)",
    )
    ap.add_argument(
        "--pool-size",
        type=int,
        default=1800,
        help="stratified ATC candidate pool size for headroom (NETWORK)",
    )
    ap.add_argument(
        "--freeze-only",
        action="store_true",
        help="skip network + parsing; just freeze from what is already in the DB",
    )
    ap.add_argument(
        "--delay", type=float, default=0.1, help="throttle between DailyMed fetches (s)"
    )
    args = ap.parse_args()

    pool = candidate_pool(args.pool_size, offline=args.freeze_only)
    print(f"[corpus] candidate pool: {len(pool)} unique ingredients")

    if args.freeze_only:
        freeze(pool, snapshot=args.snapshot)
        return 0

    # 1 · fetch the delta up to target+buffer good labels (resumable, quality-gated).
    from pharmarag.ingest.dailymed import build_snapshot

    print(
        f"[corpus] fetching SPLs into {args.snapshot} "
        f"(stop after {args.target + args.buffer} good labels) …"
    )
    build_snapshot(
        pool,
        snapshot=args.snapshot,
        delay=args.delay,
        resume=True,
        quality_filter=True,
        candidates_per_drug=3,
        stop_after_kept=args.target + args.buffer,
    )

    # 2 · parse discovery — persist chunks, learn which labels yield real content.
    #     Only parse candidates not already in the DB (already-parsed are known good).
    from build_index import run_build

    already = parseable_drugs()
    fresh = [d for d in pool if d not in already]
    print(
        f"[corpus] parse discovery: {len(fresh)} new candidates "
        f"({len(already)} already parseable) …"
    )
    run_build(args.snapshot, dry_run=True, restrict=fresh)

    # 3 · shared-SPL repair: names whose label another name already claimed get
    #     re-fetched with claimed setids excluded, then parsed.
    _, skipped_shared, _ = plan_freeze(pool, snapshot=args.snapshot)
    if skipped_shared:
        refetch_shared(skipped_shared, snapshot=args.snapshot, delay=args.delay)
        run_build(args.snapshot, dry_run=True, restrict=skipped_shared)

    # 4 · freeze exactly `target` parseable ingredients in pool order.
    freeze(pool, snapshot=args.snapshot)
    print(
        "[corpus] next: uv run python scripts/build_index.py --snapshot "
        f"{args.snapshot}   (embeds + indexes exactly {args.target})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
