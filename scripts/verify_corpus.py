"""Post-build corpus verification — all the checks the RUNBOOK used to do with
fragile inline one-liners, in one PowerShell-safe script.

    uv run python scripts/verify_corpus.py

Exits non-zero if any hard invariant fails, so it can gate a build.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharmarag.config import DATA, DB_MAIN  # noqa: E402
from pharmarag.db import verify_audit_immutability  # noqa: E402
from pharmarag.ingest.selection import CORPUS_SIZE, load_expanded  # noqa: E402


def _q(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def main() -> int:
    ok = True

    def check(label: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label:42s} {detail}")

    print("PharmaRAG corpus verification")
    print("-" * 72)

    # --- selection ---
    frozen = load_expanded()
    check(
        "frozen selection == 1000 unique",
        len(frozen) == CORPUS_SIZE and len(set(frozen)) == CORPUS_SIZE,
        f"{len(frozen)} drugs, {len(set(frozen))} unique",
    )

    # --- corpus DB ---
    conn = sqlite3.connect(DB_MAIN)
    try:
        docs = _q(conn, "SELECT COUNT(*) FROM documents")
        chunks = _q(conn, "SELECT COUNT(*) FROM chunks WHERE retrievable=1")
        drugs_with_chunks = _q(
            conn, "SELECT COUNT(DISTINCT lower(ingredient_name)) FROM chunks WHERE retrievable=1"
        )
        overdosage = _q(
            conn, "SELECT COUNT(*) FROM chunks WHERE loinc_section_code='34088-5' AND retrievable=1"
        )
        interactions = _q(
            conn, "SELECT COUNT(*) FROM chunks WHERE loinc_section_code='34073-7' AND retrievable=1"
        )
        table_rows = _q(
            conn, "SELECT COUNT(*) FROM chunks WHERE content_type='table_row' AND retrievable=1"
        )
        cached = _q(conn, "SELECT COUNT(*) FROM embedding_cache")
    finally:
        conn.close()

    print(
        f"  ---- documents={docs:,}  retrievable_chunks={chunks:,}  "
        f"drugs_with_chunks={drugs_with_chunks}"
    )
    print(
        f"  ---- interactions={interactions:,}  table_rows={table_rows:,}  "
        f"embeddings_cached={cached:,}"
    )

    frozen_set = {d.strip().lower() for d in frozen}
    conn = sqlite3.connect(DB_MAIN)
    try:
        in_corpus = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT lower(ingredient_name) FROM chunks WHERE retrievable=1"
            )
        }
    finally:
        conn.close()
    missing = sorted(frozen_set - in_corpus)
    check(
        "every frozen drug has retrievable chunks",
        not missing,
        "all present" if not missing else f"{len(missing)} missing: {missing[:5]}",
    )

    # --- ADR-005: Overdosage never indexed ---
    check("ADR-005 Overdosage NOT retrievable", overdosage == 0, f"{overdosage} (must be 0)")

    # --- ADR-047: audit append-only ---
    check(
        "ADR-047 audit log append-only", verify_audit_immutability(), "UPDATE + DELETE both blocked"
    )

    # --- Qdrant index ---
    try:
        from pharmarag.index.store import collection_stats, get_client

        cl = get_client()
        try:
            stats = collection_stats(cl)
        finally:
            cl.close()
        points = int(stats.get("points", 0))
        check("Qdrant index populated", stats.get("exists") and points > 0, f"{points:,} points")
    except Exception as exc:
        check("Qdrant index populated", False, f"error: {type(exc).__name__}: {exc}")

    # --- derived artifacts (built by build_entities / build_graph) ---
    gaz = (DATA / "gazetteer.json").is_file()
    lasa = (DATA / "lasa_table.json").is_file()
    graph = any((DATA / "graph").glob("*.json")) if (DATA / "graph").is_dir() else False
    print(
        f"  ---- gazetteer={'yes' if gaz else 'NO — run build_entities.py'}  "
        f"lasa={'yes' if lasa else 'no'}  "
        f"graph={'yes' if graph else 'NO — run build_graph.py'}"
    )

    print("-" * 72)
    print("RESULT:", "ALL HARD INVARIANTS PASS ✓" if ok else "FAILURES ABOVE ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
