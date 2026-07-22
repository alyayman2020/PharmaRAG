"""Ad-hoc queries against the main DB, without shell quoting pain.

PowerShell mangles `python -c "... COUNT(*) ..."`: `\"` is bash escaping, so the
string closes early and PowerShell parses the rest itself, failing on `*`. Piping a
here-string works but is tedious, so the common queries live here instead.

    uv run python scripts/dbq.py                 # summary
    uv run python scripts/dbq.py chunks          # a named query
    uv run python scripts/dbq.py --list          # what's available
    uv run python scripts/dbq.py --sql           # read SQL from stdin
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharmarag.db import session

QUERIES: dict[str, str] = {
    "chunks": "SELECT content_type, COUNT(*) n FROM chunks GROUP BY content_type ORDER BY n DESC",
    "retrievable": "SELECT retrievable, COUNT(*) n FROM chunks GROUP BY retrievable",
    "sections": "SELECT loinc_section_code, COUNT(*) n FROM chunks GROUP BY loinc_section_code ORDER BY n DESC",
    "documents": "SELECT COUNT(*) documents, COUNT(DISTINCT corpus_version) corpus_versions FROM documents",
    "drugs": "SELECT ingredient_name, COUNT(*) n FROM chunks GROUP BY ingredient_name ORDER BY n DESC",
    # A drug with no prose has only table rows — it cannot answer a warnings or
    # interactions question. Empty result is the healthy case.
    "prose_gaps": (
        "SELECT ingredient_name, COUNT(*) n FROM chunks WHERE ingredient_name NOT IN "
        "(SELECT DISTINCT ingredient_name FROM chunks WHERE content_type='prose') "
        "GROUP BY ingredient_name ORDER BY ingredient_name"
    ),
    # Same source document indexed under two ingredient names — chunk_id hashes text
    # only, so the later pass silently overwrites the earlier one's prose.
    "collisions": (
        "SELECT set_id, COUNT(DISTINCT ingredient_name) names, "
        "GROUP_CONCAT(DISTINCT ingredient_name) drugs FROM chunks "
        "GROUP BY set_id HAVING names > 1"
    ),
}


def show(conn: sqlite3.Connection, sql: str) -> None:
    rows = conn.execute(sql).fetchall()
    if not rows:
        print("  (no rows)")
        return
    for r in rows:
        # sqlite3.Row: iteration yields VALUES, so `.keys()` is required here — SIM118's
        # suggested rewrite would print values as keys.
        print("  " + "  ".join(f"{k}={r[k]}" for k in r.keys()))  # noqa: SIM118


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="named query (default: run them all)")
    ap.add_argument("--list", action="store_true", help="list named queries")
    ap.add_argument("--sql", action="store_true", help="read one SQL statement from stdin")
    args = ap.parse_args()

    if args.list:
        for name in QUERIES:
            print(name)
        return 0

    with session() as conn:
        if args.sql:
            show(conn, sys.stdin.read())
            return 0
        if args.query:
            if args.query not in QUERIES:
                print(f"unknown query {args.query!r}; try --list", file=sys.stderr)
                return 2
            show(conn, QUERIES[args.query])
            return 0
        for name, sql in QUERIES.items():
            print(f"[{name}]")
            show(conn, sql)
    return 0


if __name__ == "__main__":
    sys.exit(main())
