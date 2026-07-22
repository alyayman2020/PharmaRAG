"""SQLite access. Three separate files, never merged (ADR-046)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pharmarag.config import DB_CHECKPOINTS, DB_MAIN, DB_MLFLOW, ensure_dirs

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(path: Path = DB_MAIN) -> sqlite3.Connection:
    # timeout: block up to 30s for a lock instead of raising "database is locked"
    # the instant two connections write at once (e.g. a build while a test or the
    # app reads). WAL mode lets readers proceed while a writer holds the DB, so
    # concurrent access degrades to "slower", never "crashed".
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def session(path: Path = DB_MAIN) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> dict[str, str]:
    """Create all three database files. Idempotent."""
    ensure_dirs()
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with session(DB_MAIN) as conn:
        conn.executescript(ddl)

    # Placeholders so the three-file separation is visible on disk from day one.
    for p in (DB_CHECKPOINTS, DB_MLFLOW):
        c = connect(p)
        c.execute("PRAGMA journal_mode = WAL")
        c.close()

    return {
        "main": str(DB_MAIN),
        "checkpoints": str(DB_CHECKPOINTS),
        "mlflow": str(DB_MLFLOW),
    }


def verify_audit_immutability() -> bool:
    """Prove the ADR-047 triggers actually fire. Called by tests and the runbook.

    Uses a fresh probe id each run — the probe row cannot be deleted afterwards,
    which is the whole point.
    """
    import uuid

    probe = f"__probe__{uuid.uuid4().hex[:12]}"
    with session(DB_MAIN) as conn:
        conn.execute(
            "INSERT INTO audit_log (query_id, timestamp_utc, normalized_query,"
            " prompt_template_version) VALUES (?, '1970-01-01T00:00:00Z','probe','v0')",
            (probe,),
        )
    ok_update = ok_delete = False
    conn = connect(DB_MAIN)
    try:
        conn.execute("UPDATE audit_log SET reason_code='x' WHERE query_id=?", (probe,))
    except sqlite3.IntegrityError:
        ok_update = True
    try:
        conn.execute("DELETE FROM audit_log WHERE query_id=?", (probe,))
    except sqlite3.IntegrityError:
        ok_delete = True
    conn.close()
    return ok_update and ok_delete
