"""DEPRECATED shim — the authoritative DailyMed module is ``pharmarag.ingest.dailymed``.

This file used to be a second, divergent copy of the acquisition code that used a
raw ``httpx.Client`` (no Avast CA bundle, so SSL failed on this machine) and was
not resumable. Its one good idea — the label quality filter — has been merged into
the package module, which is now the single source of truth. Kept as a thin
re-export so any old runbook path (`python scripts/dailymed.py`) keeps working.
"""

from __future__ import annotations

from pharmarag.ingest.dailymed import (  # noqa: F401
    archive_bytes,
    archive_path,
    build_snapshot,
    download_mapping_zip,
    fetch_spl_xml,
    label_quality,
    load_manifest,
    search_spls,
    snapshot_id,
)

if __name__ == "__main__":
    import sys

    print(
        "scripts/dailymed.py is deprecated. Use:\n"
        "  uv run python scripts/build_corpus_1000.py   (select + snapshot to 1000)\n"
        "  uv run python scripts/build_index.py --snapshot <snap>",
        file=sys.stderr,
    )
    sys.exit(2)
