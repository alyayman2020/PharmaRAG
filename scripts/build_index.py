"""Milestone A2-A4: parse -> chunk -> embed -> index.

Usage:
    uv run python scripts/build_index.py --snapshot dailymed-2026-07-20
    uv run python scripts/build_index.py --snapshot ... --dry-run   # no OpenAI calls

By default the build indexes exactly the frozen corpus selection
(``data/corpus_selection.json`` via ``load_expanded``) intersected with the
snapshot manifest — so the vector store reflects exactly the canonical corpus,
not whatever happens to be archived. Pass ``--all`` to index every archived
document in the manifest (used during corpus discovery, before the freeze).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/spl_parser/src"))

from pharmarag.chunking.chunker import chunk_document
from pharmarag.config import SECTION_NAMES, SNAPSHOTS, ensure_dirs, settings
from pharmarag.db import init_db, session
from pharmarag.ingest.dailymed import archive_path, load_manifest
from pharmarag.ingest.selection import load_expanded, selection_path
from spl_parser import parse_spl


def persist(chunks: list, parents: list, doc, drug: str, sha: str, corpus_version: str) -> None:
    now = dt.datetime.now(dt.UTC).isoformat()
    with session() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (set_id, doc_version, effective_time, title,"
            " ingredient_name, rxcui, brand_names, application_type, is_canonical, sha256,"
            " corpus_version, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc.set_id,
                doc.doc_version,
                doc.effective_time,
                doc.title,
                drug,
                None,
                "[]",
                None,
                1,
                sha,
                corpus_version,
                now,
            ),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO parents (parent_chunk_id, set_id, loinc_section_code,"
            " section_name, section_path, parent_part, text, token_count) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    p.parent_chunk_id,
                    p.set_id,
                    p.loinc_section_code,
                    p.section_name,
                    p.section_path,
                    p.parent_part,
                    p.text,
                    p.token_count,
                )
                for p in parents
            ],
        )
        for c in chunks:
            d = c.to_dict()
            for k in (
                "rxcui_all",
                "brand_names",
                "pharm_class_epc",
                "footnote_text",
                "units_present",
                "population_tags",
                "dose_values",
            ):
                d[k] = json.dumps(d[k])
            d["retrievable"] = int(d["retrievable"])
            d["is_canonical"] = int(d["is_canonical"])
            d["is_variant"] = int(d["is_variant"])
            cols = ",".join(d)
            marks = ",".join("?" * len(d))
            conn.execute(
                f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({marks})", tuple(d.values())
            )


def _select_docs(manifest: dict, *, index_all: bool, restrict: Iterable[str] | None) -> list[dict]:
    """Documents to index, intersected with the canonical corpus when frozen."""
    docs = [d for d in manifest["documents"] if "sha256" in d]
    if restrict is not None:
        allow = {n.strip().lower() for n in restrict}
    elif not index_all and selection_path().is_file():
        allow = {n.strip().lower() for n in load_expanded()}
    else:
        return docs
    return [d for d in docs if str(d["drug"]).strip().lower() in allow]


def run_build(
    snapshot: str,
    *,
    dry_run: bool = False,
    index_all: bool = False,
    restrict: Iterable[str] | None = None,
) -> dict[str, int]:
    """Parse -> chunk -> (embed -> index). Returns summary counts."""
    ensure_dirs()
    init_db()

    if not (SNAPSHOTS / snapshot / "manifest.json").is_file():
        available = (
            sorted(p.name for p in SNAPSHOTS.iterdir() if p.is_dir()) if SNAPSHOTS.is_dir() else []
        )
        print(f"[build] no manifest.json for snapshot {snapshot!r}", file=sys.stderr)
        print(
            f"[build] available: {', '.join(available) or '(none — run the ingest step first)'}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    manifest = load_manifest(snapshot)
    docs = _select_docs(manifest, index_all=index_all, restrict=restrict)
    print(f"[build] indexing {len(docs)} documents from {snapshot}")

    all_chunks: list = []
    indexed_drugs: set[str] = set()
    for entry in docs:
        path = archive_path(str(entry["sha256"]))
        if not path.exists():
            print(f"[build]   MISSING archive for {entry['drug']}")
            continue
        try:
            doc = parse_spl(path, SECTION_NAMES)
            chunks, parents = chunk_document(
                doc,
                corpus_version=snapshot,
                ingredient_name=str(entry["drug"]),
                rxcui=str(entry.get("rxcui") or entry["drug"]),
            )
        except Exception as exc:
            print(f"[build]   PARSE FAILED {entry['drug']}: {type(exc).__name__}: {exc}")
            continue
        retrievable = [c for c in chunks if c.retrievable]
        if not retrievable:
            print(f"[build]   {entry['drug']:24s} 0 retrievable chunks — skipped")
            continue
        persist(chunks, parents, doc, str(entry["drug"]), str(entry["sha256"]), snapshot)
        indexed_drugs.add(str(entry["drug"]).strip().lower())
        all_chunks.extend(retrievable)
        print(
            f"[build] {entry['drug']:24s} {len(chunks):4d} chunks "
            f"({len(chunks)-len(retrievable)} non-retrievable), {len(parents):3d} parents"
        )

    print(f"\n[build] {len(indexed_drugs)} drugs, {len(all_chunks)} retrievable chunks total")
    if dry_run:
        print("[build] --dry-run: stopping before embedding. No OpenAI calls made.")
        return {"drugs": len(indexed_drugs), "chunks": len(all_chunks), "points": 0}

    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient  # noqa: F401

    from pharmarag.embed.client import embed_texts, estimate_cost
    from pharmarag.index.store import collection_stats, create_collection, get_client
    from pharmarag.index.upsert import upsert_chunks

    total_tokens = sum(c.token_count for c in all_chunks)
    print(
        f"[build] ~{total_tokens:,} tokens -> est. ${estimate_cost(total_tokens):.4f} "
        "(cache hits are $0)"
    )

    dense = embed_texts([c.embed_text for c in all_chunks])
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse = list(bm25.embed([c.raw_text for c in all_chunks]))  # ADR-021: RAW text

    client = get_client()
    create_collection(client, recreate=True)
    written = upsert_chunks(client, [c.to_dict() for c in all_chunks], dense, sparse)
    stats = collection_stats(client)
    print(f"[build] wrote {written} points | {stats}")
    return {"drugs": len(indexed_drugs), "chunks": len(all_chunks), "points": int(stats["points"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    _default = settings.corpus_version if settings.corpus_version != "dev" else None
    ap.add_argument(
        "--snapshot",
        nargs="?",
        default=_default,
        const=_default,
        help="snapshot directory under data/snapshots/ (default: $CORPUS_VERSION from .env)",
    )
    ap.add_argument("--dry-run", action="store_true", help="parse + chunk only, no embedding")
    ap.add_argument(
        "--all",
        action="store_true",
        help="index every archived doc (ignore corpus_selection.json freeze)",
    )
    args = ap.parse_args()
    if not args.snapshot:
        ap.error("--snapshot is required (or set CORPUS_VERSION in .env)")
    run_build(args.snapshot, dry_run=args.dry_run, index_all=args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
