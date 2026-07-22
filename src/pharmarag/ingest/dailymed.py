"""DailyMed acquisition, snapshotting, and content-addressed archiving (ADR-004).

NETWORK REQUIRED. Every function here is written against DailyMed's documented
bulk-download and web-service shapes.

The snapshot is immutable and content-addressed: any citation is re-verifiable
against the exact bytes retrieved, even after a monthly refresh. That is the
audit trail satisfied at the DATA layer rather than bolted on.

This is the single authoritative DailyMed module. It merges three concerns that
used to live in two divergent copies (``scripts/dailymed.py`` was the other):

* **Avast-safe TLS** — all HTTP goes through ``pharmarag.http.client``, which
  carries ``PHARMARAG_CA_BUNDLE`` so calls survive Avast's TLS-inspecting proxy.
  The old ``scripts`` copy used a raw ``httpx.Client`` and would fail SSL here.
* **Resumable snapshots** — a 1,000-drug run is a long network job; the manifest
  is rewritten every ``save_every`` fetches and a re-run skips drugs already
  archived. Kill it and restart and it continues where it stopped.
* **Quality filter + multi-candidate** — taking ``hits[0]`` blindly is how a
  ~936-drug corpus ends up averaging 25 chunks/drug instead of ~310: the labels
  parse, they are just nearly empty (OTC monographs, kit/convenience packs,
  discontinued stubs). We try several search hits per drug and keep the FIRST
  that passes the quality gate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

from pharmarag.config import ARCHIVE, SNAPSHOTS
from pharmarag.http import client as http_client

# DailyMed bulk downloads. VERIFY these before a full run — NLM reorganizes paths.
BULK_INDEX = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"
RXNORM_MAPPING_URL = "https://dailymed.nlm.nih.gov/dailymed/rxnorm_mappings.zip"
PHARM_CLASS_MAPPING_URL = "https://dailymed.nlm.nih.gov/dailymed/pharmacologic_class_mappings.zip"

# Web service for fetching a single SPL by setid.
SPL_XML_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
SPL_SEARCH_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

# A full prescribing-information label is large. Anything much smaller is an OTC
# monograph, a kit/convenience-pack entry, or a discontinued stub — it parses fine
# and indexes a handful of near-useless chunks, which inflates the corpus while
# contributing nothing retrievable.
MIN_LABEL_BYTES = 120_000
MIN_TARGET_SECTIONS = 3
# LOINC codes for the sections this project actually retrieves.
_QUALITY_SECTIONS = ("34073-7", "34070-3", "34068-7", "43685-7", "43684-0")


def snapshot_id(date: dt.date | None = None) -> str:
    return f"dailymed-{(date or dt.date.today()).isoformat()}"


def archive_bytes(payload: bytes) -> str:
    """Store content-addressed. Returns the sha256."""
    sha = hashlib.sha256(payload).hexdigest()
    dest = ARCHIVE / sha[:2] / f"{sha}.xml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(payload)
    return sha


def archive_path(sha: str) -> Path:
    return ARCHIVE / sha[:2] / f"{sha}.xml"


def search_spls(drug_name: str, *, limit: int = 5) -> list[dict[str, object]]:
    """Find candidate SPL setids for a drug name."""
    # http_client() carries PHARMARAG_CA_BUNDLE, so DailyMed calls survive Avast's
    # TLS-inspecting proxy — the same reason http.py exists for RxNav/OpenAI.
    with http_client(timeout=45) as c:
        r = c.get(SPL_SEARCH_URL, params={"drug_name": drug_name, "pagesize": limit})
        r.raise_for_status()
        return list(r.json().get("data", []))


def fetch_spl_xml(setid: str) -> bytes:
    with http_client(timeout=90) as c:
        r = c.get(SPL_XML_URL.format(setid=setid))
        r.raise_for_status()
        return r.content


def download_mapping_zip(url: str, dest_dir: Path) -> list[Path]:
    """Fetch and extract a DailyMed mapping archive.

    These are gold: SetID -> RxCUI and SetID -> pharmacologic class, offline,
    with zero API round trips (ADR-001).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with http_client(timeout=300) as c:
        r = c.get(url)
        r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest_dir)
        return [dest_dir / n for n in z.namelist()]


def label_quality(xml: bytes) -> tuple[bool, str]:
    """Reject minimal labels before they pollute the index.

    Returns ``(ok, reason)`` — ``reason`` is empty when the label passes.
    """
    if len(xml) < MIN_LABEL_BYTES:
        return False, f"too small ({len(xml):,}B < {MIN_LABEL_BYTES:,})"
    text = xml.decode("utf-8", errors="ignore")
    present = sum(1 for code in _QUALITY_SECTIONS if code in text)
    if present < MIN_TARGET_SECTIONS:
        return False, f"only {present}/{len(_QUALITY_SECTIONS)} target sections"
    return True, ""


def _write_manifest(
    snap_dir: Path, snap: str, manifest: list[dict[str, object]], *, quality_filter: bool
) -> dict[str, object]:
    kept = sum(1 for m in manifest if "sha256" in m)
    out = {
        "corpus_version": snap,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": "DailyMed SPL web service v2",
        "count": kept,
        "requested": len(manifest),
        "rejected_low_quality": sum(
            1 for m in manifest if str(m.get("error", "")).startswith("no label passed")
        ),
        "quality_filter": {
            "min_bytes": MIN_LABEL_BYTES,
            "min_target_sections": MIN_TARGET_SECTIONS,
        }
        if quality_filter
        else None,
        "documents": manifest,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def build_snapshot(
    drug_names: list[str],
    *,
    snapshot: str | None = None,
    delay: float = 0.1,
    resume: bool = True,
    save_every: int = 20,
    quality_filter: bool = True,
    candidates_per_drug: int = 3,
    stop_after_kept: int | None = None,
    exclude_setids: frozenset[str] | set[str] | None = None,
) -> dict[str, object]:
    """Fetch one good SPL per drug name, archive it, and write a snapshot manifest.

    For each drug we try up to ``candidates_per_drug`` search hits and keep the
    FIRST whose label passes ``label_quality`` (ADR-004). Resumable by design: the
    manifest is rewritten every ``save_every`` fetches and a re-run skips any drug
    already carrying a ``sha256`` or a permanent ``error``.

    ``stop_after_kept`` lets a caller stop once the snapshot holds N good labels —
    used by the corpus-sizing bridge to fetch only the delta up to a target.

    ``exclude_setids`` skips search hits whose setid is already claimed by another
    drug in the manifest — used when re-fetching a name whose first label turned
    out to be a shared/combination SPL, so it gets its OWN document.
    """
    excluded = frozenset(exclude_setids or ())
    snap = snapshot or snapshot_id()
    snap_dir = SNAPSHOTS / snap
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    done: set[str] = set()
    manifest_path = snap_dir / "manifest.json"
    if resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Keep both successes and permanent "no SPL found" so re-runs don't retry
        # drugs DailyMed simply doesn't have.
        manifest = [
            m
            for m in prior.get("documents", [])
            if "sha256" in m or m.get("error") == "no SPL found"
        ]
        done = {str(m["drug"]) for m in manifest}
        print(f"[snapshot] resuming {snap}: {len(done)} already resolved/skipped", flush=True)

    kept = sum(1 for m in manifest if "sha256" in m)
    total = len(drug_names)
    fetched = 0
    for i, name in enumerate(drug_names, 1):
        if stop_after_kept is not None and kept >= stop_after_kept:
            print(
                f"[snapshot] reached target of {stop_after_kept} good labels — stopping", flush=True
            )
            break
        if name in done:
            continue
        try:
            hits = search_spls(name, limit=candidates_per_drug)
        except Exception as exc:
            manifest.append({"drug": name, "error": str(exc)})
            continue
        if not hits:
            manifest.append({"drug": name, "error": "no SPL found"})
            continue

        chosen: dict[str, object] | None = None
        chosen_setid, chosen_sha, chosen_bytes = "", "", 0
        reasons: list[str] = []
        for hit in hits[:candidates_per_drug]:
            setid = str(hit.get("setid", ""))
            if not setid:
                continue
            if setid in excluded:
                reasons.append(f"{setid[:8]}: setid already claimed by another drug")
                continue
            try:
                xml = fetch_spl_xml(setid)
            except Exception as exc:
                reasons.append(f"{setid[:8]}: {exc}")
                continue
            if quality_filter:
                ok, why = label_quality(xml)
                if not ok:
                    reasons.append(f"{setid[:8]}: {why}")
                    continue
            chosen_setid, chosen_sha, chosen_bytes = setid, archive_bytes(xml), len(xml)
            chosen = {
                "drug": name,
                "setid": chosen_setid,
                "sha256": chosen_sha,
                "title": hit.get("title", ""),
                "bytes": chosen_bytes,
            }
            break

        if chosen is None:
            manifest.append(
                {"drug": name, "error": "no label passed quality filter", "attempts": reasons[:3]}
            )
            print(
                f"[snapshot] {i:>4}/{total} {name:24s} SKIPPED — "
                f"{reasons[0] if reasons else 'no candidates'}",
                flush=True,
            )
            continue

        manifest.append(chosen)
        done.add(name)
        kept += 1
        fetched += 1
        print(
            f"[snapshot] {i:>4}/{total} {name:24s} setid={chosen_setid[:8]}… "
            f"sha={chosen_sha[:12]}… {chosen_bytes:>9,}B  (kept {kept})",
            flush=True,
        )
        if fetched % save_every == 0:
            _write_manifest(snap_dir, snap, manifest, quality_filter=quality_filter)
        time.sleep(delay)

    out = _write_manifest(snap_dir, snap, manifest, quality_filter=quality_filter)
    errors = sum(1 for m in manifest if "error" in m)
    print(f"[snapshot] {snap}: {out['count']} archived, {errors} errors/skips", flush=True)
    return out


def load_manifest(snapshot: str) -> dict[str, object]:
    data: dict[str, object] = json.loads(
        (SNAPSHOTS / snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    return data
