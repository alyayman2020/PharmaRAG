"""Section-aware chunking with parent-document retrieval (ADR-011, ADR-012, ADR-014).

Three invariants, all unit-tested in tests/test_chunk_invariants.py:

  1. Never split across a LOINC section boundary.
  2. Atomic sections (interactions, contraindications, populations) get ZERO
     overlap. Overlapping two dose rows can splice "250 mg q12h" and
     "500 mg q24h" into "500 mg q12h" — a 2x overdose assembled from two
     individually correct statements.
  3. No chunk below MIN_CHUNK_TOKENS: below ~60 tokens the ADR-014 prefix
     dominates the vector.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from pharmarag.chunking.metadata import (
    detect_population_tags,
    detect_units,
    extract_dose_values,
)
from pharmarag.config import (
    CHUNK_POLICIES,
    DEFAULT_POLICY,
    MAX_PARENT_TOKENS,
    MIN_CHUNK_TOKENS,
    NON_RETRIEVABLE_SECTIONS,
    ChunkPolicy,
)
from pharmarag.embed.prefix import build_embed_text
from pharmarag.tokens import count_tokens

if TYPE_CHECKING:
    from spl_parser.models import Section, SPLDocument

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_BULLET = re.compile(r"^\s*[-•*]\s+", re.MULTILINE)


@dataclass(slots=True)
class Parent:
    parent_chunk_id: str
    set_id: str
    loinc_section_code: str
    section_name: str
    section_path: str
    parent_part: str | None
    text: str
    token_count: int


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    set_id: str
    parent_chunk_id: str
    doc_version: int
    effective_time: str | None
    corpus_version: str
    content_sha256: str
    chunk_policy: str
    token_count: int
    section_path: str
    source_url: str
    embed_text: str  # PREFIXED   -> dense vector  (ADR-014)
    display_text: str
    raw_text: str  # UNPREFIXED -> BM25          (ADR-021)
    rxcui: str | None
    rxcui_all: list[str]
    ingredient_name: str
    brand_names: list[str] = field(default_factory=list)
    pharm_class_epc: list[str] = field(default_factory=list)
    loinc_section_code: str = ""
    section_name: str = ""
    content_type: str = "prose"
    table_id: str | None = None
    footnote_text: list[str] = field(default_factory=list)
    retrievable: bool = True
    units_present: list[str] = field(default_factory=list)
    population_tags: list[str] = field(default_factory=list)
    dose_values: list[dict[str, object]] = field(default_factory=list)
    application_type: str | None = None
    is_canonical: bool = True
    is_variant: bool = False
    conflict_of: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- splitting
def _split_units(text: str, policy: ChunkPolicy) -> list[str]:
    """Break section text into candidate units before size enforcement."""
    if not policy.split:
        return [text] if text.strip() else []

    if policy.atomic:
        # Bullets first — SPL contraindications and interactions are usually lists.
        if _BULLET.search(text):
            items = [_BULLET.sub("", b).strip() for b in text.split("\n")]
            return [i for i in items if i]
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _enforce_max(units: list[str], policy: ChunkPolicy) -> list[str]:
    """Split anything over the policy cap on sentence boundaries."""
    out: list[str] = []
    for unit in units:
        if count_tokens(unit) <= policy.max_tokens:
            out.append(unit)
            continue
        buf: list[str] = []
        size = 0
        for sent in _SENTENCE.split(unit):
            st = count_tokens(sent)
            if buf and size + st > policy.max_tokens:
                out.append(" ".join(buf))
                # ADR-012: one-sentence overlap on prose, zero on atomic.
                buf = buf[-policy.overlap_sentences :] if policy.overlap_sentences else []
                size = sum(count_tokens(b) for b in buf)
            buf.append(sent)
            size += st
        if buf:
            out.append(" ".join(buf))
    return out


def _enforce_floor(units: list[str]) -> list[str]:
    """ADR-014: merge any chunk under the floor upward with its next sibling."""
    if not units:
        return units
    out: list[str] = []
    carry = ""
    for unit in units:
        candidate = f"{carry} {unit}".strip() if carry else unit
        if count_tokens(candidate) < MIN_CHUNK_TOKENS:
            carry = candidate
            continue
        out.append(candidate)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}".strip()
        else:
            out.append(carry)  # whole section is tiny — keep it rather than drop evidence
    return out


# --------------------------------------------------------------------------- parents
def _make_parents(
    set_id: str, loinc: str, section_name: str, section_path: str, text: str
) -> list[Parent]:
    total = count_tokens(text)
    if total <= MAX_PARENT_TOKENS:
        return [
            Parent(
                f"p-{uuid.uuid4().hex[:12]}",
                set_id,
                loinc,
                section_name,
                section_path,
                None,
                text,
                total,
            )
        ]

    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        pt = count_tokens(para)
        if buf and size + pt > MAX_PARENT_TOKENS:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += pt
    if buf:
        parts.append("\n\n".join(buf))

    n = len(parts)
    return [
        Parent(
            f"p-{uuid.uuid4().hex[:12]}",
            set_id,
            loinc,
            section_name,
            section_path,
            f"{i + 1}/{n}",
            p,
            count_tokens(p),
        )
        for i, p in enumerate(parts)
    ]


# --------------------------------------------------------------------------- public


def _table_row_chunks(
    doc: SPLDocument,
    top_section: Section,
    section: Section,
    parent_id: str,
    *,
    corpus_version: str,
    ingredient_name: str,
    rxcui: str | None,
    rxcui_all: list[str] | None,
    brand_names: list[str] | None,
    pharm_class_epc: list[str] | None,
    application_type: str | None,
    is_canonical: bool,
    source_url: str,
    loinc: str,
    retrievable: bool,
) -> list[Chunk]:
    """Milestone B1 — emit one atomic chunk per linearized table row (ADR-008).

    Three invariants, all safety-relevant:
      * ONE ROW = ONE CHUNK, ZERO OVERLAP. Overlapping rows can splice
        "250 mg q12h" and "500 mg q24h" into "500 mg q12h".
      * Rows are EXEMPT from the 60-token floor and from prefixing. The
        linearization already names the drug and restates its qualifier, so it
        is self-contained; prefixing would name the drug twice and skew the
        vector (ADR-014).
      * `dose_values` is extracted from the linearized sentence, so the K3
        qualifier check has the CrCl band available to verify against.
    """
    from spl_parser.tables import linearize

    out: list[Chunk] = []
    for table in section.tables:
        for lr in linearize(table, drug=ingredient_name):
            sha = hashlib.sha256(lr.sentence.encode("utf-8")).hexdigest()
            uid = hashlib.sha256(
                f"{doc.set_id}|{loinc}|{lr.table_id}|{lr.row_index}|{lr.column}|{sha}".encode()
            ).hexdigest()
            # ADR-014: table rows are exempt from the ingredient prefix.
            embed_text = build_embed_text(
                ingredient_name=ingredient_name,
                section_name=top_section.section_name,
                body=lr.sentence,
                content_type="table_row",
            )
            out.append(
                Chunk(
                    chunk_id=f"t-{uid[:20]}",
                    set_id=doc.set_id,
                    parent_chunk_id=parent_id,
                    doc_version=doc.doc_version,
                    effective_time=doc.effective_time,
                    corpus_version=corpus_version,
                    content_sha256=sha,
                    chunk_policy="table-row-atomic",
                    token_count=count_tokens(lr.sentence),
                    section_path=section.section_path,
                    source_url=source_url,
                    embed_text=embed_text,
                    display_text=lr.sentence,
                    raw_text=lr.sentence,
                    rxcui=rxcui,
                    rxcui_all=rxcui_all or ([rxcui] if rxcui else []),
                    ingredient_name=ingredient_name,
                    brand_names=brand_names or [],
                    pharm_class_epc=pharm_class_epc or [],
                    loinc_section_code=loinc,
                    section_name=top_section.section_name,
                    content_type="table_row",
                    table_id=lr.table_id,
                    footnote_text=lr.footnotes,
                    retrievable=retrievable,
                    units_present=detect_units(lr.sentence),
                    population_tags=detect_population_tags(f"{section.section_path} {lr.sentence}"),
                    dose_values=extract_dose_values(lr.sentence),
                    application_type=application_type,
                    is_canonical=is_canonical,
                )
            )
    return out


def _parent_path(parents: list[Parent], parent_id: str) -> str:
    for p in parents:
        if p.parent_chunk_id == parent_id:
            return p.section_path
    return ""


def chunk_document(
    doc: SPLDocument,
    *,
    corpus_version: str,
    ingredient_name: str,
    rxcui: str | None = None,
    rxcui_all: list[str] | None = None,
    brand_names: list[str] | None = None,
    pharm_class_epc: list[str] | None = None,
    application_type: str | None = None,
    is_canonical: bool = True,
    source_url_base: str = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=",
) -> tuple[list[Chunk], list[Parent]]:
    """Turn a parsed SPL document into chunks and parents."""
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    source_url = f"{source_url_base}{doc.set_id}"

    for top in doc.sections:
        loinc = top.loinc_code
        if not loinc:
            continue
        policy = CHUNK_POLICIES.get(loinc, DEFAULT_POLICY)
        # ADR-005: still chunked and recorded, just flagged. index/upsert.py
        # refuses to write these to Qdrant.
        retrievable = loinc not in NON_RETRIEVABLE_SECTIONS

        # ADR-014: the floor is enforced across ALL units in a LOINC section,
        # not per-subsection. Applying it per-subsection lets a short section
        # preamble ("See subsections.") slip through as a 16-token chunk that is
        # mostly prefix. Merging across subsections is legal — merging across
        # LOINC section boundaries is not.
        pending: list[tuple[str, str, str, list[str]]] = []  # (unit, path, subtitle, footnotes)

        for sec in top.walk():
            if not sec.has_content or not sec.text.strip():
                continue

            body = sec.text
            sec_parents = _make_parents(doc.set_id, loinc, top.section_name, sec.section_path, body)
            parents.extend(sec_parents)
            parent_id = sec_parents[0].parent_chunk_id
            subtitle = sec.title if sec.depth > 0 else None
            footnotes = [f for t in sec.tables for f in t.footnotes]

            # B1: table rows are emitted directly, bypassing prose splitting.
            chunks.extend(
                _table_row_chunks(
                    doc,
                    top,
                    sec,
                    parent_id,
                    corpus_version=corpus_version,
                    ingredient_name=ingredient_name,
                    rxcui=rxcui,
                    rxcui_all=rxcui_all,
                    brand_names=brand_names,
                    pharm_class_epc=pharm_class_epc,
                    application_type=application_type,
                    is_canonical=is_canonical,
                    source_url=source_url,
                    loinc=loinc,
                    retrievable=retrievable,
                )
            )

            for unit in _enforce_max(_split_units(body, policy), policy):
                pending.append((unit, parent_id, subtitle or "", footnotes))

        # Merge sub-floor units forward, carrying the parent of the first piece.
        merged: list[tuple[str, str, str, list[str]]] = []
        carry: tuple[str, str, str, list[str]] | None = None
        for unit, pid, subtitle, fns in pending:
            if carry is not None:
                unit = f"{carry[0]} {unit}".strip()
                pid, subtitle, fns = carry[1], carry[2] or subtitle, carry[3] or fns
                carry = None
            if count_tokens(unit) < MIN_CHUNK_TOKENS:
                carry = (unit, pid, subtitle, fns)
                continue
            merged.append((unit, pid, subtitle, fns))
        if carry is not None:
            if merged:
                last = merged[-1]
                merged[-1] = (f"{last[0]} {carry[0]}".strip(), last[1], last[2], last[3])
            else:
                merged.append(carry)  # whole section is tiny — keep it, never drop evidence

        for unit, parent_id, subtitle, footnotes in merged:
            sha = hashlib.sha256(unit.encode("utf-8")).hexdigest()
            # The unit's OWN path via its parent — not the stale `sec` loop
            # variable, which would tag every merged chunk with the LAST
            # subsection's path (e.g. a Pregnancy chunk tagged as Lactation).
            unit_path = _parent_path(parents, parent_id)
            embed_text = build_embed_text(
                ingredient_name=ingredient_name,
                section_name=top.section_name,
                subsection=subtitle or None,
                body=unit,
                content_type="prose",
            )
            chunks.append(
                Chunk(
                    chunk_id=f"c-{sha[:16]}",
                    set_id=doc.set_id,
                    parent_chunk_id=parent_id,
                    doc_version=doc.doc_version,
                    effective_time=doc.effective_time,
                    corpus_version=corpus_version,
                    content_sha256=sha,
                    chunk_policy=policy.name,
                    token_count=count_tokens(unit),
                    section_path=unit_path,
                    source_url=source_url,
                    embed_text=embed_text,
                    display_text=unit,
                    raw_text=unit,
                    rxcui=rxcui,
                    rxcui_all=rxcui_all or ([rxcui] if rxcui else []),
                    ingredient_name=ingredient_name,
                    brand_names=brand_names or [],
                    pharm_class_epc=pharm_class_epc or [],
                    loinc_section_code=loinc,
                    section_name=top.section_name,
                    content_type="prose",
                    footnote_text=footnotes,
                    retrievable=retrievable,
                    units_present=detect_units(unit),
                    population_tags=detect_population_tags(f"{unit_path} {unit}"),
                    dose_values=extract_dose_values(unit),
                    application_type=application_type,
                    is_canonical=is_canonical,
                )
            )

    return chunks, parents
