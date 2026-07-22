"""LOINC-coded section extraction from HL7 SPL XML.

The point of parsing raw XML rather than openFDA JSON (ADR-007): SPL *declares*
its section boundaries with LOINC codes and preserves table structure. openFDA
reformats section content into text, which very likely destroys tables — and no
downstream cleverness recovers a table that was linearized upstream.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lxml import etree

from spl_parser.models import Section, SPLDocument, Table

NS = {"v3": "urn:hl7-org:v3"}
LOINC_CODESYSTEM = "2.16.840.1.113883.6.1"

_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _localname(el: Any) -> str:
    tag = el.tag
    return tag.split("}")[-1] if isinstance(tag, str) and "}" in tag else str(tag)


# --------------------------------------------------------------------------- tables
def _expand_row(tr: Any) -> list[str]:
    """Flatten one <tr>, repeating cells across colspan.

    rowspan is handled by the caller's carry map, because a spanning cell must
    reappear in the rows beneath it — otherwise a dose row loses the CrCl band
    that qualifies it, which is exactly the ADR-013 `qualifier` failure.
    """
    cells: list[str] = []
    for cell in tr:
        if _localname(cell) not in {"td", "th"}:
            continue
        text = _clean("".join(cell.itertext()))
        span = int(cell.get("colspan", "1") or 1)
        cells.extend([text] * max(span, 1))
    return cells


def _extract_table(tbl: Any, table_id: str) -> Table:
    caption_el = tbl.find("v3:caption", NS)
    caption = _clean("".join(caption_el.itertext())) if caption_el is not None else None

    headers: list[list[str]] = []
    rows: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # col index -> (text, rows remaining)

    for part in tbl:
        name = _localname(part)
        if name not in {"thead", "tbody", "tfoot", "tr"}:
            continue
        trs = [part] if name == "tr" else [c for c in part if _localname(c) == "tr"]
        for tr in trs:
            row = _expand_row(tr)
            # Re-insert any cell still spanning down from an earlier row.
            for idx in sorted(carry):
                text, left = carry[idx]
                if left > 0:
                    row.insert(min(idx, len(row)), text)
                    carry[idx] = (text, left - 1)
            carry = {i: v for i, v in carry.items() if v[1] > 0}

            # Register new rowspans for the rows below.
            col = 0
            for cell in tr:
                if _localname(cell) not in {"td", "th"}:
                    continue
                rspan = int(cell.get("rowspan", "1") or 1)
                if rspan > 1:
                    carry[col] = (_clean("".join(cell.itertext())), rspan - 1)
                col += int(cell.get("colspan", "1") or 1)

            if name == "thead":
                headers.append(row)
            elif row:
                rows.append(row)

    footnotes = [
        _clean("".join(fn.itertext())) for fn in tbl.iter() if _localname(fn) == "footnote"
    ]
    return Table(
        table_id=table_id,
        caption=caption,
        headers=headers,
        rows=rows,
        footnotes=[f for f in footnotes if f],
    )


# --------------------------------------------------------------------------- text
def _section_text(section_el: Any) -> str:
    """Text belonging to THIS section only.

    Nested <component><section> subsections and <table> elements are excluded:
    subsections become their own Section objects, and tables are lifted
    separately so they never get flattened into prose (ADR-008).
    """
    text_el = section_el.find("v3:text", NS)
    if text_el is None:
        return ""

    parts: list[str] = []
    for child in text_el:
        name = _localname(child)
        if name == "table":
            continue
        if name == "list":
            for item in child:
                if _localname(item) == "item":
                    t = _clean("".join(item.itertext()))
                    if t:
                        parts.append(f"- {t}")
            continue
        t = _clean("".join(child.itertext()))
        if t:
            parts.append(t)

    if not parts and text_el.text:
        parts.append(_clean(text_el.text))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- sections
def _numbering(title: str | None) -> str:
    """Pull a leading section number out of the title, e.g. '5.2'."""
    if not title:
        return ""
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", title)
    return m.group(1) if m else ""


def _parse_section(
    section_el: Any,
    depth: int,
    parent_path: str,
    table_counter: list[int],
    section_names: dict[str, str],
) -> Section:
    code_el = section_el.find("v3:code", NS)
    loinc: str | None = None
    if code_el is not None and code_el.get("codeSystem") == LOINC_CODESYSTEM:
        loinc = code_el.get("code")

    title_el = section_el.find("v3:title", NS)
    title = _clean("".join(title_el.itertext())) if title_el is not None else None

    display = (code_el.get("displayName") if code_el is not None else None) or ""
    name = section_names.get(loinc or "", "") or _clean(display).title() or (title or "Untitled")

    num = _numbering(title)
    label = title or name
    section_path = (
        f"{parent_path} › {label}".strip(" ›") if parent_path else ((num and label) or label)
    )

    tables: list[Table] = []
    text_el = section_el.find("v3:text", NS)
    if text_el is not None:
        for tbl in text_el.iter():
            if _localname(tbl) == "table":
                table_counter[0] += 1
                tables.append(_extract_table(tbl, f"t{table_counter[0]:04d}"))

    subsections: list[Section] = []
    for comp in section_el.findall("v3:component", NS):
        for sub in comp.findall("v3:section", NS):
            subsections.append(
                _parse_section(sub, depth + 1, section_path, table_counter, section_names)
            )

    return Section(
        loinc_code=loinc,
        section_name=name,
        title=title,
        text=_section_text(section_el),
        section_path=section_path,
        tables=tables,
        subsections=subsections,
        depth=depth,
    )


# --------------------------------------------------------------------------- document
def _effective_time(root: Any) -> str | None:
    el = root.find("v3:effectiveTime", NS)
    raw = el.get("value") if el is not None else None
    if not raw or len(raw) < 8:
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def parse_spl(
    source: str | Path | bytes, section_names: dict[str, str] | None = None
) -> SPLDocument:
    """Parse an SPL XML file or byte string into an SPLDocument.

    `section_names` maps LOINC code -> friendly name; pass
    pharmarag.config.SECTION_NAMES. Falls back to the label's own displayName.
    """
    names = section_names or {}
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)

    if isinstance(source, bytes):
        root = etree.fromstring(source, parser=parser)
    else:
        root = etree.parse(str(source), parser=parser).getroot()

    set_id_el = root.find("v3:setId", NS)
    set_id = (set_id_el.get("root") if set_id_el is not None else None) or "UNKNOWN"

    ver_el = root.find("v3:versionNumber", NS)
    try:
        doc_version = int(ver_el.get("value") or 0) if ver_el is not None else 0
    except ValueError:
        doc_version = 0

    title_el = root.find("v3:title", NS)
    title = _clean("".join(title_el.itertext())) if title_el is not None else ""

    counter = [0]
    sections: list[Section] = []
    for body in root.iter():
        if _localname(body) != "structuredBody":
            continue
        for comp in body.findall("v3:component", NS):
            for sec in comp.findall("v3:section", NS):
                sections.append(_parse_section(sec, 0, "", counter, names))
        break

    ingredients = sorted(
        {
            (el.get("displayName") or "").strip()
            for el in root.iter()
            if _localname(el) == "activeMoiety" and el.get("displayName")
        }
    )

    return SPLDocument(
        set_id=set_id,
        doc_version=doc_version,
        effective_time=_effective_time(root),
        title=title,
        sections=sections,
        ingredient_names=[i for i in ingredients if i],
    )
