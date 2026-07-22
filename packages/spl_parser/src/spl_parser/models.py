"""Parsed SPL data structures.

Deliberately plain dataclasses with no pharmarag imports — this package is
extracted to its own public repo in Phase 5 (ADR-010).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Table:
    """A table lifted out of an SPL <text> block.

    Track A captures structure but does not yet linearize rows — that is
    milestone B1 (ADR-008). `rows` preserves cell text with colspan/rowspan
    already expanded so B1 only has to do the linearization.
    """

    table_id: str
    caption: str | None
    headers: list[list[str]]
    rows: list[list[str]]
    footnotes: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        """Debug rendering only. NOT what gets embedded — see ADR-008."""
        out: list[str] = []
        if self.caption:
            out.append(f"**{self.caption}**")
        for h in self.headers:
            out.append(" | ".join(h))
        if self.headers:
            out.append(" | ".join("---" for _ in self.headers[-1]))
        out.extend(" | ".join(r) for r in self.rows)
        return "\n".join(out)


@dataclass(slots=True)
class Section:
    """One LOINC-coded SPL section, or a titled subsection of one."""

    loinc_code: str | None
    section_name: str
    title: str | None
    text: str
    section_path: str  # "5.2 Hepatotoxicity" — the citation anchor
    tables: list[Table] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)
    depth: int = 0

    def walk(self) -> list[Section]:
        """Self plus every descendant, depth-first."""
        out = [self]
        for s in self.subsections:
            out.extend(s.walk())
        return out

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.tables)


@dataclass(slots=True)
class SPLDocument:
    set_id: str
    doc_version: int
    effective_time: str | None  # ISO date; surfaced never filtered (ADR-018)
    title: str
    sections: list[Section] = field(default_factory=list)
    ingredient_names: list[str] = field(default_factory=list)
    brand_names: list[str] = field(default_factory=list)
    application_type: str | None = None

    def section_by_loinc(self, code: str) -> Section | None:
        for s in self.sections:
            if s.loinc_code == code:
                return s
        return None

    def coverage(self, codes: frozenset[str] | set[str]) -> dict[str, bool]:
        """Which target sections are present and non-empty.

        Drives the ADR-009 canonical-label selection: an ANDA label with more
        complete section coverage beats one with less.
        """
        present = {s.loinc_code: s.has_content for s in self.sections if s.loinc_code}
        return {c: present.get(c, False) for c in sorted(codes)}
