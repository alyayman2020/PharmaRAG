"""Drug-name gazetteer and matching rules (ADR-036, ADR-041).

Substring matching is the specific failure this module exists to prevent:
"codeine" matches inside "hydrocodone" and "oxycodone"; "prednisone" inside
"prednisolone". A false match here silently widens the retrieval filter and
pulls an unrelated drug's label into evidence.

Every drug-name match in the project — extraction (ADR-036) and the K4 display
gate (ADR-041) — routes through `find_mentions` so the rules cannot drift apart.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Stems where a shorter name is a substring of a longer, clinically different one.
SUBSTRING_BLOCKLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("codeine", "hydrocodone"),
        ("codeine", "oxycodone"),
        ("prednisone", "prednisolone"),
        ("prednisolone", "methylprednisolone"),
        ("nifedipine", "nimodipine"),
        ("nifedipine", "nicardipine"),
        ("cortisone", "hydrocortisone"),
        ("testosterone", "methyltestosterone"),
        ("amphetamine", "dextroamphetamine"),
        ("amphetamine", "methamphetamine"),
    }
)


@dataclass(frozen=True, slots=True)
class Mention:
    text: str
    normalized: str
    rxcui: str | None
    start: int
    end: int


def normalize(name: str) -> str:
    """Lowercase, strip salt forms and dose forms."""
    s = name.lower().strip()
    s = re.sub(
        r"\b(hydrochloride|hcl|sodium|potassium|calcium|sulfate|sulphate|besylate|"
        r"maleate|mesylate|tartrate|succinate|fumarate|citrate|acetate|phosphate|"
        r"bromide|chloride|dihydrate|monohydrate|anhydrous)\b",
        "",
        s,
    )
    s = re.sub(
        r"\b(tablet|capsule|injection|solution|suspension|cream|ointment|"
        r"film coated|extended release|oral|topical|iv|im)\b",
        "",
        s,
    )
    return re.sub(r"[^a-z0-9\s-]", " ", s).strip()


class Gazetteer:
    """name -> rxcui, with longest-match-first, word-boundary-anchored lookup."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self._map: dict[str, str] = {}
        self._pattern: re.Pattern[str] | None = None
        self._reverse: dict[str, str] | None = None
        if entries:
            self.add_many(entries)

    def add(self, name: str, rxcui: str) -> None:
        key = normalize(name)
        if key:
            self._map[key] = rxcui
            self._pattern = None
            self._reverse = None

    def add_many(self, entries: dict[str, str]) -> None:
        for name, rxcui in entries.items():
            self.add(name, rxcui)

    def __len__(self) -> int:
        return len(self._map)

    def rxcui(self, name: str) -> str | None:
        return self._map.get(normalize(name))

    def canonical_name(self, rxcui: str) -> str | None:
        """The identity's own name, for display when the matched text is an alias.

        A brand alias ("lipitor") and its ingredient ("atorvastatin") share an
        rxcui; showing the user the brand they typed would name a partition the
        system did not search. Built lazily and cached, shortest-name-wins so an
        ingredient beats a longer brand that maps to the same identity.
        """
        if self._reverse is None:
            rev: dict[str, str] = {}
            for name, rx in self._map.items():
                if rx not in rev or len(name) < len(rev[rx]):
                    rev[rx] = name
            self._reverse = rev
        return self._reverse.get(rxcui)

    def _compiled(self) -> re.Pattern[str]:
        if self._pattern is None:
            # Longest first so "methylprednisolone" wins over "prednisolone".
            names = sorted(self._map, key=len, reverse=True)
            alt = "|".join(re.escape(n) for n in names) or r"(?!x)x"
            self._pattern = re.compile(rf"\b({alt})\b", re.IGNORECASE)
        return self._pattern

    def find_mentions(self, text: str) -> list[Mention]:
        """Word-boundary-anchored, longest-match-first, blocklist-filtered."""
        out: list[Mention] = []
        taken: list[tuple[int, int]] = []
        for m in self._compiled().finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue  # already covered by a longer match
            norm = normalize(m.group(1))
            if self._blocked(norm, text, s, e):
                continue
            taken.append((s, e))
            out.append(Mention(m.group(1), norm, self._map.get(norm), s, e))
        return sorted(out, key=lambda x: x.start)

    def _blocked(self, norm: str, text: str, start: int, end: int) -> bool:
        window = text[max(0, start - 12) : min(len(text), end + 12)].lower()
        return any(short == norm and long in window for short, long in SUBSTRING_BLOCKLIST)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._map, indent=0, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Gazetteer:
        return cls(json.loads(path.read_text(encoding="utf-8")))
