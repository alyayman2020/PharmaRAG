"""Precomputed look-alike / sound-alike table (ADR-020).

Cost: $0, minutes to build. Feeds the Tier-3 abstention band — the strongest
LASA mitigation in the project, because it refuses to retrieve for the wrong
drug rather than detecting the error after generation.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    a_flags = [False] * len(a)
    b_flags = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - window), min(i + window + 1, len(b))
        for j in range(lo, hi):
            if not b_flags[j] and b[j] == ch:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    k = 0
    transpositions = 0
    for i, ch in enumerate(a):
        if a_flags[i]:
            while not b_flags[k]:
                k += 1
            if ch != b[k]:
                transpositions += 1
            k += 1
    t = transpositions / 2
    jaro = (matches / len(a) + matches / len(b) + (matches - t) / matches) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def soundex(name: str) -> str:
    """Cheap phonetic key for sound-alike detection."""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    s = "".join(c for c in name.lower() if c.isalpha())
    if not s:
        return ""
    out = s[0].upper()
    prev = codes.get(s[0], "")
    for ch in s[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            out += code
        if ch not in "hw":
            prev = code
    return (out + "000")[:4]


def build_lasa_table(names: list[str], threshold: float = 0.90) -> dict[str, list[str]]:
    """Pairwise LASA table. O(n^2) — ~1.1M comparisons at n=1500, seconds."""
    table: dict[str, set[str]] = {}
    keys = {n: soundex(n) for n in names}
    for a, b in combinations(sorted(set(names)), 2):
        look_alike = jaro_winkler(a, b) >= threshold
        sound_alike = keys[a] == keys[b] and keys[a] != ""
        if look_alike or sound_alike:
            table.setdefault(a, set()).add(b)
            table.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in sorted(table.items())}


def save(table: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")


def load(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
