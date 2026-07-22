"""Brand-name aliases for the gazetteer (ADR-020, ADR-036).

A clinician asks about *Lipitor*, not *atorvastatin*. Without brand aliases the
resolver returns NO_EVIDENCE for a drug that is fully indexed — a false refusal
caused by vocabulary, not by coverage, which is the worst kind because the
evidence is right there.

Brands come from RxNorm via RxNav (free, no key). For each ingredient we take
the RxCUI, then the related concepts of term type BN (Brand Name). The map is
harvested once, cached to disk, and folded into the gazetteer at load time —
so resolution stays a local dictionary lookup with no network on the query path.

The alias resolves TO the ingredient, and the resolver surfaces that as a
substitution (ADR-020): the user is told "Lipitor -> atorvastatin" rather than
silently answered about a name they did not type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pharmarag.config import settings

RXCUI_BY_NAME = "{base}/REST/rxcui.json"
RELATED_BY_RXCUI = "{base}/REST/rxcui/{rxcui}/related.json"

# Brand strings that are useless or actively harmful as aliases: single letters,
# pure numbers, or a brand identical to the ingredient itself.
_MIN_BRAND_LEN = 3


def brands_path() -> Path:
    from pharmarag.config import DATA

    return DATA / "brand_names.json"


def _rxcui_for(name: str) -> str | None:
    from pharmarag.http import get_json

    data: Any = get_json(
        RXCUI_BY_NAME.format(base=settings.rxnav_base),
        params={"name": name, "search": "1"},
    )
    ids = (data or {}).get("idGroup", {}).get("rxnormId") or []
    return str(ids[0]) if ids else None


def _brands_for(rxcui: str) -> list[str]:
    from pharmarag.http import get_json

    data: Any = get_json(
        RELATED_BY_RXCUI.format(base=settings.rxnav_base, rxcui=rxcui),
        params={"tty": "BN"},
    )
    groups = (data or {}).get("relatedGroup", {}).get("conceptGroup") or []
    out: list[str] = []
    for g in groups:
        for c in g.get("conceptProperties") or []:
            nm = str(c.get("name", "")).strip()
            if nm:
                out.append(nm)
    return out


def harvest(
    ingredients: list[str], *, progress_every: int = 50
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build ({brand -> ingredient}, {ambiguous brand -> ingredients}).

    Network required, $0. A brand claimed by more than one corpus ingredient is a
    COMBINATION product — Caduet is amlodipine+atorvastatin, Janumet is
    sitagliptin+metformin. Aliasing it to whichever ingredient happened to be
    harvested first would answer about half the product and silently omit the
    other half, which is precisely the class of error this system refuses to
    make. Those brands are excluded from the alias map and returned separately so
    the exclusion is visible rather than assumed.
    """
    claims: dict[str, set[str]] = {}
    uniq = sorted(set(ingredients))
    for i, ing in enumerate(uniq, 1):
        try:
            rxcui = _rxcui_for(ing)
            if rxcui:
                for brand in _brands_for(rxcui):
                    b = brand.strip().lower()
                    if len(b) >= _MIN_BRAND_LEN and b != ing.strip().lower():
                        claims.setdefault(b, set()).add(ing)
        except Exception as exc:  # one drug's failure must not end the harvest
            print(f"[brands] {ing}: {type(exc).__name__}: {exc}")
        if progress_every and i % progress_every == 0:
            print(f"[brands] {i}/{len(uniq)} ingredients, {len(claims)} brand candidates")

    mapping = {b: next(iter(ings)) for b, ings in claims.items() if len(ings) == 1}
    ambiguous = {b: sorted(ings) for b, ings in claims.items() if len(ings) > 1}
    return mapping, ambiguous


def save(mapping: dict[str, str], path: Path | None = None) -> Path:
    p = path or brands_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, indent=0, sort_keys=True), encoding="utf-8")
    return p


def load(path: Path | None = None) -> dict[str, str]:
    p = path or brands_path()
    if not p.is_file():
        return {}
    data: dict[str, str] = json.loads(p.read_text(encoding="utf-8"))
    return data
