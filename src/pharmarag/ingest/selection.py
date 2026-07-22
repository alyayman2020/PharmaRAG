"""Drug selection (ADR-003).

Stratified union, not top-N. The corpus is ENGINEERED AGAINST THE EVALUATION —
that is the defensible answer when someone asks "why these drugs?".

The five hand-built strata below (``track_a_slice``) guarantee every safety
mechanism has something to bite on. The full corpus scales that to exactly
``CORPUS_SIZE`` ingredient-level concepts by stratified ATC selection
(:mod:`pharmarag.ingest.select`), and the resolved result is FROZEN to
``data/corpus_selection.json`` so ``load_expanded`` is deterministic and offline.

The freeze is produced by ``scripts/build_corpus_1000.py`` (the bridge that the
old docstring called ``expand_corpus.py`` — it never existed; this replaces it).
``load_expanded`` reads that file and returns exactly ``CORPUS_SIZE`` names.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ADR-003 — the corpus is exactly this many unique ingredient-level concepts.
CORPUS_SIZE = 1000

# 1 · High prescription volume — real-world relevance.
HIGH_VOLUME = [
    "atorvastatin",
    "levothyroxine",
    "lisinopril",
    "metformin",
    "amlodipine",
    "metoprolol",
    "omeprazole",
    "losartan",
    "gabapentin",
    "sertraline",
]

# 2 · High interaction burden — CYP substrates, inhibitors, inducers.
INTERACTION_HEAVY = [
    "clarithromycin",
    "ketoconazole",
    "fluconazole",
    "rifampin",
    "carbamazepine",
    "ritonavir",
    "verapamil",
    "diltiazem",
    "fluoxetine",
    "amiodarone",
]

# 3 · Narrow therapeutic index — highest harm potential.
NARROW_THERAPEUTIC_INDEX = [
    "warfarin",
    "digoxin",
    "lithium",
    "phenytoin",
    "theophylline",
    "tacrolimus",
    "cyclosporine",
    "colchicine",
]

# 4 · Renally cleared, dose-adjusted — stresses the dosing-threshold path.
RENAL_ADJUSTED = [
    "vancomycin",
    "gabapentin",
    "levofloxacin",
    "enoxaparin",
    "rivaroxaban",
    "dabigatran",
    "apixaban",
    "allopurinol",
    "famotidine",
    "baclofen",
]

# 5 · Documented LASA pairs — directly targets Challenge #1.
LASA_PAIRS = [
    "citalopram",
    "celecoxib",  # Celexa / Celebrex
    "hydralazine",
    "hydroxyzine",
    "lamotrigine",
    "terbinafine",  # Lamictal / Lamisil
    "clonidine",
    "klonopin",
    "prednisone",
    "prednisolone",
    "glipizide",
    "glyburide",
]

# Ranked backup list (ADR-003). When a stratified candidate fails DailyMed
# resolution or the label quality gate, the corpus-sizing bridge substitutes from
# this list — in order — so a documented, high-value single-ingredient drug fills
# the slot rather than the corpus silently shrinking. These are common, well-
# labelled US ingredients across therapeutic areas, chosen to almost always have a
# full prescribing-information SPL on DailyMed.
BACKUP_INGREDIENTS = [
    "acetaminophen",
    "ibuprofen",
    "naproxen",
    "aspirin",
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "doxycycline",
    "cephalexin",
    "prednisone",
    "albuterol",
    "montelukast",
    "fluticasone",
    "loratadine",
    "cetirizine",
    "pantoprazole",
    "esomeprazole",
    "ranitidine",
    "ondansetron",
    "metoclopramide",
    "hydrochlorothiazide",
    "furosemide",
    "spironolactone",
    "carvedilol",
    "atenolol",
    "propranolol",
    "clopidogrel",
    "simvastatin",
    "pravastatin",
    "rosuvastatin",
    "ezetimibe",
    "insulin glargine",
    "glimepiride",
    "pioglitazone",
    "sitagliptin",
    "escitalopram",
    "paroxetine",
    "venlafaxine",
    "duloxetine",
    "bupropion",
    "mirtazapine",
    "trazodone",
    "quetiapine",
    "olanzapine",
    "risperidone",
    "aripiprazole",
    "lorazepam",
    "alprazolam",
    "diazepam",
    "zolpidem",
    "pregabalin",
    "topiramate",
    "levetiracetam",
    "valproic acid",
    "oxcarbazepine",
    "tramadol",
    "morphine",
    "oxycodone",
    "hydrocodone",
    "cyclobenzaprine",
    "tizanidine",
    "methotrexate",
    "azathioprine",
    "hydroxychloroquine",
    "sulfasalazine",
    "tamsulosin",
    "finasteride",
    "sildenafil",
    "tadalafil",
    "estradiol",
    "levonorgestrel",
    "medroxyprogesterone",
    "testosterone",
    "prednisolone",
    "dexamethasone",
    "hydrocortisone",
    "budesonide",
    "mesalamine",
    "ursodiol",
    "lactulose",
    "nortriptyline",
    "amitriptyline",
    "doxepin",
    "clonazepam",
    "buspirone",
]


def track_a_slice() -> list[str]:
    """~50 unique ingredients spanning all five strata (the safety core)."""
    seen: dict[str, None] = {}
    for group in (
        HIGH_VOLUME,
        INTERACTION_HEAVY,
        NARROW_THERAPEUTIC_INDEX,
        RENAL_ADJUSTED,
        LASA_PAIRS,
    ):
        for name in group:
            seen.setdefault(name.lower(), None)
    return sorted(seen)


def strata() -> dict[str, list[str]]:
    return {
        "high_volume": HIGH_VOLUME,
        "interaction_heavy": INTERACTION_HEAVY,
        "narrow_therapeutic_index": NARROW_THERAPEUTIC_INDEX,
        "renal_adjusted": RENAL_ADJUSTED,
        "lasa_pairs": LASA_PAIRS,
    }


def selection_path(path: Path | str | None = None) -> Path:
    from pharmarag.config import DATA

    return Path(path) if path else DATA / "corpus_selection.json"


def save_expanded(
    drugs: list[str], *, path: Path | str | None = None, meta: dict[str, object] | None = None
) -> Path:
    """Freeze the resolved corpus selection to ``data/corpus_selection.json``.

    Enforces the ADR-003 invariant at write time: exactly ``CORPUS_SIZE`` unique,
    lower-cased names in deterministic order.
    """
    import json

    norm = [d.strip().lower() for d in drugs]
    if len(set(norm)) != len(norm):
        raise ValueError("corpus selection contains duplicates after normalization")
    if len(norm) != CORPUS_SIZE:
        raise ValueError(f"corpus selection must be exactly {CORPUS_SIZE}, got {len(norm)}")
    p = selection_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"corpus_size": CORPUS_SIZE, "drugs": norm, **(meta or {})}, indent=2),
        encoding="utf-8",
    )
    return p


def load_expanded(path: Path | str | None = None) -> list[str]:
    """Return the frozen corpus selection — exactly ``CORPUS_SIZE`` unique names.

    Reads ``data/corpus_selection.json`` (written by ``scripts/build_corpus_1000.py``).
    If that freeze does not exist yet, it falls back to the Track A safety slice so
    downstream scripts still run, but prints a LOUD warning to stderr — the silent
    49-drug fallback that masked the real corpus size is gone (ADR-003).
    """
    import json

    p = selection_path(path)
    if not p.is_file():
        print(
            f"[selection] WARNING: {p} not found — falling back to the "
            f"{len(track_a_slice())}-drug Track A safety slice. Run "
            "`uv run python scripts/build_corpus_1000.py` to build the full "
            f"{CORPUS_SIZE}-drug corpus.",
            file=sys.stderr,
        )
        return track_a_slice()
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data["drugs"])
