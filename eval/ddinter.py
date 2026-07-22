"""DDInter 2.0 as a HELD-OUT recall oracle (ADR-001, ADR-043).

DDInter is CC BY-NC-SA 4.0 and ShareAlike is viral. It is therefore NEVER
indexed and NEVER redistributed — it lives outside the corpus and is used only
to measure recall. You get the whole evaluative value with the repo licence
clean.

Because it is never indexed, querying the system with DDInter pairs is a genuine
out-of-distribution recall test against independent expert-curated ground truth.

Download manually from ddinter.scbdd.com and place the CSV at
eval/data/ddinter.csv (gitignored). Expected columns include drug names and a
severity level; the loader is tolerant of column naming.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pharmarag.db import session

DEFAULT_PATH = Path(__file__).parent / "data" / "ddinter.csv"


@dataclass(slots=True)
class DDIPair:
    drug_a: str
    drug_b: str
    severity: str = ""

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.drug_a.lower(), self.drug_b.lower())))  # type: ignore[return-value]


def load_pairs(path: Path | None = None) -> list[DDIPair]:
    p = path or DEFAULT_PATH
    if not p.is_file():
        return []
    out: list[DDIPair] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        a = pick("drug_a", "druga", "drug1", "drug_1", "name_a")
        b = pick("drug_b", "drug2", "drug_2", "name_b")
        sev = pick("level", "severity", "interaction_level")
        if not (a and b):
            return []
        for row in reader:
            out.append(
                DDIPair(
                    str(row[a]).strip(),
                    str(row[b]).strip(),
                    str(row.get(sev, "")).strip() if sev else "",
                )
            )
    return out


def corpus_ingredients() -> set[str]:
    with session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT LOWER(ingredient_name) n FROM documents "
            "WHERE ingredient_name IS NOT NULL"
        ).fetchall()
    return {r["n"] for r in rows}


def pairs_in_corpus(pairs: list[DDIPair]) -> list[DDIPair]:
    """DDInter pairs where BOTH drugs are indexed. Only these can be a
    retrieval miss; the rest are corpus-coverage gaps (ADR-043)."""
    have = corpus_ingredients()
    return [p for p in pairs if p.drug_a.lower() in have and p.drug_b.lower() in have]


def pair_documented_in_labels(pair: DDIPair) -> bool:
    """Does OUR corpus actually document this interaction?

    ADR-043's critical distinction. DDInter contains interactions FDA labels
    never mention. A 'miss' on those is a CORPUS COVERAGE limitation and
    refusing is the CORRECT behaviour — not a system failure. Conflating the two
    makes a correctly-refusing system look broken.
    """
    with session() as conn:
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE retrievable=1 AND loinc_section_code='34073-7' "
            "AND LOWER(ingredient_name)=? AND LOWER(display_text) LIKE ? LIMIT 1",
            (pair.drug_a.lower(), f"%{pair.drug_b.lower()}%"),
        ).fetchone()
        if row:
            return True
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE retrievable=1 AND loinc_section_code='34073-7' "
            "AND LOWER(ingredient_name)=? AND LOWER(display_text) LIKE ? LIMIT 1",
            (pair.drug_b.lower(), f"%{pair.drug_a.lower()}%"),
        ).fetchone()
    return bool(row)


def coverage_report(sample: int = 400) -> dict[str, object]:
    """The README line: how much of DDInter our corpus actually documents."""
    import random

    pairs = load_pairs()
    if not pairs:
        return {"error": f"no DDInter CSV at {DEFAULT_PATH}"}
    in_corpus = pairs_in_corpus(pairs)
    rng = random.Random(42)
    probe = rng.sample(in_corpus, min(sample, len(in_corpus)))
    documented = sum(1 for p in probe if pair_documented_in_labels(p))
    return {
        "ddinter_pairs_total": len(pairs),
        "both_drugs_in_corpus": len(in_corpus),
        "sampled": len(probe),
        "documented_in_labels": documented,
        "corpus_coverage_rate": round(documented / len(probe), 3) if probe else 0.0,
    }
