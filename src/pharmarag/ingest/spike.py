"""MILESTONE A1.0 — the falsification spike. RUN THIS FIRST. ~30 minutes.

ADR-007 commits 22 hours to a custom lxml SPL parser on the inference that
openFDA's JSON destroys table structure. openFDA states it "converts the data
into JSON" and that content is "reformatted to make it easier to read" — strong
circumstantial evidence, not proof.

Never spend 22 hours on an inference you can falsify in 30 minutes.

  * openFDA PRESERVES tables  -> you just saved ~14 hours. Revisit ADR-007.
  * openFDA FLATTENS tables   -> proceed with documented evidence, not a guess.

Writes docs/adr/007-openfda-table-fidelity.md either way.

NETWORK REQUIRED. Not runnable in an offline sandbox.
"""

from __future__ import annotations

import json
import sys

import httpx

from pharmarag.config import ROOT, settings

# Drugs with two-dimensional renal dosing tables (CrCl bands x indication/age).
CANDIDATES = ["vancomycin", "levofloxacin", "enoxaparin", "rivaroxaban", "dabigatran"]


def fetch_openfda(generic: str) -> dict[str, object] | None:
    url = f"{settings.openfda_base}/drug/label.json"
    params: dict[str, str | int] = {"search": f'openfda.generic_name:"{generic}"', "limit": 1}
    with httpx.Client(timeout=30) as c:
        r = c.get(url, params=params)
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        return results[0] if results else None


def analyse(record: dict[str, object]) -> dict[str, object]:
    """Does openFDA's representation retain table structure?"""
    dosage = record.get("dosage_and_administration")
    findings: dict[str, object] = {
        "field_type": type(dosage).__name__,
        "is_list_of_strings": isinstance(dosage, list) and all(isinstance(x, str) for x in dosage),
        "contains_html_table_tags": False,
        "contains_pipe_or_tab_structure": False,
        "sample": "",
    }
    text = " ".join(dosage) if isinstance(dosage, list) else str(dosage or "")
    findings["contains_html_table_tags"] = "<table" in text.lower() or "<td" in text.lower()
    findings["contains_pipe_or_tab_structure"] = ("|" in text) or ("\t" in text)
    findings["sample"] = text[:600]
    findings["verdict"] = (
        "TABLES PRESERVED"
        if findings["contains_html_table_tags"]
        else "TABLES LIKELY FLATTENED TO TEXT"
    )
    return findings


def main() -> int:
    out_dir = ROOT / "docs" / "adr"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = [
        "# ADR-007 · openFDA table fidelity spike",
        "",
        "Ran before committing ~22 h to a custom SPL parser.",
        "",
    ]
    verdicts: list[str] = []

    for generic in CANDIDATES:
        print(f"[spike] fetching {generic} ...", flush=True)
        try:
            rec = fetch_openfda(generic)
        except Exception as exc:
            print(f"[spike]   error: {exc}")
            report.append(f"## {generic}\n\nFetch failed: `{exc}`\n")
            continue
        if rec is None:
            report.append(f"## {generic}\n\nNo openFDA record.\n")
            continue
        f = analyse(rec)
        verdicts.append(str(f["verdict"]))
        print(f"[spike]   {f['verdict']}")
        report.append(
            f"## {generic}\n\n"
            f"- field type: `{f['field_type']}`\n"
            f"- list of plain strings: `{f['is_list_of_strings']}`\n"
            f"- HTML table tags present: `{f['contains_html_table_tags']}`\n"
            f"- pipe/tab structure: `{f['contains_pipe_or_tab_structure']}`\n"
            f"- **verdict: {f['verdict']}**\n\n"
            f"```\n{f['sample']}\n```\n"
        )

    preserved = sum(v == "TABLES PRESERVED" for v in verdicts)
    decision = (
        "openFDA preserves table structure. REVISIT ADR-007 — the custom parser "
        "may be unnecessary and ~14 h can be saved."
        if preserved and preserved == len(verdicts)
        else "openFDA does not reliably preserve table structure. ADR-007 CONFIRMED: "
        "parse raw SPL XML with lxml. Challenge #3 would be unsolvable from "
        "openFDA JSON alone."
    )
    report.insert(3, f"**Decision:** {decision}\n")
    path = out_dir / "007-openfda-table-fidelity.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n[spike] wrote {path}")
    print(f"[spike] DECISION: {decision}")
    (out_dir / "007-spike-raw.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
