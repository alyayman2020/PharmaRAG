"""Presentation layer for the Streamlit demo (ADR-049).

Kept apart from `main.py` so the demo's control flow stays readable: the app
file reads as pipeline-in, evidence-out, and every rule about how a claim, a
citation, or a refusal *looks* lives here.

Design intent: this is a clinical-safety instrument, not a chat toy. The visual
language is therefore restrained — one accent colour, generous whitespace,
tabular numerals for anything measured — and the elements that carry safety
meaning (safety tier, refusal reason, guardrail verdicts) are the only ones
allowed to use colour. Everything renders correctly in both light and dark: no
hard-coded background or text colours, only tokens that flip with the theme.
"""

from __future__ import annotations

import html

# Safety tier -> (label, colour token). Tier drives colour everywhere it appears
# so the reader learns one mapping and it holds across the whole surface.
TIER_META: dict[int, tuple[str, str]] = {
    1: ("Critical", "var(--pr-tier1)"),
    2: ("Warning", "var(--pr-tier2)"),
    3: ("Dosing", "var(--pr-tier3)"),
    4: ("Reference", "var(--pr-tier4)"),
}

SECTION_TIER: dict[str, int] = {
    "Boxed Warning": 1,
    "Contraindications": 1,
    "Drug Interactions": 2,
    "Warnings and Precautions": 2,
    "Warnings": 2,
    "Dosage and Administration": 3,
    "Use in Specific Populations": 3,
    "Clinical Pharmacology": 4,
}

REFUSAL_COPY: dict[str, tuple[str, str]] = {
    "NO_EVIDENCE_IN_CORPUS": (
        "No evidence in the corpus",
        "The labelling indexed here does not answer this. The system refuses "
        "rather than fall back on the model's own memory.",
    ),
    "BELOW_CONFIDENCE_THRESHOLD": (
        "Below the confidence threshold",
        "Passages were retrieved, but none scored high enough to answer safely.",
    ),
    "OUT_OF_SCOPE": (
        "Out of scope",
        "This system answers questions about drug labelling, not personal " "medical decisions.",
    ),
    "UNSAFE_QUERY": (
        "Refused on safety grounds",
        "This query seeks information that could cause harm.",
    ),
    "AMBIGUOUS_DRUG": (
        "Ambiguous drug name",
        "More than one drug matches closely. Guessing between look-alike names "
        "is the error class this system exists to prevent, so it asks instead.",
    ),
    "POPULATION_ONLY_SWEEP": (
        "Needs a specific drug",
        "The corpus is organised by drug. A sweep across every label is a "
        "report, not a retrieval.",
    ),
    "GUARDRAIL_BLOCKED": (
        "Blocked by verification",
        "An answer was generated but failed post-hoc verification, so it was "
        "withheld. Failing closed is the designed behaviour.",
    ),
    "EXPANSION_TOO_BROAD": (
        "Class expansion too broad",
        "The drug class matched more members than can be searched precisely.",
    ),
}

GUARDRAIL_LABELS: dict[str, tuple[str, str]] = {
    "citations": ("Citations", "Every claim's citation resolves to assembled context"),
    "dose": ("Dose integrity", "Units, magnitudes, frequencies and qualifiers verified"),
    "lasa": ("Drug identity", "No unauthorised or look-alike drug name in the answer"),
    "grounding": ("Grounding", "Each claim entailed by its own cited source"),
}

CSS = """
<style>
:root {
  --pr-accent:   #0f766e;
  --pr-accent-2: #0d9488;
  --pr-tier1:    #b91c1c;
  --pr-tier2:    #b45309;
  --pr-tier3:    #1d4ed8;
  --pr-tier4:    #4b5563;
  --pr-ok:       #15803d;
  --pr-surface:  rgba(15, 118, 110, 0.05);
  --pr-border:   rgba(128, 128, 128, 0.26);
  --pr-muted:    rgba(128, 128, 128, 0.95);
}
@media (prefers-color-scheme: dark) {
  :root {
    --pr-accent:   #2dd4bf;
    --pr-accent-2: #5eead4;
    --pr-tier1:    #f87171;
    --pr-tier2:    #fbbf24;
    --pr-tier3:    #60a5fa;
    --pr-tier4:    #9ca3af;
    --pr-ok:       #4ade80;
    --pr-surface:  rgba(45, 212, 191, 0.07);
  }
}

/* ---------- masthead ---------- */
.pr-hero { padding: .35rem 0 1.15rem 0; border-bottom: 1px solid var(--pr-border);
           margin-bottom: 1.35rem; }
.pr-hero h1 { font-size: 2.05rem; font-weight: 700; letter-spacing: -.024em;
              margin: 0 0 .4rem 0; line-height: 1.15; }
.pr-hero h1 .pr-mark { color: var(--pr-accent); }
.pr-hero p { margin: 0; font-size: 1.01rem; line-height: 1.6; color: var(--pr-muted);
             max-width: 62ch; }
.pr-rule { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .95rem; }
.pr-rule span { font-size: .69rem; font-weight: 600; letter-spacing: .07em;
                text-transform: uppercase; padding: .26rem .6rem; border-radius: 999px;
                border: 1px solid var(--pr-border); color: var(--pr-muted); }

/* ---------- AI disclosure (EU AI Act Art. 50) ---------- */
.pr-disclose { display: flex; gap: .7rem; align-items: flex-start; padding: .8rem .95rem;
               border: 1px solid var(--pr-border); border-left: 3px solid var(--pr-accent);
               border-radius: 8px; background: var(--pr-surface); margin-bottom: 1.4rem; }
.pr-disclose .pr-ic { font-size: 1.05rem; line-height: 1.35; }
.pr-disclose div { font-size: .855rem; line-height: 1.55; }
.pr-disclose b { font-weight: 650; }

/* ---------- section headings ---------- */
.pr-h { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
        color: var(--pr-muted); margin: .3rem 0 .7rem 0; }

/* ---------- answer ---------- */
.pr-answer { border: 1px solid var(--pr-border); border-left: 3px solid var(--pr-accent);
             border-radius: 10px; padding: 1.15rem 1.3rem; background: var(--pr-surface);
             margin-bottom: 1.1rem; }
.pr-answer .pr-sum { font-size: 1.08rem; line-height: 1.62; font-weight: 500; margin: 0; }

.pr-claim { display: flex; gap: .8rem; padding: .72rem 0; border-top: 1px solid var(--pr-border); }
.pr-claim:first-of-type { border-top: none; padding-top: .95rem; margin-top: .35rem;
                          border-top: 1px solid var(--pr-border); }
.pr-claim .pr-n { flex: 0 0 auto; width: 1.45rem; height: 1.45rem; border-radius: 50%;
                  border: 1px solid var(--pr-border); display: flex; align-items: center;
                  justify-content: center; font-size: .72rem; font-weight: 700;
                  color: var(--pr-accent); font-variant-numeric: tabular-nums; }
.pr-claim .pr-body { flex: 1 1 auto; }
.pr-claim .pr-txt { font-size: .945rem; line-height: 1.6; }
.pr-cites { margin-top: .34rem; display: flex; gap: .3rem; flex-wrap: wrap; }
.pr-cite { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .68rem;
           padding: .13rem .42rem; border-radius: 5px; border: 1px solid var(--pr-border);
           color: var(--pr-muted); }

/* ---------- DDI facets ---------- */
.pr-facets { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
             gap: .7rem; margin: 1rem 0 .2rem 0; }
.pr-facet { border: 1px solid var(--pr-border); border-radius: 8px; padding: .68rem .8rem; }
.pr-facet .k { font-size: .66rem; font-weight: 700; letter-spacing: .08em;
               text-transform: uppercase; color: var(--pr-muted); margin-bottom: .28rem; }
.pr-facet .v { font-size: .89rem; line-height: 1.5; }

/* ---------- guardrails ---------- */
.pr-guards { display: flex; gap: .45rem; flex-wrap: wrap; margin: .2rem 0 1.15rem 0; }
.pr-guard { display: inline-flex; align-items: center; gap: .34rem; font-size: .755rem;
            font-weight: 550; padding: .3rem .62rem; border-radius: 999px;
            border: 1px solid var(--pr-border); }
.pr-guard.ok   { color: var(--pr-ok); }
.pr-guard.fail { color: var(--pr-tier1); }

/* ---------- refusal ---------- */
.pr-refusal { border: 1px solid var(--pr-border); border-left: 3px solid var(--pr-tier2);
              border-radius: 10px; padding: 1.15rem 1.3rem; margin-bottom: 1rem; }
.pr-refusal .pr-code { display: inline-block; font-family: ui-monospace, Menlo, Consolas, monospace;
                       font-size: .69rem; font-weight: 600; letter-spacing: .04em;
                       padding: .22rem .55rem; border-radius: 5px; color: var(--pr-tier2);
                       border: 1px solid var(--pr-border); margin-bottom: .7rem; }
.pr-refusal h4 { margin: 0 0 .42rem 0; font-size: 1.06rem; font-weight: 650; letter-spacing: -.01em; }
.pr-refusal .pr-why { font-size: .9rem; line-height: 1.6; color: var(--pr-muted); margin: 0; }
.pr-refusal .pr-said { font-size: .945rem; line-height: 1.6; margin: .7rem 0 0 0;
                       padding-top: .7rem; border-top: 1px solid var(--pr-border); }
.pr-byd { margin-top: .85rem; padding: .68rem .8rem; border-radius: 7px;
          background: var(--pr-surface); font-size: .875rem; line-height: 1.55; }

/* ---------- evidence ---------- */
.pr-src { border: 1px solid var(--pr-border); border-radius: 9px; padding: .85rem 1rem;
          margin-bottom: .7rem; }
.pr-src-top { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap;
              margin-bottom: .5rem; }
.pr-tier { font-size: .655rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
           padding: .19rem .5rem; border-radius: 4px; border: 1px solid currentColor; }
.pr-drug { font-weight: 650; font-size: .93rem; }
.pr-sec  { font-size: .82rem; color: var(--pr-muted); }
.pr-meta { margin-left: auto; font-size: .715rem; color: var(--pr-muted);
           font-variant-numeric: tabular-nums; }
.pr-quote { font-size: .855rem; line-height: 1.62; padding: .62rem .75rem; border-radius: 6px;
            background: var(--pr-surface); white-space: pre-wrap;
            max-height: 15rem; overflow-y: auto; }
.pr-srcid { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .67rem;
            color: var(--pr-muted); }

/* ---------- stage timeline ---------- */
.pr-stage { display: flex; align-items: baseline; gap: .65rem; padding: .3rem 0;
            font-size: .825rem; border-bottom: 1px dashed var(--pr-border); }
.pr-stage:last-child { border-bottom: none; }
.pr-stage .s-n { font-weight: 600; flex: 0 0 8.6rem; }
.pr-stage .s-d { color: var(--pr-muted); flex: 1 1 auto; word-break: break-word; }
.pr-stage .s-t { font-variant-numeric: tabular-nums; color: var(--pr-muted); font-size: .77rem; }

/* ---------- misc ---------- */
.pr-sub { font-size: .84rem; color: var(--pr-muted); margin-bottom: .85rem;
          padding-left: .6rem; border-left: 2px solid var(--pr-accent); }
.pr-foot { margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid var(--pr-border);
           font-size: .77rem; line-height: 1.6; color: var(--pr-muted); }
div[data-testid="stMetricValue"] { font-size: 1.02rem; }
.stButton > button { text-align: left; justify-content: flex-start; font-size: .855rem;
                     line-height: 1.35; padding: .55rem .8rem; min-height: 3.1rem; }
</style>
"""


def e(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


def tier_for(section: str) -> int:
    """Map a section path to its safety tier by longest matching section name."""
    s = str(section or "")
    best, best_len = 4, 0
    for name, tier in SECTION_TIER.items():
        if name.lower() in s.lower() and len(name) > best_len:
            best, best_len = tier, len(name)
    return best


def hero() -> str:
    return (
        '<div class="pr-hero">'
        '<h1><span class="pr-mark">◆</span> PharmaRAG</h1>'
        "<p>Drug interactions, contraindications, and dosing — answered from primary "
        "FDA labelling, with a verifiable citation behind every claim and a typed "
        "refusal when the evidence is not there.</p>"
        '<div class="pr-rule">'
        "<span>Evidence-first</span><span>Cited per claim</span>"
        "<span>Refuses when unsure</span><span>Fully audited</span>"
        "</div></div>"
    )


def disclosure() -> str:
    return (
        '<div class="pr-disclose"><div class="pr-ic">⚠︎</div><div>'
        "<b>You are interacting with an AI system.</b> Educational demonstration "
        "only — not a medical device, not clinical decision support, and not medical "
        "advice. Always verify against current prescribing information."
        "<br><span style='opacity:.75'>Disclosure per EU AI Act Article 50.</span>"
        "</div></div>"
    )


def guardrail_row(results: dict[str, bool]) -> str:
    if not results:
        return ""
    out = []
    for key, ok in results.items():
        label, why = GUARDRAIL_LABELS.get(key, (key.title(), ""))
        out.append(
            f'<span class="pr-guard {"ok" if ok else "fail"}" title="{e(why)}">'
            f'{"✓" if ok else "✗"} {e(label)}</span>'
        )
    return f'<div class="pr-guards">{"".join(out)}</div>'


def answer_block(summary: str, claims: list[dict]) -> str:
    rows = []
    for i, c in enumerate(claims, 1):
        cites = "".join(
            f'<span class="pr-cite">{e(cid)}</span>' for cid in (c.get("citation_ids") or [])
        )
        rows.append(
            f'<div class="pr-claim"><div class="pr-n">{i}</div><div class="pr-body">'
            f'<div class="pr-txt">{e(c.get("text", ""))}</div>'
            f'<div class="pr-cites">{cites}</div></div></div>'
        )
    return f'<div class="pr-answer"><p class="pr-sum">{e(summary)}</p>{"".join(rows)}</div>'


def facets(payload: dict) -> str:
    keys = [
        ("ddi_severity", "Severity"),
        ("ddi_mechanism", "Mechanism"),
        ("ddi_management", "Management"),
    ]
    cards = [
        f'<div class="pr-facet"><div class="k">{lbl}</div>'
        f'<div class="v">{e(payload[k])}</div></div>'
        for k, lbl in keys
        if payload.get(k)
    ]
    return f'<div class="pr-facets">{"".join(cards)}</div>' if cards else ""


def refusal_block(reason_code: str, summary: str, what_would_help: str) -> str:
    title, why = REFUSAL_COPY.get(
        reason_code, ("Refused", "The system declined to answer this query.")
    )
    help_html = (
        f'<div class="pr-byd"><b>What would help:</b> {e(what_would_help)}</div>'
        if what_would_help
        else ""
    )
    said = f'<p class="pr-said">{e(summary)}</p>' if summary else ""
    return (
        f'<div class="pr-refusal"><span class="pr-code">{e(reason_code or "REFUSAL")}</span>'
        f"<h4>{e(title)}</h4><p class=\"pr-why\">{e(why)}</p>{said}{help_html}</div>"
    )


def source_card(src: dict) -> str:
    tier = tier_for(src.get("section", ""))
    label, colour = TIER_META[tier]
    eff = src.get("effective_time") or "date unknown"
    url = str(src.get("url") or "")
    link = f'<a href="{e(url)}" target="_blank" rel="noopener">DailyMed ↗</a>' if url else ""
    return (
        '<div class="pr-src"><div class="pr-src-top">'
        f'<span class="pr-tier" style="color:{colour}">{label}</span>'
        f'<span class="pr-drug">{e(src.get("drug", ""))}</span>'
        f'<span class="pr-sec">{e(src.get("section", ""))}</span>'
        f'<span class="pr-meta">score {e(src.get("score", ""))} · {e(eff)} · {link}</span>'
        "</div>"
        f'<div class="pr-quote">{e(src.get("text", ""))}</div>'
        f'<div class="pr-srcid" style="margin-top:.45rem">id {e(src.get("id", ""))}</div>'
        "</div>"
    )


def stage_rows(stages: list) -> str:
    return "".join(
        f'<div class="pr-stage"><span class="s-n">{e(s.name)}</span>'
        f'<span class="s-d">{e(s.detail)}</span>'
        f'<span class="s-t">{s.ms:,.0f} ms</span></div>'
        for s in stages
    )
