# Model Card — PharmaRAG (Track A)

## Intended use
Educational and portfolio demonstration of retrieval-augmented generation over
FDA Structured Product Labeling. **Not placed on the market or put into service
for a clinical purpose.** Not a medical device. Not clinical decision support.

## System components & versions
- **Corpus:** 1,000 frozen US drug ingredients, snapshot `dailymed-2026-07-20`
  (content-addressed archive; the frozen selection is
  `data/corpus_selection.json`).
- **Vector store:** Qdrant **server** v1.18.3 (native binary, hybrid
  dense + sparse in one collection, entity-prefiltered search). Embedded local
  mode remains only for tests — it cannot hold this corpus in memory.
- **Embeddings:** `text-embedding-3-small` (1536-dim, SHA-keyed cache).
- **Sparse:** Qdrant/bm25 via fastembed, computed locally.
- **Reranker:** BAAI/bge-reranker-v2-m3 (GPU) / cross-encoder ms-marco-MiniLM-L-6-v2
  (CPU fallback); scores every candidate, no skip path.
- **Synthesis:** `gpt-5.4-nano`, strict JSON schema. **Grounding evaluator (K2):**
  `gpt-5.4-mini`. **Input guard:** `gpt-5.4` (optional, per-query toggle).
- Model, prompt-template, corpus, graph, and calibrator versions are stamped
  into every append-only audit record.

## Regulatory position
- **EU AI Act Article 50(1)** (AI disclosure, applies from 2 August 2026) — the
  deployed demo displays a persistent AI-disclosure banner.
- **Articles 12/13/14** (record-keeping, transparency, oversight) — high-risk
  obligations now apply from **2 December 2027** for Annex III standalone
  systems following the Digital Omnibus. This system is engineered against them
  ahead of the deadline; it is not itself a high-risk system in deployment.
- **FDA** — design *correspondence* only, not a regulatory claim: FDA's clinical
  decision support guidance includes a criterion that the healthcare
  professional can independently review the basis for a recommendation.
  Per-claim citations to primary labeling are that mechanism.
- Nothing here is legal advice.

## Limitations — the most important section

1. **1,000 US drug ingredients, frozen at snapshot.** Not exhaustive; refresh is
   manual and label effective dates are surfaced, never filtered.
2. **US FDA labeling only.** No UK/EU/WHO dosing conventions.
3. **Overdosage sections deliberately excluded** from the retrievable index
   (ADR-005). Antidote and supportive-management information is unavailable.
   A deliberate safety trade, not an oversight.
4. **Abstention is UNCALIBRATED in the active configuration.** A Platt
   calibrator IS fit against the current corpus and validates honestly
   (**ECE 0.064** on 604 out-of-fold judgments, 5-fold cross-fit, no leakage) —
   but it is **not activated**, because the ADR-026 abstention thresholds are
   still expressed on the raw-score scale and would mass-abstain under the
   calibrated scale (the fitted transform maps even the best-scoring candidate
   below the 0.60 include threshold). Until the threshold migration lands and
   the scorecard is re-measured under it, every audit record carries
   `calibrator_version=uncalibrated`. Two further honesty notes: the
   calibration labels are machine-generated (distant supervision + LLM judge,
   75% agreement; 154 disagreements logged for human review), and the reranker
   signal's discrimination on them is modest (AUC 0.64).
5. **Grounding verification (K2) depends on an LLM judge.** Per-claim entailment
   runs on `gpt-5.4-mini` with a deterministic numeric pre-check, fails closed
   when the evaluator is unavailable, and drops-and-discloses unverifiable
   non-safety claims (ADR-039) — but the judge is itself a model. Its per-claim
   verdicts are written to the audit record so any answer can be re-reviewed.
   The numeric pre-check is deliberately asymmetric: a number carrying a **unit**
   that is absent from the cited source is a fabricated dose and hard-refuses at
   every tier, while a bare numeral degrades to `PARTIALLY_SUPPORTED` and takes
   the tier route (tier 1 refuses; others drop and disclose). Treating both
   identically produced false refusals on correct answers.
6. **Compound-regimen analysis is not on the demo path.** Pairwise expansion,
   class-burden / additive-risk traversal (QT burden, "triple whammy"), and the
   LangGraph orchestration exist as Track B modules with tests, but the demo
   surfaces (Streamlit, API) run the linear pipeline, which answers one
   drug-pair question at a time.
7. **English only.**
8. **No weight-based pediatric calculation.** Retrieval only, never computation.
9. **Challenge #5 (chemical-name tokenization) is an accepted, unmitigated
   risk** at the embedding layer, from ADR-002. Entity-first retrieval and BM25
   reduce its blast radius; they do not fix the embeddings.
10. **Some source labels contain run-together text.** A minority of DailyMed SPL
    table cells ship with words fused in the published XML itself — e.g.
    `Potentiatestheelectrophysiologicandhemodynamiceffects`. This is upstream
    data, reproduced faithfully rather than repaired: a dictionary-based splitter
    could silently corrupt a drug name, which is a worse failure than awkward
    text. It degrades BM25 term matching on the affected rows; dense retrieval
    and the synthesis model both handle it, and the meaning is preserved.
11. **Brand-name coverage is RxNorm-derived and deliberately conservative.**
    Brand aliases (Lipitor → atorvastatin) come from RxNav and are surfaced as
    substitutions, never applied silently. Brands that map to more than one
    corpus ingredient — combination products such as Caduet or Janumet — are
    **excluded** rather than resolved to one component, because answering about
    half a combination product while silently omitting the other half is exactly
    the failure class this system refuses to make.

## Evaluation

Full-pipeline scorecard, measured 2026-07-22 against the current corpus and
vector store: 229-item silver set (LLM-generated + verified, **not
pharmacist-reviewed — all figures provisional** until the silver→gold review),
LLM input guard on, cold cache, $0.36 per run.

| Metric | Result |
|---|---|
| Unsafe-query leak rate | 0.000 (20/20 refused) |
| Dose-error escape rate | 0.000 (CI gate) |
| LASA-substitution escape rate | 0.000 (CI gate) |
| Citation validity | 1.000 |
| Ungrounded-claim rate | 0.000 |
| Correct refusal on true corpus gaps | 0.978 |
| False-refusal rate | 0.201 |
| Retrieval miss rate | 0.321 |
| Context recall | 0.679 |

148/229 answered with cited, guardrail-verified output; the rest refused with
typed reason codes. The dominant error mode is over-refusal, which is the
designed failure direction: coverage is traded for the guarantee that what is
answered is grounded.

## Data
FDA Structured Product Labeling via DailyMed (US Government, public domain).
DDInter 2.0 is **not redistributed** — reserved as a held-out evaluation oracle.
