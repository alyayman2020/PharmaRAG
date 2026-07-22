# Architecture Decision Records — Index

Code and docs throughout this repo cite decisions by number (`ADR-023`). This
index is the authoritative map from number to decision. Each entry is the
decision as stated where it is enforced in code; entries marked 📄 have a full
ADR document in this directory, the rest are recorded in the docstrings and
comments of the module that implements them (linked).

A wrong answer is treated as a safety event; most of these decisions exist to
make a specific failure mode structurally impossible rather than merely
unlikely.

| ADR | Decision | Enforced in |
|---|---|---|
| 001 | DDInter 2.0 is a **held-out** evaluation oracle — never redistributed, never an ingestion source, zero API round trips | `eval/ddinter.py` |
| 002 | Chemical-name tokenization is an **accepted, unmitigated risk** at the embedding layer; entity-first retrieval and BM25 reduce its blast radius | `src/pharmarag/embed/` |
| 003 | Entity-first design: resolve drug identity **before** retrieval; search runs inside that partition via indexed payload filters | `src/pharmarag/index/store.py` |
| 004 | DailyMed acquisition with content-addressed (SHA-256) immutable archiving; first label passing `label_quality` wins; resumable by design | `src/pharmarag/ingest/dailymed.py` |
| 005 | Overdosage sections are ingested and archived but **never indexed** — enforced by absence, not by a WHERE clause a developer can forget | `src/pharmarag/config.py`, `src/pharmarag/index/upsert.py` |
| 006 | RxClass harvesting for the knowledge graph; full GraphRAG (multi-pass LLM extraction) **rejected on budget** | `src/pharmarag/graph/rxclass.py` |
| 007 | 📄 openFDA rejected as table source — it strips table markup; DailyMed SPL XML preserves fidelity | [007-openfda-table-fidelity.md](007-openfda-table-fidelity.md) |
| 008 | Dual table representation: structured rows preserved separately from prose, one atomic chunk per linearized row (dose never separated from its qualifier) | `packages/spl_parser/tables.py`, `src/pharmarag/chunking/` |
| 009 | Canonical-label selection cascade: NDA/BLA first, then section coverage, then recency; variant conflicts diffed, not merged | `src/pharmarag/ingest/select.py` |
| 010 | `spl_parser` is a standalone package, extracted to its own public repo in Phase 5 | `packages/spl_parser/` |
| 011 | Section-aware chunking with parent-document retrieval (child chunks embed, parents assemble) | `src/pharmarag/chunking/chunker.py` |
| 012 | Per-section chunk policies — a single global size would merge unrelated interaction statements into one vector | `src/pharmarag/config.py` |
| 013 | Qualifier rule: a dose without its condition (renal band, hepatic class, age) is a wrong dose; population tags detected and enforced | `src/pharmarag/chunking/metadata.py` |
| 014 | Chunks under 60 tokens are mostly prefix — merge upward with sibling | `src/pharmarag/config.py` |
| 015 | Full 1536-dim embeddings, no Matryoshka truncation | `src/pharmarag/config.py` |
| 016 | SHA-keyed embedding cache; re-running unchanged chunks costs $0; incremental writes make builds resumable | `src/pharmarag/embed/client.py` |
| 017 | One Qdrant collection with named dense + sparse vectors — splitting would force app-side score fusion | `src/pharmarag/index/store.py` |
| 018 | Payload indexes on every filter field; `effective_time` deliberately absent — staleness is surfaced, never filtered | `src/pharmarag/index/store.py` |
| 019 | One persistence layer: SQLite stores corpus, cache, audit; LangGraph checkpoints use SqliteSaver on the same store | `src/pharmarag/db/` |
| 020 | Typed entity resolution with a Tier-3 abstention band; population-only sweeps refuse; ambiguity surfaces candidates | `src/pharmarag/entity/resolve.py` |
| 021 | BM25 sparse vectors computed over **raw** text (not display text) | `scripts/build_index.py` |
| 022 | Hybrid dense + sparse retrieval fused with RRF | `src/pharmarag/retrieve/search.py` |
| 023 | Retrieval is entity-prefiltered with 4 prefetch branches → RRF; an **empty candidate set is a hard refusal**, never a fallback to unfiltered search | `src/pharmarag/retrieve/search.py` |
| 024 | bge-reranker-v2-m3 on GPU, ms-marco-MiniLM CPU fallback | `src/pharmarag/rerank/reranker.py` |
| 025 | Reranking **never skips** — it is the scoring stage that feeds abstention, and thin candidate sets are exactly where abstention matters most | `src/pharmarag/rerank/reranker.py` |
| 026 | Raw reranker scores are not probabilities; Platt calibration is opt-in via `CALIBRATOR_VERSION` so a stale fit is never silently reused | `src/pharmarag/rerank/reranker.py` |
| 027 | Context assembled in **safety-tier order** (boxed warnings and contraindications lead regardless of relevance), asymmetric relevance floors, 8k token cap | `src/pharmarag/generate/context.py` |
| 028 | Citation **integrity** guardrail: every claim's ID must resolve and must have been in the assembled context | `src/pharmarag/guardrails/citations.py` |
| 029 | Synthesis returns strict JSON against a schema; refusal is a first-class `answer_type` | `src/pharmarag/generate/schema.py` |
| 030 | Model tiering by context size: nano sees the full 12k context, mini evaluates 2k claim+source pairs — input shrinkage is what makes verification affordable | `src/pharmarag/generate/synthesize.py` |
| 031 | LangGraph only where agency earns its keep: the compound loop, bounded retry, and resume-on-disambiguation — not the linear path | `src/pharmarag/orchestrate/` |
| 032 | Orchestration framework lands **after** there is branching to orchestrate; the Track A pipeline is a plain function | `src/pharmarag/pipeline.py` |
| 033 | Compound regimens: pairwise decomposition (one pair per visit, capped at 20) **plus** class-burden additive-risk check — QT burden and "triple whammy" are class-count properties no pairwise pass can produce | `src/pharmarag/graph/traverse.py` |
| 034 | Evaluator loop is bounded: exactly one retry, then refuse — never a third attempt | `src/pharmarag/orchestrate/nodes.py` |
| 035 | Knowledge graph is a deterministic property graph (NetworkX), not an LLM-extracted one | `src/pharmarag/graph/build.py` |
| 036 | One set of drug-name matching rules everywhere — naive substring matching finds "codeine" inside "hydrocodone" | `src/pharmarag/entity/gazetteer.py` |
| 037 | Class-term expansion capped at 25 RxCUIs; overflow is surfaced | `src/pharmarag/config.py` |
| 038 | Guard model tiering by consequence: the input guard (highest consequence) gets the strongest model; the guard runs **first, always, and is never cached** | `src/pharmarag/guardrails/input_guard.py` |
| 039 | K2 grounding: per-claim entailment with tier-routed consequences; `PARTIALLY_SUPPORTED` is never a silent pass — safety-tier-1 refuses, others drop **and disclose** | `src/pharmarag/guardrails/grounding.py` |
| 040 | Dose guardrail: every numeric dose in an answer must appear in a cited source | `src/pharmarag/guardrails/dose.py` |
| 041 | LASA guardrail: drug names in the answer are checked against resolved identity and the confusable-name table | `src/pharmarag/guardrails/lasa_gate.py` |
| 043 | Safety-metric decomposition: "missed interaction" splits into retrieval miss vs. corpus-coverage gap; accuracy-vs-coverage across thresholds is the headline artifact | `eval/metrics.py` |
| 044 | Golden-dataset schema, labeling rubric, and target distribution; un-cited candidates are treated as irrelevant; blind re-label round for intra-rater agreement | `eval/schema.py` |
| 045 | CI gates: escape rates (dose, LASA, unsafe leak) must be **zero** — any escape fails the build | `eval/metrics.py` |
| 046 | Three SQLite files (corpus, checkpoints, MLflow), never merged | `src/pharmarag/db/` |
| 047 | Append-only audit log capturing everything needed to reconstruct any answer months later: versions, hashes, scores, verdicts | `src/pharmarag/audit.py` |
| 048 | FastAPI service with SSE stage streaming | `src/pharmarag/api/main.py` |
| 049 | One Streamlit surface with progressive disclosure; scripted example chips each prove a specific ADR | `app/main.py` |
| 050 | Exact-match answer caching on the canonical resolved query; **semantic caching rejected on safety grounds**; version fields mandatory in the key; guard verdicts never cached | `src/pharmarag/cache.py` |

ADR-042 was never assigned.
