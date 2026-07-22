# PharmaRAG — Run Book

The complete operational guide: every command to build, verify, run, and evaluate
PharmaRAG **from a fresh clone on a new machine**, in order, with the **expected
output, wall-clock time, and dollar cost** of each step. All costs assume the same
models the project pins (`text-embedding-3-small`, `gpt-5.4` family); all timings
were measured on an ordinary consumer laptop (CPU-only, no GPU).

> Educational / portfolio demonstration. Not a medical device, not clinical
> decision support, not medical advice. See `README.md` for the full regulatory
> positioning and architecture.

---

## 0 · What you need before starting

| Requirement | Detail |
|---|---|
| **OS** | Windows 10/11 with PowerShell (the helper scripts are `.ps1`; everything else is portable Python) |
| **Python** | 3.12+ — managed automatically by `uv` |
| **[`uv`](https://docs.astral.sh/uv/)** | The only supported package manager. **Never `pip install`**; add dependencies with `uv add` |
| **OpenAI API key** | Used for embeddings, synthesis, and the LLM input guard. Set a hard spend cap in the OpenAI dashboard first — the whole project fits in **under $5 of API spend** |
| **Qdrant server binary** | Single `.exe`, no Docker. Download the Windows asset from [qdrant/qdrant releases](https://github.com/qdrant/qdrant/releases) into `.qdrant\` (~80 MB) |
| **Disk** | **~20 GB free**: label archive ~0.2 GB, SQLite ~2.5 GB, Qdrant server storage ~13 GB |
| **RAM** | 8 GB works; 16 GB is comfortable. The Qdrant server memory-maps its storage, so RAM stays flat regardless of index size |
| **GPU** | Optional. Without one, the reranker automatically falls back to a CPU cross-encoder (`ms-marco-MiniLM-L-6-v2`, ADR-024) |
| **Network** | Needed for the one-time downloads (DailyMed labels, RxNav/RxClass, OpenAI). All downloads except OpenAI are free |

### Total budget for a complete from-scratch build

| Phase | Wall-clock | API cost |
|---|---|---|
| Setup + preflight (§1–2) | ~15 min | $0 |
| Corpus build: select → download → index (§4.1–4.2) | ~30–45 min | ~$0.40 |
| Entities + brand harvest + graph (§4.4–4.5) | ~1 h unattended | $0 |
| Evaluation dataset + calibration + scorecard (§6) | ~35 min machine | ~$2 |
| Human review of the eval set (§6.5) | ~45 min of a person | $0 |
| **Total** | **~half a day, mostly unattended** | **≈ $2.50** |

Re-runs are much cheaper: every stage is resumable, the embedding cache makes
re-embeds of unchanged text $0, and the label archive is never re-downloaded.

---

## 1 · One-time setup

```powershell
# from the repository root
uv sync
```
> **Expect:** `uv` resolves the locked dependencies (`uv.lock`) into `.venv`,
> ending with `Installed N packages`. ~2–5 min on first run. No errors.

```powershell
Copy-Item .env.example .env    # then open .env and paste your OPENAI_API_KEY
```
> **Expect:** a new `.env`. The minimum working contents:
> ```dotenv
> OPENAI_API_KEY=sk-...
> CORPUS_VERSION=dev              # will become dailymed-<date> after §4.1
> GRAPH_VERSION=none              # will become graph-dailymed-<date> after §4.5
> CALIBRATOR_VERSION=uncalibrated # becomes v1 only after §6.3
> DEVICE=auto
> QDRANT_URL=http://localhost:6333
> ```
> Behind a corporate/antivirus TLS-inspecting proxy (Zscaler, Avast, Kaspersky —
> anything that re-signs certificates), also set `PHARMARAG_CA_BUNDLE` to a PEM
> bundle containing the proxy's root CA (see §8). Never commit `.env` or any
> `certs/*.pem`.

```powershell
powershell -File scripts\start_qdrant.ps1
```
> **Expect:** `Qdrant up on http://localhost:6333`. Launches the native Qdrant
> binary from `.qdrant\qdrant.exe` with storage in `data\qdrant_server`; if it is
> already running the script says so and exits cleanly. Stop it any time with
> `Stop-Process -Name qdrant` (safe — storage is on disk).
>
> **Why a server is required:** without `QDRANT_URL`, the client opens the store
> in embedded mode and loads the whole collection into Python memory — at the
> full corpus size that means 6+ GB of RAM and minutes of silent hang. The
> server memory-maps instead: instant startup, flat RAM, and the UI + API can
> run simultaneously.

---

## 2 · Preflight checks ($0, no network)

```powershell
uv run python --version
```
> **Expect:** Python 3.12 or newer.

```powershell
uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"
```
> **Expect:** `cuda: True` or `cuda: False` — **both are fine**. `False` means the
> CPU reranker fallback will be used. To enable the GPU reranker
> (`bge-reranker-v2-m3`) later, install a CUDA torch wheel:
> `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`

```powershell
uv run pytest -m deterministic -q
```
> **Expect:** all tests pass (e.g. `80 passed`; on a fresh clone a couple of
> corpus-dependent tests skip until §4 has run). This is the free safety-gate
> suite — no network, no OpenAI, runs in ~1–2 min.

---

## 3 · (Optional) Reset to a pristine database

Skip this on a fresh clone. If you have a previous build and want a clean
rebuild, this keeps every downloaded label and re-derives everything else:

```powershell
# stop any running build first
Remove-Item data\pharmarag.db -ErrorAction SilentlyContinue
Remove-Item data\qdrant_server -Recurse -Force -ErrorAction SilentlyContinue
```
> `data\archive`, `data\snapshots`, and `data\corpus_selection.json` are
> untouched, so §4 re-parses from labels already on disk — no re-downloading.
> Deleting `pharmarag.db` also drops the embedding cache, so the re-embed costs
> the full ~$0.40 again. **Never delete the whole `data\` folder** — the label
> archive takes the longest to re-fetch. §9 lists what is safe to delete.

---

## 4 · Build the pipeline

Run in order. Every step is resumable and safe to re-run. Pick one snapshot name
of the form `dailymed-YYYY-MM-DD` (today's date) and use it consistently — it
becomes `CORPUS_VERSION` and is stamped into every cache key and audit record.

### 4.1 — Freeze exactly 1000 drugs + download their labels

```powershell
uv run python scripts/build_corpus_1000.py --snapshot dailymed-YYYY-MM-DD
```
> **What it does:** builds a deterministic stratified candidate pool across ATC
> classes plus five safety strata (high-volume, interaction-heavy,
> narrow-therapeutic-index, renally-adjusted, LASA pairs), downloads one
> *quality-passing* SPL label per drug from DailyMed into a content-addressed
> (SHA-256) archive, and freezes **exactly 1000** canonical ingredients to
> `data/corpus_selection.json`.
> **Expect (final lines):**
> ```
> [corpus] froze 1000 canonical ingredients (1000 distinct SPL documents) -> ...corpus_selection.json
> [corpus] next: uv run python scripts/build_index.py --snapshot dailymed-YYYY-MM-DD
> ```
> **Time:** ~10–20 min cold (network-bound); seconds on re-run — already-archived
> labels are skipped, and an interrupted run resumes where it stopped.
> **Cost:** $0 (DailyMed is free). Re-freeze offline from an existing archive
> with `--freeze-only`. Afterwards set `CORPUS_VERSION=dailymed-YYYY-MM-DD` in `.env`.

### 4.2 — Parse → chunk → embed → index

```powershell
uv run python scripts/build_index.py --snapshot dailymed-YYYY-MM-DD   # Qdrant must be up (§1)
```
> **What it does:** parses the SPL XML (tables kept intact), chunks by clinical
> section with per-section policies, linearizes dosing-table rows, embeds every
> chunk (SHA-cached), and rebuilds the Qdrant collection with named dense +
> sparse vectors. Overdosage chunks are written to SQLite but **refused at index
> write time** (ADR-005).
> **Expect:** a per-drug log line, a cost-estimate line, then a final count:
> ```
> [build] indexing 1000 documents from dailymed-YYYY-MM-DD
> [build] atorvastatin            514 chunks (1 non-retrievable),  36 parents
> ... (1000 lines) ...
> [build] ~19,000,000 tokens -> est. $0.38 (cache hits are $0)
> [build] wrote ~232,000 points | {'exists': True, 'points': ~232000}
> ```
> **Reference numbers from a real build:** 1000 documents → **233,735 chunks**
> (232,673 retrievable + 1,062 held-out Overdosage), 135,154 parents,
> **232,674 Qdrant points**, ~19 M embedded tokens.
> **Time:** ~15–25 min first run (CPU parsing + API embedding); minutes on re-run
> with a warm cache. **Cost: ≈ $0.40 once** — re-embedding unchanged chunks is $0.
> **Free dry run:** add `--dry-run` to parse + chunk without touching OpenAI.

### 4.3 — Verify the build

```powershell
uv run python scripts/verify_corpus.py
```
> Runs every post-build invariant in one command. **Expect** all `[PASS]`, ending
> `RESULT: ALL HARD INVARIANTS PASS ✓`:
> ```
>   [PASS] frozen selection == 1000 unique           1000 drugs, 1000 unique
>   [PASS] every frozen drug has retrievable chunks  all present
>   [PASS] ADR-005 Overdosage NOT retrievable        0 (must be 0)
>   [PASS] ADR-047 audit log append-only             UPDATE + DELETE both blocked
>   [PASS] Qdrant index populated                    ~232,000 points
> ```
> Seconds with the server running.

### 4.4 — Gazetteer, LASA table, brand aliases

```powershell
uv run python scripts/build_entities.py --brands
```
> **What it does:** harvests brand names from RxNav for every frozen ingredient
> (so `Lipitor` resolves to `atorvastatin`), writes `data/brand_names.json`,
> then builds the drug gazetteer and the LASA (look-alike/sound-alike) confusable
> table strictly from the frozen corpus.
> **Expect:**
> ```
> [brands] harvesting from RxNav for 1000 ingredients (network, $0)
> [brands] 50/1000 ingredients, NNN brand candidates
> ...
> [brands] NNNN unambiguous aliases -> ...brand_names.json
> [brands] NN combination-product brands excluded (map to >1 corpus ingredient)
> [entities] restricted to frozen corpus: 1000/... drugs
> [entities] gazetteer: ~3000 names -> ...gazetteer.json
> [entities] LASA table: NN names with confusable neighbours
> ```
> **Time:** the harvest is ~2,000 sequential RxNav calls — **~25–55 min**
> depending on your network (measured: ~53 min through a TLS-inspecting proxy).
> Unattended; **$0**. Once `brand_names.json` exists, re-running *without*
> `--brands` rebuilds gazetteer + LASA from it in seconds, offline.
>
> Combination brands (Caduet, Janumet) are **deliberately excluded** — resolving
> them to one component would silently answer about half the product. Every
> brand→ingredient swap is surfaced as a substitution in answers and audit rows.

### 4.5 — Knowledge graph (RxClass → NetworkX)

```powershell
uv run python scripts/build_graph.py
```
> **What it does:** harvests RxClass memberships (drug → pharmacologic class →
> mechanism/risk) and builds the deterministic NetworkX property graph used for
> class-term expansion and the additive-risk check.
> **Expect:** graph stats JSON, then
> ```
> [graph] wrote ...data\graph\graph-dailymed-YYYY-MM-DD.json
> [graph] set GRAPH_VERSION=graph-dailymed-YYYY-MM-DD in .env
> ```
> **Time:** ~10 min with network (the harvest is cached to
> `data/rxclass_harvest.json`); seconds afterwards with `--offline`. **$0.**
> Set the printed `GRAPH_VERSION` value in `.env`.

---

## 5 · Run the demo

Start the Qdrant server first (§1). With `QDRANT_URL` set, the UI and API can
run at the same time.

**UI (Streamlit):**
```powershell
uv run streamlit run app/main.py
```
> **Expect:** `Local URL: http://localhost:8501`. The page shows the AI-disclosure
> strip, the system sidebar (device, corpus version, calibration status), and
> example-query chips split into **Grounded answers** vs **Refusals — by design**.
> Click a chip → pipeline stages stream → a cited answer card (summary, numbered
> claims with citation chips, and a verification row showing the four guardrail
> verdicts) or a typed refusal.

**API (FastAPI + SSE):**
```powershell
uv run uvicorn pharmarag.api.main:app --port 8000
```
> **Expect:** `Uvicorn running on http://127.0.0.1:8000`. Then:
> ```powershell
> curl http://localhost:8000/health
> ```
> returns JSON with `"status":"ok"`, the device, the corpus version, and
> `"index":{"exists":true,"points":...}`. A `Dockerfile` is included if you
> prefer to containerize the API.

**Per-query cost** (measured from the audit log): a full answered query —
guard, synthesis, and grounding evaluation — averages **~$0.002**; refusals
average ~$0.0015. Exact-match cache hits are $0.

### Five queries that each prove a design decision

| Query | Expect |
|---|---|
| `Atorvastatin and clarithromycin interaction` | cited answer (interaction + management) |
| `Lipitor dosing` | brand → atorvastatin substitution surfaced |
| `interaction between <any corpus drug> and grapefruit` | grounded answer or `NO_EVIDENCE_IN_CORPUS` refusal — never a guess |
| `I have CKD, how much metformin should I take?` | refusal `OUT_OF_SCOPE` (personal medical advice) + reformulation; audit row PHI-redacted |
| `warfarin, amiodarone and simvastatin` | pairwise interaction checks + additive-risk fields |

---

## 6 · Evaluation: dataset → calibration → scorecard

The eval harness measures what generic RAG metrics cannot: missed interactions,
false refusals, and escape rates that must be zero. Full sequence
**~35 min machine time, ~$2**, plus one human review pass.

### 6.1 — Generate the evaluation dataset (~15 min, ~$1.20)
```powershell
uv run python eval/autogen.py --all
```
> Writes `eval/data/silver.jsonl` (~230 items across 8 categories). Out-of-corpus
> refusal items are *computed* against the live corpus (a drug is verified
> absent, not assumed); DDI/dosing items are LLM-written then verified by a
> second pass, which drops roughly a fifth.

### 6.2 — Generate calibration labels (~10 min, ~$0.40)
```powershell
uv run python eval/autolabel.py --queries 40 --topk 20
```
> Combines distant supervision with an LLM relevance judge; disagreements land in
> `eval/data/calibration_disagreements.jsonl` — reviewing *only those* is the
> highest-value human pass available.

### 6.3 — Fit the abstention calibrator (~10 s, $0)
```powershell
uv run python eval/run_eval.py --calibrate
```
> **Expect:** `[calib] ECE (out-of-fold) = 0.0x` and a written
> `data/calibrator.json`. Until `CALIBRATOR_VERSION=v1` is set in `.env`,
> abstention honestly reports `uncalibrated` and uses a plain sigmoid — do
> **not** describe the system as "calibrated" before then.
>
> ⚠️ **Do not flip `CALIBRATOR_VERSION=v1` blindly.** The abstention thresholds
> (`THRESHOLD_INCLUDE`/`THRESHOLD_FLAG`, ADR-026) are expressed on the
> raw-score scale; a fitted Platt transform compresses scores, and if even
> top candidates map below the include threshold the system will mass-abstain.
> Activate only together with a threshold migration to the calibrated scale,
> then re-run §6.4 to measure the system under the new configuration. Note
> also that a `CALIBRATOR_VERSION` value in the ambient shell environment
> overrides `.env` (env vars win over `.env` by design) — verify with:
> `uv run python -c "from pharmarag.config import settings; print(settings.calibrator_version)"`

### 6.4 — Score the system (~1 h on CPU, ~$0.40)
```powershell
uv run python eval/run_eval.py --llm-guard
```
> Runs the **full pipeline** (LLM input guard included) over every dataset item
> and writes `eval/data/results.jsonl` + `eval/data/scorecard.json`. **Expect** a
> progress line every 25 items, then the scorecard: refusal rates by category,
> retrieval miss vs. corpus-coverage decomposition, and the ADR-045 CI gates —
> dose-error, LASA, and unsafe-leak **escape rates must be 0** or the run exits
> non-zero. Measured on a 229-item run: 62 min on CPU, **$0.36** (cost scales
> with how many items reach synthesis).

### 6.5 — Human review: silver → gold (~45 min, $0, the step that matters)
```powershell
uv run streamlit run eval/review_app.py
```
> Accept / edit / reject at ~10 s per item. Until this pass, every metric is
> **provisional** — the scorecard carries `n_gold / n_silver` and `run_eval`
> prints a silver warning. Do not claim a "pharmacist-labeled golden dataset"
> while `n_gold` is 0. Keep `rejected.jsonl`; the rejection rate is a finding.

### 6.6 — (Optional) DDInter recall oracle
```powershell
uv run python eval/run_eval.py --coverage
```
> Requires `eval/data/ddinter.csv`, downloaded manually from ddinter.scbdd.com
> (CC BY-NC-SA — gitignored, **never committed, never indexed**; ADR-001). Reports
> how much of an independent interaction database the corpus documents.

---

## 7 · Quality gates (run before any commit)

```powershell
uv run pytest -q          # all deterministic tests — expect all pass, ~1–2 min, $0
uv run ruff check .       # expect: All checks passed!
uv run mypy src           # expect: Success: no issues found
```

A bare `pytest` can never spend money — the `llm_judge` marker is excluded by
default in `pyproject.toml`. The paid judge suite (`uv run pytest -m llm_judge`,
~$1–2) is for tagged releases only.

---

## 8 · Troubleshooting

| Symptom | Cause & fix |
|---|---|
| SSL / certificate-verify errors on DailyMed, RxNav, or OpenAI | A TLS-inspecting proxy (corporate or antivirus) is re-signing certificates — not a network outage. Export the proxy's root CA to a PEM file and set `PHARMARAG_CA_BUNDLE=certs/<your-bundle>.pem` in `.env`. Never commit certificates |
| `Connection refused` on localhost:6333 | Qdrant server not running — `powershell -File scripts\start_qdrant.ps1` |
| `qdrant.exe not found` | Download the Windows release asset from qdrant/qdrant GitHub releases into `.qdrant\` |
| Streamlit hangs minutes on "Running pipeline…" with no stages | `QDRANT_URL` unset → embedded mode is loading the whole store into RAM. Set it in `.env` and start the server |
| "storage already accessed by another instance" | Two processes opened embedded mode. Set `QDRANT_URL`; with the server, UI + API run simultaneously |
| `load_expanded()` warns and returns a small number | `data/corpus_selection.json` missing — run §4.1 |
| `build_index` says "no manifest.json" | The snapshot was never fetched — run §4.1 with your snapshot name |
| `cuda: False` | CPU-only torch wheel; harmless (CPU reranker fallback). §2 has the CUDA install |
| A `msvcrt` traceback printed at interpreter exit | Benign Windows shutdown noise from the Qdrant client destructor. Ignore |
| Brand harvest looks hung | It is ~2,000 sequential network calls with buffered output — check for an established connection to rxnav.nlm.nih.gov; results are written only at the end |

---

## 9 · The `data/` directory: what is safe to delete

| Path | What it is | Safe to delete? |
|---|---|---|
| `data/archive/` | Content-addressed SPL label XML | **No** — longest to re-download |
| `data/snapshots/` | Manifests mapping drugs → labels | **No** — needed to rebuild |
| `data/corpus_selection.json` | The frozen 1000-drug selection | **No** — the identity of the corpus |
| `data/rxclass_harvest.json` | Cached RxClass harvest | Avoid (saves ~10 min network) |
| `data/brand_names.json` | RxNav brand-alias harvest | Avoid (saves ~25–55 min network) |
| `data/pharmarag.db` | Chunks + embedding cache + audit log | Yes — rebuilds from archive; re-embed ≈ $0.40 |
| `data/qdrant_server/` | Qdrant server storage (~13 GB) | Yes — always rebuilt by `build_index` |
| `data/gazetteer.json` · `lasa_table.json` · `data/graph/` | Derived entity/graph artifacts | Yes — rebuilt in seconds/minutes |
| `.qdrant/qdrant.exe` | Qdrant server binary | Yes — re-download from GitHub releases |

---

## 10 · Cost summary

| Action | Cost | Frequency |
|---|---|---|
| Deterministic tests · ruff · mypy | $0 | every push |
| DailyMed / RxNav / RxClass fetches | $0 | one-time (cached) |
| Full corpus embed (~19 M tokens) | ~$0.40 | once; warm-cache re-runs ~$0 |
| Per interactive query | ~$0.002 | as used; cache hits $0 |
| Eval dataset + calibration (`autogen` + `autolabel`) | ~$1.60 | per dataset rebuild |
| Full scorecard run (`run_eval.py --llm-guard`) | ~$0.50–1.50 | per release |
| LLM-judge test suite | ~$1–2 | tagged releases only |

**Everything, end to end, from a bare machine: ≈ $2.50–5.** Set a hard spend cap
in the OpenAI dashboard as a backstop before exposing any free-text demo surface.
