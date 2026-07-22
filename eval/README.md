# Track C — automated pipeline

Run in this order. Total: **~50 min of your time, ~$2.**

```powershell
# 1 · generate the dataset            ~15 min, ~$1.20   (LLM writes + verifies)
uv run python eval/autogen.py --all

# 2 · generate calibration labels     ~10 min, ~$0.40
uv run python eval/autolabel.py --queries 40 --topk 20

# 3 · fit the calibrator              ~10 s, $0
uv run python eval/run_eval.py --calibrate

# 4 · score the system                ~10 min, ~$0.50
uv run python eval/run_eval.py --llm-guard

# 5 · REVIEW  ~45 min, $0  ← the only manual step, and the one that matters
uv run streamlit run eval/review_app.py
```

## Why step 5 is not optional

Steps 1–4 produce a **silver** dataset: LLM-generated, verified for internal
consistency, never seen by a pharmacist. It is genuinely useful — retrieval
recall, refusal behaviour, and calibration all measure something real, because
chunk selection is independent of the retriever.

What it cannot establish is whether a question is one a pharmacist would
actually ask, or whether the attached evidence is *sufficient* rather than
merely related. That is clinical judgement, and no amount of generation
substitutes for it.

**Until step 5, `run_eval.py` prints a silver warning on every run and the
scorecard carries `n_gold` / `n_silver`.** Do not write "pharmacist-labeled
golden dataset" in the README while `n_gold` is 0 — that is the one claim in
this project a reviewer can check in a single question, and it is your strongest
differentiator precisely because it is hard to fake.

Review is accept / edit / reject at ~10 s an item. 250 items ≈ 45 minutes.

## What is auto-generated well

| Category | Method | Quality |
|---|---|---|
| Out-of-corpus must-refuse | **Computed** against the live `documents` table | **Better than hand-written** — a drug is verified absent, not assumed |
| Unsafe must-refuse | Templates over real corpus drugs, harm/personal pairs | Good |
| LASA traps | Built from your `lasa_table.json` | Good — every pair is one the system can actually confuse |
| Compound regimens | Built from graph risk classes | Good — exercises the additive-risk path |
| DDI / dosing / contraindication | LLM writes, then a second pass **verifies** the source answers it | Fair — the verify pass drops roughly a fifth |

## Calibration labels

Two signals, combined in `autolabel.py`:

- **Distant supervision** — gold chunk or same parent → relevant. High precision, low recall.
- **LLM judge** — same rubric a human would apply. Higher recall, noisier.

Distant supervision alone trains a calibrator that is over-confident that things
are *not* relevant, which produces **over-abstention** — the failure a
recall-first system must avoid. Disagreements between the two are written to
`calibration_disagreements.jsonl`. **Reviewing only those is the highest-value
human pass available**, because they are exactly where the labels are uncertain.
```powershell
uv run python -c "from eval.schema import read_jsonl; print(len(read_jsonl('eval/data/calibration_disagreements.jsonl')))"
```

## Files

| File | Purpose |
|---|---|
| `autogen.py` | dataset generation (silver) |
| `autolabel.py` | calibration judgments |
| `review_app.py` | **silver → gold, ~10 s/item** |
| `labeling_app.py` | full manual labeling (unused if you autogen) |
| `calibrate.py` | Platt fit, 5-fold cross-fit, reliability diagram |
| `metrics.py` | safety scorecard (ADR-043 decomposition) |
| `ddinter.py` | held-out recall oracle |
| `run_eval.py` | the harness |
| `data/silver.jsonl` | generated, unreviewed |
| `data/golden.jsonl` | reviewed — the real dataset |
| `data/rejected.jsonl` | items you rejected (keep them; the rejection rate is itself a finding) |
