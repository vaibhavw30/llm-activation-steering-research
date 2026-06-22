# Geometry-of-Truth: Linear vs Non-Linear Probe Experiment

## Master Build Guide (Claude Code Prompts)

**Goal:** Measure whether "truth" is encoded linearly or non-linearly in an LLM's activations, by comparing a linear probe (logistic regression) against XGBoost across every layer, on multiple datasets. Produce CSVs and plots for a research meeting.

**Your constraints:** ~20 min to set up, 5+ hours of unattended runtime, CPU only.

**Model choice (given your runtime budget):** `google/gemma-2-2b`.

- It matches the model used in Julian's A-LQR paper — directly relevant.
- Bigger model = cleaner "truth" representations = results closer to Marks & Tegmark's findings.
- 26 layers × ~7,500 statements on CPU fits comfortably in 5 hours.
- If Gemma's HuggingFace license gate blocks you (it needs a one-click accept), fall back to `Qwen/Qwen2.5-1.5B` (ungated) — same code, no other change.

---

## What You Already Have

Four real dataset CSVs are in `got_datasets/` (provided alongside this guide):

| File                          | Statements | Character                                                        |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| `cities.csv`                  | 1,496      | Clean, curated ("The city of X is in country Y") — expect linear |
| `sp_en_trans.csv`             | 354        | Spanish-English translation — clean, different domain            |
| `companies_true_false.csv`    | 1,199      | Company business descriptions — messier                          |
| `common_claim_true_false.csv` | 4,450      | General world claims — messiest, most likely non-linear          |

The **contrast between clean (cities) and messy (common_claim)** is the actual insight: clean concepts should be linearly separable (small XGBoost gap), messy ones may show non-linear headroom (larger gap).

Format: every CSV has a `statement` column (text) and a `label` column (1=true, 0=false). Some have extra metadata columns you ignore.

---

## Setup (the 20-minute part)

Paste this to Claude Code first:

> Set up a Python project for an LLM interpretability experiment. Create a virtualenv and install: CPU-only torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `transformers accelerate pandas numpy scikit-learn xgboost matplotlib`. The dataset CSVs are already in `./got_datasets/`. Confirm the install works by importing all packages and printing their versions.

---

## PROMPT 1 — Activation Extraction Script

> Create `extract.py`. It loads `google/gemma-2-2b` on CPU in float32 with `output_hidden_states=True`, reads a dataset CSV from `got_datasets/`, and for each statement extracts the residual-stream activation at the **last token** (the period at the end of the statement) at **every layer**. Save to `acts_<dataset>.npz` containing: `activations` of shape `(num_layers+1, num_statements, hidden_dim)`, `labels` (int array), `statements` (object array), and `model` (string).
>
> Requirements:
>
> - CPU only — float32, no `.to("cuda")`, no device_map.
> - Batch the forward passes (batch size 16) with `padding=True`; set `tokenizer.pad_token = tokenizer.eos_token` if pad token is None; for each row take the activation at the **last non-pad token** (use the attention mask to find its index), not just position -1.
> - `truncation=True, max_length=64` (statements are short).
> - Take dataset filename as a command-line argument: `python extract.py cities.csv`.
> - Print progress every batch and the final tensor shape.
> - If Gemma fails to load due to gating, print a clear message telling me to either accept the license at huggingface.co/google/gemma-2-2b or switch MODEL_NAME to `Qwen/Qwen2.5-1.5B`.

After it's written, run a 20-statement smoke test before the full run:

> Add a `--limit N` flag to extract.py that only processes the first N statements. Run `python extract.py cities.csv --limit 20` to confirm the pipeline works end-to-end and prints a sensible activation shape (should be roughly (27, 20, 2304) for Gemma-2-2B). Then remove the limit for the real run.

---

## PROMPT 2 — Launch the Full Extraction (the 5-hour part)

> Write a shell script `run_all_extract.sh` that runs `extract.py` on all four datasets in sequence, logging output to `extract.log`:
>
> ```
> python extract.py cities.csv
> python extract.py sp_en_trans.csv
> python extract.py companies_true_false.csv
> python extract.py common_claim_true_false.csv
> ```
>
> Make it resilient: if one dataset fails, continue to the next. Print a timestamp before each. I'll launch this with `nohup bash run_all_extract.sh > extract.log 2>&1 &` so it survives my terminal closing.

**Order matters:** smallest first (sp_en_trans, cities) so you get results to analyze quickly even if the big one (common_claim) is still running. Reorder if you like — put `sp_en_trans.csv` and `cities.csv` first.

---

## PROMPT 3 — Analysis Script (run after each .npz appears)

> Create `analyze.py`. It loads `acts_<dataset>.npz` and, for every layer, does:
>
> 1. **Train/test split** (80/20, `random_state=42`, stratified on label).
> 2. **Linear probe:** standardize features (StandardScaler fit on train), train LogisticRegression(max_iter=2000), record test accuracy.
> 3. **XGBoost:** XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), no scaling, record test accuracy.
> 4. **Direction comparison (on full data):** compute the contrastive mean-difference direction `v_mean = unit(mean(act[label==1]) - mean(act[label==0]))` and the logistic-regression gradient direction `v_grad = unit(coef / scaler.scale_)`. Record their cosine similarity.
>
> Save `results_<dataset>.csv` (columns: layer, linear*acc, xgb_acc, gap) and `directions*<dataset>.csv`(columns: layer, cos_mean_vs_grad). Print a per-layer table and a summary: best layer, max gap across layers.
Take dataset name (no .csv) as argument:`python analyze.py cities`.

---

## PROMPT 4 — Plots

> Add to `analyze.py` (or a separate `plots.py`) three matplotlib figures per dataset, saved as PNGs with `matplotlib.use("Agg")`:
>
> 1. `plot_<ds>_accuracy.png` — linear_acc and xgb_acc vs layer, two lines, legend.
> 2. `plot_<ds>_gap.png` — bar chart of gap (xgb − linear) vs layer, horizontal line at 0.
> 3. `plot_<ds>_directions.png` — cos_mean_vs_grad vs layer, dashed line at 1.0 labeled "identical", y-axis from -1 to 1.05.
>    Title each with the dataset name.

---

## PROMPT 5 — Cross-Dataset Summary (the money plot)

> Create `summary.py` that reads all `results_<dataset>.csv` files and produces:
>
> 1. A single bar chart `plot_summary_maxgap.png` showing the **maximum non-linear gap** (best layer's xgb_acc − linear_acc) for each dataset, so I can see at a glance which concepts have non-linear headroom.
> 2. A printed table: dataset, best-layer linear acc, best-layer xgb acc, max gap.
> 3. A 4-sentence auto-generated summary stating, for each dataset, whether truth appears linearly encoded (gap < 0.02) or shows non-linear structure (gap ≥ 0.02).

This summary is what you actually present. The hypothesis to look for: **cities/sp_en_trans show ~0 gap (linear), common_claim shows a larger gap (non-linear headroom).**

---

## Execution Timeline

| Time         | Action                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------- |
| 0:00–0:05    | Setup prompt, install packages                                                           |
| 0:05–0:15    | PROMPT 1, write extract.py, run `--limit 20` smoke test                                  |
| 0:15–0:20    | PROMPT 2, launch `nohup bash run_all_extract.sh ...` and walk away                       |
| (during run) | As each `acts_*.npz` appears, run PROMPT 3+4 on it — you don't have to wait for all four |
| (after run)  | PROMPT 5 for the cross-dataset summary                                                   |

The extraction is the only slow part. The analysis on each `.npz` takes seconds-to-minutes on CPU, so you can analyze `cities` while `common_claim` is still extracting.

---

## What You Walk Into the Meeting With

- **Three plots per dataset** + one cross-dataset summary plot.
- **The headline finding:** "On clean truth statements (cities, translations), linear and non-linear probes are equivalent — consistent with Marks & Tegmark. On messier claims (common_claim), XGBoost opens a gap of [X] points, suggesting non-linear structure where concepts are entangled."
- **The direction finding:** "mean-diff and classifier-gradient directions agree with cosine [Y] on clean data."
- **The caveat that shows maturity:** "Cosine agreement isn't the real test — the non-identifiability work shows different directions can steer equivalently, so the next step is behavioral evaluation through A-LQR."
- **Open questions for Julian:** which concepts to target next (toxicity? sycophancy?), and access to the A-LQR code for behavioral eval.

---

## Failure Modes & Fixes

| Symptom                            | Fix                                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| Gemma won't load (gated)           | Accept license at huggingface.co/google/gemma-2-2b, or set `MODEL_NAME="Qwen/Qwen2.5-1.5B"`    |
| Runs out of RAM                    | Lower batch size to 8 or 4; or use `Qwen/Qwen2.5-0.5B`                                         |
| XGBoost acc = 1.000 everywhere     | Likely train/test leak — verify the split happens before fitting                               |
| Logistic regression won't converge | Already at max_iter=2000; confirm StandardScaler is applied                                    |
| All cosines ≈ 1.0                  | Expected for clean linear concepts — that's the finding, not a bug                             |
| Extraction too slow                | It's fine to let it run overnight; or process only cities + common_claim for the core contrast |

---

## Minimum Viable Result (if everything goes sideways)

Run extraction + analysis on **just `cities.csv` and `common_claim_true_false.csv`**. The single comparison — "clean concept is linear, messy concept has a non-linear gap of X" — is the entire core finding. Everything else is supporting evidence. Don't let a stalled run on the big dataset stop you from presenting the cities result.
