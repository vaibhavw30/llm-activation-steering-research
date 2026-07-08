# LLM Activation Steering Research — Truth: Geometry vs. Causality in gemma-2-2b

**One-line finding:** the "truth direction" in gemma-2-2b is **easy to read but hard to push** —
a linear probe decodes it at ~99%, yet it is *not* one of the model's dominant causal directions
(DCT never surfaces it, and steering it only weakly changes truthfulness). *Decodable ≠ causally
dominant.*

## Start here
- **Continuing this work (esp. on a new machine):** [`docs/PROJECT_CONTEXT_AND_ROADMAP.md`](docs/PROJECT_CONTEXT_AND_ROADMAP.md) — self-contained onboarding + exact resume state + research directions. **Read this first if you are picking the project up.**
- **Explaining the results (to yourself or a PI):** [`docs/PI_MEETING_RESULTS.md`](docs/PI_MEETING_RESULTS.md) — concise, with an evidence table of real generated text.
- **Full detail on the DCT-vs-truth funnel:** [`docs/FUNNEL_RESULTS.md`](docs/FUNNEL_RESULTS.md)
- **The whole project, soup-to-nuts:** [`docs/MASTER_EXPLAINER.md`](docs/MASTER_EXPLAINER.md) *(personal, gitignored)*
- **How DCT works + how it was set up:** [`docs/DCT_METHODOLOGY.md`](docs/DCT_METHODOLOGY.md)

## Repository layout

```
├── README.md                 ← you are here
├── CLAUDE.md                 ← original Project-1 build guide (geometry of truth)
├── src/                      ← ALL code (run scripts as `python src/<name>.py` from repo root)
│   ├── extract.py analyze.py summary.py            (Project 1: geometry of truth)
│   ├── dct.py dct_train.py sae_comparison.py       (the DCT paper's code — third-party)
│   ├── run_dct_minimal.py run_dct_data.py          (DCT training)
│   ├── apply_dct_vector.py interpret_top10.py steer_supervised.py dct_steer_utils.py  (steering)
│   ├── funnel_utils.py compare_directions.py subspace_top_k.py cross_dataset.py       (analysis)
│   ├── export_truth_dir.py                          (make truth dirs for the cluster)
│   ├── viz_funnel.py viz_findings.py viz_steer.py   (figures)
│   └── judge_results.py                             (LLM-as-a-judge scoring)
├── docs/                     ← all writeups (see "Start here")
├── deltaai/                  ← NCSA DeltaAI (GH200) launch scripts + runbooks
├── got_datasets/             ← input CSVs (cities, sp_en_trans, companies, common_claim)
├── activations/              ← extracted activations acts_<ds>.npz  (gitignored, large)
├── results/                  ← Project-1 outputs (probe CSVs + plots, committed)
└── (repo root)               ← DCT/funnel outputs land here, gitignored & regenerable:
                                 dct_V/U/meta_<ds>, truth_dir_<ds>.npz, interpret_top10_<ds>.md,
                                 steer_supervised_<ds>.{md,csv}, plot_*.png, compare_<ds>.csv
```

> **Why outputs sit at the repo root (not in a subfolder):** they're gitignored (so the tracked
> repo stays clean) and regenerable, and both the local scripts and the cluster `rsync` flow read/
> write them here. Relocating them would mean re-pathing ~10 scripts + the cluster runbooks — a
> change to a working pipeline for zero git-cleanliness gain. `git status` is already clean.

## How to run (all commands from the repo root)

**Project 1 — geometry of truth** (CPU, `.venv`):
```bash
.venv/bin/python src/extract.py cities.csv     # → activations/acts_cities.npz
.venv/bin/python src/analyze.py cities         # → results/ CSVs + plots
.venv/bin/python src/summary.py                # cross-dataset summary
```

**DCT + funnel** — training runs on the GH200 (see `deltaai/`), analysis runs locally:
```bash
# local analysis (needs dct_V_<ds>.pt pulled back from the cluster):
.venv/bin/python src/compare_directions.py --dataset cities
.venv/bin/python src/subspace_top_k.py --dataset cities
.venv/bin/python src/cross_dataset.py
.venv/bin/python src/viz_findings.py           # summary figures
# optional LLM-judge (needs ANTHROPIC_API_KEY):
.venv/bin/python src/judge_results.py --mode interpret --dataset cities
```

**On DeltaAI (GH200):** follow `deltaai/MY_RUN_STEPS.md` (first-time) or `deltaai/FUNNEL_RUN_STEPS.md`
(the funnel jobs). Job scripts: `deltaai/run_dct.slurm`, `run_interpret.slurm`, `run_steer.slurm`.

## Environments
- `.venv` — CPU, geometry + local analysis (torch 2.12, transformers 5.x, scikit-learn, anthropic).
- `.venv-dct` — local DCT (torch 2.6, **transformers 4.51.3** — required by the DCT paper's code).
- `.venv-dct-gpu` — on DeltaAI, built on the cluster's torch module (see `deltaai/setup_env.sh`).
