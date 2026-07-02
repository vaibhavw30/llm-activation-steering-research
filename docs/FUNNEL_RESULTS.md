# DCT-Interpretation Funnel — Results

*Companion to `DCT_VS_TRUTH_FINDINGS.md` (the original null) and `DCT_METHODOLOGY.md` (how DCT
works). This doc explains, in plain terms, how we figured out **what the null actually means**.*

---

## Bottom line (read this first)

We found that **the "truth direction" is real and readable, but it is not one of the model's
main causal levers.** Concretely:

- A supervised probe reads truth off gemma-2-2b's activations at ~99% accuracy (clean data).
- DCT — which finds the directions the model is *most causally sensitive to* — does **not** find
  truth among its top directions, individually or in combination.
- When we steer the supervised truth direction directly (both signs, −120…+120), it has only a
  **weak, noisy** effect — a few real falsehoods at strong "−" steering, a mild "+→more factual"
  nudge, but mostly just output degradation at high magnitude. Not a clean truth dial.

**One-line thesis (refined): being *readable* is not the same as being *causally dominant*.**
Truth is easy to read out and has a small causal effect, but it is not one of the biggest knobs
the model actually turns. In the PI's framing, this is **Ending A** (defined below), now supported
by five independent angles.

---

## The situation, and why we ran a "funnel"

Earlier we found a **null**: DCT's single best direction did not match our supervised truth
direction (cosine ≈ 1.2× a random baseline). A null like that is ambiguous — **"no match" has
three possible causes**, and we hadn't ruled any of them out:

| # | Possible explanation | Plain meaning |
|---|---|---|
| 1 | Truth genuinely isn't a top DCT direction | The interesting finding — truth isn't a big causal lever |
| 2 | The supervised direction isn't really causal | Our "truth direction" is just a correlate; DCT never had a real target to match |
| 3 | Truth is *spread across several* DCT directions | It's there, but a single-vector cosine can't see a combination |

The PI's key push: **we'd been asking "do DCT's vectors match truth?" but never "what ARE DCT's
vectors?"** The funnel is four diagnostics that answer that and separate #1/#2/#3.

---

## The four diagnostics (what each one asks)

| Step | Script | Question, in plain English |
|---|---|---|
| **1. Interpret** | `interpret_top10.py` | Steer with DCT's top-10 directions and *read the output* — what does each one actually do? Any about truth? |
| **2. Steer supervised** | `steer_supervised.py` | Inject our *supervised* truth direction and watch behavior — does it actually change how truthful the model is? |
| **3. Subspace** | `subspace_top_k.py` | Can you rebuild the truth direction by mixing DCT's top directions? (tests explanation #3) |
| **4. Cross-dataset** | `cross_dataset.py` | Does a truth-detector built from DCT directions still work on a *different* dataset? (is it a real, general truth carrier?) |

Steps 1–2 ran on the GH200 (they generate text). Steps 3–4 ran locally (just math on saved
vectors). All four point the same way.

---

## Step 1 — What DCT's top directions actually do

**How to read this:** for each top-10 vector we injected it during generation and compared the
steered vs. unsteered text. The "theme" is our label for what the vector changes.

**cities** (steered at scale 47.7) — the top directions are about:
- **Geography / places** (#447, #375, #62, #38): they inject cities, countries, coordinates. When
  they scramble entities you get *false geography* — e.g. "the capital of Japan is in Germany,"
  "Tokyo was founded by JFK," "the moon is a satellite of Jupiter."
- **Encyclopedic register** (#139, #381, #117): Wikipedia-style phrasing, often *more* accurate.
- **Format** (#173 → code, #114 → French + quiz style, #32 → statistics/numbers).

**common_claim** (steered at scale 86.7) — the top directions are about:
- **Tone / sentiment** (#355 → weird, dark word-salad; #69 → morbid/negative).
- **Format** (#441 → math-quiz/formal register).
- At this high magnitude, mostly **coherence breakdown** (the text turns to gibberish).

**What this means:** *none* of the top directions is a clean "truth" knob. They control **topic,
formatting, and tone** — not truthfulness. When false statements appear, it's a *side effect* of
the vector scrambling entities, not a controlled "make it lie" switch.

A nice sub-finding: cities' top directions are **all about geography** — which makes sense,
because DCT was trained on the cities data ("The city of X is in country Y"). So DCT surfaced
"talk-about-places" directions, i.e. **what DCT finds is shaped by the data you feed it** (a point
the PI specifically wondered about).

> **PI gate:** "If none of the top 10 are about truth, the null is legitimate." → None are. ✅

---

## Step 2 — Does our supervised truth direction actually steer? (± sweep)

**How to read this:** we injected the supervised truth direction (mean-diff and gradient
versions) at **nine magnitudes from −120 to +120** ("+" = toward truth, "−" = away from it) and
scored the 8 factual completions for correctness with a keyword heuristic. Chart:
`plot_findings_steering.png`.

**Result: a weak, noisy causal effect — not a clean truth dial.**
- **Real falsehoods do appear** at strong *negative* gradient steering on cities: "Two plus two
  equals **three**," "the sun rises in the **west**," "Tokyo, the largest city in **Canada**." So
  the direction *can* push the model into lying — evidence it's genuinely causal, not a dead correlate.
- **A mild "+ → more factual" trend** shows up on common_claim (gradient): "The capital of Japan
  is *a city…*" → "**Tokyo**, 13M people" as strength rises.
- **But the dominant pattern is magnitude, not sign.** Factual accuracy mostly just *degrades at
  large |steering|* in **either** direction (cities/gradient falls to ~chance at both −120 *and*
  +120), instead of cleanly dropping on "−" and rising on "+." The mean-diff version is flat/noisy.

**What this means:** steering the supervised direction *does* change truthfulness (so it's
**somewhat causal** — not just a readable correlate), but the effect is **weak and noisy** —
closer to "hard steering degrades the output" than "a knob that sets how truthful the model is."
So even the *supervised* truth direction is only a **minor** causal lever. That's exactly why DCT
— whose whole job is to find the model's *biggest* levers — never surfaces it.

---

## Step 3 — Is truth a *combination* of DCT's top directions?

This tests explanation #3: maybe truth is spread across several DCT vectors, so no single one
matches, but together they capture it.

**(A) "Truth-in-span": can you rebuild the truth arrow from the top-k DCT arrows?** We measure how
much of the truth direction lies inside the space spanned by the top-k DCT vectors, vs. what you'd
expect by pure chance (chance ≈ k / 2304).

| Dataset | k=10 (chance 0.004) | k=20 (0.009) | k=50 (0.022) |
|---|---|---|---|
| cities @ L11 (mean-diff / grad) | 0.008 / 0.003 | 0.012 / 0.009 | 0.027 / 0.025 |
| common_claim @ L13 | 0.017 / 0.002 | 0.025 / 0.004 | 0.043 / 0.016 |

Every number is essentially at the chance line → **truth is not reconstructable from the top DCT
directions.** Explanation #3 is ruled out.

**(B) "Classify from DCT features": can DCT's directions predict true/false?** We project the
activations onto the top-k DCT directions and train a simple classifier on just those numbers,
comparing against **random** directions (and the full-probe ceiling).

| Dataset | k=10 DCT / random | k=50 DCT / random | full-probe ceiling |
|---|---|---|---|
| cities @ L11 | 0.944 / 0.967 | 0.991 / 0.993 | 0.994 |
| common_claim @ L13 | 0.669 / 0.633 | 0.718 / 0.690 | 0.722 |

**What this means:** DCT's directions predict truth **no better than random directions** (cities:
actually a touch worse; common_claim: a hair better). Everything scores decently only because
truth is so linearly readable that *any* projection preserves it — the fair comparison is
DCT-vs-random, and DCT carries **no special truth signal.**

---

## Step 4 — Do DCT directions generalize as a truth carrier?

A real "truth direction" should detect truth on data it wasn't built from. We test transfer: build
a detector on dataset A, test it on dataset B.

**(A) Supervised probe transfer** (train row → test column):

| train ＼ test | cities | common_claim |
|---|---|---|
| cities | 1.000 | **0.602** |
| common_claim | **0.761** | 1.000 |

**(B) DCT-directions-as-detector transfer** (top-20 directions from the train row):

| train ＼ test | cities | common_claim |
|---|---|---|
| cities | 0.972 | **0.500** |
| common_claim | **0.502** | 0.701 |

**What this means:** the *supervised* truth direction transfers across datasets (0.60–0.76 — well
above the 0.50 chance line). DCT directions transfer at **chance (0.50)** — they don't carry a
general notion of truth, only whatever was specific to the data they came from. The supervised
direction is the more robust, general truth signal.

**Bonus (single-direction view, `viz_funnel.py`):** the single best DCT direction classifies truth
at ~the random-direction level (cities 0.72 vs random 0.70; common_claim 0.61 vs random 0.55), far
below the supervised direction (0.97 / 0.74). Plots: `plot_funnel_cosine_<ds>.png`,
`plot_funnel_dctclass_<ds>.png`.

---

## Putting it together — the verdict (Ending A)

| Question | Answer | From |
|---|---|---|
| Are DCT's top directions about truth? | No — topic / format / tone | Step 1 |
| Does the supervised truth direction steer behavior? | Yes but weakly/noisily — some falsehoods at extreme "−", mild "+→factual"; mostly magnitude-degradation | Step 2 (± sweep) |
| Is truth a *combination* of top DCT directions? | No (≈ chance) | Step 3 |
| Do DCT directions generalize as a truth detector? | No (≈ chance) | Step 4 |

The PI framed three possible endings; we landed cleanly on the first:

- **A — Truth is causally real but not a dominant DCT lever.** ✅ *(our result)*
- B — Methodology was off (truth hidden in a DCT combination, or a DCT vector secretly steers
  truth). ❌ ruled out by Steps 1 & 3.
- C — The supervised direction is a dead, non-causal correlate. ❌ ruled out by Step 2 (it steers).

**Refined thesis:** *decodable ≠ causally **dominant**.* Truth is easy to read (99%) and has a
mild causal effect when steered, but it is **not** one of the model's biggest causal directions at
these layers — DCT's strongest levers are topic, format, and tone instead.

---

## Expectations vs. reality

The gap between what we expected and what we found *is* the result:

| We expected | We found |
|---|---|
| DCT (unsupervised + causal) would **rediscover the truth direction**, confirming truth is a primary causal axis. | DCT's top directions are about **topic / format / tone**; truth is absent from them — as a single vector, as a combination, and as a detector that generalizes. |
| If the truth direction is causal, **steering it should cleanly flip truth↔false** (a dial: "−" → lies, "+" → more truthful). | Steering has only a **weak, noisy** effect — a few real falsehoods at extreme "−", a mild "+→factual" nudge, but mostly just **degradation at high magnitude**. |
| Clean data (cities) would give a **crisp** causal truth lever; messy data (common_claim) a diffuse one. | **Both behave the same:** truth is readable everywhere but never a dominant causal lever — the *geometry* (probe) story and the *causal* (DCT + steering) story simply diverge. |

**The one-sentence takeaway: truth is easy to read but hard to push** — highly decodable, only
weakly causal, and not among the model's dominant causal directions. That divergence between
"readable" and "causally dominant" is the finding.

---

## Caveats & honest next checks

- **Steering signal is weak and the score is coarse.** The ± sweep (both signs, 9 magnitudes) is
  done, but the factual-accuracy score is a keyword heuristic over just 8 prompts (1/8 granularity),
  so small trends are noisy. **An LLM-judge scorer is now built** (`judge_results.py`) — running it
  (`--mode steer`) re-scores every completion TRUE/FALSE/INCOHERENT, separating "made it lie" from
  "just degraded it"; `--mode interpret` auto-labels the top-10 vectors + whether each manipulates
  truthfulness (replacing the hand-read). Needs `ANTHROPIC_API_KEY`; outputs `judge_*.csv` +
  `plot_judge_steering_<ds>.png`.
- **Scale confound:** at large |steering| the output degrades/incoheres, which muddies "is it
  false vs. just broken?" — the effect is partly magnitude-degradation, not purely truth.
- **Only 512 factors / one target layer.** A truth lever might appear with `--num-factors 1024` or
  a deeper target layer — worth a check.
- **This is all geometry + qualitative steering.** The definitive test of a causal truth direction
  is a proper **behavioral evaluation (A-LQR)**, not cosines.

---

## Figures (generated by `viz_findings.py` + `viz_funnel.py`)

| File | Shows |
|---|---|
| `plot_findings_decode_vs_causal.png` | **The thesis in one chart** — probe reads truth at ~99% (cities) / 72% (common_claim), but the best DCT direction ≈ a random direction. Decodable ≠ causally dominant. |
| `plot_findings_alignment.png` | Best-of-512 DCT vector vs the truth direction, against a random baseline — the null. |
| `plot_findings_subspace.png` | Truth-in-span of the top-k DCT directions vs chance — truth isn't a combination either. |
| `plot_findings_transfer.png` | Cross-dataset transfer: supervised directions generalize (off-diagonal 0.60/0.76); DCT directions don't (0.50 ≈ chance). |
| `plot_findings_steering.png` | Step 2 ± sweep: factual accuracy vs steering magnitude — weak/noisy, mostly magnitude-degradation (not a clean truth dial). |
| `plot_funnel_cosine_<ds>.png` | Heatmap: each top-10 DCT vector's \|cos\| with mean-diff / gradient. |
| `plot_funnel_dctclass_<ds>.png` | Per-vector single-direction truth accuracy vs supervised & random references. |

Regenerate: `.venv/bin/python src/viz_findings.py`, `viz_steer.py`, and `viz_funnel.py --dataset <ds>`.

---

## Reproducibility

- **Local (geometry `.venv`):** `subspace_top_k.py`, `cross_dataset.py`, `viz_funnel.py`,
  `export_truth_dir.py` (shared `funnel_utils.py`). Read `activations/acts_<ds>.npz` +
  `dct_V/U_<ds>.pt`; analyze at the truth-peak layer (cities 11, common_claim 13).
- **Cluster (`.venv-dct-gpu`, GH200):** `interpret_top10.py`, `steer_supervised.py` (shared
  `dct_steer_utils.py`); launchers `deltaai/run_interpret.slurm`, `run_steer.slurm`
  (`--time=00:30:00`). Top DCT vectors ranked by ‖U_i‖ (downstream-effect magnitude).
- All funnel outputs (`interpret_top10_*.md`, `steer_supervised_*`, `truth_dir_*.npz`,
  `plot_funnel_*.png`) are gitignored — regenerate from the scripts.
