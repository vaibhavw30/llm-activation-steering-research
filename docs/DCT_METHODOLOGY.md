# DCT — Complete Guide (Methodology + Setup + Running It)

*Everything in one place: how "Deep Causal Transcoding" (the MELBO/DCT paper) works, the ML
concepts behind it, the environments and scripts, the debugging journey, the cluster work,
and how to talk about all of it with your PI. Companion to the geometry-of-truth docs
(`EXPLAINER.md`, `MEETING_SUMMARY.md`, `DIRECTIONS_FINDINGS.md`) and the cluster runbook
(`deltaai/GPU_SETUP.md`, `deltaai/MY_RUN_STEPS.md`).*

**How this doc is organized:**
- **Part I — The science (§0–§11):** what DCT is, the ML concepts, how we test it, how to
  explain it.
- **Part II — The practice (§12–§17):** the three environments, the scripts and exact
  commands, the debugging journey, the cluster, and what to do right now.
- **§18 — Glossary.**

---

# PART I — THE SCIENCE

---

## 0. Where this fits in your project

You have **two methods** for finding "directions" inside gemma-2-2b's activations:

| | Geometry-of-truth (docs 1–3) | DCT (this doc) |
|---|---|---|
| Uses labels? | **Yes** (supervised) | **No** (unsupervised) |
| How it finds a direction | Train a probe on true/false labels | Search for directions that *causally change behavior* |
| What it gives you | "the truth direction" (one per layer) | hundreds of *steering* directions |
| Evidence type | Correlational (this direction predicts truth) | Causal (this direction *changes* the model) |

**The whole point of bringing in DCT:** if an *unsupervised, causal* method (DCT) lands on
the *same* direction your *supervised* probe found, that's powerful corroboration that the
truth direction is real — not an artifact of your labels or your classifier. `compare_directions.py`
(Stage 4) measures exactly this.

---

## 1. The 60-second version (say this to your PI)

> "DCT is an unsupervised method that searches a model's activation space for directions
> that, when added to an early layer, cause the biggest possible change at a later layer —
> i.e. directions the model is *causally sensitive* to. It finds hundreds of them at once
> from as little as a single prompt, and auto-calibrates how hard to push so the output
> changes without turning to gibberish. We're running their exact implementation on
> gemma-2-2b, then checking whether any of these label-free 'steering' directions line up
> with the supervised 'truth' direction we found earlier. Agreement would be causal
> evidence for the truth direction; we benchmark the alignment against a random baseline."

Everything below unpacks that paragraph.

---

## 2. Background concepts (the foundation)

### 2.1 Activations and the "residual stream"
A transformer processes text through a stack of layers (gemma-2-2b has 26). At each layer,
every token has a **hidden state** — a list of 2,304 numbers (its "activation"). Think of
this as the model's internal scratchpad. The running scratchpad that flows from layer to
layer is called the **residual stream**. Earlier layers hold surface features (word
identity, syntax); later layers hold abstract ones (is this claim true? is this toxic?).

### 2.2 What a "direction" is, and "steering"
That list of 2,304 numbers is a point in a 2,304-dimensional space. A **direction** is just
an arrow in that space (a unit vector). **Steering** = adding a chosen direction to the
hidden state mid-computation, nudging the model's "thought" along that axis. If you add a
"French" direction, the model starts answering in French; add a "refusal" direction and it
refuses. The research dream is to find directions that correspond to *meaningful, controllable*
concepts.

### 2.3 Supervised vs unsupervised — the key contrast
- **Supervised** (your probes): you *tell* the model what to look for via labels, then read
  out the direction that separates the classes. Risk: you only ever find what you labeled,
  and a probe can pick up on a *correlate* of truth rather than truth itself.
- **Unsupervised** (DCT): you give it *no labels* and ask "what directions does this model
  actually respond to?" It discovers an inventory of causal directions; *afterward* you check
  which (if any) correspond to concepts you care about. Less biased, but you have to
  interpret what it found.

---

## 3. DCT's core question and the "slice" trick

DCT asks one question:

> **"What small perturbation, added at an early (source) layer, produces the largest change
> at a later (target) layer?"**

Why source→target instead of the whole model? Because the interesting transformations happen
in the *middle* of the network. DCT isolates a chunk — say layers 6→18 — and studies it as a
standalone function.

**`SlicedModel` (in `dct.py`)** implements this: you hand it a hidden state from layer 6, and
it runs *only* layers 6 through 18 and returns the layer-18 hidden state. It literally
mutates the model to skip the other layers, then restores it. So you can treat
"layers 6→18" as a math function `f(x)` you can poke at.

> **Analogy:** instead of studying an entire factory, you isolate one assembly line
> (stations 6–18), feed it a part, and watch what comes out the other end.

---

## 4. The steering objective — what DCT maximizes

**`DeltaActivations` (in `dct.py`)** measures the *effect* of a steering direction `θ`:

```
Δ(θ) = f(x + θ) − f(x)
```

In words: "take the normal layer-6 state `x`, add the steering vector `θ`, run it through the
slice, and see how much the layer-18 output *moved* compared to no steering." A big `‖Δ‖`
means `θ` is a direction the model is very sensitive to — pushing on it has large downstream
consequences.

DCT's goal: **find the `θ`s that make `Δ` as large as possible** (subject to `θ` having a
fixed size — otherwise you'd just make `θ` huge). Those are the model's "levers."

---

## 5. Finding many directions at once — `ExponentialDCT`

Finding one lever is easy; DCT finds **hundreds of distinct levers simultaneously**. Three
ideas make this work:

### 5.1 Forward-mode autodiff (the `jvp` — and why our cluster bug happened)
To know which direction increases `Δ` fastest, DCT needs the **derivative** of the slice
function — specifically a **Jacobian-vector product (`jvp`)**: "if I nudge the input in
direction `v`, which way does the output move, and how fast?"

There are two flavors of autodiff:
- **Backprop (reverse-mode)** — what training normally uses ("output → which inputs caused it").
- **Forward-mode (`jvp`)** — "input nudge → resulting output nudge." DCT uses this because it's
  the natural fit for "perturb the input, measure the output change."

> **This is exactly the cluster bug we hit.** The fast fused attention kernels
> (`scaled_dot_product_attention`, "flash attention") implement backprop but *not* forward-mode
> autodiff — so `jvp` through them throws `NotImplementedError`. The fix was loading the model
> with `attn_implementation="eager"` (plain, un-fused attention math), which *does* support
> forward-mode. That's why our scripts force eager attention. **PI-ready phrasing:** "DCT's
> calibration uses forward-mode autodiff, which the fused attention kernels don't support, so
> we run eager attention."

### 5.2 Why "Exponential"
The objective doesn't just sum up the effects — it weights them through an exponential, which
*emphasizes the largest effects*. This pushes the optimizer toward a few strong, clean
directions rather than many weak, mushy ones.

### 5.3 Orthogonalization — keeping the directions diverse
If you just optimized for "biggest effect" repeatedly, you'd find the same dominant direction
over and over. DCT **orthogonalizes** the set (via QR decomposition / a "soft" variant) so
each new direction is *different* from the others — covering many distinct behaviors instead
of redundant copies.

> **Analogy:** you don't want 100 flashlights all pointing the same way; you want them
> spread out to light up the whole room. Orthogonalization spreads the directions out.

### 5.4 What you get: `V` and `U`
`ExponentialDCT.fit(...)` returns two matrices:
- **`V`** (2304 × num_factors): the **input steering vectors** — the directions you *add* at
  the source layer. **These are the discovered "levers," and what we compare to your truth
  direction.**
- **`U`** (2304 × num_factors): the **output effects** — what each lever *does* downstream.

(In our smoke run, num_factors = 64, so V is 2304 × 64 = 64 candidate levers.)

---

## 6. Calibration — the "Goldilocks" push strength

How hard should you push along `θ`? Too gentle → output barely changes (you learn nothing).
Too hard → output becomes gibberish (you've shoved the model off the rails). You need the
*just right* magnitude.

**`SteeringCalibrator`** automates this. It tries random directions, measures how the actual
effect compares to the *linear prediction* of the effect (the ratio of "real change" to
"derivative-predicted change"), and solves for the push size `R` where that ratio hits a
target (0.5 by default) — i.e. the point where the slice is starting to behave non-linearly
but hasn't broken. In our run it picked `input_scale ≈ 25.7`.

> **PI phrasing:** "They auto-calibrate the steering norm so perturbations are large enough to
> matter but small enough to stay in the model's normal operating regime — it solves the
> over-steering problem instead of hand-tuning a magnitude."

---

## 7. Why DCT is a notable method (the "so what")

1. **Unsupervised + causal.** Most interpretability either needs labels (probes) or is only
   correlational (SAEs, see §9). DCT finds directions defined by their *causal* effect, with
   no labels.
2. **Data-efficient.** It can extract a rich set of directions from a *single prompt* — because
   it's analyzing the model's internal sensitivity, not learning statistics from a big dataset.
3. **A behavior inventory.** One run yields hundreds of steering directions; you then interpret
   which correspond to language, refusal, sentiment, *truth*, etc.

---

## 8. The non-identifiability caveat (ties back to your geometry docs)

Your `MEETING_SUMMARY.md` and `DIRECTIONS_FINDINGS.md` raised it, and it's central here:
**different directions can produce equivalent behavior.** Two methods agreeing on a vector
(high cosine) is *suggestive* but not proof that it's "the" causal truth direction — and even
strong steering doesn't prove uniqueness. This is *why* combining methods matters: a supervised
probe direction (correlational) that *also* shows up as a DCT direction (causal) is much
harder to dismiss as an artifact than either alone. It's also why the honest next step is
**behavioral evaluation**, not just more cosines.

---

## 9. How we actually test it — Stage 4 (`compare_directions.py`)

At the DCT **source layer**, we put four kinds of directions in the same space and measure
cosine similarity (1 = identical axis, 0 = unrelated):

- **DCT vectors** `V` (unsupervised, causal)
- **mean-difference direction** (supervised: average true-activation minus average false)
- **logistic-gradient direction** (supervised: what your linear probe uses)
- **(optional) Gemma-Scope SAE features** — a *third*, independent unsupervised method
  (Sparse Autoencoders decompose activations into many interpretable features). Reusing the
  paper's `sae_comparison.py` approach lets us ask "do DCT, SAEs, and probes converge?"

**The random baseline is the crux.** In 2,304 dimensions, two random vectors have cosine
≈ 0 with tiny spread (std ≈ 1/√2304 ≈ 0.02); the *max* over 64 random vectors is ≈ 0.06. So:
- best DCT alignment ≈ 0.06 → **no better than chance** (what we saw with the *smoke* vectors,
  which came from an unrelated prompt — a clean negative control ✅).
- best DCT alignment ≫ 0.06 (say 0.3, "5× random") → **DCT genuinely rediscovered the truth axis**.

The script prints the ratio "× random" and a verdict, so you're never fooled by a number that
*looks* big but isn't.

---

## 10. Explaining it to your PI — talking points & likely questions

**Lead with the thesis:** "Supervised probes told us *where* truth lives; DCT tests whether
that direction is *causal* and *discoverable without labels*."

**Anticipated PI questions and crisp answers:**

- *"Why DCT and not just steer along the probe direction?"* — The probe direction is
  correlational and label-dependent. DCT is label-free and defined by causal effect; agreement
  between the two is the strong claim.
- *"What layer are you comparing at?"* — The DCT source layer (e.g. 6). DCT's input vectors
  live there, so we compute the probe directions at the *same* residual-stream layer for an
  apples-to-apples cosine.
- *"How do you know an alignment is real?"* — We benchmark against a random-vector baseline in
  the same dimension; we report the ratio to random, not the raw cosine.
- *"Why does the sign of the cosine not matter?"* — A steering vector and its negative are the
  same axis (one pushes toward true, the other toward false), so we report |cos| for *axis*
  alignment and keep the sign to see direction.
- *"Single prompt — isn't that too little data?"* — For *Stage 1* yes, it's just a plumbing
  check. For the real comparison we run DCT on batches of true/false statements (Stage 3,
  `run_dct_data.py`), mirroring the paper's combined-observations training.
- *"What would falsify the hypothesis?"* — If, on clean `cities` truth, *no* DCT vector beats
  the random baseline, then truth isn't a direction the model is causally organized around at
  that layer (or DCT's hyperparameters/layer are wrong).

**Things to proactively flag (shows maturity):** the non-identifiability caveat (§8); that
cosine is geometry, not behavior; and that the real validation is behavioral steering eval
(A-LQR), not alignment numbers alone.

---

## 11. How the methodology shaped the cluster work

- **Forward-mode autodiff → eager attention.** (§5.1) The calibrator's `jvp` is unsupported by
  fused attention kernels → we force `attn_implementation="eager"`.
- **Compute cost.** The `jvp` + multi-direction optimization through a deep slice is heavy:
  ~90 min for the tiny smoke run on CPU; paper-scale (512–1024 directions, 30 iters, many
  statements) would take *weeks* on CPU → hence the **DeltaAI GH200** pivot.
- **Version pin.** `SlicedModel` mutates transformers internals, so we pin `transformers==4.51.3`
  (newer versions reorganized those internals and break the slice). See `deltaai/GPU_SETUP.md`.

---

# PART II — THE PRACTICE

## 12. The three Python environments (and why there are three)

DCT and your geometry-of-truth work need **different, conflicting** library versions, so we
keep them in separate virtual environments. Never mix them.

| Env | Where | Key versions | Used for |
|---|---|---|---|
| **`.venv`** | laptop | py3.13, torch 2.12, **transformers 5.x**, scikit-learn | Geometry-of-truth (extract/analyze) **and Stage 4** (`compare_directions.py`) |
| **`.venv-dct`** | laptop | py3.13, torch 2.6, **transformers 4.51.3** | CPU DCT testing (Stages 1–3 locally, slow) |
| **`.venv-dct-gpu`** | DeltaAI | module torch (ARM+CUDA), **transformers 4.51.3** | The real GPU DCT runs |

**Why the split:** DCT's `SlicedModel` reaches into transformers' internals, which were
reorganized after 4.51.3 — so DCT **requires** `transformers==4.51.3`. But your geometry work
already runs on transformers 5.x. Rather than downgrade (and risk breaking the finished
geometry pipeline), we isolate DCT in its own env. Stage 4 is the one DCT step that runs in
`.venv` — because it only does cosine math on saved vectors and needs scikit-learn (which
lives in `.venv`), and the activations are already on your laptop.

> **PI phrasing:** "DCT pins an older transformers because it monkeypatches model internals;
> we sandbox it so it can't disturb the geometry pipeline."

## 13. The scripts and exact commands (the four stages)

The paper's own code (don't edit): **`dct.py`** (the method), **`dct_train.py`** (their
driver, which we mirror), **`sae_comparison.py`** (their SAE comparison). Our thin wrappers
around it:

| Stage | Our script | What it does | Run it with |
|---|---|---|---|
| 1 | `run_dct_minimal.py` | Verify DCT end-to-end on one prompt → `dct_V.pt`/`dct_U.pt` | `.venv-dct/bin/python` (CPU) or `--device cuda` on cluster |
| 2 | `apply_dct_vector.py` | Inject a vector during generation; unsteered vs steered text | same env as Stage 1 |
| 3 | `run_dct_data.py` | DCT on your true/false statements → `dct_V_<ds>.pt` | **GPU** (`.venv-dct-gpu`) — the real run |
| 4 | `compare_directions.py` | Cosine of DCT vectors vs your truth direction + random baseline | **`.venv/bin/python`** (laptop) |

**Exact commands:**
```bash
# Stage 1 — verify (CPU local; ~90 min) or on GPU (~minutes)
.venv-dct/bin/python src/run_dct_minimal.py --num-factors 64 --max-iters 10            # CPU
python src/run_dct_minimal.py --device cuda --num-factors 64 --max-iters 10            # cluster

# Stage 2 — see it steer behavior
.venv-dct/bin/python src/apply_dct_vector.py --vectors 0,1,2,3 --scale 25.72

# Stage 3 — the real research run (GPU)
python src/run_dct_data.py --dataset cities --device cuda \
    --num-factors 512 --num-iters 30 --num-samples 64 --balanced

# Stage 4 — the payoff (laptop, after rsync-ing dct_V_<ds>.pt back)
.venv/bin/python src/compare_directions.py --dataset cities
```

Outputs: Stage 1 → `dct_V.pt`/`dct_U.pt`; Stage 3 → `dct_V_<ds>.pt`/`dct_U_<ds>.pt`/`dct_meta_<ds>.json`;
Stage 4 → printed table + `compare_<ds>.csv`. All are gitignored (regenerable).

## 14. The debugging journey (five real blockers — and the lesson in each)

Getting the paper's cluster-scale code to run on a laptop, then a GH200, surfaced five issues.
These are worth understanding — they're typical of running research code on new hardware.

1. **Wrong file paths.** The guide assumed `melbo-dct-paper/src/dct.py`; the files were flat in
   the repo root (`dct.py`, `dct_train.py`, `sae_comparison.py`). *Lesson: verify the actual
   tree before trusting a README.*
2. **transformers version.** `SlicedModel` mutates model internals → needed `transformers==4.51.3`.
   *Lesson: research code is often pinned to exact versions; honor the pins.*
3. **`dtype` vs `torch_dtype`.** transformers renamed the kwarg between versions; our loader now
   tries both. *Lesson: small API drift breaks old code in subtle ways.*
4. **A false "MISMATCH" alarm.** Our slice sanity check used `torch.allclose`'s default
   `atol=1e-8`, which is far too strict on near-zero activation values — it reported a mismatch
   even though the slice was *bit-for-bit faithful* (cosine 1.000000). We fixed the check to
   report cosine + max-diff. *Lesson: don't trust a pass/fail boolean — quantify the actual
   discrepancy before concluding something's broken.*
5. **Forward-mode autodiff vs fused attention** (the big one). The calibrator's `jvp` is
   unsupported by the fused `scaled_dot_product_attention` (flash) kernel → `NotImplementedError`.
   Fix: `attn_implementation="eager"`. *Lesson: §5.1 — autodiff mode and kernel choice interact.*

> All five are now baked into our scripts, so you won't re-hit them. This list is for *your
> understanding* (and a good "what went wrong and how I diagnosed it" story for your PI).

## 15. The cluster, in a nutshell (NCSA DeltaAI)

- **Hardware:** GH200 "Grace-Hopper" nodes — **ARM64 CPU** + Hopper GPU (~96 GB), 4 GPUs/node,
  partition **`ghx4`**. gemma-2-2b (~10 GB fp32) fits on **one** GPU — no multi-GPU needed.
- **Why GPU at all:** Stage 1 took ~90 min on CPU; Stage 3 at paper scale would take *weeks* on
  CPU but **tens of minutes to ~2 h** on one GH200.
- **ARM means** we *don't* pip-install torch (x86 wheels won't run) — we use DeltaAI's
  `module load python/miniforge3_pytorch` and layer transformers 4.51.3 on top
  (`deltaai/setup_env.sh`).
- **Login:** `ssh vwudaru@dtai-login.delta.ncsa.illinois.edu`, **NCSA password + NCSA Duo**
  (separate from GaTech). Jobs via **SLURM** (`sbatch deltaai/run_dct.slurm`).
- **Budget:** ~312 GPU-hours left on project **CIS260948**. Stage 1 ≈ 0.1 h; Stage 3 ≈ 1–2 h
  each. Keep `--time` tight so a hang can't drain it.
- Full step-by-step: **`deltaai/MY_RUN_STEPS.md`** (your personal runbook).

## 16. End-to-end: what to do right now

1. **Unblock the cluster login** — finish NCSA Duo enrollment; get a shell on DeltaAI. *(This is
   the only thing currently blocking you; everything code-side is ready.)*
2. **On the cluster:** `bash deltaai/setup_env.sh` → `hf download google/gemma-2-2b` → edit
   `--account` in `deltaai/run_dct.slurm` → `sbatch` it. Confirm Stage 1 prints
   `V shape (2304, 64)`.
3. **Run Stage 3** (uncomment the `run_dct_data.py` lines) for `cities` and
   `common_claim_true_false`.
4. **Bring back** `dct_V_*.pt` to your laptop (rsync).
5. **Run Stage 4 locally:** `.venv/bin/python src/compare_directions.py --dataset cities` — read the
   "× random" ratio and verdict. Repeat for the other datasets.
6. **Interpret with your PI** using §1, §9, §10.

## 17. File & folder map

```
dct.py, dct_train.py, sae_comparison.py   ← the paper's code (don't edit)
run_dct_minimal.py     Stage 1 (verify)
run_dct_data.py        Stage 3 (DCT on true/false data, GPU)
apply_dct_vector.py    Stage 2 (steer generation)
compare_directions.py  Stage 4 (DCT vs truth direction)  ← run in .venv
got_datasets/          true/false CSVs (shared with geometry work)
activations/           geometry-of-truth activations (gitignored, big)
deltaai/               setup_env.sh, run_dct.slurm, GPU_SETUP.md, MY_RUN_STEPS.md(personal)
docs/                  EXPLAINER, MEETING_SUMMARY, DIRECTIONS_FINDINGS (geometry) + this file
dct_V*.pt / dct_U*.pt / compare_*.csv   ← outputs (gitignored, regenerable)
```

---

## 18. Glossary (quick reference)

| Term | Plain meaning |
|---|---|
| **Activation / hidden state** | The 2,304 numbers representing a token at a given layer |
| **Residual stream** | The running hidden state passed layer-to-layer |
| **Direction / steering vector** | A unit arrow in activation space; adding it nudges behavior |
| **Source / target layer** | Where you inject the perturbation / where you measure its effect |
| **`SlicedModel`** | Runs only layers source→target as a standalone function |
| **`DeltaActivations`** | The downstream change `f(x+θ)−f(x)` caused by a steering vector |
| **`jvp` / forward-mode autodiff** | "Input nudge → output nudge" derivative; needs eager attention |
| **Orthogonalization** | Forcing the discovered directions to be diverse, not redundant |
| **Calibration / `input_scale` (R)** | The auto-tuned push strength (big enough to matter, not break) |
| **`V` / `U`** | Input steering vectors / their downstream output effects |
| **num_factors** | How many directions DCT finds at once (64 in our smoke run) |
| **Cosine similarity** | Alignment of two directions: 1=same, 0=unrelated |
| **Random baseline** | Max cosine you'd get by chance — the bar real alignment must clear |
| **SAE (Sparse Autoencoder)** | A separate unsupervised method that splits activations into interpretable features |
| **Non-identifiability** | Different directions can cause the same behavior → agreement matters, geometry isn't proof |
```
