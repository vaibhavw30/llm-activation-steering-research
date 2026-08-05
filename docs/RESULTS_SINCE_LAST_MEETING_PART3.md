# Part 3: everything since the last meeting doc

*Written 2026-07-30. Scope, repo review, experiment-by-experiment walkthrough, figure guide,
what is built but not yet run, and where this goes next.*

---

## 0. Which doc this continues, and what window it covers

The last meeting doc was the pair committed together in `e39ddc5` on **2026-07-22**:

- [`docs/PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md`](PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md) (the judge and pipeline half)
- [`docs/RESULTS_SINCE_LAST_MEETING_PART2.md`](RESULTS_SINCE_LAST_MEETING_PART2.md) (the results half)

Those were written for the **2026-07-23 meeting**. Part 2 has since grown a 155-line section
(still uncommitted in the working tree) recording the PI's feedback from that meeting. Everything
in this document is what happened from 2026-07-22 to 2026-07-30.

**The window, quantified:**

| Measure | Value |
|---|---|
| Commits since `e39ddc5` | **71** |
| Files touched | 125 |
| Lines added / removed | **+18,401 / −55** |
| Test suite | 182 → **298 passed, 1 skipped** (runs in 5.7 s) |
| New cluster jobs written | 9 SLURM scripts (`run_reach_*` ×7, `run_refusal_*` ×4, minus overlap) |
| New runbooks | `REACH_RUN.md`, `REACH_H0_RUN.md`, `REFUSAL_RUN.md` |
| New implementation plans | 4 (reachability audit, Horizon 0, horizons overarching, Horizon 1) |
| Repo totals now | 10,320 LOC in `src/`, 3,765 LOC in `tests/` (52 test files), 19,629 lines of docs |

The window contains **one abandoned branch of work** (the 2026-07-23 conditional-steering round,
built and never run) and **one complete new research programme** (backward reachability), plus its
audit, its validations, and half of the follow-up.

---

## 1. What the meeting changed

Two pieces of feedback. The first was small and is done. The second reframed the whole project.

### 1.1 "Make a better visualizer for the linearity cosine shift score"

Done. `src/viz_mag_linearity.py` produces `plot_mag_linearity_v2.png`: two panels across all four
datasets. Left panel is cos(direction, prefix-induced shift Δ^Q); v_Q sits at 0.84 to 0.98, while
the supervised truth axis and DCT's top lever sit inside an |cos| < 0.1 "orthogonal band". Right
panel is ε_Q per direction, with the truth axis and DCT lever **above** the ε_Q = 1 line
(1.28 to 1.40), meaning they explain the shift worse than predicting no shift at all.

The old figure showed only ε_Q for v_Q and buried the actual finding, which is the cosine
structure: the direction that carries a prompt's question-vs-statement framing is essentially
orthogonal to the truth direction.

### 1.2 The control-theory reframe (the load-bearing one)

The PI's proposal, decoded:

> Stop hunting for single steering **directions**. Define a **target set** in the output space,
> specifically in J-space (the Jacobian-lens space), for example "activations whose truth readout
> says FALSE while staying coherent". Then compute the **backward-reachable set** of that target
> at the source layer: the set of source-layer perturbations that land in the target. Because the
> map is locally linear, the tools are just linear algebra: preimages of halfspaces under a linear
> map, Jacobian transposes, SVD.

Why this is the right upgrade for us: every prior result in the project is a statement that some
**single direction** fails to be a truth lever. Reachability turns the null into a claim about
**sets**. Either the "assert a falsehood, stay coherent" region is not reachable from bounded
source-layer perturbations at all (which makes the null theorem-shaped rather than an absence of
evidence), or it is reachable but thin or off-axis (which explains why every single-direction
probe missed it, and hands us the steering set directly).

Four references came with it. Deep-read summaries are in Part 2 §F2; the load-bearing one is
**arXiv:2509.21528** (Karnik & Bansal, latent reachability with backward reachable tubes over
generation steps). Our case is structurally easier than theirs: one linearized 9-block hop through
depth instead of a recurrent nonlinear rollout through time, so closed-form set preimages replace
their learned value function. One reference, arXiv:1910.13272, does not match the described paper
and still needs confirming with the PI.

---

## 2. The reachability formalism: every object named

This is the piece to be able to say out loud, so here it is stripped down. The PI's ask was for one
place where every linear-algebra object between token input and token output is written down and
justified, rather than described in words. That is §2.1 through §2.7. The LaTeX source with the
proofs and the cone dual is [`docs/math_map.tex`](math_map.tex); this section is the same content
in the document that gets read, and it is the authority for the symbols used everywhere below.

### 2.1 The forward map, and what is linear in it

Width `d = 2304`, `L = 26` decoder blocks, vocabulary `V = 256,000`. Write `T` for the number of
positions in a prompt.

| Object | Shape | What it is | Where it comes from |
|---|---|---|---|
| `E` | `V × d` | token embedding, **tied**, so the same matrix is the unembedding | `model.get_input_embeddings()` |
| `h^(l)_t` | `d` | residual stream at layer `l`, position `t` | `output_hidden_states=True`, `hidden_states[l]` |
| `gamma` | `d` | gain of the final RMSNorm | `model.model.norm.weight` |
| `z_t` | `d` | post-norm activation, `z = (h^(L) / rms(h^(L))) * (1 + gamma)`, `rms(h) = ‖h‖/√d` | `hidden_states[-1]` |
| `u_t` | `V` | logits, `u = E z` | the tied head |
| `c = 30.0` | scalar | final logit softcapping, `ũ = c·tanh(u/c)` | `config.final_logit_softcapping` |

Only two of these steps are linear: the unembedding `u = E z`, and RMSNorm *once `‖h‖` is held
fixed*. Everything else (attention, MLP, the softcap, softmax) is not. Two facts keep the algebra
usable anyway. First, the softcap is strictly monotone, so it preserves the argmax and therefore
preserves every decision boundary we care about; it changes margins but not which token wins.
Second, the block stack is smooth, so a first-order model of it is testable rather than assumed,
and Phase 5 and Horizon 0.1 test it.

### 2.2 The two readouts, which are the same algebra with a different `w`

Everything downstream depends on one choice: the scalar we want to move.

**Probe readout (what the audit ran).** The supervised truth probe at the target layer gives a
direction `w ∈ R^d` and a threshold `t02` (the value at which the probe assigns `p = 0.2` to
"true"). The FALSE set is the halfspace `{u : w·u ≤ t02}`.

**Token readout (what the token-space program runs).** For statement `i`, let `j_top` be the token
the model actually emits and `j_tgt` the token we want emitted. Then

```
a_i = E[j_tgt] - E[j_top]  ∈ R^d ,      r_i(z) = a_i · z
```

and `r_i > 0` is *exactly* "the model emits `j_tgt` instead of `j_top`" at temperature 0. This is
the readout that cannot dissociate from behavior, because it is the decision. It is the same
certificate with `w := a_i` and `t := 0` (`src/token_jac.py:1-27`).

The full argmax cone is `C_j = {z : (E[j] - E[k])·z ≥ 0 for all k ≠ j}`, so it has 255,999 faces.
`a_i` is the single most-violated face. `delta_cone` is the least-norm perturbation satisfying all
of them, solved as a QP in `token_geom.cone_budget`; `delta_face` is the one-face relaxation.

### 2.3 The certificate, stated once

Fix a sign convention once, because mixing the two is a real bug we have hit: the **target set is
`{z : w·z ≥ t}`**, and a statement starts *outside* it. Fix an injection site `S` and let `A_S` be
the linear map from a perturbation at that site to the readout coordinates (§2.4). A perturbation
`δ` moves the readout by `w·A_S δ = (A_S^T w)·δ` to first order, so the perturbations that land in
the target set form a halfspace, because the preimage of a halfspace under a linear map is a
halfspace:

```
{ δ : (A_S^T w) · δ ≥ g }
```

with the three scalars the whole audit turns on:

- **`g = t - w·z`** is the *gap*: how far the unperturbed statement sits from the boundary, positive
  because it starts outside. For the token readout `t = 0` and `g = -a_i·z`, which is exactly the
  logit margin `M_i` of the incumbent token over the target token.
- **`m = ‖A_S^T w‖`** is the *controllability margin*: readout movement per unit of perturbation, in
  the best possible direction. `A_S^T w` is the **pullback vector**, and it is the only object that
  needs computing.
- **`ε* = g / m`** is the *budget*: the smallest perturbation norm that reaches the target set. The
  minimiser is `δ* = (g/m²)·A_S^T w`, which is the pullback vector itself, rescaled.

The probe instantiation in §3 through §5 flips the orientation, because there the target set is the
FALSE halfspace `{z : w·z ≤ t02}`, the `≤` side. Reading it against the convention above means
`w := -w_probe` and `t := -t02`, which is the same thing as leaving `w_probe` alone and writing
`g = w_probe·z - t02` and `{δ : (A_S^T w_probe)·δ ≤ -g}`. Both are correct; only mixing them is
not. `g` is positive in both.

**Why this is cheap.** `A_S^T w` is one vector-Jacobian product per statement (`torch.func.vjp`, or
`torch.autograd.grad` of the summed scalar in the batched implementations). No `2304 × 2304`
Jacobian is ever formed. That is what makes a per-statement certificate over hundreds of statements
affordable.

### 2.4 The three injection sites, and their `A_S`

The certificate is not one number, it is one number *per site*, and the sites differ by a large
factor. Measured medians, `n = 200` per dataset:

| Site | `A_S` | `A_S^T` | `m` median (cities) | `m` median (common_claim) |
|---|---|---|---|---|
| post-norm, add to `z` | `I` | `I` | 2.383 | 2.138 |
| pre-norm, add to `h^(L)` | `(√d/‖h‖)·diag(1+gamma)·P_perp` | `(√d/‖h‖)·P_perp·diag(1+gamma)` | 0.532 | 0.432 |
| layer `l`, add to `h^(l)` | the block-stack Jacobian `J_l` | one VJP | 0.607 (l=16) | 0.676 (l=8) |

The pre-norm map is written explicitly in `token_geom.rmsnorm_site` (`src/token_geom.py:132-157`)
with `P_perp = I - ĥĥ^T`. Two consequences fall straight out of it and are the reason this site is
treated separately at all: **`P_perp` annihilates the radial component of any perturbation**, so
any steering vector aligned with the activation itself is simply deleted; and what survives is
rescaled by `√d/‖h‖`, a contraction. The measured cost of moving from post-norm to pre-norm is the
`rmsnorm_penalty`, median **4.38 on cities** and **5.18 on common_claim**: the same behavioral
change costs about 4.4x to 5.2x more perturbation norm one step earlier in the network.

For the probe instantiation used in §3 through §5, the site is a source layer and the readout a
target layer, both inherited from the earlier DCT work and recorded in `dct_meta_<ds>.json`:

| Dataset | source → target | `input_scale` |
|---|---|---|
| `cities` | layer **11 → 20** | 47.72 |
| `common_claim_true_false` | layer **13 → 22** | 86.73 |

`input_scale` is the activation's own norm yardstick, produced by DCT's `SteeringCalibrator`. Every
perturbation budget below should be read against it. A budget of 2.8 on cities means "a nudge of
about 6% of the activation's own size".

**The hop map.** `src/reach_hop.py` defines `F(δ)` = the target-layer activation you get when you
add `δ` to the source-layer activation, broadcast at every position, and run the remaining blocks.
This is a function from `R^2304` to `R^2304`, and `J = ∂F/∂δ` at `δ = 0` is the `J` used everywhere
in §3 through §5. That earlier notation is the special case `A_S := J`, `w :=` the probe direction,
`t := t02` of the certificate in §2.3.

**Local linearity.** `F(δ) ≈ J·δ + F(0)`. This is not assumed, it is measured (Phase 5 and
Horizon 0.1), and it holds far better than expected. The important qualification, which cost us a
wrong reading once: the measured `R² = 0.999` licenses **linearity** of readout against scale, not
**gain**. The same measurement found the realized slope 8x to 35x below the predicted `‖J^T w‖`.
Linearity says the response is a straight line; it does not say we predicted its slope.

### 2.5 The broadcast operator, written out

Every certificate on disk was computed with the perturbation added at **every position**, not at
one aligned position. That is not a detail of the implementation, it is a different linear map, and
it was previously only described in words. Write it down.

Let `B : R^d → R^{T×d}` be the broadcast, `B δ = (δ, δ, …, δ) = 1_T ⊗ δ`. The map actually being
inverted is not `J` but `J ∘ B`, and its adjoint sums the position gradients:

```
(J ∘ B)^T w  =  B^T J^T w  =  Σ_t  (∂r / ∂h^(l)_t)     (sum over positions)
```

This is literally `g.sum(dim=1)` in `src/token_jac.py:82` and `src/reach_jlens.py:97`. The
single-position convention is `∂r / ∂h^(l)_last` with no sum. Both are computed side by side, and
their ratio is the **broadcast gain**:

```
broadcast_gain(l) = median_i  ‖Σ_t ∂r_i/∂h^(l)_t‖ / ‖∂r_i/∂h^(l)_last‖
```

Measured: **1.355 at cities layer 16** and **2.016 at common_claim layer 8**. Two things follow.
The broadcast site has the larger margin, so a budget quoted under broadcast is *easier* to meet
than the same number quoted per position, and the honest per-position comparison is `m_last`.
And ActAdd-style single-position injection is not the same experiment as ours: comparing our
budgets to a published ActAdd budget without dividing through by this ratio compares two different
operators.

### 2.6 Alignment `alpha`, and what the `_asis` numbers are a cosine between

`ε*` is the budget along the *optimal* direction, which is the pullback vector. Nobody steers with
the pullback vector; they steer with a named concept direction `u`. The cost of that substitution
is one cosine:

```
alpha(u) = |(A_S^T w) · u| / ‖A_S^T w‖ ,        eps_required(u) = ε* / alpha(u)
```

At the post-norm token site `A_S = I` and `A_S^T w = a_i`, so `alpha = |a_i·u| / ‖a_i‖`
(`src/token_geom.py:366-368`). `alpha` is the fraction of a flip that a unit of `u` buys, and
`1/alpha` is the multiplier on the honest budget. Chance alignment between two random unit vectors
in `R^2304` is `E|cos| = √(2/πd) = 0.0166`, which is the bar to read these against. (The other
number that gets quoted here, `1/√d = 0.0208`, is the RMS cosine, not the mean absolute cosine.
`alpha` is an absolute cosine, so `0.0166` is the right null and it is the one used in
`math_map.tex` and in the token-space findings.)

| Direction | `alpha`, cities | vs chance | `alpha`, common_claim | vs chance |
|---|---|---|---|---|
| `md_full` (final-layer mean-diff truth direction) | 0.00398 | 0.24x | 0.00940 | 0.57x |
| `mean_diff_tgt_asis` | 0.01325 | 0.80x | 0.01560 | 0.94x |
| `probe_grad_tgt_asis` | 0.01355 | 0.82x | 0.01471 | 0.88x |

Medians over `n = 200` per dataset. Every truth direction sits **at or below** chance alignment
with the direction that decides the next token, and the best of them buys 0.94 of what a random
vector would. That is the diagnosis, and it is a statement about `alpha`, not about `ε*`: the
certificate was never wrong about the budget, we were spending it almost perpendicular to the
thing that moves the token. Read through `ε(u) = ε*/alpha`, this is a 74x to 79x multiplier on
the honest budget for cities.

**The `_asis` convention, stated plainly, because it has been a footnote and needs to stop being
one.** The `_asis` directions are taken verbatim from `reach_dirs_<ds>.npz`. They were fitted at
the **target layer** `h^(l_tgt)`, and the `alpha` above reads them in **post-norm coordinates**
`z`. No change of basis is applied between the two (`src/token_geom.py:395-400`). So
`alpha_mean_diff_tgt_asis = 0.01325` is the cosine between `a_i` and *the raw coordinate vector we
actually injected*, not between `a_i` and a properly transported version of that direction. It
answers "how aligned is the direction we really steered with to the thing that moves the next
token", which is the operational question and the one we need. It is **not** a claim that the two
objects live in the same basis, and it is not a coordinate-free statement about the concept. A
basis-correct version would push the target-layer direction forward through the remaining blocks
and the final RMSNorm before taking the cosine; that has not been computed, and until it is, the
`_asis` numbers should be quoted with this sentence attached.

### 2.7 What "reachable" would mean

`ε*` small relative to `input_scale`, for most statements, along a direction the model can actually
be pushed in, at the site where the perturbation is actually injected. All four conditions matter,
and the audit in §3 through §5 is the story of the certificate passing the first and failing the
third.

---

## 3. The five-phase audit: what each experiment does, and what it found

Run on DeltaAI GH200, 2026-07-24. Runbook: [`deltaai/REACH_RUN.md`](../deltaai/REACH_RUN.md).
Findings doc: [`REACH_AUDIT_FINDINGS.md`](REACH_AUDIT_FINDINGS.md).

### Phase 1: are the margins real? (`reach_margins.py` → `reach_analyze.py`)

**What it does.** Extracts source and target activations for every statement
(`reach_acts_<ds>.npz`), builds a *battery* of readout directions (`reach_dirs_<ds>.npz`), then
computes Jᵀw for every statement × every direction with checkpointed VJP passes
(`reach_margins_<ds>.npz`). The battery is deliberately padded with controls:

| Group | Members | Purpose |
|---|---|---|
| `truth` | `mean_diff_tgt`, `probe_grad_tgt` | the readouts we care about |
| `truth_sub` | 8 directions spanning the truth subspace | best-case within the concept |
| `dct_u` | DCT's top-4 target-layer effect directions | a "known-causal" comparison group |
| `rand` | 64 random unit directions | the null |

**What it found.**

| | cities | common_claim |
|---|---|---|
| median margin ‖Jᵀw‖ for `mean_diff_tgt` | **9.07** | **7.00** |
| random-null median margin | 1.10 | 1.34 |
| ratio truth / null | **8.3×** | **5.2×** |
| median ε\* | **2.80** | **10.69** |
| fraction reachable at `input_scale` | 1.00 | 1.00 |
| verdict | `reachable-candidate` | `reachable-candidate` |

Read that as: the truth readout is roughly an order of magnitude more controllable than a random
direction, and the budget to flip it is a small fraction of the activation's own norm. On paper,
reachable.

**Figures.**

- **`plot_reach_margins_<ds>.png`** is a violin plot of log10 ‖Jᵀw‖ for the four groups, with a
  dashed line at the random-null median. *How to read it:* the truth violins sit visibly above the
  gray random violin. That separation is the entire "reachable-candidate" verdict. If truth had
  overlapped random, the programme would have stopped here.
- **`plot_reach_curves_<ds>.png`** is the money plot of Phase 1. X axis is the perturbation budget
  ε in activation-norm units; Y axis is the fraction of TRUE statements whose predicted crossing
  needs no more than ε. A vertical dashed line marks `input_scale`. A gray band on the right marks
  "beyond linear validity", the region where Phase 5 says the first-order model is off by more than
  20%. *How to read it:* for cities the curve saturates well to the left of both the dashed line
  and the gray band, meaning the certificate is cheap and inside its own validity region. For
  common_claim the curve saturates **inside** the gray band, meaning the certificate is
  extrapolation.
- **`plot_reach_geometry_<ds>.png`** is two histograms: cos(Jᵀw, mean_diff at the source layer) and
  cos(Jᵀw, v_Q). *How to read it:* this asks "where does the optimal input direction actually
  point?" It is not the source-layer truth direction. That observation is the seed of the D3
  mechanism below.

### Phase 2: is the hop's high-gain channel the concept channel? (`reach_svd.py`)

**What it does.** Computes the *full* 2304×2304 Jacobian for a subsample of 32 statements per
dataset (expensive, hence the subsample), takes its SVD, and saves the top 64 left and right
singular vectors per statement to `reach_svd_<ds>/stmt_*.npz`. Summaries land in
`reach_svd_summary_<ds>.csv` and `reach_svd_energy_<ds>.csv`.

**What it found.** The Jacobian is not a flat random matrix, but its input side is concept-blind.

| | cities | common_claim |
|---|---|---|
| effective rank (of 2304) | 396 to 551 (**17 to 24%**) | 447 to 616 (**19 to 27%**) |
| s64 / s1 | 0.080 to 0.160 | 0.101 to 0.238 |
| overlap of top-16 right singulars with DCT's V | **0.053 to 0.076** | **0.039 to 0.059** |

The spec's stop-and-diagnose threshold on that overlap was 0.3. Both datasets fire it at about
0.05, which is essentially random. The hop routes through a compact channel of roughly 500
effective dimensions, and that channel's **input** has almost nothing to do with any concept
direction we can name.

**Figure. `plot_reach_svd_<ds>.png`** plots, for each of four vectors, how much of its energy lies
inside the top-k singular subspace as k grows. *How to read it:* the truth readout at the *target*
layer (blue and green, measured against the left singulars U) climbs fast, 74% of its energy inside
the top 64 for cities and 51% for common_claim, which is 17× to 125× what a random direction gets.
The source-layer truth direction (orange, measured against the right singulars V) barely beats
random. That asymmetry is the finding: **the truth readout is downstream-visible but not
upstream-addressable.**

### Phase 3: does any of this move behavior? (`reach_steer.py` + the OLMo judge)

**What it does.** Two arms.

- The **mean arm** steers along the dataset-mean Jᵀw direction, generating on 32 held-out prompts
  at a ladder of scales.
- The **per-statement arm** (the primary one) takes 200 TRUE statements, computes each one's own
  ε_i, and steers *that* statement at multiples of *its own* budget, then generates a continuation
  from the statement's stem and asks the judge whether the model asserted something true, false, or
  incoherent.

Readouts are logged alongside (`reach_steer_readout_<ds>.csv`) so we can see whether the probe
moved even when behavior did not.

**What it found: nothing moved.**

| | cities | common_claim |
|---|---|---|
| lie rate at the strongest toward-FALSE push | **3.5%** | **5.5%** |
| unsteered baseline lie rate | ~3% | ~3% |
| incoherent fraction | 18 to 26% | 18 to 26% |

Even at the strongest push the completions stay coherent and *true*: "The city of Busan is in South
Korea", "Abu Dhabi is in the United Emirates". The linear reachability certificate does not
transfer to behavior.

**Figures. `plot_judge_reach_<ds>.png`** (mean arm) and **`plot_judge_reach_stmt_<ds>.png`**
(per-statement arm) plot the TRUE / FALSE / INCOHERENT fractions against steering scale. *How to
read them:* flat lines. That flatness is the headline null of the whole project.

### Phase 4: were we steering at the right depth? (`reach_jlens.py`)

**What it does.** For every layer 0 through 25, computes the mean margin ‖J_lᵀw‖ for three
final-basis readouts, under two input modes:

- readouts: the **truth** probe, the question-framing direction **v_Q**, and the **verdict**
  direction (the yes-minus-no unembedding difference, which is what the judge actually elicits)
- modes: **declarative** input and **question-prefixed** input

**What it found (recomputed from `reach_jlens_<ds>.csv`).** Controllability is front-loaded.

| dataset | mode | readout | peak layer | margin at peak | margin at our source layer | fraction of peak lost |
|---|---|---|---|---|---|---|
| cities | decl | truth | 0 | 6.15 | 2.81 (L11) | 54% |
| cities | decl | verdict | 0 | 1.00 | 0.50 (L11) | 50% |
| cities | quest | truth | 0 | 7.22 | 2.49 (L11) | 66% |
| cities | quest | **verdict** | **7** | 1.42 | 0.64 (L11) | 55% |
| common_claim | decl | truth | 0 | 2.31 | 1.18 (L13) | 49% |
| common_claim | quest | truth | 0 | 3.58 | 0.95 (L13) | 73% |
| common_claim | quest | verdict | 0 | 1.54 | 0.52 (L13) | 66% |

Two things follow. First, we steered **past the peak**: by our source layers, 42% to 73% of the
available controllability is already gone, and there is no late resurgence. Second, the **verdict**
readout, the one that corresponds to what the model will actually say, has a margin 2.1× to 5.7×
smaller than the truth probe at the same layer. We were pushing the biggest lever we could measure
and it was not the lever attached to behavior.

**Figure. `plot_reach_jlens_<ds>.png`** is two panels (declarative, question-prefixed), log-y,
three curves each. *How to read it:* the curves fall from left to right. The place you would want
to intervene is the left edge. We intervened in the middle. The red verdict curve sits below the
others everywhere.

### Phase 5: was the certificate inside its own validity region? (`reach_linerr.py`)

**What it does.** Sweeps the perturbation norm along several directions and measures the relative
error between the first-order prediction J·Δ and the true F(Δ). Reports **eps20**, the norm at
which that error hits 20%. This is the honest trust radius of the linearization.

**What it found.**

| direction | cities eps20 | common_claim eps20 |
|---|---|---|
| `jtw_mean_diff_tgt` (the one the certificate uses) | **2.99** | **4.87** |
| `mean_diff_src` | 12.77 | 14.61 |
| `dct_v_top` | 5.73 | 4.76 |
| `random` | 14.35 | 20.11 |

Compare to the ε\* values from Phase 1:

- **cities: ε\* = 2.80 vs trust radius 2.99, so 0.94×. Inside.** The certificate is evaluated where
  the linear model is valid. This is the scientifically interesting case: the math is right and
  behavior still does not move.
- **common_claim: ε\* = 10.69 vs trust radius 4.87, so 2.20×. Outside.** The certificate is
  extrapolation. Empirically, 0% of statements actually cross under the predicted push and the
  median gap stays at +60, exactly as this diagnosis predicts.

There is a second, quieter finding in that table: `Jᵀw` has by far the **smallest** trust radius of
any direction tested, 4 to 5× smaller than a random direction. The direction the certificate points
along is the most curvature-sensitive direction in the space. That is worth a sentence in the paper.

**Two summary figures**, `plot_reach_audit_dissociation.png` and
`plot_reach_audit_certificate.png`, combine the above into the two-panel story (readout moves /
behavior does not; certificate inside validity for cities, outside for common_claim).
**Reproducibility note:** these two PNGs were produced ad hoc on 2026-07-24 and **no committed
script regenerates them**. Everything else in the figure list comes from `src/viz_reach.py`. If
these figures go in a paper they need a script.

---

## 4. The deep audit: four mechanisms, D1 through D4

After Phase 3 came back null, four parallel deep-dives asked *why*. They sharpened, and in two
places corrected, the initial read. Full detail in [`REACH_AUDIT_FINDINGS.md`](REACH_AUDIT_FINDINGS.md).

**D1: actuation is linear, correctly signed, and 10 to 35× attenuated.** Per-statement fits of
readout-vs-scale give **R² = 0.999** (cities) and 0.990 (common_claim), with **0% wrong-sign**. So
steering works exactly as the linear model says it should, just far too weakly: the perturbation
actually needed to cross is ε_realized ≈ **97** in both datasets, against a predicted ε\* of 2.9 and
12.3. Since 97 is 1 to 2× `input_scale`, a real crossing needs a nudge the size of the activation
itself. Worse, on cities the predicted margins do not even **rank-order** which statements actuate
best (Spearman ρ = 0.002), so the certificate loses its ordering content, not just its calibration.

**D2: behavior is inert, strictly.** The initial "mild coherence degradation" read was too
generous. The incoherence rate is not dose-dependent in 3 of 4 arms. The 131 FALSE verdicts
concentrate on a handful of problem prompts that recur identically **at scale 0**, including an
MCQ-style completion judged FALSE at every scale and two judge factual errors (Abidjan, Maturín).
The one statistically significant trend runs **backwards** (p = 0.012, driven by the toward-TRUE
extreme). Where incoherence does correlate with scale it is sign-symmetric topic derailment into
generic institutional filler, not negation. Verdict: inert. Not even an incoherence knob.

**D3: the Jacobian is structured but concept-blind on the input side.** This corrected an earlier
"near-isotropic" claim. The spectrum is not flat (effective rank 21 to 23% with a two-regime
decay), so the hop does route through a compact channel, and that channel's **output** contains the
truth axis (74% / 51% of `mean_diff_tgt`'s energy in the top-64 left singulars, 17× to 125×
random). But its **input** is concept-blind: the source-layer mean-difference direction is only
1.8× to 7× random in the top right singulars, and on common_claim the probe-gradient direction is
**at or below random** (0.58× to 0.73×). Any linear concept-steering method operating in this band
hits the same structural ceiling.

**D4: controllability is front-loaded and the behavioral lever is the weakest one.** Covered in
Phase 4 above. The dissociation is over-determined: smaller lever (verdict ≪ truth), wrong depth
(past the peak), wrong mode (question-mode verdict controllability collapses right at our source
layers), plus D3's input-side misalignment.

**The refined headline.** The certificate fails behaviorally *not* because linearization breaks
(R² stays at 0.99) but because of three **transfer failures**:

1. **Context transfer.** Jacobian gain collapses across a one-word context shift while sign and
   linearity survive.
2. **Population transfer.** The probe's threshold does not transfer from statement tokens to
   generation prefixes. Cities prefixes read g ≈ −52, "deep FALSE", while the model completes
   truthfully. The probe dissociates from behavior *at baseline*, before any steering.
3. **Input-side concept misalignment.** J's high-gain channel is not addressable from any
   source-layer concept direction.

And behavioral validation *inherently forces* transfers 1 and 2, because behavior only exists
during generation, which is never the point where the certificate is computed. A behaviorally
meaningful reachability certificate has to be robust to exactly the shifts this one is not.

---

## 5. Horizon 0: closing the audit's own loopholes

Seven validations, each closing a specific hole a reviewer would poke. Plan:
[`plans/2026-07-28-horizon0-validations.md`](superpowers/plans/2026-07-28-horizon0-validations.md).
Runbook: [`deltaai/REACH_H0_RUN.md`](../deltaai/REACH_H0_RUN.md). All seven are **complete**. Two
cluster jobs (`run_reach_samepoint.slurm`, `run_reach_h0.slurm`) plus local CPU analysis.

### 0.1 Same-point control: was the certificate ever right at its own point?

**The experiment.** The 10 to 35× shortfall in D1 had two candidate explanations: either the VJP
margins are simply miscalibrated, or they are exact at their own linearization point and the loss
happens when we move to a different context. `reach_samepoint.py` settles it by doing a
teacher-forced forward on the **full statement** with the injection hook active, and reading the
readout at the statement's last token, which is *exactly* where ε\* was computed.

**The result.** Decisive, and this is the single most important number in Horizon 0.

| | cities (n=199) | common_claim (n=197) |
|---|---|---|
| **calibration factor** (realized slope ÷ predicted ‖Jᵀw‖, same point) | **0.9997**, IQR [0.9910, 1.0117] | 0.7025, IQR [0.605, 0.802] |
| **context factor** (stem slope ÷ same-point slope) | **0.0729** | 0.1944 |
| implied collapse across one word of context | **13.7×** | 5.1× |

On cities the certificate is **exact at its own point**, to four significant figures. The entire
loss is the context shift. This converts the result from "our math might be wrong" into "the math
is exact and the transfer fails", which is a completely different, and much stronger, paper.

On common_claim the calibration is 0.70, so there is a real 30% local miscalibration there on top
of the context loss. Cities is the clean case and should carry the argument.

### 0.2 Probe refit on the generation population: is D2 a threshold shift or an absence of signal?

**The experiment.** `reach_stemprobe.py` extracts the target-layer activation at each generation
stem's last token, then labels each stem for free using the **scale-0 judge verdicts already in
hand** (TRUE means the model completed that stem truthfully). Then four arms: the old threshold,
a recalibrated 1-D threshold on the same direction, a fully refit LR probe under 5-fold CV, and an
XGBoost arm to test for nonlinear headroom.

**The result.** No truth signal at stem tokens at all.

| | cities | common_claim |
|---|---|---|
| labeled stems | 158 | 178 |
| base rate P(TRUE) | 0.943 (9 FALSE) | 0.944 (10 FALSE) |
| old-threshold accuracy | 0.057 | 0.944 |
| old-threshold **balanced** accuracy | **0.500** | **0.500** |
| median g for TRUE stems | −52.6 | +82.0 |
| median g for FALSE stems | −52.0 | +79.3 |
| recalibrated 1-D threshold accuracy | 0.943 (= base rate) | 0.944 (= base rate) |
| refit LR, 5-fold CV | 0.930 | 0.944 |
| XGBoost, 5-fold CV | 0.943 (gap **+0.013**) | 0.944 (gap **+0.000**) |

Read the two accuracy rows together. On cities the old threshold calls almost everything FALSE
(accuracy 0.057); on common_claim it calls almost everything TRUE (accuracy 0.944). Both look
different but both have **balanced accuracy exactly 0.500**, which is chance. The TRUE and FALSE
medians are within 1 to 3 units of each other on a scale where the threshold sits 100+ units away.
Recalibrating recovers nothing beyond the base rate. Refitting recovers nothing. **XGBoost recovers
nothing**, which rules out the obvious "the boundary is just nonlinear here" objection.

So the answer is the stronger one: this is not a threshold shift, it is an absence of signal. At
the token where generation actually begins, the target-layer truth probe has no information about
whether the model is about to say something true. **This finding appears unclaimed in the
literature** and is the strongest supporting figure the paper has.

**The caveat, stated plainly:** balanced accuracy at a 0.94 base rate rests on only 9 and 10 FALSE
examples respectively. The direction of the result is unambiguous; its precision is not. A larger
labeled stem set would make this bulletproof and is cheap (the judge is already built).

### 0.3 Judge hardening: does the null survive removing the bad prompts?

**The experiment.** D2 identified roughly 15 prompts that fail at baseline, before any steering:
MCQ-style completions the judge cannot score, and two prompts where the judge itself is factually
wrong. `reach_judge_harden.py` drops them and recomputes every fraction. Free, local, no GPU.

**The result.** The null survives and gets cleaner.

| dataset, mean arm | n | FALSE at scale 0 | FALSE at strongest push | INCOHERENT at scale 0 | INCOHERENT at strongest push |
|---|---|---|---|---|---|
| cities, raw | 32 | 0.031 | 0.031 | 0.156 | 0.188 |
| cities, **hardened** | 26 | **0.000** | **0.000** | **0.000** | **0.038** |
| common_claim, raw | 32 | 0.031 | 0.031 | 0.156 | 0.188 |
| common_claim, **hardened** | 26 | **0.000** | 0.038 | **0.000** | 0.038 |

After hardening, the baseline is a clean 100% TRUE and the strongest push barely dents it. The
apparent "mild degradation" in the raw numbers was almost entirely the bad prompts.

### 0.4 and 0.5 Stem-context Jacobian: does *any* input direction survive the context shift?

**The experiment.** `reach_stemjac.py` runs a second VJP pass at the **stem** context and compares
it, statement by statement, to the cached full-statement Jᵀw. Then it computes a first-order robust
certificate ε\*_robust = g / (‖Jᵀw‖ + ‖ΔJᵀw‖), using the measured spread as the uncertainty.

**The result.** The context shift is not a small perturbation. It changes both the size and the
direction of the optimal input.

| | cities (n=200) | common_claim (n=200) |
|---|---|---|
| median margin, full context | 7.30 | (see below) |
| median margin, stem context | 1.76 | |
| **gain ratio stem / full** | **0.241** | **0.385** |
| **cos(Jᵀw_full, Jᵀw_stem)** | **0.305**, IQR [0.263, 0.339] | **0.392** |
| median ε\* | 2.87 | 12.21 |
| median **ε\*_robust** | **67.0** | **155.3** |
| robust ÷ nominal | **22×** | **13×** |

Two things. First, one word of context knocks the gain down by 3 to 4× *and* rotates the direction
to a cosine of about 0.3, so it is not the same direction weakened, it is a different direction.
Second, the honest robust certificate is 13× to 22× the nominal one, which puts it **above**
`input_scale`. Stated as a certificate rather than a null: **once you account for the measured
context sensitivity, the FALSE set is not reachable within the activation's own norm.**

### 0.6 Newton re-steering: is the problem curvature or context?

**The experiment.** If the one-shot gain collapse were caused by curvature (the Jacobian changing
as you move along the path), then recomputing Jᵀw at the steered state and iterating should fix it.
`reach_newton.py` does 3 iterations at the same point on 32 statements. This is a deliberate
**negative control**: the D1 diagnosis predicts it should *not* help.

**The result.** It converges essentially exactly, and barely rotates.

| | cities | common_claim |
|---|---|---|
| statements | 32 | 32 |
| median \|g\| by step (0 → 3) | 20.77 → 3.30 → 0.071 → **0.0** | 83.62 → 5.49 → 0.054 → **0.0** |
| final median \|g\| | **4.65e-05** | **2.74e-05** |
| final max \|g\| | 0.017 | 0.117 |
| statements hitting the step cap | **0 / 32** | **0 / 32** |
| cos(final direction, initial Jᵀw) | **0.971** | 0.724 |

The residual gap drops six orders of magnitude in three steps, nothing gets capped, and on cities
the final direction is a 0.97 cosine to where we started. Curvature is not the obstacle. D1's
"context, not curvature" claim is confirmed by a control designed to falsify it.

### 0.7 Recovering the singular vectors

The per-statement U64 and V64 vectors from Phase 2 had been computed on the cluster but excluded by
the rsync-back glob. They are now local: `reach_svd_cities/` and
`reach_svd_common_claim_true_false/`, 32 `stmt_*.npz` files each. Cost: an rsync. This unblocked
the SAE decomposition below, and unblocks Horizon 2.2 (balanced truncation) and 2.3 (in-channel
steering).

---

## 6. Horizon 1 Track B: SAE feature forensics (ran locally, cities only)

**The question.** D3 says the hop's high-gain input channel is "concept-blind" and D1/D2 say the
context shift changes the direction. Both statements are geometric. Can we say what these
directions *are*, in interpretable terms, rather than just that they fail to align?

**The method.** GemmaScope 16k JumpReLU SAEs at layer 11 (source) and layer 20 (target) for
gemma-2-2b, which is the best-covered model in GemmaScope. Each direction is decomposed into **32
decoder atoms via orthogonal matching pursuit with a least-squares refit on the support**.

The methodological point matters: we deliberately do **not** run the SAE encoder on these vectors.
arXiv:2411.08790 shows that is misleading, because the encoder is trained on activations and a
steering direction is not an activation. OMP against the decoder is the correct operation, and
because we decompose only nine fixed vectors, exact OMP is affordable and strictly better than the
gradient-pursuit approximation that paper proposes for streaming use. `src/sae_decompose.py`
carries a guard that refuses to run if `dct_meta` names a model other than `google/gemma-2-2b`,
since the atoms live in that model's residual basis.

**Result 1: the readout is legible, the input side is not.** Cumulative explained variance at 32
atoms:

| direction | space | explained |
|---|---|---|
| `w_mean_diff_tgt` (the truth readout) | target L20 | **0.504** |
| `V64_common_2` | source L11 | 0.348 |
| `V64_common_3` | source L11 | 0.328 |
| `common_v1` | source L11 | 0.282 |
| `jtw_stem_mean` | source L11 | 0.276 |
| `V64_common_0` | source L11 | 0.269 |
| `jtw_mean` | source L11 | 0.254 |
| `jtw_full_matched_mean` | source L11 | 0.253 |
| `V64_common_1` | source L11 | 0.228 |

The target-layer truth readout is about **2× more SAE-legible** than any source-side direction.
This is a new, independent restatement of D3: the thing we are trying to control is describable in
feature terms, and the thing we have to push on is not.

**Result 2: the context shift swaps the feature set outright.** Pairwise Jaccard of the top-32
feature supports:

| pair | Jaccard | shared features |
|---|---|---|
| `jtw_full_matched_mean` vs `jtw_mean` (sanity control) | **0.684** | 26 of 32 |
| `common_v1` vs `jtw_full_matched_mean` | **0.362** | 17 of 32 |
| `common_v1` vs `jtw_stem_mean` | 0.164 | |
| **`jtw_full_matched_mean` vs `jtw_stem_mean` (the D2 pair)** | **0.049** | **3 of 32** |
| any `V64_common_j` vs anything else | ≤ 0.12 | |

The sanity control at 0.684 proves the measurement is not degenerate: the matched subset really
does agree with the full-dataset mean. Against that, the full-context and stem-context directions
share **3 features out of 32**. This is a feature-level account of D1 and D2: the one-word context
shift does not weaken a direction, it substitutes a different set of features.

**Figures.**

- **`plot_sae_explained_cities.png`**: cumulative explained variance against OMP rank, one line per
  direction. *How to read it:* how sparse each direction is in feature space. A line that rises
  steeply and plateaus high is a direction the SAE "understands". The readout line is the top one;
  all the source-side lines cluster low.
- **`plot_sae_overlap_cities.png`**: bar chart of pairwise Jaccard, sorted, with the full-vs-stem
  pair highlighted in red. *How to read it:* the red bar is near the floor while the control bars
  are far above it. That contrast is the figure's whole point.

**Limits, stated up front:** cities only, one SAE width (16k), one layer pair, and no
absorption/dark-matter robustness pass yet. Running common_claim and a second width is cheap CPU
work and should happen before this goes in a paper.

---

## 7. What is built and **not** run

This section is the one to read carefully before the next meeting.

### 7.1 The refusal positive control (Horizon 1.1): the publication gate

**Nothing has run on the cluster.** No `refusal_*` artifacts exist anywhere in the repo. The four
SLURM scripts still carry the `ACCOUNT_NAME` placeholder, meaning none has been submitted.

**Why it exists.** The truth run produced a `readout-only` outcome: the certificate is locally
exact, steering past ε\* moves the readout, and completions do not change. That is consistent with
two very different stories:

1. **The instrument is broken.** Our hook convention, our choice of Jᵀw as the input direction, or
   our thresholds are wrong, and this pipeline would fail on any concept.
2. **Truth is not a behaviorally actuatable variable in gemma-2-2b.** The instrument works and the
   finding is real.

Only a concept with a **known** behavioral handle can distinguish those. Refusal is that concept:
Arditi et al. (arXiv:2406.11717) established a single refusal direction, with public code,
validated down to small Gemma models, at a known steering layer. Running the *unmodified* pipeline
on it, same hook, same ε\* arithmetic, same verdict logic, is the positive control.

**What is already done.** The dataset is built locally:

| file | rows | contents |
|---|---|---|
| `got_datasets/refusal.csv` | **976** | balanced, 488 label-1 (harmless, from Alpaca) and 488 label-0 (harmful, from AdvBench) |
| `got_datasets/refusal_holdout.csv` | **64** | never enters direction fitting; supplies both the screening prompts and the mean-arm generation prompts, so behavioral evaluation is leakage-free |

Two design decisions in that build are load-bearing:

- **Polarity: label 1 = harmless.** The pipeline computes ε\* for label-1 statements crossing into
  the label-0 halfspace, so this makes the certificate describe *inducing* refusal on a harmless
  instruction. That direction has full headroom, because baseline refusal on harmless prompts is
  near zero. The opposite direction (ablating refusal) would require a base model that refuses
  unprompted, which gemma-2-2b may never do. `reach_steer` sweeps ± scales, so we observe the
  ablation direction for free without betting on it.
- **Linearization point: `--prompt-mode full`.** A refusal instruction *is* the generation prompt,
  so the certificate's linearization point is the prompt's own last token. The Horizon-0
  context-shift confound (the entire D1 story) **cannot arise here by construction**. That is a
  deliberate design choice, not an accident, and it is why the control is a clean test of the
  instrument rather than a rerun of the same failure.

**The four jobs, gated in sequence.** Runbook:
[`deltaai/REFUSAL_RUN.md`](../deltaai/REFUSAL_RUN.md).

| # | job | wall cap | what it does | gate |
|---|---|---|---|---|
| 1 | `run_refusal_screen.slurm` | 00:40 | measures unsteered refusal rates for `gemma-2-2b` and `gemma-2-2b-it` | **STOP HERE.** harmful-prompt refusal ≥ 0.10 on the base model → keep the base model (no model confound with the truth run). < 0.10 → fall back to `-it`, re-prep the dataset **on the laptop** with `--chat-template`, and do **not** re-run the screen (it would double-template every prompt) |
| 2 | `run_refusal_prep.slurm` | 03:00 | extraction, layer-sweep meta, `SteeringCalibrator` scale, direction exports | check `dct_meta_refusal.json`: `source_layer ≥ 5` and `input_scale` is a number, not `null`. `num_factors: null` is **by design** (the minimal control trains no DCT factors) |
| 3 | `run_refusal_reach.slurm` | 08:00 | smoke run at `--limit 8` with a hard stop, then P1 margins + analyze, P3 mean and per-statement steering, substring judging, and the verdict | check the `[reach] slice fidelity cos=…` line; a `SlicedModel unfaithful` death means the Jacobians would be of the wrong map |
| 4 | `run_refusal_spotcheck.slurm` | 01:30 | OLMo re-scoring, raw agreement and Cohen's kappa on 60 rows per arm | kappa < 0.6 → the substring rate needs a stated caveat in the writeup |

Total wall cap 13h10m; realistic spend is 3 to 6 GPU-hr against roughly 475 remaining.

**The verdict engine, `src/reach_control.py`.** This is 643 lines and is the most carefully guarded
script in the repo, because it produces the single number the programme turns on. It assembles the
2×2 of (readout crossed) × (behavior moved) and names one of four cells:

| verdict | meaning | what we do |
|---|---|---|
| `actuatable` | crossing the boundary changes behavior | **the instrument is valid**, the truth result is a fact about truth, proceed to the main-venue paper |
| `readout-only` | readout moves, behavior does not, i.e. refusal fails the same way truth did | **STOP.** Take it to Julian before spending more. Prime suspects: the all-position hook convention, or Jᵀw as the input direction |
| `inert` | behavior moves without a readout crossing | off-target steering |
| `no-crossing` | neither | underpowered sweep, the `1.5 × input_scale` cap is clamping scales below ε\* |

Design details worth knowing, because they are the difference between a result and an artifact:

- Rows are indexed by **frac = scale / ε\*, not raw scale.** In the per-statement arm every
  statement has its own ε_i, so raw scales are all distinct and grouping by scale would give n=1
  buckets everywhere.
- `MIN_BUCKET_N = 5` bars any bucket with fewer than 5 rows from deciding a verdict cell. This is
  explicitly a **clamp-artifact filter, not a power gate**: the code comments note that a 5/5 rate
  still has a 95% Wilson interval of (0.57, 1.00). Undersized buckets are still shown, flagged, and
  never dropped. If *every* non-baseline bucket is undersized, the script refuses to emit a verdict
  at all and exits non-zero, which is correct: that is a failed measurement, not a `no-crossing`.
- **The sign trap.** `MIN_DELTA = 0.10` is a symmetric "did anything change" screen. But the
  prediction here is directional: with label 1 = harmless, crossing should make the model refuse
  **more**. A refusal rate that *falls* by 10 points still returns `actuatable`, and that would be
  evidence *against* actuation. The sidecar JSON therefore carries `deciding_frac`,
  `deciding_delta_vs_baseline` (signed) and `refusal_fell_at_deciding_frac`, and the log prints a
  warning. The likely mundane cause of a fall: the refusal marker list includes "as an ai" and "as
  a language model", which are benign openers on harmless Alpaca instructions, so a large
  perturbation that degrades fluency removes them and lowers the measured rate.
- **The primary arm is `stmt`, not `mean`.** The per-statement arm has roughly 200 rows per bucket
  against 32 for the mean arm, and steers each subject at its *own* ε_i. At n = 32, four prompts
  changing status is enough to cross `MIN_DELTA`, so a mean-arm `actuatable` must never be read on
  its own.
- The artifact of record is the sidecar `reach_control_<ds>_<arm>.json`, not the job log. It
  carries the verdict, the per-frac table with counts and Wilson intervals, three row-accounting
  counts, and a self-describing `column_semantics` dict.

**One free action available right now:** `reach_control.py` has never been run on **anything**. No
`reach_control_*.csv` or `.json` exists. Running it on the existing truth artifacts would put the
truth result on record in the same pre-registered vocabulary (`readout-only`), costs zero GPU, and
makes the eventual truth-vs-refusal contrast exactly like-for-like instead of two differently
worded conclusions.

### 7.2 The 2026-07-23 conditional-steering and U-anchor round: built, never run, superseded

Six feature commits on 2026-07-23, a 9.2 KB design spec, and a 68.7 KB implementation plan:

- **Arm A1**: MAG verdict-mode steering with p_yes / p_no logit readout
- **Arm A2**: composed v_Q-gate plus mean_diff conditional steering, with figures
- **Arm B**: U-space anchored warm DCT (`--anchor-space u`, `u_anchor_lambda`), anchoring the
  factor-0 *effect* direction rather than the input direction, with direction assembly, geometry,
  a parameterized steer script, and verdict-curve and geometry-ladder figures
- Cluster jobs `run_dct_uwarm.slurm`, `run_dct_uwarm_judge.slurm`, `run_mag_cond_judge.slurm` and
  the runbook `docs/CONDITIONAL_UANCHOR_RUNBOOK.md`

**Artifact check: nothing.** No `dct_uwarm_*` files, no conditional-steering CSVs, no
`plot_*uwarm*` figures. The reachability pivot landed the next day and this round was never
submitted.

This needs an explicit decision rather than quiet abandonment. Note that Arm B is conceptually the
*scalar* version of the reachability idea (it anchors the effect toward the truth readout, where
reachability replaces "one anchored effect direction" with "the full preimage of the effect set"),
so it is not junk. But it is superseded, and running it now would consume GPU that the refusal gate
has a stronger claim on. My read: retire it, and say so in the writeup, keeping the code as the
scalar precursor of the set-based method.

---

## 8. Cluster inventory: every job, and whether it ran

Account `bhhv-dtai-gh`, partition `ghx4`, NCSA DeltaAI GH200. Roughly **475 GPU-hr remain**.

| SLURM script | Round | Status |
|---|---|---|
| `run_dct.slurm`, `run_interpret.slurm`, `run_steer.slurm`, `run_judge.slurm` | original funnel | ran (pre-window) |
| `run_mag_extract.slurm`, `run_mag_steer.slurm`, `run_mag_judge.slurm` | MAG battery | ran (pre-window) |
| `run_length_steer.slurm`, `run_length_judge.slurm` | length control | ran (pre-window) |
| `run_dct_warm.slurm`, `run_dct_warm_judge.slurm` | warm DCT | ran (pre-window) |
| `run_dct_uwarm.slurm`, `run_dct_uwarm_judge.slurm`, `run_mag_cond_judge.slurm` | 2026-07-23 conditional/U-anchor | **written, never submitted** |
| `run_reach_margins.slurm` | audit P1 | ran 2026-07-24 |
| `run_reach_svd.slurm` | audit P2 | ran 2026-07-24 |
| `run_reach_steer.slurm`, `run_reach_judge.slurm` | audit P3 | ran 2026-07-24 |
| `run_reach_linerr.slurm` | audit P5 | ran 2026-07-24 |
| `run_reach_samepoint.slurm` | Horizon 0.1 | ran 2026-07-29 |
| `run_reach_h0.slurm` | Horizon 0.2 / 0.4 / 0.6 | ran 2026-07-29 |
| `run_refusal_screen.slurm` | Horizon 1.1 gate | **written, never submitted** |
| `run_refusal_prep.slurm` | Horizon 1.1 | **written, never submitted** |
| `run_refusal_reach.slurm` | Horizon 1.1 | **written, never submitted** |
| `run_refusal_spotcheck.slurm` | Horizon 1.1 | **written, never submitted** |

**The environment landmine to remember.** `dct.SlicedModel` divides gemma-2 inputs by √d, expecting
the HF model to re-multiply `inputs_embeds` by its normalizer. **transformers ≥ 5 no longer does**,
which silently made the hop map unrelated to the real forward (cosine 0.04 to 0.23). Every hop job
now prints a startup fidelity probe: `[reach] slice fidelity cos=1.000000 (input compensation x…)`.
If a job instead dies with `SlicedModel unfaithful`, **stop**, because its Jacobians would be of the
wrong map. The cluster's `.venv-dct-gpu` (transformers 4.51.3) is faithful with factor 1, and the
audit's margins were computed there and stand. The local `.venv` is 5.12.1 and relies on the
compensation.

---

## 9. Repo review

**Code.** 10,320 LOC across 60 modules in `src/`. The reachability programme added roughly 2,300
LOC of it: `reach_hop.py` (the hop map and autograd rows), `reach_margins.py` (staged extraction
with atomic checkpointing), `reach_analyze.py`, `reach_svd.py`, `reach_steer.py`, `reach_jlens.py`,
`reach_linerr.py`, `reach_samepoint.py`, `reach_stemprobe.py`, `reach_stemjac.py`,
`reach_newton.py`, `reach_judge_harden.py`, `reach_control.py`, plus `viz_reach.py`. Horizon 1
added `refusal_screen.py`, `prep_refusal.py`, `refusal_judge.py`, `make_reach_meta.py`,
`calibrate_scale.py`, `sae_load.py`, `sae_decompose.py`, `viz_sae.py`.

**Tests.** 52 test files, 3,765 LOC, **298 passing and 1 skipped in 5.7 seconds**. The suite grew
from 182 to 298 in this window. That speed matters: the suite is run before every cluster
submission, which is the only cheap defense against burning an 8-hour GPU job on a typo.

**Review discipline.** Of the 71 commits, 24 are `fix(...)` commits landed during review passes,
including several that would have produced wrong conclusions rather than crashes: the
baseline-relative crossing definition, excluding never-steered statements from the per-statement
baseline, requiring same-frac co-occurrence before declaring `actuatable`, gating every verdict cell
on `MIN_BUCKET_N`, and threading the model id so the SAE decomposition cannot silently use the
wrong dictionary. Three commits exist purely to reconcile the plan's stated test count with reality
(267 → 285 → 298), which is a small thing but the right instinct.

**Known local hazards, all now guarded in code or documented:**

- Never `import xgboost` in the same process as `torch`: two libomp copies segfault on macOS ARM.
  This is why `reach_stemprobe.py --fit` is deliberately torch-free.
- Never `np.load(...)["key"]` inside a loop. `reach_margins_cities.npz["jtw"]` re-decompresses each
  time; in a 200-iteration loop that allocates roughly 100 GB and dies.
- Three virtualenvs exist because of two hard transformers version conflicts (DCT pins 4.51.3, the
  OLMo judge needs newer).

**Hygiene gap worth naming.** 72 untracked paths sit at the repo root, and they include essentially
every result artifact: `reach_*.npz`, `reach_*.csv`, `judge_*.csv`, all the `plot_reach_*` and
`plot_sae_*` PNGs, and `sae_features_cities.csv`. `.gitignore` covers the older generations
(`plot_findings_*`, `judge_steer_*`, `mag_acts_*`) but was never extended to the reach and SAE
generations, so these are untracked by accident rather than by policy. Two of the audit's headline
figures also have no generating script. Before anything goes into a paper, both need fixing.

---

## 10. Where this goes from here

Ordered by information per GPU-hour and by what blocks what.

**1. Submit `run_refusal_screen.slurm`.** Forty minutes of GPU. Everything downstream in Horizon 1
is gated on it, and it is the only thing standing between the current state and a publishable
result. The gate is pre-registered: base-model harmful-prompt refusal ≥ 0.10 keeps the base model
and gives a contrast with no model confound; below 0.10 falls back to the instruct model with a
stated caveat. Note the `-it` license must be accepted on HuggingFace before submitting, because
the job screens both candidates.

**2. Then prep → reach → spot-check.** Three more jobs, each gated on a check of the previous one's
output. Read the sidecar JSON, not the log. Read the `stmt` arm as primary. Check the sign of
`deciding_delta_vs_baseline` before believing an `actuatable`.

**What the two outcomes mean:**

- **`actuatable`.** The instrument is validated. The truth dissociation becomes a fact about truth
  rather than a fact about our method, and the paper upgrades from a workshop note to a main-venue
  candidate: a certified-reachable-but-behaviorally-inert dissociation, with a four-mechanism
  decomposition and an instrument-validation contrast. Per the roadmap's literature search, no
  published work reports this.
- **`readout-only`.** Refusal fails the same way truth did, which means the instrument cannot
  demonstrate actuation even where the literature says actuation exists. **Stop and take it to
  Julian.** The two prime suspects are named in advance: the all-position steering-hook convention,
  and the choice of Jᵀw as the input direction. This outcome is not a failure of the project, it is
  a different and more contestable claim about open-loop steering generally (which would echo
  arXiv:2407.12404), and it should not be made without the PI.

**3. Free CPU work, in parallel, no gate:**

- Run `reach_control.py` on the existing truth artifacts so the truth result is stated in the
  pre-registered vocabulary. Zero GPU.
- Extend the SAE forensics to `common_claim` and to a second SAE width, and add the residual
  dark-matter fraction check. Currently cities-only at 16k, which is too thin to publish.
- Enlarge the labeled stem set for 0.2 so the balanced-accuracy result does not rest on 9 and 10
  FALSE examples. The judge already exists.
- Write generating scripts for the two audit summary figures, and extend `.gitignore` to the reach
  and SAE artifact generations.

**4. Decide the fate of the 2026-07-23 round** rather than leaving it in limbo.

**5. Horizon 2, with Julian, after the refusal verdict.** The three items are already scoped: the
maximum-contrast closed-loop configuration (source at layers 5 to 8 per D4, verdict-direction cost,
receding-horizon re-planning, which is the direct fix for D1's context-transfer failure and exactly
the open-loop-to-closed-loop upgrade A-LQR formalizes); balanced-truncation of the empirical
Gramians from the 32 recovered full Jacobians, now that 0.7 is local; and in-channel steering along
the top right singular vectors to test D3 causally.

---

## 11. How to present this to Julian in three minutes

**The one-sentence version.** We built per-statement backward-reachability certificates for a truth
readout across a 9-layer hop, verified them to be locally exact (calibration factor 0.9997), showed
that steering along them moves the readout exactly as predicted (R² = 0.999, zero sign errors), and
found that behavior does not change at all, and we can now decompose *why* into four measured
mechanisms.

**The three numbers to lead with.**

1. **0.9997.** The same-point calibration factor on cities. The certificate is exact where it is
   defined. This is what makes the null a finding rather than a bug.
2. **0.500.** The balanced accuracy of the target-layer truth probe at generation-prefix tokens,
   with a nonlinear refit recovering nothing. The probe carries no information about what the model
   is about to say. This one appears unclaimed in the literature.
3. **0.049.** The Jaccard overlap between the full-context and stem-context optimal input
   directions in SAE feature space, 3 shared features out of 32, against a non-degenerate control
   pair at 0.362. One word of context does not weaken the direction, it substitutes a different one.

**The framing for his line of work.** Our R² = 0.999 corroborates A-LQR's (arXiv:2604.19018) core
local-linearity assumption directly and quantitatively. Our D1 (context-transfer gain collapse) and
D2 (population-transfer probe failure) are precisely the open-loop failure modes that his
closed-loop design implicitly answers. Cities, which is readout-controllable and behavior-inert
inside its own validity radius, is the cleanest possible test case to hand to A-LQR.

**The honest column, which should be said out loud rather than waited for.** Everything so far is a
null, and it stays a null pending the refusal positive control, which is built and gated but not
yet submitted. Single model, one hop pair per dataset, one probe family. The D2 balanced-accuracy
result rests on 9 and 10 FALSE examples at a 0.94 base rate. The common_claim same-point
calibration is 0.70, not 1.0, so cities carries the clean argument. The SAE work is cities-only at
one width. There is no human-versus-judge kappa on the actual steering completions, only on the
clean statements the 0.970 validation gate used. And arXiv:1910.13272 from the meeting notes still
does not match the paper it was described as.
