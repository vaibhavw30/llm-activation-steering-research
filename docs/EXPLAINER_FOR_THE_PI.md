# Since the last meeting: what you asked for, what I did, and what came back

*Window: the 2026-07-23 meeting through 2026-08-05. 91 commits. Written to be read cold,
in order, by someone who was in that meeting and has not looked at the repo since.*

**How this document is organised.** The meeting produced a small ask (a better
visualizer), one large reframe (control theory), and eight numbered notes. Sections 1
through 5 are the chronology: the reading I did first, the algebra I wrote out, the
experiment I designed and why each decision went the way it did, and the cluster runs.
Section 6 answers the eight notes one at a time with a verdict and the numbers. Sections
7 through 10 are the synthesis, the figure guide, the honest column, and what I want from
you next.

Every number is read off a named artifact on disk or off a committed findings document
that cites one. Where an earlier document of mine disagrees with an artifact, the artifact
wins and I say so in place.

---

## 0. The two-minute version

Three things happened.

**The reframe worked, and it produced a certificate.** Turning "find a steering direction"
into "compute the backward-reachable set of a target set" gave a per-statement budget
`eps* = g/m` that is one vector-Jacobian product to compute. I ran it, audited it in five
phases, and then spent a week trying to break it. It does not break: on `cities` the
certificate is **exact at its own linearization point, calibration factor 0.9997**. It also
does not work: at the strongest push the model's lie rate goes from 3% to **3.5%**. That
combination, a formally correct and behaviorally inert certificate, is the result.

**Diagnosing why produced the strongest single number in the project.** At the token where
generation actually begins, the truth probe's **balanced accuracy is 0.500**, chance, and
an XGBoost refit recovers **+0.013**. The probe carries no information about what the model
is about to say. In SAE feature terms, removing one word of context replaces the actuator's
feature support outright: Jaccard **0.049**, 3 shared features of 32, against a
non-degenerate control at **0.684**.

**Your token-space proposal fixed it.** Defining the target set in token space instead of
by a fitted probe makes the readout *be* the behavior, so the dissociation cannot happen by
construction. At the output layer the closed-form cone displacement flips the emitted token
on **200 of 200** statements on both datasets. Pulled back through the Jacobian to layer 16
it flips **34.0%** at the certified budget and **67.5%** at twice it, while the old
probe-halfspace direction flips **0.000** in every arm at every budget on both datasets.

And the one number that retires the old story: at the budget the old certificate calls a
full trip into the FALSE halfspace, the old direction consumes a median **1.7%** of the
logit margin that actually decides the token. We were not pushing weakly on the right
thing. We were pushing on something statistically indistinguishable from noise.

---

## 1. What the meeting asked for

Four things came out of the meeting, and this document is structured so each one is
traceable to an answer.

### 1.1 The small ask: a better visualizer for the linearity cosine shift score

**Done, first, before anything else.** `src/viz_mag_linearity.py` produces
[`plot_mag_linearity_v2.png`](../plot_mag_linearity_v2.png), two panels across all four
datasets.

The old figure plotted only `eps_Q` for `v_Q` and buried the actual finding. The finding is
the cosine structure, so the new left panel is
`cos(direction, prefix-induced shift Delta^Q)`: the question-framing direction `v_Q` sits at
**0.84 to 0.98**, while the supervised truth axis and DCT's top lever both sit inside an
`|cos| < 0.1` orthogonal band. The right panel is `eps_Q` per direction, with the truth axis
and DCT lever **above** the `eps_Q = 1` line at **1.28 to 1.40**, meaning they explain the
prefix-induced shift *worse* than predicting no shift at all.

Plain English: the direction that carries "am I being asked a question about this" is real,
crisp, and essentially perpendicular to the direction that carries "this statement is true."
Those are two different things inside the model, and we had been conflating them.

### 1.2 The large ask: stop hunting directions, compute a backward-reachable set

Your proposal, as I wrote it down:

> Stop searching for single steering **directions**. Define a **target set** in the output
> space, specifically in J-space, for example "activations whose truth readout says FALSE
> while staying coherent". Then compute the **backward-reachable set** of that target at the
> source layer: the set of source-layer perturbations that land in the target. Because the
> map is locally linear, the tools are just linear algebra: preimages of halfspaces under a
> linear map, Jacobian transposes, SVD.

Why this was the right upgrade, and I want to be explicit that I think it was: every result
in the project up to that point was a statement that some **single direction** failed to be
a truth lever. That is an absence of evidence and a reviewer can always say we picked the
wrong direction. Reachability turns it into a claim about **sets**, and sets have only two
interesting outcomes, both of which are findings. Either the "assert a falsehood, stay
coherent" region is not reachable from bounded source-layer perturbations at all, which
makes the null theorem-shaped; or it is reachable but thin or off-axis, which explains why
every single-direction probe missed it and hands us the steering set directly.

### 1.3 The eight notes

These are your meeting notes, decoded into testable claims. The table is reproduced from
[`NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md` §II.1](NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md),
which is where I first wrote them down, before any of them had been run. Section 6 gives
each one a verdict.

| # | Your note, compressed | What it asserts |
|---|---|---|
| 1 | Naively steer on the output; if it causes the same problem, mostly making it incoherent | A no-Jacobian control isolates *formulation error* from *linear-steering weakness* |
| 2 | If incoherent, the issue is the linear feature in the final layer | If even direct final-layer steering only degrades text, truth is not behaviorally linear there |
| 3 | Check the steering vector is what you want it to be | Verify the injected vector is what we think, where we think |
| 4 | Small perturbations in different directions have vastly different chains (chaotic latent space) | Directional sensitivity is wildly heterogeneous, so a norm budget is the wrong currency |
| 5 | SAE is a useful alternative for classifying the target set | Define the target set by interpretable features, not a probe hyperplane |
| 6 | Temperature 0, deterministic, then invert that mapping of the tokens | Define the target set in token space and pull it back |
| 7 | Write out all the math from input to output and justify | One place where every object is named |
| 8 | ActAdd sweeps all layers, some layers do nothing, mean difference is heuristic | Our layer choice (11 and 13) came from DCT and was never swept |

### 1.4 The four references

You gave me four. I read all four properly before writing any code, and section 2 covers
what each turned out to be. One of them does not match its description and I need you to
confirm it.

---

## 2. Chapter 1: the reading

This is the part that happened first, and it changed the design of everything downstream, so
it is worth its own section rather than a citation list.

### 2.1 The four references, deep-read

**arXiv:2509.21528, "Preemptive Detection and Steering of LLM Misalignment via Latent
Reachability" (Karnik and Bansal, Stanford Safe and Intelligent Autonomy Lab, Sept 2025).**
The load-bearing one. They treat greedy LLM generation as a discrete-time control system in
residual-stream latent space `z_t`, define a failure set as the sub-zero level set of a
scalar margin `l(z)` (a toxicity-classifier score), and compute a **Backward Reachable
Tube** `B = {z : V(z) <= 0}` via a Bellman-style value recursion learned by an MLP. That is
DeepReach-style neural reachability: approximate, no certificates. Steering is a
least-restrictive filter, intervening only when `V(z)` drops below a margin. They report
about 98% detection, flagging unsafe trajectories 7 to 10 tokens early, and 54 to 85% fewer
unsafe generations across five LLMs. No code released.

The mapping to us, and the thing that determined my whole approach: their `l` is our probe
margin, and their token-step backward tube is our **depth-wise** preimage. Our case is
*structurally easier*. They have a recurrent nonlinear rollout through time; we have one
linearized 9-block hop through depth. That means **closed-form set preimages can replace
their learned value function**, and a learned value function is exactly the part of their
pipeline that cannot be certified. So I did not import their machinery. I wrote the
closed-form version.

**arXiv:2603.00140, "Steering Away from Memorization: Reachability-Constrained RL for
Text-to-Image Diffusion" (same lab, Feb 2026).** Same skeleton on diffusion: failure set
from a memorization margin, backward tube approximated by a safety critic trained with the
discrete-time Hamilton-Jacobi recursion, minimal perturbations chosen by constrained RL.
Two things transferred. First, **they compress the control space before doing reachability**
(77x768 CLIP embedding down to a 64-dim VAE latent), which told me to work in a reduced
basis rather than raw `d = 2304`; that is why Phase 2 computes an SVD and keeps the top 64
singular vectors. Second, their constrained-MDP objective is the template for the trade-off
every one of our experiments hits: "minimal perturbation that reaches the FALSE set
**subject to a coherence constraint**."

**Anthropic, "A Global Workspace in Language Models" (transformer-circuits, July 2026).**
This is where the "J-space" vocabulary in your notes comes from. The **Jacobian lens** is
`J_l = E[d h_final / d h_l]` averaged over pretraining prompts, a per-layer linear map to
output-vocabulary space that corrects for representational drift. J-space is the set of
points expressible as sparse nonnegative combinations of J-lens vectors, so it is a
**capacity-limited cone, not a direction**, sitting in the middle third of the network and
carrying under 10% of activation variance.

Their key functional result for us is **selective engagement**: the same information can sit
in the residual stream, be linearly decodable, and be *causally inert* for automatic tasks
while being decisive under explicit-report framing. That is our entire decodable-is-not-causal
null restated as workspace gating, and it independently predicts the MAG finding from before
the meeting, that the "being asked about truth" direction is real, linear, and orthogonal to
the truth-content axis.

**arXiv:1910.13272, "Feedback Linearization for Uncertain Systems via RL" (Westenbroek et
al., Tomlin/Sastry lab, 2019).** **This ID is not a reachability paper**, and I need you to
confirm whether it was the one you meant. It is model-free feedback linearization: learn the
correction to a nominal linearizing controller `u = A(x)^-1 (v - b(x))` so a nonlinear plant
tracks a linear reference model. The conceptual echo is real, because `A(x)^-1` is literally
"map desired outputs back through the inverse of the local linear map", which is the same
algebra as our Jacobian-transpose step. But it contains no target sets, no set propagation,
and no reachability tooling. Either it was intended as the feedback-linearization
formalization of "invert the output map", which would make sense, or it is a typo for a
set-propagation survey. **This is still open and it is the one loose end from the meeting.**

### 2.2 The tooling survey, and why I wrote plain linear algebra instead of importing a framework

Three tiers, decided before writing code:

- **Tier 0, plain linear algebra.** For a single linearized hop, the backward-reachable set
  of a halfspace target is closed-form: pull `w` back through `J^T`, intersect with the
  norm-budget ball. No framework needed. **This is what I built.**
- **Tier 1, neural-network preimage tools.** INVPROP (arXiv:2302.01404, in alpha-beta-CROWN)
  provably bounds preimages of linearly-constrained output sets and has been verified at
  167k neurons; PREMAP (JMLR 2025) is its successor with both under- and over-approximations.
  Nothing in this literature has been run on a multi-block transformer with attention. We
  would be first, which is both the opportunity and the risk.
- **Precedent at our exact scale.** LiSeCo (arXiv:2405.15454) already does linear-probe
  halfspace constraints with closed-form minimal-norm projection **on gemma-2-2b**. It is a
  forward safe-set controller rather than backward reachability, but it is proof the
  probe-as-halfspace machinery works at `d = 2304`.

Classical grid-based Hamilton-Jacobi is exact and dies at about 6 dimensions, so it can only
enter through learned value functions and never directly at `d = 2304`. That is a hard fact
about the method and it is why the Bansal-lab papers use an MLP.

**The decision, and the reason:** start at Tier 0. Our hop is one linearization, the target is
a halfspace, and the preimage of a halfspace under a linear map is a halfspace. Anything more
elaborate would have added approximation error on top of a question we could answer exactly.
If Tier 0 had come back ambiguous I would have escalated. It did not; it came back sharp.

### 2.3 The deep-research pass, and the three things it changed

I ran a full literature pass (the prompt is committed at
[`DEEP_RESEARCH_PROMPT_REACHABILITY.md`](DEEP_RESEARCH_PROMPT_REACHABILITY.md)) after the
audit came back null but before spending GPU on the token-space program. It changed three
things, and two of them saved GPU-hours.

**1. The framing of the contribution.** Our headline was going to be "decodable is not
causal". That is not new. Elazar et al. (amnesic probing, TACL 2021) said it, LEACE (2023)
section 5.3 showed a decodability peak at a different layer from the causal peak, and
AxBench (ICML 2025) states **on our exact model** that better classification does not yield
better steering. A reviewer who knows Elazar rejects that framing on sight. The claim has to
be: *we make that gap quantitative and certifiable*. The certificate is the contribution,
not the gap.

**2. A counter-explanation I had not considered, and had to rule out.** Tan et al. (NeurIPS
2024) report that in many datasets nearly half of examples are **anti-steerable**, which
would flatten an aggregate while individual statements move in both directions. Our headline
was an aggregate FALSE rate, so this had to be answered before anything else was believable.
I tested it on data already on disk, no GPU, and the answer is different per dataset, which
is why the two are never pooled anywhere in this document.

| | `cities` | `common_claim` |
|---|---:|---:|
| statements | 199 | 197 |
| **inert** (byte-identical at every scale) | **103 (51.8%)** | 8 (4.1%) |
| churn (text moved, verdict did not) | 68 (34.2%) | 133 (67.5%) |
| mover (verdict moved) | 28 (14.1%) | 56 (28.4%) |
| movers with the direction / against | **24 / 4** | 35 / 21 |
| exact two-sided sign test | **p = 0.00018** | p = 0.081 |

`cities` is **inert, not cancelling**: half the statements do not move a single byte, and
among those that do the effect goes the right way 24 times against 4. Anti-steerability
cannot explain a null when nothing moves in either direction. `common_claim` **is** partly
the Tan regime, so there the distribution must be reported instead of the mean. Figure:
[`plot_signed_steer_cities.png`](../plot_signed_steer_cities.png) and
[`plot_signed_steer_common_claim_true_false.png`](../plot_signed_steer_common_claim_true_false.png).

**3. Two prior-art corrections.** INVPROP is arXiv:2302.01404 (my brief had no ID). And
Grivas et al. (arXiv:2203.06462) supersedes Demeter for the unargmaxability test with an
exact algorithm, reporting that the effect is rare above about `d = 200`. That last one
predicted the stolen-probability result in §5.1 before I ran it.

---

## 3. Chapter 2: the math, input space to output space (your note 7)

Your ask was for one place where every linear-algebra object between token input and token
output is written down **and justified**, rather than described in words. This chapter is
that, in the document you read; the LaTeX source carrying the same content with the proofs
typeset is [`math_map.tex`](math_map.tex), and **both are now tracked in git** so they are
shared record rather than two working files on my laptop.

Everything here is derived rather than asserted. Where a step is a modelling choice rather
than a consequence I say which.

### 3.1 The forward map, written out

Let `d = 2304` be the model width, `L = 26` the number of decoder blocks, and
`V = 256,000` the vocabulary. `E` in `R^(V x d)` is the token embedding matrix, and in this
model it is **tied**, so the same matrix serves as the unembedding. `gamma` in `R^d` is the
gain of the final RMSNorm.

Given a prompt `x_1..T`, the residual stream is

```
h^(0)_t   =  sqrt(d) * E[x_t]                                              (1)

h^(l+1)   =  h^(l) + Attn_l( RN(h^(l)) ) + MLP_l( RN(.) ),   l = 0..L-1    (2)
```

Each block mixes across positions, so `h^(l+1)_t` depends on `h^(l)_1..t`. That
position-mixing is the reason the broadcast operator in §3.11 is not a bookkeeping detail.
Then the final norm and the head:

```
z_t   =  RN_gamma( h^(L)_t )  =  ( h^(L)_t / rms(h^(L)_t) ) * (1 + gamma),
                                  rms(h) = ||h|| / sqrt(d)                 (3)

u_t   =  E z_t   in R^V                                                    (4)

u~_t  =  c * tanh(u_t / c),        c = 30   (final logit softcapping)      (5)

p_t   =  softmax( u~_t / T )                                               (6)
```

Of these six lines, exactly two are linear maps: (4), and (3) once `||h||` is held fixed.
(1) is linear but is not a site we intervene at. (2), (5) and (6) are not linear. The whole
chapter is about how much can be said exactly anyway.

### 3.2 A convention warning, because it is a real bug I hit

Equation (3) **folds the gain into `z`**, so the matrix that pairs with `z` in (4) is the
plain `E`. The other valid convention writes `z' = h^(L)/rms(h^(L))` and
`W = E * (1 + gamma)`, giving `u = W z'`.

Both produce **identical logits and identical argmax**. They do **not** produce identical
**distances**, and every budget in this chapter is a distance. Mixing them inflates every
reported `eps` by a factor that varies per coordinate. Fix the convention once and state it,
which is what this paragraph is for. (The one place the other convention is correct is the
stolen-probability check in §5.1, where the effective unembedding genuinely is
`W = E * (1 + gamma)`, because there we are asking about `h` and not about `z`.)

### 3.3 Three facts that make the output layer tractable

These are the load-bearing structural results. Everything downstream rests on them.

**Fact 1: softcapping is argmax-irrelevant.** `s -> c tanh(s/c)` is strictly increasing and
applied coordinatewise, so it **preserves the ordering** of `u_t`. Therefore every statement
about *which token wins* can be made about the raw `u_t = E z_t`, and the softcap can be
ignored. Statements about *probabilities* may not ignore it. This is why the code computes
margins against the uncapped unembedding rows while taking the argmax from the model's own
capped head: the two agree by Fact 1, and that agreement is checked rather than assumed.

**Fact 2: at `T = 0` the decoder is exactly an argmax.** So the map from `z` to the emitted
token is piecewise constant, and its pieces are

```
C_j  =  { z in R^d  :  (E[j] - E[k]) . z  >=  0   for all k != j }         (7)
```

`C_j` is an intersection of `V - 1 = 255,999` halfspaces through the origin, hence a
**convex polyhedral cone**. It is convex because each halfspace is, and a cone because the
constraints are homogeneous, so `z` in `C_j` implies `c z` in `C_j` for every `c > 0`.

**This is the object the meeting asked for: the target set, defined in token space.** It is
not a learned probe, not a heuristic, and not something a readout can dissociate from,
because membership in `C_j` **is** the behavior.

**Fact 3: RMSNorm is degree-zero homogeneous**, meaning `RN_gamma(c h) = RN_gamma(h)` for
`c > 0`. Differentiating (3) gives the Jacobian at the final norm. Write
`z = sqrt(d) * (h/||h||) * (1 + gamma)`, and use `d/dh (h/||h||) = (I - h_hat h_hat^T)/||h||`:

```
A_pre  =  dz / dh^(L)  =  ( sqrt(d) / ||h^(L)|| ) * diag(1 + gamma) * P_perp,

                              P_perp = I - h_hat h_hat^T,
                              h_hat  = h^(L) / ||h^(L)||                    (8)
```

Two consequences, both measurable, and both of which decide an experiment later:

- **`P_perp` annihilates the radial component.** `P_perp h = h - h_hat (h_hat . h) = h - h_hat ||h|| = 0`.
  So any steering vector added before the norm that is aligned with the activation itself is
  **simply deleted**. That is not attenuation, it is exact cancellation, and it is forced by
  degree-zero homogeneity: if scaling `h` does not change `z`, then the derivative in the `h`
  direction must be zero.
- **What survives is contracted** by `sqrt(d)/||h^(L)||`, which for our activations is about
  `48/726 = 0.066`, a sixteenfold contraction, partly undone by the gain `(1 + gamma)`.

The measured net cost of injecting one step earlier is the **`rmsnorm_penalty`, median 4.380
on cities and 5.177 on common_claim**. That number explains an entire null result in §6.1.

**A note on temperature, since your note 6 specified `T = 0`.** For `T > 0` the
piecewise-constant picture softens. Two regimes matter. As `T -> 0+` we recover (7) exactly.
For `T > 0`, `log p_j - log p_k = (u~_j - u~_k)/T`, so the **logit margin `M`** defined in
§3.7 converts directly into a log-odds shift of `M/T`: a perturbation consuming a fraction
`phi` of the margin multiplies the odds of the target against the incumbent by
`exp(phi M / T)`. **That is the formal reason the margin, and not the activation norm, is the
right currency**, and it is why §6.6 reports `frac_margin`. It also means the `T = 0` results
are not a special case that fails to generalize; they are the `T -> 0` limit of a quantity
that stays meaningful.

One caveat on determinism that I have not tested and should: at `T = 0`, determinism is only
as good as the kernels. Batch-invariance failure in RMSNorm, matmul and attention makes
"temperature 0" non-reproducible under **dynamic batching**, so bit-exactness claims require a
fixed batch size or batch-invariant ops.

### 3.4 The certificate, derived

This is the piece you asked for by name: the Jacobian transpose, the dot product with `w`,
and why they give a budget.

**Setup.** Fix an injection **site** `S`, which is a layer index or the input or output of
the final norm. Fix a **readout**: a vector `w` and a threshold `t` such that the property of
interest is `w . z >= t`, where `z` is the activation the readout lives on. Let `A_S` be the
Jacobian of the map from the site to `z`, evaluated at the unperturbed activation.

**Step 1: what a perturbation does to the readout.** Inject `delta` at `S`. To first order
the activation moves by `A_S delta`, so the readout moves by

```
w . ( A_S delta )  =  ( A_S^T w ) . delta                                   (9)
```

That single move of the transpose across the inner product is the whole mechanical content of
the method. It is the definition of the adjoint, `<w, A delta> = <A^T w, delta>`, and it is
what turns a question about a `d x d` matrix into a question about **one vector**. Call
`A_S^T w` the **pullback vector**. It is the only object that needs computing.

**Step 2: the preimage of a halfspace is a halfspace.** The target set is `{z : w . z >= t}`.
Landing in it means `w . (z + A_S delta) >= t`, which by (9) is

```
( A_S^T w ) . delta   >=   t - w . z   =:   g                              (10)
```

So the set of perturbations that reach the target is itself a halfspace in `R^d`, with normal
`A_S^T w` and offset `g`. **This is the entire reason the reframe is tractable**: the
backward-reachable set of a halfspace under a linear map is a halfspace, in closed form, with
no set-propagation machinery, no zonotopes, and no learned value function.

**Step 3: the least-norm point of that halfspace.** By Cauchy-Schwarz,
`(A_S^T w) . delta <= ||A_S^T w|| ||delta||`. Combined with (10), any feasible `delta`
satisfies `||A_S^T w|| ||delta|| >= g`, so

```
||delta||  >=  g / ||A_S^T w||
```

with equality if and only if `delta` is parallel to `A_S^T w`. That gives the three scalars
the whole programme turns on:

```
  g     =  t - w . z            the GAP, distance to travel
  m     =  || A_S^T w ||        the CONTROLLABILITY MARGIN, gain of the site on this readout
  eps*  =  g / m                the CERTIFIED BUDGET
  delta* = (g / m^2) * A_S^T w  the MINIMISER                              (11)
```

Check the minimiser: `(A_S^T w) . delta* = (g/m^2) ||A_S^T w||^2 = (g/m^2) m^2 = g`, so it is
feasible with equality; and `||delta*|| = (g/m^2) m = g/m = eps*`, so it attains the bound.

`eps*` is exactly a **point-to-hyperplane distance**. `m` carries all of the network dynamics
and nothing else about the network enters. `g` carries all of the task.

**The sign convention, fixed once, because mixing the two orientations is a bug I hit.** The
target set is `{z : w . z >= t}` and a statement starts *outside* it, so `g > 0`. The probe
instantiation in §6.0 flips the orientation, because there the target is the FALSE halfspace
`{z : w . z <= t02}`, the `<=` side. Reading that against the convention above means
`w := -w_probe` and `t := -t02`, which is the same thing as leaving `w_probe` alone and
writing `g = w_probe . z - t02` and `{delta : (A_S^T w_probe) . delta <= -g}`. **Both are
correct; only mixing them is not.** `g` is positive in both.

### 3.5 The cost of steering along a direction that is not the optimal one

`eps*` is the budget along `delta*`, the pullback vector. Nobody steers with the pullback
vector; they steer with a named concept direction `u`. Redo Step 3 with `delta = s u` for a
fixed unit `u`. Feasibility (10) becomes `s (A_S^T w) . u >= g`, so the smallest usable
magnitude is

```
eps(u)  =  g / | (A_S^T w) . u |  =  eps* / alpha ,

  alpha(u)  =  | (A_S^T w) . u |  /  || A_S^T w ||    in [0, 1]           (12)
```

`alpha` is just the cosine between the actuator you chose and the optimal one, and `1/alpha`
is the multiplier on the honest budget. **Equation (12) is the whole diagnosis of this
project.** A steering experiment that fixes a budget without reporting `alpha` cannot tell
"the property is not causally actuable" from "we pushed almost perpendicular to the thing
that moves it."

**What to compare `alpha` against, which is not 1.** For two random unit vectors in `R^2304`
the expected absolute cosine is `E|cos| = sqrt(2/(pi d)) = 0.0166`. That is the null. (The
other number that gets quoted, `1/sqrt(d) = 0.0208`, is the RMS cosine, not the mean absolute
cosine. `alpha` is an absolute cosine, so 0.0166 is the correct floor and it is the one used
in `math_map.tex` and in the findings.)

Measured medians, `n = 200` per dataset:

| Direction | `alpha`, cities | vs chance | `alpha`, common_claim | vs chance |
|---|---:|---:|---:|---:|
| `md_full` (final-layer mean-diff truth direction) | 0.00398 | **0.24x** | 0.00940 | **0.57x** |
| `mean_diff_tgt_asis` (the one we steered with) | 0.01325 | 0.80x | 0.01560 | 0.94x |
| `probe_grad_tgt_asis` | 0.01355 | 0.82x | 0.01471 | 0.88x |

**Every truth direction we have sits at or below chance alignment with the direction that
decides the next token.** On cities a Wilcoxon test against the chance floor gives `p < 1e-6`
for all three, so they are significantly *worse* than random; on common_claim two of the three
are statistically indistinguishable from a randomly drawn vector (`p = 0.10` and `p = 0.96`).
Read through (12), this is a **74x to 79x multiplier** on the honest budget for cities.

Note carefully what kind of statement this is. **The certificate was never wrong about the
budget. We were spending it almost perpendicular to the thing that moves the token.**

### 3.6 Instantiation 1: the probe halfspace (what the audit ran)

Site `S` is the input of decoder layer `l_src`; the readout lives at layer `l_tgt`; `w` is the
contrastive mean-difference direction

```
w  =  unit( mean of h^(tgt) over label 1  -  mean of h^(tgt) over label 0 )
```

and `t = t02` is obtained by fitting a one-dimensional logistic regression of the label on the
score `s = w . h^(tgt)` and solving for the score at which `P(y=1) = 0.2`:

```
sigmoid(a s + b) = 0.2   =>   a s + b = log(0.2/0.8) = -1.3863
                         =>   t02 = ( -1.3863 - b ) / a                    (13)
```

Here `A_S = J`, the layer-to-layer Jacobian of the block stack, and `m = ||J^T w||`. Concretely
`src/reach_hop.py` defines `F(delta)` as the target-layer activation you get when you add
`delta` at the source layer, broadcast at every position, and run the remaining blocks;
`J = dF/d(delta)` at `delta = 0`.

**Why this instantiation was always going to be structurally fragile, which I did not
appreciate before running it.** `w` is **fitted**, so `w . z >= t` is a statement about a
*readout*, and nothing forces a readout to be the thing the rest of the network consumes. We
measured exactly that: readout displacement tracked the certificate to `R² = 0.999` while
behavior did not move at all. That gap is also not new (Elazar et al. 2021, LEACE §5.3,
AxBench 2025 on this exact model), which is why the contribution has to be the certificate and
not the phenomenon.

**Local linearity is measured, not assumed.** `F(delta) ~= J delta + F(0)` is the modelling
step, and Phase 5 and Horizon 0.1 test it directly (§6.0). It holds far better than expected.
The important qualification, which cost me a wrong reading once: the measured `R² = 0.999`
licenses **linearity** of readout against scale, **not gain**. The same measurement found the
realized slope 8x to 35x below the predicted `||J^T w||`. Linearity says the response is a
straight line; it does not say we predicted its slope.

### 3.7 Instantiation 2: the argmax cone (what your note 6 proposed)

Now take the readout to be a face of the cone (7). Let `j_top` be the current argmax at the
position that chooses the next word, and `j_tgt` the token we want instead. Put

```
a  =  E[j_tgt] - E[j_top]

M  =  - a . z  =  u_{j_top} - u_{j_tgt}   >  0                             (14)
```

Then "the model emits `j_tgt` rather than `j_top`" is **exactly** `a . z >= 0`, and (11)
applies verbatim with

```
w := a ,   t := 0 ,   g = M ,   m = || A_S^T a || ,

     eps*_token  =  M / || A_S^T a ||                                      (15)
```

**Same formula, same code, same single VJP.** The only thing that changed is that `a` is not
fitted: it is read off the unembedding. So `a . z >= 0` is not a proxy for the behavior, it is
a restatement of it, and **the readout-versus-behavior dissociation becomes impossible by
construction.** That single substitution is what your note 6 bought.

Note also that `g` is now not an abstract distance to a fitted threshold. It is literally the
**logit margin** of the incumbent token over the target token, a quantity that was sitting in
the model's own output the whole time and that we had never computed.

### 3.8 From one face to the whole cone, and the QP

Equation (15) certifies crossing **one** face. Making `j_tgt` the true argmax requires all of
(7). After perturbing, we need `(E[j_tgt] - E[k]) . (z + A_S delta) >= 0` for every `k`, which
rearranges to a linear constraint in `delta`:

```
delta*  =  argmin ||delta||   subject to

  ( A_S^T ( E[j_tgt] - E[k] ) ) . delta   >=   ( E[k] - E[j_tgt] ) . z     for all k   (16)
```

Note the structure: **only the constraint normals depend on the site; the offsets are logit
gaps either way.** So moving the injection site changes the left-hand sides and leaves the
right-hand sides alone.

With `V = 256,000` constraints this looks impossible, but **the active set is tiny**. Solve the
dual on the faces seen so far, ask the full vocabulary who wins, add the violator, repeat. The
dual of (16) is

```
min over lambda >= 0 :   (1/2) lambda^T G lambda  -  b^T lambda ,
     G = A A^T ,   delta = A^T lambda                                       (17)
```

with `A` the matrix of active constraint normals, which is a few microseconds. Measured on
gemma-2-2b, **the active set has 1 to 3 faces**, and `||delta*||` exceeds the single-face bound
`M / ||A_S^T a||` by about **8.8%** (the measured ratio is 1.088). So the single face is a
good but not exact proxy, and the full solve is worth doing.

**Every solution is then verified by an explicit full-vocabulary argmax check**, so a reported
certificate is a proof rather than an optimizer's opinion. This is what makes the `oracle` arm
a harness assertion rather than a competitor: at the post-norm site, with `delta = delta*`, the
token flip is a **theorem**. If the running model does not emit `j_tgt`, the injection code is
wrong.

### 3.9 Is the target set even non-empty?

`C_j` can be empty, and if the false-answer tokens had empty cones the whole reformulation
would be dead before it started. By Demeter et al. (ACL 2020) a token is emittable for *some*
`z` if and only if its unembedding row is a **vertex of the convex hull** of all rows. By
minimax on a compact convex set,

```
max over ||z|| <= 1 of  min over k != j of  ( E[j] - E[k] ) . z

   =  min over lambda in the simplex of  || sum_k lambda_k ( E[j] - E[k] ) ||

   =  dist( E[j] ,  conv{ E[k] : k != j } )                                (18)
```

where the middle expression simplifies because `sum_k lambda_k (E[j] - E[k]) = E[j] - sum_k lambda_k E[k]`.

**This is exact in both directions, which is why it is a proof and not an estimate.** A
positive value is proved by exhibiting a **witness `z`** whose full-vocabulary argmax is `j`; a
zero value is proved by exhibiting the **convex weights** reconstructing `E[j]` from the other
rows. Solved by Frank-Wolfe, batched so one GEMM `W P^T` per iteration serves all candidates.

Result in §5.1: all 182 tested tokens are achievable. And `d = 2304` is well above the regime
where unargmaxability bites (Grivas et al. 2022 report the effect is rare above about
`d = 200`).

**The caveat that must travel with it.** (18) clears a **necessary condition only**. It says
the cone is non-empty, not that it is reachable from where the model actually sits under a
bounded perturbation. That is precisely (15), and measuring it is the main run.

### 3.10 Instantiating the site: three `A_S`, and what a failure at each one indicts

| Site `S` | `A_S` | `A_S^T a` | what a failure indicts |
|---|---|---|---|
| post-norm (`z`) | `I` | `a` | the direction, and nothing else |
| pre-norm (`h^(L)`) | equation (8) | `(sqrt(d)/&#124;&#124;h&#124;&#124;) P_perp diag(1+gamma) a` | the direction, or RMSNorm |
| layer `l` | `J_(l -> z)` | one VJP | the direction, or the dynamics |

The pre-norm adjoint is the transpose of (8), and both `P_perp` and `diag(1+gamma)` are
symmetric, so the transpose just reverses their order. That adjoint pair is verified
numerically against a finite-difference Jacobian in the test suite.

Measured medians of `m = ||A_S^T w||`, `n = 200` per dataset:

| Site | `m` median (cities) | `m` median (common_claim) |
|---|---:|---:|
| post-norm | 2.383 | 2.138 |
| pre-norm | 0.532 | 0.432 |
| layer `l` | 0.607 (L16) | 0.676 (L8) |

**The post-norm site is the decisive one, and it is exactly the naive experiment your note 1
asked for.** There the map from perturbation to logits is *exactly* linear:
`u(z + delta) = E z + E delta`. Nothing nonlinear remains to blame. That gives a clean
trichotomy, written down before the run:

| observation at the post-norm site | conclusion |
|---|---|
| target token appears, text stays fluent | the last layer is fine; the hop broke it |
| text degenerates | the linear-feature assumption fails at the last layer |
| nothing changes at `eps = eps*` | **the harness is broken** |
| nothing changes at the budget we used before | the budget never bought a flip |

§6.1 and §6.2 report which rows fired.

For the probe instantiation, the sites were inherited from the earlier DCT work and recorded in
`dct_meta_<ds>.json`: `cities` layer **11 to 20** with `input_scale` 47.72, `common_claim`
layer **13 to 22** with `input_scale` 86.73. `input_scale` is the activation's own norm
yardstick, produced by DCT's calibrator, and every budget should be read against it. A budget
of 2.8 on cities means a nudge of about 6% of the activation's own size.

### 3.11 The broadcast operator, which had been prose and is now an operator

The site table writes `A_S` as though a perturbation were a single vector in `R^d`. **It is
not.** Every certificate we have computed added `delta` at **every position** of the prompt.
That is a different linear map and it deserves its own symbol.

Let `B : R^d -> R^(T x d)` be the broadcast, `B delta = 1_T (x) delta`. The map actually being
inverted is `A_S . B`, not `A_S`. Because `B` is a sum of coordinate injections, its adjoint
**sums the per-position gradients**:

```
( A_S B )^T a  =  B^T A_S^T a  =  sum over t of  d( a . z ) / d h^(l)_t     (19)
```

versus the single-position convention `d( a . z ) / d h^(l)_last`. The adjoint identity is one
line: `<B delta, Y> = sum_t delta . Y_t = delta . (sum_t Y_t) = <delta, B^T Y>`.

Equation (19) is literally `g.sum(dim=1)` at [`src/token_jac.py:82`](../src/token_jac.py) and
[`src/reach_jlens.py:97`](../src/reach_jlens.py); the right-hand object is `m_last`. Their
ratio is the **broadcast gain**

```
rho(l)  =  median over statements of
             || sum_t  d r_i / d h^(l)_t ||  /  || d r_i / d h^(l)_last ||  (20)
```

measured at **1.355** for cities layer 16 and **2.016** for common_claim layer 8.

Two consequences. The broadcast site has the larger margin, so **a budget quoted under
broadcast is easier to meet than the same number quoted per position**, and `m_last` is the
honest per-position figure. And **ActAdd-style single-position injection instantiates the
right-hand map, not the left-hand one**, so comparing our `eps` to a published ActAdd `eps`
without dividing through by `rho` compares two different operators. That is directly relevant
to your note 8.

The "we inject `eps sqrt(T)` of energy, not `eps`" remark I made when I found the defect is a
*consequence* of `B`, not a substitute for writing `B` down. Writing it down is what let me
predict `rho` in advance and then compare it to the realized 2.5x in §6.3.

### 3.12 The cross-layer convention behind the `_asis` cosines

`alpha` in (12) is a cosine, so both arguments must live in the same space. **For the
directions labelled `_asis` they do not**, and the label exists to say so.

Those directions are read verbatim from `reach_dirs_<ds>.npz`, where they were fitted at the
**target layer** `h^(l_tgt)`. The `alpha` reported in §3.5 takes their cosine against `a` in
**post-norm coordinates** `z`, with **no transport between the two**
(`src/token_geom.py:395-400`). So `alpha_mean_diff_tgt_asis = 0.01325` is the cosine between
`a` and *the raw coordinate vector we actually injected*.

That is the operationally correct quantity, because it answers what the direction we really
steered with bought us, and it is **not** a coordinate-free statement about the concept. The
basis-correct version would transport the target-layer direction forward first:

```
u_tr        =  J_(l_tgt -> z) u  /  || J_(l_tgt -> z) u ||

alpha_tr(u) =  | a . u_tr |  /  || a ||                                     (21)
```

which costs one JVP per statement and **has not been computed**. Until it is, every `_asis`
number in this document carries this paragraph. This is the one quantity your note 7 leaves
open, and it is a missing measurement rather than a missing object.

### 3.13 When is a norm budget the right currency? (your note 4, formalised)

Everything above prices a perturbation by its Euclidean norm. **That is a modelling choice**,
and the meeting challenged it directly: small perturbations in different directions can have
vastly different chains. Stated precisely, define for a site `S`, a unit direction `u` and a
budget `eps` the **realized gain** and the **realized slope**

```
G_S(u, eps)      =  || z(eps u) - z(0) ||  /  eps

sigma_S(u, eps)  =  a . ( z(eps u) - z(0) )  /  eps                        (22)
```

where `z(.)` is the post-final-norm activation the perturbed forward pass **actually
produces**. These are measured, not predicted: **no Jacobian appears in (22)**, which is what
makes them an independent check on everything else in this chapter.

The objection bundles two different claims and they separate cleanly.

**Anisotropy: fix `eps`, vary `u`.** If `G_S` is constant in `u` the site is isotropic and
`eps` is a fair price. The statistic is the spread `G^p90 / G^p10` over directions. But note
that **anisotropy alone is not an obstruction**: it is exactly what `alpha` already measures at
the post-norm site, and `eps(u) = eps*/alpha` from (12) is precisely the rescaling that repairs
it. An anisotropic-but-linear site needs a **change of units, not a change of theory.**

**Nonlinearity: fix `u`, vary `eps`.** For a linear `A_S`, `G_S(u, eps) = ||A_S u||` with **no
dependence on `eps` whatsoever**. So

```
kappa_S(u)  =  G_S(u, eps_max) / G_S(u, eps_min)                           (23)
```

is a **pure nonlinearity reading**, and `kappa = 1` is the exact statement that the first-order
certificate is valid over the whole budget range. **This is the claim that would actually
threaten the programme**: if `kappa` were far from 1 and varied with `u`, then no fixed budget,
rescaled or not, certifies anything, and the certificate would have to be restated as a bound
on realized gain rather than on `eps`. §6.4 reports the measurement.

**One more harness assertion falls out of this for free.** At the post-norm site `A_S = I`, so
(22) collapses to `G = 1` and `sigma = a . u` identically, for every direction and every budget.
That is **arithmetic**, so a post-norm run that does not reproduce it is measuring something
other than what it claims. Measured: 1.0000 everywhere, 13 directions, 4 budgets, both datasets.

### 3.14 Why all of this is affordable

The reason a per-statement certificate over hundreds of statements is possible at all is (9).
`A_S^T w` is a **vector-Jacobian product**: one backward pass of the scalar `w . z`. Forming
the full `2304 x 2304` Jacobian instead would cost `d` passes, about three orders of magnitude
more, per statement.

So the cost structure of the whole programme is:

| what | cost | where it is used |
|---|---|---|
| `A_S^T w` for one statement | 1 VJP (one backward pass) | every certificate, all 200 statements, all 26 layers |
| full `J` for one statement | `d` passes | Phase 2 only, hence 32 statements not 200 |
| the cone QP (17) | microseconds, CPU | every token-space certificate |
| the hull check (18) | Frank-Wolfe, batched, CPU | once, 182 tokens |

That is why the layer sweep in §6.8 could afford all 26 layers on 200 statements under both
injection conventions, and why Phase 2's SVD is the one place the sample drops to 32.

---

## 4. Chapter 3: designing the experiment, decision by decision

This section is the "why did you do it that way" answer for every load-bearing choice. Where
a decision turned out wrong I say so.

### 4.1 Model, datasets, and sample sizes

**Model: `google/gemma-2-2b`, fp32.** Chosen to match your A-LQR paper (arXiv:2604.19018) so
that anything we find is directly comparable to it, and because it is the best-covered model
in GemmaScope, which the SAE work in §5.2 needs. 26 blocks, `d = 2304`, about 9.7 GB across
three shards.

**Datasets: a deliberate clean-to-messy gradient.** `cities` (1,496 rigid-template
statements) and `common_claim_true_false` (4,450 heterogeneous world claims). The whole point
of the pair is that they fail by *different* mechanisms, which §2.3 confirmed empirically.

**They are never pooled, anywhere.** This is a rule, not a preference. `cities` uses a
semantically false country as its target; `common_claim` uses the runner-up token, so on
`common_claim` generic disruption can land on target by accident. **`cities` is the
discriminating dataset and leads every comparison in this document.**

**`n = 200` statements per dataset.** The binding constraint is that Phase 1 computes a
per-statement VJP and Phase 2 a full `2304 x 2304` Jacobian. 200 is where per-statement
certificates stay affordable; Phase 2 drops to 32 for the same reason. A known consequence:
on `common_claim` the legacy direction covers only **90 of 200** statements, restricted to 44
at label 1, because `reach_margins` capped that dataset at 2,000 of its 4,450 rows. Every
common_claim cross-direction number in this document is on that restricted set and I flag it
each time.

### 4.2 The direction battery, and why it is padded with controls

Phase 1 does not compute the margin for the truth direction alone. It computes it for four
groups, and the groups are the experiment:

| Group | Members | Purpose |
|---|---|---|
| `truth` | `mean_diff_tgt`, `probe_grad_tgt` | the readouts we care about |
| `truth_sub` | 8 directions spanning the truth subspace | best case within the concept |
| `dct_u` | DCT's top-4 target-layer effect directions | a known-causal comparison group |
| `rand` | 64 random unit directions | the null |

Without the `rand` group, "the truth margin is 9.07" means nothing. With it, "9.07 against a
random null of 1.10" is a claim. Every headline number in this project is a ratio against a
control that was designed in before the measurement, and where I could not build one in I say
so.

### 4.3 The token-space design, and the four directions

For the token-space program the same skeleton is reused with `w := a_i`, `t := 0`. Four
directions are swept side by side in the same arm, at the same budgets, on the same
statements:

| name | what it is | role |
|---|---|---|
| `oracle` | the least-norm displacement into the target token's argmax cone, solved as a QP and verified in closed form against all 256,000 vocabulary entries | **the harness assertion.** Its flip is a theorem, so it is not a competitor. If it does not fire, the injection code is wrong |
| `md_full` | the final-layer contrastive mean-difference truth direction | the naive baseline your note 1 asks about |
| `jtw_legacy` | the old probe-halfspace pullback `J^T w` | continuity with the audit |
| `jtw_token` | the token-space pullback `J^T a` | the proposal under test |

**The `oracle` arm is the single best design decision in this program**, and it is your note
3 turned into something that stays checked. At the post-norm site the flip is provable, so
any arm where `oracle` fails is an arm whose numbers are meaningless. It fires on 200 of 200
statements on both datasets in every post-norm arm.

**The budget unit is `frac`, not raw scale.** `frac` is the fraction of the *certified*
budget spent, so `frac = 1.0` is by construction the displacement predicted to consume
exactly one logit margin. This matters because in a per-statement arm every statement has its
own `eps_i`, so raw scales are all distinct and grouping by scale would give buckets of size
one everywhere. `frac_margin` is then the fraction of the margin actually consumed, measured.

A detail I want on record because it prevents a specific kind of self-deception: the CSVs
carry a second column, `readout_delta`, which is the same measurement rescaled per statement.
It is **never reported beside `frac_margin`**, because printing both would show one number
twice and look like corroboration. A duplicate-column detector runs over all twelve arms and
reads True on every one, confirming the CSVs were not regenerated.

### 4.4 Two code defects found by reading our own code, and what they cost

Before running anything I did the "check the steering vector is what you want it to be"
homework on our existing code. Two things came out of it, and both are confounds for the
*entire* pre-meeting negative result.

**Defect A: the perturbation was broadcast to every position, including BOS.**
`dct_steer_utils.Steerer._hook` adds a `(d,)` vector to a `(batch, seq, d)` tensor, so
broadcasting gives every token position the same `delta`, including BOS, which gemma-2 uses
as a strong attention sink. ActAdd adds at a small span of positions; CAA spreads across
tokens. Two things follow: the *total* injected energy is `eps * sqrt(T)`, not `eps`, so the
"we stayed inside the trusted budget" claim may not even be in the right units; and
corrupting the attention sink is a plausible, cheap-to-test source of the incoherence we had
been attributing to over-steering. **Fix: a `--positions all|last` flag, and both conventions
run side by side.** Measured cost in §6.3: worth 2.5x on the realized hit rate.

**Defect B: `repetition_penalty = 1.3` was on, in a truth experiment, at 8 tokens.** At a
nonzero repetition penalty the decoder is not a pure argmax, so **the cone geometry of the
entire token-space argument does not literally hold for the runs we already had**. It also
penalizes tokens appearing in the prompt, which for "The city of X is in" includes the
obvious continuations. There is no defensible reason for it here. **Fix: default 1.0, and a
`--rep-penalty` flag so the old convention runs as its own arm for comparison.** Measured
cost in §6.3: it changed what we *saw* downstream and not the token we were *measuring*.

I want to flag that finding these was the highest-value hour in the window. Both were
candidate explanations for the whole negative result, and both cost one ablation each to rule
out.

### 4.5 The test suite as a pre-submission gate

The suite is now **381 passed, 1 skipped, in about 6 seconds**, up from 182 at the meeting.
That speed is the point: it runs before every cluster submission, and it is the only cheap
defense against burning an 8-hour GPU job on a typo.

31 of those tests are your note 3 turned into permanent assertions, with no model and no GPU:
the RMSNorm site adjoint and its radial annihilation, the cone certificate verified against a
full synthetic vocabulary at three sizes, **which layer each hook actually fires on**, whether
`positions=last` touches only the last row, and the `sqrt(T)` energy the broadcast convention
spends without reporting it.

Of the window's commits, 24 are `fix(...)` landed during review passes, and several would
have produced *wrong conclusions* rather than crashes: a baseline-relative crossing definition
instead of halfspace membership, excluding never-steered statements from the per-statement
baseline, requiring same-`frac` co-occurrence before declaring actuation, gating every verdict
cell on a minimum bucket size, and threading the model id so the SAE decomposition cannot
silently use the wrong dictionary.

---

## 5. Chapter 4: the cluster runs

Account `bhhv-dtai-gh`, partition `ghx4`, NCSA DeltaAI GH200, ARM64. About **475 GPU-hr
remain** of the allocation. For scale: a 64-factor DCT smoke fit takes about 90 minutes on
CPU and about 4 seconds on a GH200, roughly **800x**, which is why the project lives there.

**The environment landmine, worth knowing because it silently invalidated a run once.**
`dct.SlicedModel` divides gemma-2 inputs by `sqrt(d)`, expecting the HF model to re-multiply
`inputs_embeds` by its normalizer. **transformers >= 5 no longer does**, which silently made
the hop map unrelated to the real forward pass, cosine 0.04 to 0.23. Every hop job now prints
a startup fidelity probe and dies with `SlicedModel unfaithful` rather than producing
plausible garbage. The cluster's env is pinned to transformers 4.51.3 and is faithful with
factor 1; the audit's margins were computed there and stand.

### 5.1 The free checks that ran first, on CPU, before any GPU spend

Both of these could have killed a downstream program, and both cost nothing.

**Stolen probability (2026-08-03).** There is a known structural result that tokens whose
output embedding lies in the *interior* of the convex hull of all embeddings can **never** be
the argmax for any hidden state. If the false-answer tokens were interior points, the
token-space target set would be literally empty and the whole proposal would be dead before
it started. The test is exact in both directions by minimax on a compact convex set:
achievability of token `j` is precisely the statement that `E[j]` is a *vertex* of the convex
hull of the unembedding rows.

**Result: 182 of 182 candidate tokens ACHIEVABLE, zero non-achievable.** All 102 distinct
country first-tokens, 64 random controls, and the 16 smallest-norm rows in the entire 256,000
vocabulary. 97 were resolved immediately by a self-witness; the other 85 by Frank-Wolfe, whose
distances plateaued between 1.196 and 4.643 and never approached zero. **Every one passed an
explicit full-vocabulary witness check, so this is a proof rather than an estimate.**

A side finding worth a sentence in the paper: **embedding norm is a poor proxy for
interiority in gemma-2-2b.** The 16 lowest-norm rows in the whole vocabulary are all
achievable. If we had relied on the literature's norm-based heuristic we would have concluded
the opposite.

The caveat that must travel with it: this shows the target set is non-empty in the
*unconstrained* sense. It does **not** show such an activation is reachable under a bounded
perturbation. That is exactly `eps*_token`, and measuring it is the main run.

**SAE feature forensics (2026-08-03).** Covered in §6.5, also free, also CPU.

### 5.2 The runs, and what each produced

| Round | Jobs | Date | What it produced |
|---|---|---|---|
| Reachability audit, Phases 1 to 5 | `run_reach_margins`, `run_reach_svd`, `run_reach_steer`, `run_reach_judge`, `run_reach_linerr` | 2026-07-24 | `reach_acts_*.npz`, `reach_dirs_*.npz`, `reach_margins_*.npz`, `reach_svd_*/`, `judge_reach_*.csv`, `reach_linerr_*.csv` |
| Horizon 0 validations | `run_reach_samepoint`, `run_reach_h0` | 2026-07-29 | `reach_samepoint_summary_*.csv`, stem-probe, stem-Jacobian and Newton outputs |
| Token-space program | `run_token_geom`, `run_token_jac`, `run_token_sens`, `run_token_steer` | 2026-08-04 | `token_geom_*.{csv,npz}`, `token_acts_*.npz`, `token_jac_*.csv`, `token_sens_*.csv`, twelve `token_steer_*.csv` arms |

The token-space round was budgeted at about **8 GPU-hr of wall-clock caps against 475
remaining**, in four jobs, with **no judge job at all**. That is the big saving over the
previous design and it is a direct consequence of your note 6: the outcome variable is
`hit_target`, a mechanical comparison of the next-token argmax against a target token id.
No LLM verdict is needed to score it. The judge is only required for the free-text
`completion` column, which is optional and last.

**Built and not run, which I need a decision on:**

- **The refusal positive control (four SLURM jobs, gated in sequence).** This is the
  publication gate for the *probe-halfspace* result and §7.2 explains why. The dataset is
  built (`refusal.csv`, 976 balanced rows; `refusal_holdout.csv`, 64 rows that never enter
  direction fitting). Nothing has been submitted.
- **The 2026-07-23 conditional-steering and U-anchor round.** Six feature commits, a design
  spec and an implementation plan, written the day before the reachability pivot and never
  submitted. Arm B is conceptually the *scalar* version of the reachability idea, so it is not
  junk, but it is superseded. **My read: retire it and say so in the writeup**, keeping the
  code as the scalar precursor of the set-based method. I would rather you overrule me than
  leave it in limbo.

---

## 6. Chapter 5: the results, one note at a time

The audit ran first and produced the null. The token-space program ran second and fixed it. I
present them interleaved under your eight notes, because that is the order you will want to
check them in.

### 6.0 First, the audit, because everything else is a response to it

Five phases, run 2026-07-24, on the probe-halfspace target set. Full detail in
[`REACH_AUDIT_FINDINGS.md`](REACH_AUDIT_FINDINGS.md).

**Phase 1: are the margins real?** Yes, and on paper the target set is reachable.

| | cities | common_claim |
|---|---:|---:|
| median margin `&#124;&#124;J^T w&#124;&#124;` for the truth readout | **9.07** | **7.00** |
| random-null median margin | 1.10 | 1.34 |
| ratio truth / null | **8.3x** | **5.2x** |
| median `eps*` | **2.80** | **10.69** |
| fraction reachable at `input_scale` | 1.00 | 1.00 |

The truth readout is roughly an order of magnitude more controllable than a random direction,
and the budget to flip it is a small fraction of the activation's own norm. Figures:
[`plot_reach_margins_cities.png`](../plot_reach_margins_cities.png) (violins, truth visibly
above the random null; that separation *is* the verdict) and
[`plot_reach_curves_cities.png`](../plot_reach_curves_cities.png) (fraction reachable against
budget, with `input_scale` marked and a gray band for "beyond linear validity").

**Phase 2: is the hop's high-gain channel the concept channel?** No, and this is the first
real crack.

| | cities | common_claim |
|---|---|---|
| effective rank (of 2304) | 396 to 551 (17 to 24%) | 447 to 616 (19 to 27%) |
| overlap of top-16 right singulars with DCT's V | **0.053 to 0.076** | **0.039 to 0.059** |

The pre-registered stop-and-diagnose threshold on that overlap was **0.3**. Both datasets
fire it at about 0.05, essentially random. The hop routes through a compact channel of roughly
500 effective dimensions, and that channel's **input** has almost nothing to do with any
concept direction we can name. Meanwhile the truth readout at the *target* layer has 74%
(cities) and 51% (common_claim) of its energy inside the top-64 *left* singulars, 17x to 125x
what a random direction gets. **The truth readout is downstream-visible but not
upstream-addressable.** Figure:
[`plot_reach_svd_cities.png`](../plot_reach_svd_cities.png), where that asymmetry is the whole
point.

**Phase 3: does any of this move behavior?** No.

| | cities | common_claim |
|---|---:|---:|
| lie rate at the strongest toward-FALSE push | **3.5%** | **5.5%** |
| unsteered baseline lie rate | ~3% | ~3% |

Even at the strongest push the completions stay coherent and *true*: "The city of Busan is in
South Korea." Figures:
[`plot_judge_reach_stmt_cities.png`](../plot_judge_reach_stmt_cities.png) and its mean-arm
sibling. **The flatness of those lines is the headline null of the whole project.**

**Phase 4: were we steering at the right depth?** No. Controllability is front-loaded: the
peak is at layer 0 for almost every readout and mode, and by our source layers **42% to 73%
of the available controllability is already gone**, with no late resurgence. Worse, the
**verdict** readout (the yes-minus-no unembedding difference, which is what the judge actually
elicits) has a margin **2.1x to 5.7x smaller** than the truth probe at the same layer. We were
pushing the biggest lever we could measure and it was not the lever attached to behavior.
Figure: [`plot_reach_jlens_cities.png`](../plot_reach_jlens_cities.png).

**Phase 5: was the certificate inside its own validity region?** For cities, yes; for
common_claim, no. `eps20` is the perturbation norm at which the first-order prediction is off
by 20%.

| | cities | common_claim |
|---|---:|---:|
| `eps20` along the certificate's own direction | 2.99 | 4.87 |
| `eps*` from Phase 1 | 2.80 | 10.69 |
| ratio | **0.94x, inside** | **2.20x, outside** |

**Cities is the scientifically interesting case: the math is right and behavior still does not
move.** There is a second, quieter finding in that sweep: `J^T w` has by far the *smallest*
trust radius of any direction tested, 4 to 5x smaller than a random direction. The direction
the certificate points along is the most curvature-sensitive direction in the space.

**The four mechanisms (D1 to D4).** After Phase 3 came back null, four parallel deep-dives
asked why. The refined headline is that the certificate fails behaviorally **not because
linearization breaks** (R² stays at 0.99) but because of three **transfer failures**: context
transfer (Jacobian gain collapses across a one-word context shift while sign and linearity
survive), population transfer (the probe's threshold does not transfer from statement tokens
to generation prefixes), and input-side concept misalignment (D3 above). And behavioral
validation *inherently forces* the first two, because behavior only exists during generation,
which is never the point where the certificate is computed.

**Horizon 0 then closed the audit's own loopholes**, seven validations run 2026-07-29. The
three that matter:

*Same-point control, the single most important number in the window.* Is the certificate
right at its own linearization point, or is the math simply miscalibrated? Teacher-force the
full statement with the hook active and read the readout at exactly the token where `eps*` was
computed.

| | cities (n=199) | common_claim (n=197) |
|---|---:|---:|
| **calibration factor** (realized slope / predicted, same point) | **0.9997**, IQR [0.9910, 1.0117] | 0.7025 |
| **context factor** (stem slope / same-point slope) | **0.0729** | 0.1944 |
| implied collapse across one word of context | **13.7x** | 5.1x |

On cities the certificate is **exact to four significant figures at its own point**. The
entire loss is the context shift. This converts the result from "our math might be wrong" into
"the math is exact and the transfer fails", which is a completely different and much stronger
claim.

*Probe refit at generation prefixes, the strongest supporting result.* Label each generation
stem for free using the scale-0 judge verdicts already in hand, then try four ways to read
truth off it.

| | cities | common_claim |
|---|---:|---:|
| base rate P(TRUE) | 0.943 | 0.944 |
| old-threshold accuracy | 0.057 | 0.944 |
| old-threshold **balanced** accuracy | **0.500** | **0.500** |
| median `g` for TRUE stems / FALSE stems | -52.6 / -52.0 | +82.0 / +79.3 |
| recalibrated 1-D threshold | 0.943 (= base rate) | 0.944 (= base rate) |
| refit LR, 5-fold CV | 0.930 | 0.944 |
| XGBoost, 5-fold CV | 0.943 (gap **+0.013**) | 0.944 (gap **+0.000**) |

Read the two accuracy rows together. On cities the old threshold calls almost everything
FALSE; on common_claim almost everything TRUE. Both look different and both have **balanced
accuracy exactly 0.500**. The TRUE and FALSE medians are within 1 to 3 units of each other on
a scale where the threshold sits 100+ units away. Recalibrating recovers nothing beyond the
base rate. Refitting recovers nothing. **XGBoost recovers nothing**, which rules out the
obvious "the boundary is just nonlinear here" objection. **This is not a threshold shift, it
is an absence of signal**, and I have not found it claimed in the literature.

*Newton re-steering, a negative control designed to falsify the diagnosis.* If the gain
collapse were caused by curvature, recomputing `J^T w` at the steered state and iterating
should fix it. Three iterations on 32 statements: median `|g|` goes 20.77, 3.30, 0.071, 0.0,
finishing at **4.65e-05**, with **0 of 32** hitting the step cap and a final direction at
cosine **0.971** to where we started. **Curvature is not the obstacle.** The "context, not
curvature" claim is confirmed by a control built to break it.

Now the eight notes.

### 6.1 Note 1: naive output-layer steering. **CONFIRMED**

You predicted that a no-Jacobian steer at the output would reproduce the failure and mostly
produce incoherence. It did, exactly.

`cities`, post-norm site, repetition penalty 1.0, at `frac = 1.0` (cross-tab rows are label 1,
n = 101; margin and coherence columns are the full 200):

| direction | flips | median `&#124;frac_margin&#124;` | degenerate | edit ratio |
|---|---:|---:|---:|---:|
| `oracle` | **101 / 101** | 1.0010 | 0.035 | 0.568 |
| `md_full` | 1 / 101 | 1.0877 | **0.885** | 0.862 |
| `jtw_legacy` | 0 / 101 | 0.0174 | 0.000 | 0.057 |

Country outcome on the same arm, against an unsteered baseline of 0.515 correct and 0.010
target: `md_full` at `frac = 1.0` gives **0.015 correct, 0.985 none, 0.000 target**. At
`frac = -2.0` its degenerate fraction reaches 0.980, and **740 of its 1,800 completions in
that arm are empty**: the model emits a newline and stops.

"Mostly making it incoherent" turned out to be literal. The text stops existing. And because
the same site, same hook and same budget unit with `oracle` flips every statement at a
degeneracy of 0.035, **the incoherence is a property of the direction, not of steering at the
last layer.** That is exactly the isolation you said this control would provide.

**One arm in this batch is a null for an arithmetic reason and I want to be upfront that it is
uninformative rather than negative.** The pre-norm arm reads 0.000 for every direction
including `oracle`. A displacement certified *after* the norm has to survive the norm, which
costs a median `rmsnorm_penalty` of **4.380** on cities and **5.177** on common_claim, and the
sweep stopped at `frac = 2.0`. It was never given enough budget. §8 turns this into a cheap
falsifiable prediction.

### 6.2 Note 2: does incoherence indict the linear feature at the final layer? **REFUTED**

I pre-registered the expectation of a refutation in the raw extraction dump *before* running
the systematic measurement, and the measurement agreed. Recording that the expectation was set
first.

The decisive comparison needs no new arm. It is three directions inside a single arm, at a
single budget, at the one site where the linearity question has an exact answer:
`u(z + d) = Ez + Ed`, up to a monotone softcap that preserves the argmax and therefore the
cone.

**The sharpest fact in the dump:** on cities at `frac = 1.0`, `md_full` consumes a median
**1.0877** of the logit margin, *more* than `oracle`'s **1.0010**, and flips **1** of 101
statements against `oracle`'s **101**.

Consuming the margin along the target face is **necessary and not sufficient**. `md_full` has
to spend roughly `1/alpha` of the optimal norm to buy that margin, and a displacement that
large moves the other 255,999 logits too, so a newline wins the argmax instead of the target
country.

If incoherence indicted "the linear feature in the final layer", then no linear intervention
at that layer should produce a clean targeted flip. One does, on every statement, in the same
arm, at the same budget scale. **So the incoherence indicts the direction, not the layer.**
`md_full`'s median alignment with the token decision is `alpha = 0.00398`, below the 0.0166 a
random unit vector would score. The final layer's linear structure is the one part of the
pipeline that is provably exactly linear.

### 6.3 Note 3: is the steering vector what we think it is? **ANSWERED, with a defect report**

The honest answer has three parts, and one of them is a confession.

**Check 1, the oracle assertion.** At the post-norm site the flip is a theorem verified in
closed form against the full 256,000-token vocabulary. If the running model does not emit the
target, the injection code is wrong. It emits it on **200/200** on cities and **200/200** on
common_claim, in every post-norm arm including the repetition-penalty 1.3 arm.

**Check 2, the arithmetic assertion.** At the post-norm site `A_S = I`, so the realized gain
must be exactly 1 for every direction at every budget. Measured across 13 directions at 4
budgets spanning `frac` 0.01 to 4.0 on 40 statements: **1.0000 everywhere, both datasets.**

**Check 3, the two defects, now quantified rather than argued.**

*Repetition penalty 1.3 changed what we saw downstream and not the token we were measuring.*
cities post-norm `md_full` at `frac = 1.0`: degenerate **0.885** at rp 1.0 against **0.000** at
rp 1.3, and empty completions **740** against **20** across the arm. But the flip counts are
**identical cell by cell** between the two arms, and the `oracle` country outcome barely moves.
The old runs' incoherence readings were inflated by the decoder; their flip counts were not.

*Broadcasting to every position was worth about 2.5x on cities.* At layer 16, `jtw_token` hits
**0.340** broadcast against **0.135** at the last position only at `frac = 1.0`, and 0.675
against 0.285 at `frac = 2.0`. On common_claim at layer 8 the same comparison is flat, 0.811
against 0.800, and the mean stem length explains the difference: 6.32 words on cities against
9.74 on common_claim.

**The confession.** The vector is what we think it is, at the post-norm site, which is the only
site carrying an arm that can assert it. **There is no oracle assertion at cities layer 16 or
common_claim layer 8**, so this is not a general verification at the layer sites. And what was
under-specified was never the vector: it was the **position set** it is added at, which changes
the realized flip rate by 2.5x on cities while the first-order `broadcast_gain` predicts only
1.355. **A steering specification is a direction and a position set, and only the first half
was ever written down.** That is a real methodological gap in the field's conventions and not
just in ours.

### 6.4 Note 4: is the latent space chaotic? **ANSWERED, and the two halves get opposite answers**

The claim bundles two different assertions and they separate cleanly. The measurement uses two
statistics with no Jacobian anywhere in them: the realized gain
`G(u, eps) = ||z(eps u) - z(0)|| / eps`, and `kappa(u)`, the ratio of `G` at the largest budget
to `G` at the smallest, which is exactly 1 for a linear site.

| site | spread of median gain across directions | `kappa` |
|---|---:|---|
| cities, post-norm | 1.000 | 1.000 exactly, all directions |
| cities, pre-norm | 1.31 | 0.997 to 1.001 |
| cities, layer 16 broadcast | 9.91 with `jtw`, **1.74 without it** | 0.98 to 1.06, except `jtw` at 0.66 |
| common_claim, layer 16 broadcast | 7.90 with `jtw`, **1.42 without it** | 0.97 to 1.02, except `jtw` at 0.85 |

**The scale-dependence half, the part that would have broken the whole programme, is absent.**
`kappa` is 1.0000 exactly at the post-norm site on both datasets, within **0.3%** of 1 pre-norm
and **6.1%** at layer 16 on cities, within 1.5% and 3.3% on common_claim. A first-order
certificate is valid across the entire range we sweep. If this had come back at 2x or 5x, the
certificate would have been meaningless and I would have had to say so.

**The anisotropy half is present at depth, and it is structured rather than turbulent.** At
layer 16 exactly one direction stands out, and it is the pulled-back token direction: its gain
is **8.7x** the median random direction on cities and 7.7x on common_claim, while everything
else, *including both truth directions*, sits inside a 1.74x band that also contains all eight
random directions. That is signal, not chaos.

**And the algebra already prices it.** `eps(u) = eps*/alpha` is exactly the change of units
that repairs anisotropy. An anisotropic but linear site needs different **units**, not a
different theory.

**Does `alpha` actually predict behavior?** This had never been checked, and it is the test
that could have embarrassed the whole framework. On **common_claim**: Spearman `rho = 0.316`,
`p = 5.3e-06`, n = 200, with quartile hit rates rising monotonically **0.010, 0.075, 0.120,
0.155**. On **cities** the test is **underpowered, not null**: cities draws its targets from
only 16 distinct target tokens, so `alpha_md_full` takes only 35 distinct values and the
quartile split collapses to three unequal bins. Cities reads `rho = 0.129, p = 0.068`, and that
number is evidence for nothing in either direction.

**The verdict, stated plainly: the objection is right in substance and wrong in mechanism.** A
norm budget is a fair currency in the sense that the site is linear over the swept range. It is
the wrong currency in the sense that what a fixed norm *buys* depends entirely on alignment,
and the unit that should have been reported all along is the fraction of the logit margin
consumed. §6.6 finally reports it.

### 6.5 Note 5: SAEs to define the target set. **NOT ANSWERABLE FROM THESE FILES**

I have to give you a null answer on this one, and I would rather say so than manufacture a
number.

**Why.** `sae_features_cities.csv` decomposes the truth readout at **layer 20** and every
steering vector at **layer 11**. For common_claim it is layer 22 against layer 13. GemmaScope
trains a **separate dictionary per layer**, so feature id 1371 at layer 11 and feature id 1371
at layer 20 are unrelated atoms. A feature-id overlap between the readout and the steering
vectors is therefore not a quantity these files contain, and **no cross-layer overlap number is
reported anywhere in my documents for either dataset.** Answering your note needs one
decomposition run with both vector families at the *same* layer, which is a cluster job, not a
re-read.

**What the SAE work does support**, and it is worth having even though it does not answer the
note. Method: GemmaScope 16k JumpReLU SAEs, 32 decoder atoms per direction via orthogonal
matching pursuit with a least-squares refit on the support. One methodological point matters:
we deliberately do **not** run the SAE encoder on these vectors, because arXiv:2411.08790 shows
that is misleading (the encoder is trained on activations, and a steering direction is not an
activation). OMP against the decoder is the correct operation.

*Result 1, the readout is legible and the actuator is not.* Cumulative explained variance at 32
atoms: the truth readout at the target layer reaches **0.504**, while every source-side
direction we could inject sits at **0.228 to 0.348**. The thing we are trying to control is
describable in feature terms; the thing we have to push on is not. This is D3 restated in
interpretable terms, and it is a stronger statement than the cosine we had been quoting.

*Result 2, the strong one: the context shift swaps the feature set outright.* Pairwise Jaccard
of the top-32 supports:

| pair | Jaccard | shared features |
|---|---:|---:|
| same quantity, different statement subsets (**the control**) | **0.684** | 26 of 32 |
| full-context vs stem-context (**the D2 pair**) | **0.049** | **3 of 32** |
| two random 32-subsets of 16,384 (chance floor) | ~0.001 | |

**The 0.684 control is what makes 0.049 interpretable, and it was not part of the original
design.** Measuring the same quantity on a different sample of statements reproduces 68% of the
feature support. Removing *one word* from the context reproduces 5%, which is 14x below that
noise ceiling. **Deleting the final word of the statement nearly completely replaces the set of
features the actuator is built from.** That is the mechanism behind the 0.500 balanced accuracy,
and it is a far more legible figure for a talk than any cosine. Figures:
[`plot_sae_explained_cities.png`](../plot_sae_explained_cities.png) and
[`plot_sae_overlap_cities.png`](../plot_sae_overlap_cities.png), where the red bar sits near the
floor while the control bars are far above it.

### 6.6 Note 6: temperature 0, invert the token mapping. **CONFIRMED. This is the main result**

**The method.** At temperature 0 the decoder is exactly an argmax, so the set of activations
emitting token `j` is a convex polyhedral cone with 255,999 faces. Take
`a = E[j_tgt] - E[j_top]`, put `t = 0`, and the gap `g` becomes the logit margin `M`. Then
`eps*_token = M / ||A_S^T a||` is the **same formula, the same single VJP, and the same code**
as the probe certificate. What changes is that `a` is read off the unembedding rather than
fitted, so `a.z >= 0` is not a proxy for the behavior but a restatement of it.

**The geometry, first, because it reframes everything.** On cities: median activation norm
189.17, median logit margin to the cheapest false country 13.76, median least-norm cone
displacement 6.369. So **a flip costs a median 3.4% of the activation norm.** On common_claim
the same numbers are 168.75, 3.60, 1.738, so **1.1%**. The cone budget over the single-face
bound is 1.088, meaning the binding cone is 2 to 3 faces rather than 1. Figure:
[`plot_token_budget_cities.png`](../plot_token_budget_cities.png).

**Result 1: the token pullback actuates and the probe pullback does not, at matched budgets, in
the same experiment.**

cities (n = 200; post-norm rows are label 1, n = 101; layer rows are hit rates on shared
statements):

| site | direction | `frac` 1.0 | `frac` 2.0 |
|---|---|---:|---:|
| post-norm | `oracle` | **1.000** (101/101) | **1.000** (101/101) |
| layer 16, broadcast | `jtw_token` | **0.340** | **0.675** |
| layer 16, last position | `jtw_token` | 0.135 | 0.285 |
| every arm, every budget | `jtw_legacy` | **0.000** | **0.000** |

common_claim (`jtw_legacy` covers 90 of 200, 44 at label 1, so layer rates are on those 90):

| site | direction | `frac` 1.0 | `frac` 2.0 |
|---|---|---:|---:|
| post-norm | `oracle` | **1.000** (44/44) | **1.000** (44/44) |
| layer 8, broadcast | `jtw_token` | 0.811 | 0.844 |
| every arm, every budget | `jtw_legacy` | **0.000** | **0.000** |

common_claim's target is the runner-up token, so generic disruption can land on target there;
its high rates should not be read as a stronger version of the cities result. Figure:
[`plot_token_steer_cities.png`](../plot_token_steer_cities.png).

**Result 2: the currency, finally reported.** Median fraction of the logit margin actually
consumed:

| dataset, arm | direction | median at `frac` 1.0 | n |
|---|---|---:|---:|
| cities, post-norm | `oracle` | 1.0010 | 200 |
| cities, post-norm | `md_full` | 1.0877 | 200 |
| cities, post-norm | `jtw_legacy` | **0.0174** | 200 |
| cities, layer 16 broadcast | `jtw_token` | 1.0735 | 200 |
| cities, layer 16 broadcast | `jtw_legacy` | **0.0082** | 200 |
| common_claim, layer 8 broadcast | `jtw_token` | 1.9693 | 90 |
| common_claim, layer 8 broadcast | `jtw_legacy` | **0.0272** | 90 |

**On cities, at the budget the old certificate calls a full trip into the FALSE halfspace, the
legacy direction consumes 1.7% of the deciding margin at the output layer and 0.8% at layer
16.** That single line is the quantitative replacement for the entire qualitative story about
why the old null was uninformative.

**Result 3: it buys a semantic flip without destroying the text.** At cities layer 16
broadcast, `jtw_token` moves the fraction of completions naming the *target false country*
from **0.010** unsteered to **0.075** at `frac = 1.0` and **0.120** at `frac = 2.0`, at a
degenerate fraction of 0.005 and 0.020. Compare `md_full` at the output layer: target 0.000 at
degeneracy 0.885. The token direction produces a *lie*; the truth direction produces *silence*.

**Result 4, the most heavily caveated number here, and I am flagging it rather than leading
with it.** On cities at `frac = 1.0`, the swept scale is a median 2.42x the reconstructed
legacy budget, and by that reconstruction **88.1% of statements crossed the probe's own FALSE
threshold while 0.000 flipped the token.** Two caveats travel with the crossing figure and with
nothing else on that line. The R² that licenses the reconstruction (0.9991) licenses
**linearity, not gain**, and the realized slope was 8x to 35x below prediction, so 88.1% is an
optimistic upper bound rather than a count. And the legacy budget is calibrated only at this
site. Read it as "the readout very probably crossed on most statements and the token never
moved", not as a crossing count.

**The honest limit on all four results, which I want said out loud rather than waited for.**
The certificate is a **first-token** claim and the completion recovers. On cities `oracle`
flips the first token on **200 of 200** statements, and yet only **0.085** of those same 200
completions name the target country while **0.700** still name the correct one. Reaching the
cone at position `t` does not keep the model there at position `t+1`. Extending the target set
over a horizon is the natural next problem, and it is exactly what the BRT-Align style
recursion in your citation is for.

### 6.7 Note 7: write the math out. **ANSWERED, with one number outstanding**

Done, and covered in §3 above. Every object between token input and token output is named and
justified in
[`RESULTS_SINCE_LAST_MEETING_PART3.md` §2.1 to §2.7](RESULTS_SINCE_LAST_MEETING_PART3.md), with
proofs, the cone dual and the non-emptiness argument in [`math_map.tex`](math_map.tex). Both are
tracked in git.

Two objects that had previously existed only as prose are now written down as operators: the
**broadcast operator `B`** (§3.11) and the **`_asis` convention** (§3.12).

**The residual, and it is a missing measurement rather than a missing object.** §3.12 writes
down the basis-correct version of the cross-layer cosine, and it **has not been computed**. So
every `_asis` alpha in this document is a cosine against the raw vector that was actually
injected, and must keep travelling with the convention paragraph that says so. That is the
operational question and the one the programme needs, but it is not a coordinate-free statement
about the concept. One forward pass per direction per dataset.

### 6.8 Note 8: sweep the layers, mean difference is heuristic. **ANSWERED**

**Result A: control authority is front-loaded, and 11 and 13 were not the cheapest layers.**
All 26 layers swept, 200 statements each, both injection conventions. Median certified budget,
the norm needed at that layer to flip the target token under broadcast:

| dataset | layer 0 | layer 8 | layer 11 | layer 13 | layer 16 | layer 25 |
|---|---:|---:|---:|---:|---:|---:|
| cities | **8.73** | 21.44 | 23.11 | 23.21 | 23.00 | 67.24 |
| common_claim | **2.92** | 4.60 | 5.10 | 5.34 | 6.60 | 19.93 |

Cost rises with depth over the whole sweep. Cities' layer 11 costs **2.6x** its layer 0, and
common_claim's layer 13 costs **1.8x** its layer 0. The arms we actually ran, 16 and 8, are
also not the cheapest. **This reproduces the audit's Phase 4 finding in token space with a
completely independent readout**, which is the strongest form of corroboration available here.
Figure: [`plot_token_layers_cities.png`](../plot_token_layers_cities.png).

**Result B: "mean difference is heuristic" is confirmed, quantitatively.** This is the alpha
table from §3.5: all six candidate truth directions sit at or below the 0.0166 chance floor.
The behavioral consequence is the 0.000 hit rate for `jtw_legacy` in every arm at every budget,
and 1 flip in 101 for `md_full` on cities. Figure:
[`plot_token_alpha_cities.png`](../plot_token_alpha_cities.png), which plots each direction
against the chance floor.

**Result C, a correction to my own record.** `broadcast_gain` at cities layer 16 is **1.355**,
read from `token_jac_cities.csv`. Earlier project notes carry a different, larger figure for
this quantity; **that figure is wrong and should not be quoted again.** At common_claim layer 8
the artifact reads 2.016, which also corrects the value in the implementation plan.

**Reading.** Sweeping layers is the right methodological instinct and it does not rescue this
direction. Mean-difference fails at the output layer, before the norm, and at the swept layer
alike, at every budget. **Depth is not the binding constraint; alignment is.** Depth *does*
matter for the direction that works, and `jtw_token` was only ever steered at one layer per
dataset, so choosing a cheaper one is free improvement.

---

## 7. Chapter 6: what I think this means

### 7.1 The one-sentence version

We built per-statement backward-reachability certificates for a truth readout across a 9-layer
hop, verified them locally exact (0.9997), showed that steering along them moves the readout
exactly as predicted (R² = 0.999, zero sign errors), found that behavior does not change at
all, decomposed why into four measured mechanisms, and then fixed it by changing the target set
to the one your note 6 proposed.

### 7.2 The three numbers I would lead with

1. **0.9997.** Same-point calibration on cities. The certificate is exact where it is defined.
   This is what makes the null a finding rather than a bug.
2. **0.500.** Balanced accuracy of the target-layer truth probe at generation-prefix tokens,
   with a nonlinear refit recovering nothing. The probe carries no information about what the
   model is about to say.
3. **0.049.** Jaccard overlap between full-context and stem-context optimal input directions in
   SAE feature space, 3 shared features of 32, against a non-degenerate control at 0.684. One
   word of context does not weaken the direction, it substitutes a different one.

And the fourth, from the new program: **1.7%**, the fraction of the deciding logit margin the
old direction was consuming at the budget we called a full crossing.

### 7.3 What actually went wrong, in one sentence

I certified arrival at a probe's halfspace at layer 20 while the judge measured the argmax at
layer 26, with five unmodeled operations in between. Your token-space suggestion collapses that
gap: at temperature 0 the target set is exactly a polyhedral cone in the unembedding basis, so
the same `g/m` formula applies with `w = E[j_tgt] - E[j_top]` and `t = 0`, and **the
dissociation becomes impossible by construction.**

### 7.4 The gate that still stands over the old result

The probe-halfspace result produced a `readout-only` outcome, which is consistent with two very
different stories: either **the instrument is broken** and this pipeline would fail on any
concept, or **truth is not a behaviorally actuatable variable in gemma-2-2b** and the finding is
real. Only a concept with a *known* behavioral handle separates them, which is why the refusal
positive control exists (Arditi et al., arXiv:2406.11717, established a single refusal
direction with public code, validated on small Gemma models). It is built and gated and has not
run.

**Note that the token-space certificate does not need that gate in the same way**: its oracle
arm is a positive control by construction, and it passes on 200 of 200 statements on both
datasets. That is a real structural advantage of the new formulation over the old one.

### 7.5 How this connects to your line of work

Our R² = 0.999 corroborates A-LQR's core local-linearity assumption directly and
quantitatively. Our context-transfer gain collapse and population-transfer probe failure are
precisely the **open-loop** failure modes that a closed-loop design implicitly answers. And the
first-token horizon problem in §6.6 is the cleanest possible handoff: the certificate reaches
the cone at one position and the model leaves it at the next, which is exactly where a
receding-horizon controller that re-plans each token stops being an analogy and becomes the
structurally correct object.

---

## 8. What runs next

Ordered by information per GPU-hour.

**1. The cheapest falsifiable thing on the list: sweep the pre-norm arm to `frac = 6`.** I
explain the pre-norm null as arithmetic (§6.1). That explanation makes a **prediction, stated
here before running it: sweeping the pre-norm oracle arm to `frac = 6` should recover its hit
rate to near 1.0 on both datasets**, since about 4.4 is needed on cities and about 5.2 on
common_claim. If it does not, my explanation is wrong and the RMSNorm Jacobian in the
certificate needs re-deriving. One job, no new code.

**2. Steer where the budget is cheap.** The sweep says layer 0 is 2.6x cheaper than layer 11 on
cities, and the arms that ran are at 16 and 8. Re-run `jtw_token` in the layer 0 to 8 window,
which is also the window Phase 4 identified as best-conditioned for the behavioral verdict
readout. Maximum contrast, no new machinery.

**3. Add an oracle arm at the layer sites.** Every layer-site number currently has no in-arm
ceiling. The cone QP already produces the displacement; pulling it back through the same VJP
costs nothing and turns 0.340 from a bare number into a fraction of what was provably
available.

**4. Submit the refusal screen (40 minutes of GPU).** Everything downstream in that horizon is
gated on it, and the gate is pre-registered: base-model harmful-prompt refusal at or above 0.10
keeps the base model with no model confound; below 0.10 falls back to the instruct model with a
stated caveat.

**5. Free CPU work, no gate.** Answer note 5 properly with one same-layer SAE decomposition.
Compute the transported cross-layer alpha. Enlarge the labeled stem set so the 0.500 result does
not rest on 9 and 10 FALSE examples. Write generating scripts for the two audit summary figures,
which currently have none.

**6. The horizon problem, with you.** Extending the target set over a horizon is the natural
next problem and the natural A-LQR handoff.

**Three things I need from you:**

- **Confirm arXiv:1910.13272** (§2.1). Intended as the feedback-linearization framing, or a typo
  for a set-propagation reference?
- **A decision on the 2026-07-23 conditional-steering round** (§5.2). My read is retire it; I
  would rather be overruled than leave it in limbo.
- **Whether the refusal gate is still worth 3 to 6 GPU-hr** now that the token-space oracle arm
  provides a positive control by construction. It still validates the *probe-halfspace*
  instrument, which is what the audit chapters rest on, but it is no longer the only positive
  control in the project.

---

## 9. Figure guide

All PNGs live at the repo root. Every one below has a generating script in `src/` except the two
noted.

**The reachability audit**

| Figure | What it shows | How to read it |
|---|---|---|
| [`plot_reach_margins_cities.png`](../plot_reach_margins_cities.png) · [common_claim](../plot_reach_margins_common_claim_true_false.png) | Violins of `log10 &#124;&#124;J^T w&#124;&#124;` for the four direction groups, dashed line at the random-null median | The truth violins sit visibly above the gray random violin. That separation is the entire "reachable-candidate" verdict; had they overlapped, the programme stops here |
| [`plot_reach_curves_cities.png`](../plot_reach_curves_cities.png) · [common_claim](../plot_reach_curves_common_claim_true_false.png) | Fraction of statements reachable against budget, with `input_scale` marked and a gray "beyond linear validity" band | Cities saturates well left of both markers, so the certificate is cheap and inside its validity region. common_claim saturates **inside** the gray band, so its certificate is extrapolation |
| [`plot_reach_geometry_cities.png`](../plot_reach_geometry_cities.png) · [common_claim](../plot_reach_geometry_common_claim_true_false.png) | Histograms of `cos(J^T w, mean_diff@source)` and `cos(J^T w, v_Q)` | Asks where the optimal input direction actually points. It is not the source-layer truth direction. This is the seed of the D3 mechanism |
| [`plot_reach_svd_cities.png`](../plot_reach_svd_cities.png) · [common_claim](../plot_reach_svd_common_claim_true_false.png) | Energy of four vectors inside the top-k singular subspace as k grows | The target-layer readout (against left singulars) climbs fast; the source-layer truth direction (against right singulars) barely beats random. **That asymmetry is the finding** |
| [`plot_reach_jlens_cities.png`](../plot_reach_jlens_cities.png) · [common_claim](../plot_reach_jlens_common_claim_true_false.png) | Per-layer margins for three readouts, two input modes, log-y | The curves fall left to right. You want to intervene at the left edge; we intervened in the middle. The red verdict curve sits below the others everywhere |
| [`plot_judge_reach_stmt_cities.png`](../plot_judge_reach_stmt_cities.png) · [mean arm](../plot_judge_reach_cities.png) | TRUE / FALSE / INCOHERENT fractions against steering scale | Flat lines. **That flatness is the headline null of the whole project** |
| [`plot_reach_audit_dissociation.png`](../plot_reach_audit_dissociation.png) · [`plot_reach_audit_certificate.png`](../plot_reach_audit_certificate.png) | Two-panel summaries: readout moves and behavior does not; certificate inside validity for cities, outside for common_claim | **Reproducibility warning: these two were produced ad hoc on 2026-07-24 and no committed script regenerates them.** They need a script before they go in a paper |

**The token-space program**

| Figure | What it shows | How to read it |
|---|---|---|
| [`plot_token_steer_cities.png`](../plot_token_steer_cities.png) · [common_claim](../plot_token_steer_common_claim_true_false.png) | Hit rate against `frac`, one line per direction, per arm | **The result.** `oracle` at the ceiling, `jtw_token` climbing with budget, `jtw_legacy` flat on zero |
| [`plot_token_budget_cities.png`](../plot_token_budget_cities.png) · [common_claim](../plot_token_budget_common_claim_true_false.png) | Distribution of what a flip costs: margin, cone displacement, fraction of `&#124;&#124;z&#124;&#124;` | The 3.4% headline. A flip is cheap in absolute terms, which is what makes the alignment story the whole story |
| [`plot_token_alpha_cities.png`](../plot_token_alpha_cities.png) · [common_claim](../plot_token_alpha_common_claim_true_false.png) | Alignment of each candidate direction with the token decision, against the chance floor | **The single most damning figure.** Every truth direction sits on or below the 0.0166 line |
| [`plot_token_layers_cities.png`](../plot_token_layers_cities.png) · [common_claim](../plot_token_layers_common_claim_true_false.png) | Certified budget against layer, all 26, both injection conventions, with source and target layers marked | Cost rises with depth. Our source-layer markers sit well right of the minimum. This is note 8 answered in one picture |
| [`plot_token_sens_cities_layer16_all.png`](../plot_token_sens_cities_layer16_all.png) and siblings, one per site | Realized gain against budget for 13 or 14 directions | Flat lines mean `kappa = 1` and a linear site. The one line that is not flat is the pulled-back token direction, and it is also the one 8.7x above the pack |
| [`plot_signed_steer_cities.png`](../plot_signed_steer_cities.png) · [common_claim](../plot_signed_steer_common_claim_true_false.png) | Per-statement split into inert, churn and mover, with the sign of each mover | Cities is a wall of inert; common_claim is mostly churn with movers split both ways. This is the anti-steerability objection answered per dataset |

**SAE forensics**

| Figure | What it shows | How to read it |
|---|---|---|
| [`plot_sae_explained_cities.png`](../plot_sae_explained_cities.png) · [common_claim](../plot_sae_explained_common_claim_true_false.png) | Cumulative explained variance against OMP rank, one line per direction | How sparse each direction is in feature space. The readout line is on top; the source-side lines cluster low |
| [`plot_sae_overlap_cities.png`](../plot_sae_overlap_cities.png) · [common_claim](../plot_sae_overlap_common_claim_true_false.png) | Sorted pairwise Jaccard bars, full-vs-stem highlighted red | The red bar near the floor against control bars far above it. That contrast is the figure's whole point |

**From before the meeting, still load-bearing**

| Figure | What it shows |
|---|---|
| [`plot_mag_linearity_v2.png`](../plot_mag_linearity_v2.png) | **Your visualizer ask.** Left: `v_Q` at cos 0.84 to 0.98 while truth and DCT sit in a `&#124;cos&#124; < 0.1` band. Right: `eps_Q` per direction, truth and DCT above the "explains nothing" line |

---

## 10. The honest column

This is the section to read out loud rather than wait to be asked about.

- **Everything in the token program is first-token.** The certificate makes no claim about the
  rest of the completion, and completions demonstrably recover: `oracle` flips 200 of 200 first
  tokens while only 0.085 of those completions name the target country and 0.700 still name the
  correct one.
- **The refusal positive control has not run**, so for the *probe-halfspace* certificate,
  "the instrument works and truth is not actuatable" and "the instrument does not work" remain
  unseparated.
- **`n = 200` per dataset**, subsampled from 1,496 and 4,450. One model, fp32, two datasets, both
  English, both short declarative statements.
- **`jtw_legacy` covers 90 of 200 statements on common_claim, 44 at label 1.** Every common_claim
  cross-direction number is on that restricted set.
- **The 0.500 balanced-accuracy result rests on 9 and 10 FALSE examples** at a 0.94 base rate. The
  direction is unambiguous; the precision is not. Fixing it is cheap and I have not done it.
- **common_claim's same-point calibration is 0.7025, not 1.0**, so there is a real 30% local
  miscalibration there on top of the context loss. **Cities carries the clean argument.**
- **common_claim's target is the runner-up token**, so generic disruption can land on target
  there. No conclusion in this document rests on common_claim alone.
- **No oracle assertion exists at the layer sites**, so layer-site hit rates have no in-arm
  ceiling to be read against.
- **The crossing figures in §6.6 Result 4 are reconstructed, not measured**, and carry two
  caveats: the R² licenses linearity and not gain, and the legacy budget is calibrated only at
  the post-norm site.
- **Temperature 0 only.** Nothing here transfers to sampling without re-derivation.
- **The `_asis` alphas are cross-layer carryovers.** The transported version is defined and
  uncomputed.
- **Note 5 is unanswered, not answered negatively.** It needs one same-layer decomposition run.
- **No judge ran in the token program.** `hit_target` is an argmax check and the coherence and
  country readings are string-level. Nothing there is a truth verdict.
- **The SAE work is thin**: one width (16k), one layer pair, no absorption or dark-matter
  robustness pass.
- **There is no human-versus-judge agreement measure on the actual steering completions**, only
  on the clean statements the 0.970 validation gate used.
- **arXiv:1910.13272 still does not match the paper it was described as.**
- **Hygiene gap:** most result artifacts sit untracked at the repo root by accident rather than
  policy, and two headline audit figures have no generating script. Both need fixing before
  anything is submitted.

---

## 11. Glossary

| Symbol | Plain English |
|---|---|
| `d = 2304` | width of the residual stream, the vector the model carries between layers |
| `V = 256,000` | vocabulary size |
| `E` | the embedding matrix, tied, so it is also the unembedding |
| `h^(l)_t` | the residual stream at layer `l`, position `t`, the thing we perturb |
| `z` | the activation after the final RMSNorm; logits are exactly `E z` from here |
| `w` | a readout direction: the scalar we want to move is `w.z` |
| `t` | the threshold that defines the target set |
| `g` | the **gap**, how far the statement starts from the boundary. For the token readout it is literally the logit margin |
| `m` | the **controllability margin**, readout movement per unit of perturbation in the best direction |
| `eps*` | the **certified budget**, `g/m`, the smallest perturbation that reaches the target set |
| `A_S` | the linear map from a perturbation at site `S` to readout coordinates |
| `A_S^T w` | the **pullback vector**, the only object that needs computing, one VJP per statement |
| `alpha(u)` | **alignment**, the fraction of a flip a unit of direction `u` buys. Chance is 0.0166 |
| `B` | the **broadcast operator**, which copies one perturbation to every position |
| `broadcast_gain` | how much more readout movement broadcasting buys than a single position |
| `rmsnorm_penalty` | how much more a pre-norm injection costs than a post-norm one, 4.38 and 5.18 |
| `input_scale` | the activation's own norm yardstick. Budgets should be read against it |
| `frac` | fraction of the certified budget spent. `frac = 1.0` is the predicted-just-enough push |
| `frac_margin` | fraction of the logit margin actually consumed, measured |
| `kappa` | ratio of realized gain at the largest budget to the smallest. Exactly 1 at a linear site |
| cone `C_j` | the set of activations that emit token `j`, a convex set with 255,999 faces |
| `oracle` | the least-norm displacement into that cone. Its flip is a theorem, so it is the harness assertion |
| `jtw_token` | the token pullback `J^T a`, the direction that works |
| `jtw_legacy` | the old probe pullback `J^T w`, the direction that does not |
| `md_full` | the final-layer mean-difference truth direction, the naive baseline |
| VJP | vector-Jacobian product, one backward pass, which is why per-statement certificates are affordable |
