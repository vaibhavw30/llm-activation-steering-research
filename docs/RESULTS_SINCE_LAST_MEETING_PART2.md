# Results Since the Last Meeting — Part 2: Two Unsupervised Miners, Both Put to the Test

*A from-scratch, plain-language walkthrough of the two blocks of **actual findings** produced since
`PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` was last updated (2026-07-16). That doc ended with the MAG
battery **built but not run** (§8) and a to-do list (§9). Both items on the critical path are now
**done and have results**: (A) the **MAG battery ran** on all four datasets, and (B) a new experiment,
**warm-started DCT**, ran on the GH200. This doc explains both simply and in detail, and points at every
figure. Read the Part-1 doc first if you want the backstory; §0 here recaps it in one paragraph.
Last updated: 2026-07-22.*

**Companion docs (the receipts):** `PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` (Part 1 — the pipeline,
the OLMo judge, the two DCT cluster runs), `WARM_DCT_RESULTS.md` (the full warm-DCT numbers behind
Part B here).

---

## 0. The one-paragraph recap (where Part 1 left us)

The whole project turns on one slogan: **"decodable ≠ causal."** A linear probe reads true-vs-false out
of gemma-2-2b's activations at ~99% on clean data — so truth is clearly *written down* inside the model.
But when we ask an **unsupervised** method (**DCT**, Deep Causal Transcoding) to hand us the directions
the model is most *causally driven by*, the supervised truth direction is **not** among them, and
steering along that truth direction does **not** make the model lie — it just makes it incoherent when
pushed hard (Part 1, §5: "truth is a **degradation lever, not a truth lever**"). Part 1 turned that from
a hand-read into hard numbers using an open LLM-judge (OLMo-3, validated at 0.970). **Two objections
remained open, and this doc answers both:**

1. *"Maybe truth just isn't the kind of thing **DCT's** objective finds — try a different miner."*
   → answered by **Part A: the MAG battery** (a second, geometry-based miner).
2. *"Maybe DCT's search just started in the wrong place — what if you **seed** it at the supervised truth
   axis and let it refine?"* → answered by **Part B: warm-started DCT**.

Both come back **consistent with "decodable ≠ causal,"** and Part B sharpens it further.

---

# PART A — The MAG battery (the second miner), now run

## A1. What MAG is, in plain words

DCT finds causal levers by asking "which nudge to the activations changes the output the most." **MAG
(Mining via Activation Geometry)** asks a completely different question — it never steers to find
features. Instead it **prepends a question to the input and watches where the activations move.**

Concretely: take a statement `p` ("The capital of France is Paris"). Now prepend the model's own
question about it: `Q‖p` = "Is the following statement true? The capital of France is Paris." Measure how
the internal activations shifted: **Δ<sup>Q</sup>(p) = activations(Q‖p) − activations(p)**. The idea:
*if the model internally "knows" whether the statement is true, asking it the question should move its
activations along a truth-carrying direction.* MAG builds **8 different "operators"** — 8 ways of reading
these question-conditioned activations (Direct = just the raw activations, Prefixed = the Q‖p
activations, InputDelta = the shift Δ<sup>Q</sup> itself, Verdict = the model's own yes/no answer token,
and four others).

**Why we built it:** it is a genuinely different mining principle from DCT (geometry of a question-shift,
not a steering objective). If MAG *also* fails to find causal truth, the "it's just DCT" objection dies.
If MAG succeeds where DCT failed, that's a real new lead. Either way it's informative.

We ran the full battery — five probes (E1 readability, E2 disagreement, E3 linearity, a transfer/rank
test, and E4 calibrated steering) — on all four datasets. Here is what each found.

## A2. E1 — Truth is linearly readable from *every* activation view, including the question-shift

**Figure: `plot_mag_readability.png`.** For each operator we fit a linear probe (5-fold CV) to read
truth, and compare against a **random-10-dimensions** baseline (the floor).

- On **cities** (clean), essentially every operator reads truth near-perfectly: Direct raw activations
  **0.993**, the prefix-shift InputDelta **0.993**, Prefixed **0.985**, FewShot **0.991**. Even projected
  into **DCT's own top-10 causal subspace** it's **0.947**, and top-50 **0.989**.
- On **common_claim** (messy), everything drops together but stays well above the floor: Direct **0.714**,
  InputDelta **0.720**, Prefixed **0.691**, vs random-10 at **0.633**.

**The takeaway:** the truth signal is not hiding in some exotic corner of the activations — it is present
and linearly readable in the raw stream *and* in the question-induced shift. The question-shift
(InputDelta) reads truth **as well as the raw activations do** (green ≈ grey in the figure). So MAG's
premise is sound: asking the model about truth really does move its activations along a truth-readable
direction. (`mag_readability_<ds>.csv` for the full per-operator table.)

## A3. The unsupervised arm is dead — the model **never says "false"**

This is the sharpest single MAG finding, and it's a little surprising. **Figure: `plot_mag_verdict.png`.**

The dream version of MAG is *fully unsupervised*: never use gold labels at all — just read the model's
**own** answer to "Is this statement true?" (call it y<sup>M</sup>) and use that as the label. We checked
whether that works. It does not, for a blunt reason:

- **y<sup>M</sup> is constant "yes."** Across all four datasets the model answers "yes" to
  essentially every statement — **0/354, 0/1496, 0/1200, and 1/4450** "no" answers. gemma-2-2b base,
  asked "is this true?", just says yes. It never asserts a statement is false, even for false statements.
- **The Verdict operator reads truth at chance.** The activations *at the yes/no answer position* carry
  almost no truth signal: accuracy **0.62 on cities, 0.50–0.51 on companies/common_claim, 0.48 on
  translations** — at or barely above the 0.50 coin-flip line.

**Why this matters:** the model *contains* truth (A2) but does not *express* it when you ask directly —
its verbalized verdict is a stuck "yes" that ignores the truth its own activations encode. This is
"decodable ≠ causal" showing up in a new place: truth is decodable from the geometry but is not what
drives the one behavior (the yes/no answer) that is supposed to report it. It also means the E2
"disagreement" probe is degenerate on this model (when y<sup>M</sup> is always "yes," "statements where
the model disagrees with gold" is just "all the false statements," and the reported match rates are an
artifact of that constant — we do **not** read a finding into E2).

## A4. E3 — The question-shift is a clean linear direction, but it is **orthogonal to the truth axis**

**Figure: `plot_mag_linearity.png`.** How much of the whole prefix-shift Δ<sup>Q</sup> is captured by a
single direction? We measure a residual ε_Q (lower = more of the shift lives along one line) and the
cosine of candidate directions to the actual shift.

- The MAG shift direction **v_Q** explains most of the shift: cosine **0.970 on cities, 0.836 on
  common_claim**. So "the model being asked about truth" moves its activations along essentially **one**
  direction — the question-shift is highly linear.
- **But that direction is not the truth axis.** The supervised truth direction (mean_diff) and DCT's top
  direction both sit at cosine **≈ 0** to the prefix shift (mean_diff −0.006 / +0.010; DCT −0.015 /
  +0.044). They are **orthogonal**.

**The takeaway:** there really is a crisp, single "I am being asked whether this is true" direction inside
the model — but it is a *different* direction from the "this statement is true vs false" axis. Finding the
question-shift does not hand you the truth lever. (`mag_linearity_<ds>.csv`.)

## A5. E4 — The behavioral test (honest about its limits)

E4 injects each candidate direction during generation and asks whether behavior changes. **Figure:
`plot_mag_e4_headtohead.png`** (top row cities, bottom common_claim). Two things are plotted per
direction: the **yes/no self-verdict flip rate** at τ=−1 (blue, push→truth) vs τ=+1 (orange, push→lie),
and, as small annotations, the **free-form incoherence rate** at that strength.

- The supervised / gold-contrast axes (`sup_mean_diff`, `mag_u_gold`) *do* produce **sign-dependent**
  verdict flips — e.g. cities `sup_mean_diff` flips the model's yes/no answer on 79% of statements at
  τ=−1 but ~0% at τ=+1; on common_claim it's 100% vs 0%. Read naively, that "antisymmetric" shape looks
  like a causal lie-lever.
- **Three reasons we treat this as suggestive, not a result:** (1) the metric is the **model's own yes/no
  verdict token**, not the OLMo TRUE/FALSE/INCOHERENT judge Part 1 used — a much cruder, single-token
  signal; (2) it rides on **high incoherence** (the 19–41% annotations) — the flips partly reflect the
  model breaking, not cleanly lying; (3) it's on a **tiny 24-statement** matched set. The fully-mined MAG
  operators that don't use gold labels (`mag_u_yM`, Answered, FewShot) are mostly **inert**, and the
  first principal component `resid_pc1` flips at **both** signs — the degradation signature again.

**The takeaway:** E4 is consistent with the Part-1 DCT story (the supervised axis mostly degrades; the
purely-unsupervised MAG directions don't cleanly lever truth), but because it uses a weaker metric than
the OLMo pipeline, we don't put weight on its one antisymmetric-looking wrinkle. The clean,
OLMo-judged behavioral test is **Part B**.

*(Other MAG figures, for completeness: `plot_mag_transfer.png` — do directions mined on one dataset work
on another? realized effects hover near chance (~0.5), i.e. weak transfer; `plot_mag_recovery.png` /
`plot_mag_subspace.png` — where MAG's direction ranks among DCT's levers and how much truth survives
inside DCT's top-k subspace; `plot_mag_summary_table.png` — the one-screen scorecard.)*

## A6. What MAG concludes

MAG, a mining method with a **completely different principle** from DCT, lands in the **same place**:

- Truth is **linearly readable** from the activations, including the question-conditioned shift (A2).
- The model **does not express** that truth — its own verdict is a stuck "yes" and reads at chance (A3).
- The "being asked about truth" direction is **real and linear but orthogonal** to the truth axis (A4).
- Behaviorally, the unsupervised MAG directions do **not** cleanly act as truth levers (A5).

So the Part-1 null is **not an artifact of DCT's objective.** Two unrelated unsupervised miners both fail
to find truth as a causal lever in gemma-2-2b base. That closes objection #1.

---

# PART B — Warm-started DCT (closing objection #2)

## B1. The idea, in plain words

DCT starts its causal search from random directions and refines them toward "big causal effect." The
supervised truth axis is behaviorally inert (Part 1 §5). **Warm-started DCT** asks: *what if we start
DCT's search **at** the supervised truth axis and let its causal-gradient refinement pull it around —
does it discover a nearby direction that is both truth-aligned and causally potent?* We add a soft
**anchor** of adjustable strength **λ** that ties the search back to the supervised seed:

- **λ = 0** → no anchor, DCT wanders freely (ends up ≈ a free DCT factor).
- **λ = 3** → strong anchor, the result stays ≈ the supervised truth axis.
- **λ = 0.3, 1.0** → intermediate blends.

We sweep λ ∈ {0, 0.3, 1, 3}, two seed directions (mean_diff and grad), two datasets — **16 warm fits** —
then steer gemma-2-2b with every resulting direction and score the completions with the **same OLMo
judge** and the **same two-sided strength sweep** as Part 1. Sign convention: directions point toward
TRUE, so **+τ pushes toward TRUE, −τ pushes toward FALSE (lying)**. A real truth lever would make FALSE
rise at −τ. (Full method + numbers: `WARM_DCT_RESULTS.md`.)

## B2. The anchor knob works exactly as designed (the control passed)

**Figure: `plot_dct_warm_drift_<ds>.png`.** Before asking about behavior, we confirm the λ knob actually
interpolates. It does, cleanly: the cosine between the refined direction and its supervised seed climbs
the intended ladder with λ, near-identically on both datasets:

| λ | cos(warm, seed) — cities | common_claim |
|---|---|---|
| 0.0 | 0.03 | 0.02 |
| 0.3 | 0.28 | 0.25 |
| 1.0 | 0.72 | 0.66 |
| 3.0 | 0.95 | 0.95 |

So we are genuinely sweeping from "free DCT search" to "essentially the supervised axis," and any
behavioral change across λ is caused by the anchor — not noise. The knob is a valid dial.

## B3. The behavioral answer: **no truth lever appears at any λ**

This is the headline. **Figure: `plot_dct_warm_curves_<ds>.png`** — an 11-panel grid, one per direction,
each showing the TRUE (green), FALSE (red), INCOHERENT (grey) fractions across the τ sweep.

**In every panel, on both datasets, the red FALSE line is pinned to the floor (~0.03–0.09).** No
direction — not the raw supervised axes, not DCT's free top factor, and crucially **not any of the
warm/anchored directions at any λ** — makes the model assert falsehoods in a sign-dependent way. Anchoring
an inert axis toward DCT's causal search does not manufacture a lie lever; the interpolation runs from
**inert** (high λ) to **mild degrader** (low λ) and never passes through **truth lever**.

What steering *does* do, when it does anything, is **destroy coherence** — and only on the messy dataset,
only for the DCT-derived (low-λ) directions. The cleanest example is DCT's own free top factor,
`cold_top`, on common_claim: as τ goes +1 → −1, INCOHERENT climbs **0.03 → 0.66** while FALSE stays flat
(≤0.16). Positive τ *sharpens* the model (TRUE 0.94); negative τ *breaks* it. Sign-dependent in
**coherence**, not in **truth**. And the **harder you anchor toward the supervised seed (higher λ), the
more inert the direction becomes** — the opposite of unlocking causality. cities is inert across the board
(its well-baked facts don't move at these strengths).

**Figure: `plot_dct_warm_effect_<ds>.png`** makes the dissociation a single bar chart: per direction, the
max rise above baseline in FALSE (a would-be lie lever, red) vs in INCOHERENT (a degradation lever, grey).
The FALSE bars are ~0 everywhere; only the INCOH bars move, and only for the low-λ DCT directions on
common_claim.

*(Also: `plot_dct_warm_audit_<ds>.png` re-confirms the Part-1 geometry — DCT's most **potent** cold
factor is **not** truth-aligned; `plot_dct_warm_verdict_<ds>.png` is the canonical FALSE-vs-INCOH bar at
the most-negative τ.)*

## B4. What warm-DCT concludes

The experiment's own question — *does warm-starting close the decodable-≠-causal gap?* — gets a clean
**no**, and it's a **stronger** null than Part 1. It's not merely that the raw supervised axis is inert;
it's that **no point on the entire geometric path** from the supervised axis to DCT's free causal search
is a truth lever. Truth is linearly *decodable* here but is not *steerable as truth* by any direction in
this family — the causal axis that does exist (`cold_top`) moves **coherence**, not truth, and is
orthogonal to truth. That closes objection #2.

---

## The combined bottom line for the PI

> **"Decodable ≠ causal" now survives both obvious escape routes.** Since the last meeting we (1) ran a
> **second, independent unsupervised miner (MAG)** and (2) ran a **warm-start experiment that seeds DCT at
> the supervised truth axis and lets it refine.** Both come back consistent with the original finding, and
> the warm-start makes it sharper.
>
> - **MAG** (a geometry-of-question-shift miner, unrelated to DCT's objective) finds the same thing DCT
>   did: truth is **linearly readable** from the activations (including the question-shift, InputDelta ≈
>   raw), but the model **won't express it** — its own verdict is a stuck "yes" (0–1 "no" answers out of
>   thousands) and reads truth at chance, and the crisp "being-asked-about-truth" direction is
>   **orthogonal** to the truth axis. So the Part-1 null is **not a DCT artifact.**
> - **Warm-started DCT** confirms there is **no λ** — no blend between the supervised axis and DCT's free
>   search — at which a **truth lever** appears. Steering degrades coherence (on messy claims) or does
>   nothing; it never makes the model lie in a sign-dependent way. The most causally potent direction we
>   can find moves **coherence**, not truth.
>
> Net: in gemma-2-2b base, truth is written into the activations but is **not a direction the model is
> behaviorally driven by** — and that conclusion is now robust to changing the mining method *and* to
> seeding the search at truth itself.

## Caveats we keep honest

- **MAG E4 used a weaker evaluator** than the DCT pipeline (the model's own yes/no verdict flip, on 24
  statements, with high incoherence) and was **not** run through the OLMo judge. We treat its one
  antisymmetric-looking result as suggestive only; the clean OLMo-judged behavioral test is warm-DCT.
- **Strength ceiling.** The warm-DCT τ sweep tops out at ±1·input_scale at a **single** injection layer.
  A stronger or multi-layer push might eventually force falsehoods — at the cost of ever more incoherence.
  Within the regime that keeps generations coherent, the answer is unambiguous.
- **The anchor is scale-relative** (an engineering choice, commit `8d20fb0`), so "λ=3 ≈ the supervised
  axis" is a cosine-0.95 approximation, not identity.
- **Everything here is still a *null*.** All of it says truth is the concept these miners *fail* to
  recover causally. The decisive next step is a **positive control** (below).

## What's next

1. **The positive control — `refusal`.** Every result so far is "DCT/MAG fail to find truth as a causal
   lever." The clinching move is to run the identical pipeline on a concept we **expect** to be causally
   load-bearing (refusal / safety behavior) and show the method **does** recover it. That converts the
   binary null into a **spectrum**: *a miner recovers a concept in proportion to how causally
   load-bearing it is.* The validated OLMo judge is the prerequisite that unblocks the behavioral axis of
   that spectrum.
2. **Human-vs-judge agreement (κ)** on a sample of the real steering completions — to certify the OLMo
   judge on messy generations, not just the clean statements the 0.970 gate used (carried over from
   Part 1 §9).
3. **Write `DCT_VS_MAG_ON_TRUTH.md`** — the formal head-to-head findings doc that Part A here summarizes.

---

# PI FEEDBACK (2026-07-23 meeting) — analysis and the reachability reframe

*Added after the meeting. The PI's notes, decoded against the project state, plus the literature they
point at. This section supersedes the "What's next" ordering above where they conflict.*

## F1. The feedback, item by item

**1. "Make a better visualizer for the linearity cosine shift score."**
This is E3 (§A4). The old `plot_mag_linearity.png` showed only ε_Q for v_Q and buried the actual
finding — the cosine structure. Done: `src/viz_mag_linearity.py` → **`plot_mag_linearity_v2.png`**,
two panels over all four datasets: (left) cos(direction, prefix shift Δ<sup>Q</sup>) for v_Q
(0.84–0.98) vs the supervised truth axis and DCT's top lever (both inside an |cos| < 0.1
"orthogonal band"); (right) ε_Q per direction, with the truth axis and DCT lever sitting **above**
the ε_Q = 1 "explains nothing" line (1.28–1.40 — worse than predicting zero shift).

**2–5. The control-theory reframe.** The PI's core proposal, decoded:

> Stop searching for single steering **directions**. Define a **target set** in the output space —
> specifically in **J-space** (the Jacobian-lens space of Anthropic's global-workspace paper, or
> equivalently the space DCT's U matrix lives in at the target layer) — e.g. "activations whose
> truth readout says FALSE, while staying in the coherent/on-distribution region." Then compute the
> **backward-reachable set** of that target at the source layer: the set of input-layer
> perturbations that can land in the target. The tools are from robotics safety (set propagation
> for dynamical systems); because the map is (locally) linear, "the actual tools are just linear
> algebra" — preimages of polytopes/ellipsoids under a linear map, Jacobian (pseudo)inverses, SVD.

Why this is the right formalization for us: every result in this doc is a statement that some
**single direction** fails to be a truth lever. The reachability frame upgrades the null to a
statement about **sets**: *is the "assert-falsehood, stay-coherent" region of target-layer
activation space backward-reachable at all from bounded source-layer perturbations?*
- If the backward-reachable set is **empty or off-distribution** → the null becomes a theorem-shaped
  claim ("no coherent lie is reachable from layer-11 perturbations of norm ≤ input_scale"), not an
  absence of evidence.
- If it is **non-empty but thin/curved/off-axis** → it explains *why* every single-direction probe
  missed it, and hands us the steering vector set directly (steer *within the set*, not along an
  axis).

Our own artifacts already contain the ingredients: DCT's fits are linearized maps from source →
target layer (cities 11→20, common_claim 13→22, d=2304, `input_scale` = the perturbation-norm
bound we already sweep), the linear probe `w·h > c` defines the target halfspace, and the U-space
anchored run (Arm B, built and awaiting execution) is a first-order scalar version of exactly this
idea — it anchors the *effect* toward the truth readout; the reachability version replaces "one
anchored effect direction" with "the full preimage of the effect set."

## F2. What the four references actually are (deep-read summaries)

**(a) arXiv 2509.21528 — "Preemptive Detection and Steering of LLM Misalignment via Latent
Reachability" (Karnik & Bansal, Stanford Safe & Intelligent Autonomy Lab, Sept 2025).** The
load-bearing citation. Treats greedy LLM generation as a discrete-time control system in
residual-stream latent space z_t, defines a failure set as the sub-zero level set of a scalar
margin ℓ(z) (a toxicity-classifier score), and computes a **Backward Reachable Tube**
B = {z : V(z) ≤ 0} via a Bellman-style value recursion V(z_t) = (1−γ)ℓ(z_t) + γ·min(ℓ(z_t),
V(z_{t+1})) learned by an MLP (DeepReach-style neural reachability — approximate, no certificates).
Steering is a **least-restrictive filter**: intervene only when V(z) drops below a margin, choosing
the perturbation in an L² ball that most increases V. ~98% detection, flags unsafe trajectories
7–10 tokens early, 54–85% fewer unsafe generations across 5 LLMs. **No code released yet** (project
page says "coming soon"). Direct mapping to us: their ℓ ↔ our probe margin; their token-step BRT ↔
our **depth-wise** (layer-to-layer) preimage — and our case is *easier*: one linearized 9-block hop
instead of a recurrent nonlinear rollout, so closed-form set preimages can replace their learned
value function.

**(b) arXiv 2603.00140 — "Steering Away from Memorization: Reachability-Constrained RL for
Text-to-Image Diffusion" (Karnik, Kim, Koyejo, Lee, Bansal, Feb 2026).** Same lab, same skeleton
applied to diffusion: failure set from a memorization margin, BRT approximated by a safety critic
Q<sup>safe</sup> trained with the discrete-time HJ recursion, minimal perturbations chosen by
constrained RL (SAC + Lagrange multiplier) that trade task reward against staying out of the tube.
Two transferable lessons: (i) they compress the control space (77×768 CLIP embedding → 64-dim VAE
latent) before doing reachability — we should likewise work in a reduced basis (top DCT factors /
J-lens vectors / PCA), not raw d=2304; (ii) their constrained-MDP objective is the template for our
observed trade-off: "minimal perturbation that reaches the FALSE set **subject to a coherence
constraint**" — formalizing the coherence-vs-lying dissociation that every experiment here found.

**(c) Anthropic, "A Global Workspace in Language Models" (transformer-circuits, July 2026).** This
is where the PI's "j-space / j-lens" vocabulary comes from. The **Jacobian lens**: J_ℓ =
E[∂h_final/∂h_ℓ] averaged over pretraining prompts — a per-layer linear map to output-vocabulary
space that corrects for representational drift (the "output matrices" the PI said to look at).
**J-space** = points expressible as sparse nonnegative combinations (k ≤ 25) of J-lens vectors — a
capacity-limited **cone, not a direction**, in the middle third of the network, carrying <10% of
activation variance. Their key functional result for us is **selective engagement**: the same
information can sit in the residual stream, be linearly decodable, and be *causally inert* for
automatic tasks while being decisive under explicit-report framing. That is our entire
decodable-≠-causal null restated as workspace gating — and it independently predicts our MAG
finding that the "being-asked-about-truth" direction is real, linear, and orthogonal to the truth
content axis. It also motivates Arm A1/A2 (question-mode conditionality) as the behavioral test of
exactly this gating.

**(d) arXiv 1910.13272 — "Feedback Linearization for Uncertain Systems via RL" (Westenbroek,
Fridovich-Keil, Mazumdar et al., Tomlin/Sastry lab, 2019).** ⚠️ **This ID is not a reachability
paper.** It's model-free feedback linearization: learn the correction to a nominal linearizing
controller u = A(x)⁻¹(v − b(x)) so a nonlinear plant tracks a linear reference model. The
conceptual echo is real — A(x)⁻¹ is literally "map desired outputs back through the inverse of the
local linear map," the same algebra as our Jacobian-inverse step — but it contains no target sets,
no set propagation, no reachability tooling. **Ask the PI whether this ID was intended** (as the
feedback-linearization formalization of "invert the output map") **or a typo** for a set-propagation
survey (e.g. Althoff/Frehse/Girard's set-propagation review, or hybrid-zonotope backward
reachability for neural feedback systems, arXiv 2303.10513 / 2310.06921).

**(e) The tooling landscape (surveyed alongside the four references).** Three tiers, by how much
machinery we'd import:
- **Tier 0 — plain linear algebra (start here):** for a single linearized hop h_tgt ≈ J·h_src + b,
  the backward-reachable set of a halfspace target is closed-form (pull w back through Jᵀ / J⁺,
  intersect with the norm-budget ball and the data-support region). No framework needed.
- **Tier 1 — NN preimage tools, if we want certified multi-block bounds without linearizing:**
  **INVPROP** (Kotha et al., NeurIPS 2023, in α,β-CROWN — provably bounds preimages of
  linearly-constrained output sets, GPU, verified at 167k neurons) and **PREMAP** (JMLR 2025, its
  successor: under- *and* over-approximations via branch-and-bound-refined linear relaxations).
  Nothing in this literature has been run on a multi-block transformer with attention — we would be
  the first, which is a risk and an opportunity.
- **Precedent at our exact scale:** **LiSeCo** (arXiv 2405.15454) already does linear-probe
  halfspace constraints with closed-form minimal-norm projection **on gemma-2-2b** — a *forward*
  safe-set controller, not backward reachability, but proof the probe-as-halfspace machinery works
  at d=2304. Conceptual ancestor: "Taming AI Bots" (Soatto et al., arXiv 2305.18449) — LLM
  controllability over meaning space via prompts. Classical background: zonotope/support-function
  set propagation (closed under linear maps); grid-based Hamilton–Jacobi is exact but dies at ~6
  dimensions, so it only enters via learned value functions (as in BRT-Align), never directly at
  d=2304.

## F3. The concrete next experiment (proposed)

**"Backward-reachability audit of the truth set"** — entirely linear algebra on artifacts we
already have; no cluster time needed for v0:

1. **Target set** at the target layer (20 / 22): T = {h : w·h ≤ c − δ} ∩ D, where w = the linear
   probe (or mean_diff readout) at the target layer, δ a margin ("reads FALSE with margin"), and
   D a coherence proxy (e.g. Mahalanobis ball around the activation distribution — stay
   on-distribution).
2. **Map**: DCT's learned linearization of the source→target map (and/or a fresh empirical Jacobian
   J of the 9-block map at sample points; note the exact objects DCT stores — this is the PI's
   "look how they define their output matrices" homework on our own code).
3. **Backward step**: compute the preimage J⁻¹(T − f(h₀)) intersected with the perturbation budget
   ball ‖Δh‖ ≤ input_scale at the source layer. For a halfspace target under a linear map this is
   closed-form: the preimage of {u : w·u ≤ b} under Δh ↦ JΔh is {Δh : (Jᵀw)·Δh ≤ b} — the whole
   question becomes **how large is Jᵀw, and where does it point?** If ‖Jᵀw‖ is tiny (the map's
   row space is near-orthogonal to the probe), the FALSE set is unreachable within budget — the
   null, formalized. If Jᵀw is substantial but points off the truth axis — nonlinear routing,
   found constructively.
4. **Report**: reachable/unreachable per dataset per budget, the angle of Jᵀw to mean_diff@source,
   and the volume/thinness of the reachable slice inside the coherence set.

This subsumes Arm B: the U-anchor run asks "can a scalar bias pull DCT's effect toward w"; the
audit computes the answer directly. Run Arm B as planned (it's built and cheap) — its result
becomes the behavioral validation of whatever the audit predicts.

## F4. Follow-ups filed

- Better linearity visualizer: **done** (`plot_mag_linearity_v2.png`).
- Deep-research prompt for the full literature pass (tool selection, scale limits at d=2304,
  preimage algorithms): `docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md` — paste into Claude deep
  research.
- Confirm arXiv 1910.13272 with the PI (intended vs typo).
- Access needed: none so far — all four sources were fully readable except that neither Bansal-lab
  paper has released code.
