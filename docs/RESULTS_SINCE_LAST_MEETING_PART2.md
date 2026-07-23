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
