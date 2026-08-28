# Project Progress — Full Retrospective

*Everything from the first commit (2026-06-22) to today (2026-07-30): the question chain, every
experiment, every number, the literature it sits in, and what is actually established versus still
open. Written to be read cold. Every quantity below was read off a committed doc or recomputed from
an artifact on disk; nothing is from memory.*

**Scope of the record:** 145 commits over 5.5 weeks · 11,489 LOC in `src/` · 3,765 LOC of tests
(299 collected, 1 skipped) · 18,848 lines of documentation across 23 findings docs, 10 implementation
plans, and 6 design specs · 38 SLURM job scripts and cluster runbooks in `deltaai/` · 4 datasets ·
1 model (`google/gemma-2-2b`) · 2 clusters-worth of runs on NCSA DeltaAI GH200.

---

## 0. The one-paragraph version

The project started as a replication question — *is truth encoded linearly in an LLM's activations?* —
and answered it (yes, on clean data: 99.0% linear probe accuracy, XGBoost adds +0.003). It then asked
the harder follow-up — *is that readable direction one the model is actually driven by?* — and got a
robust **no**, which survived four independent attempts to break it. After the PI reframed the problem
as **control theory**, the null was upgraded from "this direction doesn't work" to a per-statement
**backward-reachability certificate**: a computed budget ε\* guaranteeing the model's internal truth
readout crosses into the FALSE halfspace. The certificate is *arithmetically exact* (calibration factor
0.9997 at its own linearization point; R²=0.999 for the readout-vs-push fit) and **behaviorally inert**
(lie rate 3.5% under maximum push versus a 3% unsteered baseline). Diagnosing why produced four named
mechanisms, three of which are transfer failures rather than linearization failures. That
*certified-reachable-but-behaviorally-inert dissociation with mechanism* appears to be unclaimed in the
literature and is the paper. The publication gate — a refusal positive control that proves the
instrument can detect actuation when actuation exists — is fully implemented and has not yet run.

---

## 1. The question chain

Each question was answered, and its answer created the next one. This is the actual spine of the
project, and it is worth presenting this way because every negative result here is a *sharpening*, not
a dead end.

| # | Question | Answer | Where |
|---|---|---|---|
| Q1 | Is truth linearly encoded? | Yes on clean data (0.990), partly non-linear on messy (gap +0.082) | §3 |
| Q2 | Is the readable truth direction one of the model's dominant *causal* levers? | **No** — ~1.2× random alignment | §4 |
| Q3 | Is that null a hand-read artifact? | No — instrumented with a validated judge; bounded to Δ = −0.010 | §5 |
| Q4 | Is it an artifact of DCT's objective? | No — a second, unrelated miner (MAG) lands in the same place | §6 |
| Q5 | Does seeding the search *at* truth rescue it? | No — no λ on the whole path produces a truth lever | §7 |
| Q6 | Reframed by the PI: is the FALSE **set** backward-reachable at all? | Yes, formally — and the certificate is behaviorally inert | §8 |
| Q7 | Is the inertness a math error or a real dissociation? | Real — the certificate is exact at its own point (0.9997) | §9 |
| Q8 | Does the instrument work on *any* concept? | **Open.** Refusal positive control built, not run | §11 |

---

## 2. Setup and constraints (what the whole thing runs on)

**Model:** `google/gemma-2-2b` — chosen to match Julian's A-LQR paper so results are directly
comparable. 26 transformer layers (27 hidden states including embeddings), hidden dimension **2,304**,
fp32 weights ≈ 9.7 GB across 3 shards.

**Datasets** — a deliberate clean→messy gradient, because the core hypothesis was that *the answer
depends on what kind of truth you test*:

| Dataset | Statements | Character | Prediction going in |
|---|---:|---|---|
| `cities` | 1,496 | "The city of X is in country Y." — rigid template | linear |
| `sp_en_trans` | 354 | Spanish–English translation judgments | linear |
| `companies_true_false` | 1,199 | Company business descriptions | mild non-linear |
| `common_claim_true_false` | 4,450 | Heterogeneous world claims | non-linear |

All 50/50 true/false. **The prediction held, monotonically.**

**Compute reality.** A 64-factor DCT smoke fit took **~90 minutes on CPU**; the same fit takes
**~4 seconds on a GH200** — roughly **800×**. Real fits (512 factors, 64 statements, 30 iterations)
run ~9 min ≈ 0.15 GPU-hr each. That ratio is why the project moved to NCSA DeltaAI (SLURM, partition
`ghx4`, GH200 120 GB, ARM64, account `bhhv-dtai-gh`). ~475 GPU-hr remain of the allocation.

**Three virtualenvs, for a real reason.** `.venv` (torch 2.12 / transformers 5.x, has scikit-learn) for
the probe work; `.venv-dct` and `.venv-dct-gpu` pinned to **transformers 4.51.3** because `dct.py`'s
`SlicedModel` mutates model internals that transformers 5.x reorganized; `.venv-judge-gpu` for the
OLMo judge, which needs a *newer* transformers than DCT allows. Three envs is not sloppiness — it is
two hard version conflicts.

---

## 3. Chapter 1 — Geometry of Truth (the replication)

**Method.** Extract the last-token residual-stream activation at every one of 27 layers for every
statement → `(27 × N × 2304)`. Per layer, train a linear probe (logistic regression, standardized,
`max_iter=2000`) and XGBoost (300 trees, depth 4, lr 0.1), 80/20 stratified, seed 42. The metric is the
**non-linear gap** = `xgb_acc − linear_acc`.

**Result — the gap grows monotonically with messiness:**

| Dataset | Best layer | Linear acc | XGBoost acc | Gap | Verdict |
|---|---:|---:|---:|---:|---|
| cities | 11 | **0.990** | 0.993 | **+0.003** | Linear |
| sp_en_trans | 7 | 0.972 | 1.000 | +0.028 | ~Linear |
| companies_true_false | 14 | 0.917 | 0.954 | +0.038 | Mild non-linear |
| common_claim_true_false | 13 | **0.706** | 0.788 | **+0.082** | Non-linear headroom |

Gap sequence: **0.003 → 0.028 → 0.038 → 0.082**, while decodability falls **0.99 → 0.71**. Messier
truth is both *harder to read* and *less linear*. cities replicates Marks & Tegmark cleanly.

**Direction agreement** (cosine between the mean-difference direction and the probe-gradient direction,
at the best layer): sp_en_trans **0.563**, cities **0.413**, companies **0.204**, common_claim
**0.085**. Note the honest caveat that was flagged at the time and still holds: even the best cosines
(~0.5) are far below 1.0 — the two supervised estimates of "the truth direction" are *correlated but
genuinely different vectors*. Geometry alone was never going to be proof.

**Two bugs caught here, both of which would have inverted the conclusion:**

1. Gemma's tokenizer **left-pads**; the first `extract.py` assumed right-padding and grabbed the wrong
   token. cities read **0.45** — chance. Fixed by indexing the true last non-pad token via the
   attention mask.
2. `summary.py` used the *max gap across layers* instead of the *best-layer* gap, which had flagged
   cities as the **most** non-linear dataset — the exact inverse of the finding.

The lesson recorded at the time, and the one that shaped every later smoke test: **check a number, not
just a tensor shape.**

---

## 4. Chapter 2 — DCT, and the "decodable ≠ causal" result

**What DCT asks.** Deep Causal Transcoding (the MELBO/DCT paper's own implementation, used unmodified)
searches for the perturbations at a source layer that produce the **largest** change at a target layer.
Those are the directions the model is most *causally sensitive* to. It finds 512 at once, label-free,
returning `V` (input steering vectors) and `U` (their downstream effects). Config: 512 factors,
cities 11→20, common_claim 13→22.

**The synthesis test.** Compute the supervised truth directions at DCT's own source layer, then ask
three questions.

**Single-vector alignment** — is truth *one* of the 512 DCT vectors?

| Layer | Dataset | vs mean-diff | vs gradient | supervised agreement (ref) |
|---|---|---|---|---|
| 6 (control) | cities | 0.101 (1.5× rand) | 0.054 (0.8×) | 0.161 |
| 6 (control) | common_claim | 0.106 (1.8×) | 0.055 (0.6×) | 0.066 |
| **11 (truth peak)** | **cities** | **0.074 (1.2×)** | **0.075 (0.8×)** | **0.413** |
| **13 (truth peak)** | **common_claim** | **0.094 (1.1×)** | **0.057 (0.8×)** | **0.085** |

**Subspace test** — is truth a *combination* of all 512? (chance ≈ 512/2304 ≈ 0.22)

| Dataset (peak layer) | mean-diff | gradient |
|---|---|---|
| cities @ 11 | 0.288 (1.3× chance) | 0.253 (1.1×) |
| common_claim @ 13 | 0.316 (1.4× chance) | 0.223 (1.0×) |

**Why this is not a layer artifact:** at layer 11 the two *supervised* directions agree at cosine
**0.413** versus 0.161 at layer 6 — truth is maximally, cleanly defined exactly where we looked. DCT's
512 most causally-salient directions still miss it.

**The claim, stated precisely:** a direction can be read at 99% accuracy by a probe and simultaneously
sit essentially orthogonal to the space of directions the model is most sensitive to. **Decodability ≠
causal salience.** This is *causal* evidence for the non-identifiability caveat that Chapter 1 could
only raise on geometric grounds.

**What it is not:** not "DCT is broken" (it produces valid, diverse, behavior-changing vectors), and
not "truth isn't real" (probes read it at 99%). It is a bounded claim about the *relationship* between
readability and causal salience.

---

## 5. Chapter 3 — Instrumentation, and the measurement bug (the credibility chapter)

Two pieces of the Chapter-2 story were **read by hand**: I eyeballed the DCT vectors' completions and
the steering completions. This chapter turned both into counts, and in doing so found the single most
valuable bug in the project.

**The judge.** OLMo-3-7B-Instruct, chosen over the Anthropic API (needs a key, billed, closed judge is
harder to cite) and TruthfulQA-7B (wrong format, can't do interpret mode). OLMo is fully open (weights,
data, recipe), free, local, and general enough to run both judging modes from a rubric prompt.
**Validation gate: ≥0.85 agreement with gold labels. It passed at 0.970 (97/100).**

**The bug, and why it matters more than the result.** The first cluster run reported 3/10 DCT vectors
"manipulating truthfulness." Before reporting it, a hand drill-in on the *zero-steering* completions
found this:

> Prompt "Two plus two equals" → "**four.** That's the message from a new study that found people who
> eat…" — judged **FALSE**, reason "completion is unrelated to the stem."

The answer was correct. gemma-2-2b is a **base** model: it doesn't stop, it rambles for the full
generation length, and the judge was scoring the whole paragraph. Correct answers were failing on their
tails. **This is a measurement-validity bug — the numbers were real and measuring the wrong thing.**

Critically: **the 0.970 gate did not catch it**, because the gate validated the judge on clean
single-claim statements, not on long rambling generations. *A passing gate on clean inputs does not
certify labels on messy ones.* That lesson is now load-bearing in every later plan.

**The fix, and the proof it worked.** Four changes: generation cut 24 → ~8 tokens; answer-only rubric;
factual-only prompt stems; 8 → **32 prompts per strength** for power. Result: 576 judged completions per
dataset, 1,152 total. **The unsteered baseline TRUE rate rose 0.50 → 0.81** — the artifact is gone. (The
residual 0.19 is the 2B base model genuinely flubbing ~5 of 32 prompts; it is *identical* across
datasets and *symmetric* across sign, so it cannot fake a directional effect.)

**What the corrected run showed:**

- **Interpret: 0/10 vectors flip a fact on cities, 0/10 on common_claim, 0/20 pooled.** Rule of three
  gives a **95% upper bound of ~14%** on the fraction of top DCT vectors that could flip an established
  fact. The old 3/10 came from a loose rubric counting confabulation ("Tokyo founded by JFK"), register
  shifts (into code, into Vietnamese), and gibberish — while the *target fact* survived in every case.
  Top causal levers are about **geography, format, and tone**, never a truth switch.
- **Steer: no lie-asymmetry, and now bounded.** Pooled Δ = **−0.010, 95% CI [−0.042, +0.023]**;
  TOST equivalence passes on every slice; prompt-clustered bootstrap agrees. The previous run could only
  *assert* "about zero"; this run **proves no lie-asymmetry larger than ~0.04 exists**.
- **The one real effect is symmetric degradation.** TRUE rate falls from the ~0.81 plateau to ~0.55–0.65
  at **both** ±120 ends (Cochran-Armitage **z = −5.06, p < 1e-6**), while the verdict×sign omnibus is
  not significant (**χ² = 4.49, p = 0.34**). Only *magnitude* matters, not *direction*. That is the
  fingerprint of a direction that **breaks** the model rather than **flipping its truth value**.

**A correction to the previous meeting's claim, stated plainly:** the "a few real falsehoods" I had read
by hand turned out to be symmetric byproducts of degradation. The truth axis is a **degradation lever,
not a truth lever.**

**The non-linear escape hatch, closed separately.** On common_claim, XGBoost reads truth **+0.057**
better than a linear probe from the full residual stream — real non-linear structure. Project onto DCT's
top causal directions and that gap **collapses to ≈0 (−0.01)**, no better than random projections. DCT
misses truth non-linearly *as well as* linearly. It doesn't miss truth *because* truth is non-linear.

*A null result needs more defense than a positive one* — hence the full battery (McNemar pairing, TOST,
clustered bootstrap, Cochran-Armitage trend, Benjamini-Hochberg correction) in
`INVESTIGATION_steering_validity.md` §6.

---

## 6. Chapter 4 — MAG: a second miner, to kill "it's just DCT"

**The objection:** *maybe truth just isn't the kind of thing DCT's objective finds.* So we built a
miner with a completely different principle. **MAG (Mining via Activation Geometry)** never steers.
It prepends the model's own question to the input — `Q‖p` = "Is the following statement true? {p}" —
and measures the activation shift **Δ^Q(p) = m(Q‖p) − m(p)**. Eight operators (Direct, Prefixed,
Answered, Verdict, InputDelta, QuestionDelta, Interaction, FewShot) read that question-conditioned
geometry different ways. Five probes: E1 readability, E2 disagreement, E3 linearity, transfer/rank,
E4 calibrated steering. Built over 14 commits with a per-task reviewer gate.

**E1 — truth is readable from every view, including the question-shift.**
cities: Direct **0.993**, InputDelta **0.993**, Prefixed 0.985, FewShot 0.991; even projected into
DCT's own top-10 causal subspace it reads **0.947** (top-50: 0.989). common_claim: Direct **0.714**,
InputDelta **0.720**, Prefixed 0.691, against a random-10 floor of 0.633. MAG's premise is sound —
asking the model about truth really does move activations along a truth-readable direction.

**E2/E3 — the sharpest MAG finding: the model never says "false."** The fully-unsupervised dream was
to use the model's own yes/no answer `y^M` as the label. It does not work, bluntly: across all four
datasets the model answers "yes" to essentially everything — **0/354, 0/1496, 0/1200, and 1/4450**
"no" answers. And the activations *at the answer position* read truth at chance: **0.62** cities,
**0.50–0.51** companies/common_claim, **0.48** translations. The model **contains** truth but does not
**express** it when asked. (This also makes E2 degenerate, and we read no finding into it.)

**E3 — the question-shift is a crisp linear direction, orthogonal to truth.** The MAG shift direction
`v_Q` explains most of the shift (cosine **0.970** cities / **0.836** common_claim), but the supervised
truth axis and DCT's top lever both sit at cosine **≈ 0** to it (mean_diff −0.006 / +0.010; DCT −0.015 /
+0.044). There is a real "I am being asked whether this is true" direction — and it is a *different*
direction from "this statement is true." The improved visualizer built at the PI's request
(`plot_mag_linearity_v2.png`) shows `v_Q` at 0.84–0.98 while truth and DCT sit inside an |cos| < 0.1
band, and shows their ε_Q at **1.28–1.40** — *above* the "explains nothing" line of 1.0, i.e. worse than
predicting zero shift.

**E4 — behavioral, and honestly caveated.** The gold-contrast axes do produce sign-dependent verdict
flips (cities `sup_mean_diff`: 79% flip at τ=−1 versus ~0% at τ=+1; common_claim 100% vs 0%). We treat
this as **suggestive only** for three stated reasons: the metric is the model's own single verdict
token rather than the OLMo judge; it rides on 19–41% incoherence; and it is a 24-statement matched set.
The purely-mined operators (`mag_u_yM`, Answered, FewShot) are mostly inert, and `resid_pc1` flips at
**both** signs — the degradation signature again.

**Verdict: objection #1 closed.** Two unrelated unsupervised miners both fail to find truth as a causal
lever in gemma-2-2b base.

---

## 7. Chapter 5 — Warm-started DCT: seeding the search at truth itself

**The objection:** *maybe DCT just started in the wrong place.* So we added a soft anchor of strength λ
that ties DCT's causal search back to the supervised truth seed, and swept λ ∈ {0, 0.3, 1, 3} × 2 seeds
× 2 datasets = **16 warm fits**, then steered with every result and judged with the *same* OLMo judge
and the *same* two-sided sweep as Chapter 3.

**The control passed first.** cos(warm direction, supervised seed) climbs the intended ladder, nearly
identically on both datasets:

| λ | cities | common_claim |
|---|---|---|
| 0.0 | 0.03 | 0.02 |
| 0.3 | 0.28 | 0.25 |
| 1.0 | 0.72 | 0.66 |
| 3.0 | 0.95 | 0.95 |

So we genuinely swept from "free DCT search" to "essentially the supervised axis." Any behavioral change
across λ is caused by the anchor.

**The result: no truth lever at any λ.** In every one of the 11 panels, on both datasets, the FALSE
fraction is pinned to the floor (**0.03–0.09**). Not the raw supervised axes, not DCT's free top factor,
not any anchored direction at any λ. The interpolation runs from **inert** (high λ) to **mild degrader**
(low λ) and never passes through **truth lever**.

What steering *does* do, when it does anything, is destroy coherence — and only on the messy dataset,
only for the DCT-derived low-λ directions. The cleanest case is `cold_top` on common_claim: as τ goes
+1 → −1, INCOHERENT climbs **0.03 → 0.66** while FALSE stays ≤0.16 and TRUE at +τ is 0.94.
Sign-dependent in **coherence**, not in **truth**. And the harder you anchor toward truth, the **more
inert** the direction becomes — the opposite of unlocking causality.

**Verdict: objection #2 closed, and the null is now stronger than Chapter 2's.** It is not merely that
the supervised axis is inert; **no point on the entire geometric path** from the supervised axis to
DCT's free causal search is a truth lever.

---

## 8. Chapter 6 — The PI reframe, and the backward-reachability audit

**2026-07-23 meeting.** The PI (robotics/control background) redirected the project in one move:

> Stop searching for single steering **directions**. Define a **target set** at the output layer — e.g.
> "activations whose truth readout says FALSE, while staying coherent" — and compute its
> **backward-reachable set** at the source layer. The formalization is control theory; the actual tools
> are just linear algebra.

**Why this was the right call.** Every result in Chapters 2–5 is a statement that some *single
direction* fails. The reachability frame upgrades that to a statement about **sets**: is the
"assert-falsehood, stay-coherent" region backward-reachable at all from bounded source perturbations?
An empty or thin preimage turns an absence of evidence into a theorem-shaped claim.

**The math, which is genuinely simple.** For a locally linear hop `h_tgt ≈ J·h_src + b`, the preimage of
the halfspace `{u : w·u ≤ b}` is `{Δ : (Jᵀw)·Δ ≤ b}`. The whole question becomes **how large is ‖Jᵀw‖,
and where does it point?** Per statement: margin `m = ‖Jᵀw‖` computed by `torch.func.vjp`, gap
`g = w·h_tgt − t02`, minimal budget **ε\* = g/m**. No search, no repeated forward passes.

**The five-phase audit** (DeltaAI GH200, 2026-07-24; cities 11→20, common_claim 13→22):

| Phase | Question | cities | common_claim |
|---|---|---|---|
| **P1** linear reachability | Is FALSE reachable within budget? | reachable-candidate, median **ε\* = 2.80** | reachable-candidate, median **ε\* = 10.69** |
| **P2** Jacobian SVD | Do Jᵀ's dominant directions align with DCT's concept subspace? | overlap **0.062** (≪0.3 threshold) | overlap **0.047** |
| **P3** behavioral steering | Does it flip the verdict? | **No** — lie rate **3.5%** vs 3% baseline | **No** — **5.5%** vs 3% |
| **P4** per-layer J-lens | margins by layer, decl vs quest | computed | computed |
| **P5** linearization trust | Is ε\* inside the valid region? | trust radius **2.99** → **0.94× (inside)** | trust radius **4.87** → **2.20× (OUTSIDE)** |

**Two distinct failure modes, and only one of them is interesting:**

- **common_claim — the certificate is extrapolation.** ε\* = 10.69 is 2.2× the trust radius of 4.87, so
  "reachable" is evaluated far outside where the first-order model holds. Empirically **0% of statements
  cross g ≤ 0** under the predicted push; median g stays at +60. The P5 diagnosis is confirmed
  empirically. This case is disqualified, not surprising.
- **cities — a genuine readout↔behavior dissociation.** ε\* = 2.80 is **inside** the trust radius
  (0.94×), the readout **does** sit on the FALSE side, and behavior **still** does not move. This one
  cannot be blamed on linearization error. **This is the scientific case.**

In both, **P2 had already warned**: Jᵀw's top-16 singular subspace overlaps DCT's concept subspace at
only ~0.05, versus the spec's stop-and-diagnose threshold of 0.3.

### The deep audit — four mechanisms (D1–D4)

Four parallel analyses sharpened, and in one place **corrected**, the story.

**D1 — actuation is linear, correctly signed, and 8–35× attenuated.** Per-statement fits of
`g_read ~ scale`: median **R² = 0.999** (cities) / 0.990 (common_claim), **0% wrong-sign** in both. But
the realized crossing budget is **ε_realized ≈ 97 in both datasets** against ε\* of 2.9 / 12.3 — a
**35× / 8× shortfall**. Since ε_realized ≈ 1–2× `input_scale`, a real crossing needs a perturbation
comparable to the activation's own norm. Worse, for cities the predicted margins don't even rank-order
actuation (**Spearman ρ = 0.002**) — the certificate loses its *ordering* content, not just its gain
calibration. And the protocol shift causing it is tiny: ε\* is computed at the full-statement last
token; steering happens at the stem, **one word removed**.

**D2 — behavior is inert, strictly.** The earlier "mild coherence degradation" read was too generous.
INCOHERENT is **not** dose-dependent (3 of 4 arms n.s.; one marginal at p≈0.10). The 131 FALSE verdicts
concentrate on a handful of problem prompts that recur **identically at scale 0** (an MCQ-style
"a. heart. b. aorta" completion judged FALSE at every scale; judge factual errors on Abidjan/Maturín).
The single significant trend (p = 0.012) runs **backwards**. Where incoherence is scale-correlated at
all, it is topic-derailment into generic institutional filler ("Budapest is in → the process of
building a new…"), **sign-symmetric** — an off-manifold nudge, not negation. **Verdict: not even an
incoherence knob.**

**D3 — the Jacobian is structured and concept-blind on its input side** (this *corrects* the audit's
earlier "near-isotropic" claim). The spectrum is not flat: effective rank ~21–23% of 2304 with a
two-regime decay (s64/s1 ≈ 0.11–0.16). The hop routes through a compact channel, and that channel's
**output** contains the truth axis — `mean_diff_tgt` puts **74% (cities) / 51% (common_claim)** of its
energy in the top-64 left-singular vectors, **17–125× random**. But its **input** is concept-blind:
`md_src` is only **1.8–7× random**, DCT overlap 0.05, and on common_claim the probe-gradient direction
is *at or below* random (0.58–0.73×). **The truth readout is downstream-visible but not
upstream-addressable by any source-side concept direction.** Any linear concept-steering method
operating in this band hits the same structural ceiling. (Side finding: cities' local Jacobian geometry
splits by label — true statements have higher effective rank and lower s1, p < 0.001 — absent in
common_claim.)

**D4 — controllability is front-loaded; we steered past the peak.** All readouts peak at layers 0–8 and
decay monotonically, with no late resurgence; at the target layers every readout has lost **65–87% of
peak**. The behavioral *verdict* readout has **2–7× smaller absolute margin than the truth probe at
every layer**, and quest-mode verdict controllability — the pathway the judge actually elicits — crosses
*below* decl-mode right at the source layers. In 3 of 4 dataset×mode combos there is a distinct verdict
peak at **layers 5–9**, **1.5–3× better-conditioned** than the layers we used (11/13).

### The refined headline

The certificate fails behaviorally **not because linearization breaks** (R² stays ≈0.99) but because of
**three transfer failures**:

1. **Context transfer** — Jacobian gain collapses 8–35× across a one-word context shift while sign and
   linearity survive.
2. **Population transfer** — the probe's boundary `t02` does not transfer from statement tokens to
   generation prefixes. cities prefixes read g ≈ −52 ("deep FALSE") while the model completes
   *truthfully* — the probe dissociates from behavior **at baseline**, before any steering.
3. **Input-side concept misalignment** — J's high-gain channel is not addressable from source-layer
   concept directions.

And the structural point: **behavioral validation inherently forces transfers (1) and (2)**, because
behavior only exists during generation, which is never the point where the certificate is computed. A
behaviorally meaningful reachability certificate must be robust to exactly the shifts this one is not.

---

## 9. Chapter 7 — Horizon 0: closing the audit's own loopholes

Seven validations, each designed to close a specific hole a reviewer would poke. All ran. **These are
the numbers that make the audit defensible rather than merely suggestive**, and several were verified
fresh from the artifacts for this document.

### 0.1 — The same-point control (the decisive one)

Does the certificate work at its *own* linearization point? Teacher-force the full statement with the
injection hook active and read g at the statement's final token — exactly where ε\* was computed.

| | cities (n=199) | common_claim (n=197) |
|---|---:|---:|
| predicted margin `m_pred` (median) | 7.2944 | 6.2478 |
| realized slope at the same point | 7.3102 | 4.3554 |
| **calibration = realized/predicted** | **0.9997** (IQR 0.9910–1.0117) | **0.7025** (IQR 0.6046–0.8023) |
| realized slope one word earlier (stem) | 0.5409 | 0.8248 |
| **context factor** | **0.0729** → **13.7× collapse** | **0.1944** → **5.1× collapse** |

**This is the single most important number in the project.** On cities the certificate is *exact* at its
own point — calibration 0.9997. The entire 8–35× shortfall is therefore **context transfer, not a math
error**. The vjp margins, the ε\* arithmetic, and the steering hook all do what we believe they do. The
decomposition is clean, and it is what lets "the nudge is brittle" be a defensible claim rather than a
hedge. (common_claim's 0.70 shows partial local miscalibration on top of the context loss — a second,
smaller problem confined to the messy dataset.)

### 0.2 — Refitting the truth probe on generation stems (D2, quantified)

Is the prefix-transfer failure a *recalibrable threshold shift* or an *absence of signal*? Labels are the
existing scale-0 judge verdicts on the model's own completions.

| | cities (n=158) | common_claim (n=178) |
|---|---:|---:|
| base rate P(TRUE) | 0.943 | 0.944 |
| old threshold `t02`, accuracy | 0.057 | 0.944 |
| old threshold, **balanced accuracy** | **0.500** | **0.500** |
| recalibrated 1-D threshold on the same `w` | 0.943 (`t02_stem` = −251.28 vs old +89.45) | 0.944 (−264.16 vs −105.48) |
| refit LR probe, 5-fold CV | 0.930 | 0.944 |
| **XGBoost, 5-fold CV** | **0.943** (gap +0.013) | **0.944** (gap 0.000) |

**Balanced accuracy is exactly 0.500 in both datasets** — the strongest possible form of D2. This is not
a threshold that needs moving: the direction carries **no signal at all** about whether the model is
about to complete truthfully. And crucially, a **non-linear** refit recovers nothing above base rate
either, which rules out "the boundary just needs a better fit."

*Honest limit:* the ~0.94 base rate means only ~9–10 FALSE examples per dataset, so power is low. The
0.500 balanced accuracy is a strong signal, but it rests on a small positive class.

### 0.3 — Judge hardening

Drop the identified problem prompts and recompute. On cities the mean-arm FALSE floor goes from
**0.031 → 0.000** across the whole scale sweep (n 32 → 26) — i.e. **every** FALSE verdict in the cities
mean arm was a judge artifact, not a steering effect. common_claim's floor drops similarly at most
scales. The null was already robust; this removes its noise floor.

### 0.4 — Stem-vs-full Jacobian (D1 operationalized)

Recompute the margin at the *stem* context and compare directly:

| | cities (n=200) | common_claim (n=200) |
|---|---:|---:|
| `m_full` (median) | 7.2974 | 6.2499 |
| `m_stem` (median) | 1.7603 | 2.4309 |
| **ratio** | **0.241** | **0.385** |
| **cos(Jᵀw_full, Jᵀw_stem)** | **0.305** | **0.392** |

So the one-word shift does not merely shrink the gain — it **rotates the direction to cosine ~0.3**.
The certificate's direction is roughly 72° away from the direction that actually actuates at the point
where steering happens. This is the mechanism behind D1 stated geometrically.

### 0.5 — First-order robust margin

Using 0.4's measured spread: median `eps_robust` = **67.0** (cities) versus ε\* = 2.87, and **155.3**
(common_claim) versus 12.2. An honestly robustified budget is **~23× and ~13× larger** than the naive
one. (Explicitly not a certified tube — a first-order robustification.)

### 0.6 — Newton re-steering (the negative control that behaved as predicted)

Recompute Jᵀw at the steered state and iterate, 3 steps, 32 statements per dataset:

| | cities | common_claim |
|---|---:|---:|
| final median &#124;g&#124; | 4.65e-05 | 2.74e-05 |
| fraction converged (&#124;g&#124; < 1e-3) | 0.875 | 0.906 |
| final median ‖Δ‖ | 2.355 | 12.237 |
| median cos(Δ, initial Jᵀw) | **0.971** | 0.724 |
| statements hitting the trust cap | **0/32** | **0/32** |

The prediction was that Newton would **not** fix D1, because the problem is context, not curvature. It
didn't — and the reason is now visible: on cities the direction barely rotates across iterations
(cos 0.971), so there was nothing for iteration to find. The readout converges to machine-zero, and
behavior still doesn't move.

### 0.7 — Recovering the U64/V64 singular vectors

The audit's rsync-back glob had excluded the per-statement SVD directories, which were saved on the
cluster all along. Recovered as a free ~19 MB/dataset pull. This unblocked the SAE forensics below.

---

## 10. Chapter 8 — SAE feature forensics (Track B, partial)

Using GemmaScope's 16k-width JumpReLU SAEs (gemma-2-2b is the best-covered model in GemmaScope), with
**orthogonal matching pursuit against the decoder** — because naive encoding of steering vectors is
known-misleading (arXiv:2411.08790). Run on cities; 32-feature decompositions.

| Vector | Layer | Cumulative variance explained by 32 features |
|---|---:|---:|
| `w_mean_diff_tgt` (the target-layer truth readout) | 20 | **0.504** |
| `V64_common_*` (the high-gain input subspace) | 11 | 0.228 – 0.348 |
| `jtw_full` / `jtw_stem` / `jtw_mean` | 11 | 0.253 – 0.276 |
| `common_v1` | 11 | 0.282 |

**Two things fall out.** First, the **target-layer truth readout is roughly twice as SAE-legible** as
anything on the source side (0.504 vs 0.23–0.35) — the same asymmetry D3 found spectrally, now in
feature space: truth is a nameable thing downstream and a dark direction upstream. ~70–77% of the
source-side directions' energy is not captured by the top 32 SAE features at all.

Second, the D2 mechanism has a feature-level signature:

> **Jaccard(features of Jᵀw at the full statement, features of Jᵀw at the stem) = 0.049 — 3 shared
> features out of 32.**

For contrast, `common_v1` shares 17/32 with `jtw_full` (Jaccard 0.362), so the metric is not
degenerate — 0.049 is genuinely near-disjoint. **One word of context shift replaces essentially the
entire feature basis of the actuating direction.** That is D1/D2 restated in interpretable units, and
it is the most legible single statement of the mechanism we have.

*Status:* cities only, one SAE width, no robustness sweep across widths (absorption / non-canonicity,
arXiv:2409.14507, 2502.04878) and no residual "dark matter" accounting (arXiv:2410.14670) yet. Reported
as a lead with a clean number, not a finished analysis.

---

## 11. Chapter 9 — The publication gate (built, not run)

**The claim under test, stated as the plan states it.** The truth run produced a dissociation where the
certificate was *locally exact* (calibration 0.9997; Newton 32/32 converged, none capped) yet
behaviorally inert. Two things could explain that:

1. **The instrument is broken** — `‖Jᵀw‖`, `ε\* = g/m`, and the steering hook do not do what we believe,
   on *any* concept.
2. **The instrument is fine and truth is not a behaviorally actuatable variable in gemma-2-2b.** This is
   the interesting result and the paper's thesis.

**Only a positive control separates them.** Track A runs the *unmodified* audit pipeline on the Arditi
refusal direction (arXiv:2406.11717 — public code, validated down to small Gemma models, known steering
layer): same code path, same hook, same ε\* arithmetic, different concept.

**Two design decisions worth recording, because both were non-obvious:**

- **Polarity.** `refusal.csv` uses **label 1 = harmless, label 0 = harmful**, so the pipeline's
  label-1 crossing *induces* refusal on harmless prompts. The opposite direction (ablating refusal)
  requires the model to refuse in the first place, and a base model may never do so — there would be no
  headroom. The sweep covers ± scales anyway, so we observe it for free without betting the control on
  it.
- **The linearization point.** The truth per-statement arm had to chop the final word off each statement
  so the model would generate it — which is *exactly* what created the context-shift confound. A refusal
  instruction **is** the generation prompt, so `--prompt-mode full` puts the certificate's linearization
  point at the prompt's final token and **the confound cannot arise by construction**. Refusal is a
  strictly cleaner certificate test than truth ever was.

**Status: fully implemented, nothing run.** 15 tasks executed subagent-driven with a reviewer gate per
task; test suite went **182 → 298 passing** (299 collected today, including the parallel spectrum track).
Backward compatibility is absolute — the cities and common_claim code paths are bit-identical, verified
against a captured baseline. The cluster runbook `deltaai/REFUSAL_RUN.md` has a hard **STOP** at Phase 2
(the model screen), and a documented `readout-only` branch: if refusal *also* moves the readout without
moving behavior, the runbook says stop and recalibrate with Julian rather than spend more GPU time.

**The decision tree that makes this a gate and not just another experiment:**

- Refusal **passes** (gain transfers, behavior moves) + truth fails → an instrument-validated contrast;
  the paper upgrades from workshop to main-venue candidate.
- Refusal **also fails** → **STOP.** Either the thresholds are miscalibrated against Arditi's effect
  sizes, or open-loop steering is broken more generally — a bigger and much more contestable claim.

**Six wrong-conclusion defect classes were caught in review before any of this ran** — the kind of thing
that would have produced a confidently wrong number. The one worth naming: the truth-stem baseline was
**g = −53.2**, which is vacuous (a "deep FALSE" reading on prompts the model completes truthfully), and
had it been used as a reference point it would have made any steering look like it was crossing a
boundary it had already been sitting past.

---

## 12. Literature review — where this sits

**The papers the project is built on:**

- **Marks & Tegmark, "The Geometry of Truth."** The claim under replication: truth is a linear
  direction. **Our cities result reproduces it** (0.990, gap +0.003). Our common_claim result **bounds
  it**: on heterogeneous claims the linear story degrades (0.706) and non-linear structure appears
  (+0.082). The replication is not unconditional — it is a statement about *templated* truth.
- **The MELBO/DCT paper (Deep Causal Transcoding).** Used unmodified as the unsupervised causal miner.
  We are not testing DCT; we are using it as an instrument, and it does its job (valid, diverse,
  behavior-changing vectors). The finding is about truth's position relative to what DCT finds.
- **Julian's A-LQR — arXiv:2604.19018, "Local Linearity of LLMs Enables Activation Steering via
  Model-Based Linear Optimal Control" (Skifstad, Yang, Chou).** The most directly relevant paper.
  **Our R² = 0.999 is a direct corroboration of its core local-linearity assumption**, measured
  independently on a different task. And our D1/D2 are precisely the **open-loop failure modes its
  closed-loop design implicitly answers**: if gain collapses across a one-word context shift, a
  receding-horizon controller that re-plans every token is the structurally correct fix. This is the
  natural handoff.
- **Arditi et al., arXiv:2406.11717 (refusal is mediated by a single direction).** The positive control.
  Chosen because it is public, validated down to small Gemma models, and has a known steering layer —
  i.e. maximum prior probability that a working instrument detects it.
- **Karnik & Bansal, arXiv:2509.21528 (BRT-Align)** — the PI's load-bearing citation. Treats greedy
  generation as a discrete-time control system, defines a failure set as the sub-zero level set of a
  classifier margin, and learns a Backward Reachable Tube via a DeepReach-style value recursion; ~98%
  detection, flags unsafe trajectories 7–10 tokens early, 54–85% fewer unsafe generations across 5 LLMs.
  No code released. **Our case is strictly easier**: one linearized 9-block hop instead of a recurrent
  nonlinear rollout, so closed-form set preimages replace their learned value function.
- **Karnik et al., arXiv:2603.00140 (reachability-constrained RL for diffusion memorization)** — same
  lab, same skeleton on diffusion. Two transferable lessons: they **compress the control space** before
  doing reachability (77×768 → 64-dim), which argues we should work in a reduced basis rather than raw
  d=2304; and their constrained-MDP objective is the template for "minimal perturbation reaching the
  FALSE set *subject to a coherence constraint*" — formalizing the coherence-vs-lying dissociation every
  experiment here found.
- **Anthropic, "A Global Workspace in Language Models."** The source of the PI's J-space vocabulary. Its
  key functional result — **selective engagement**: the same information can sit in the residual stream,
  be linearly decodable, and be causally inert for automatic tasks while being decisive under
  explicit-report framing — is **our entire decodable-≠-causal null restated as workspace gating**. It
  also independently predicts our MAG finding that the "being-asked-about-truth" direction is real,
  linear, and orthogonal to the truth content axis.
- **arXiv:1910.13272** — the PI cited this, and it resolved to *feedback linearization via RL*
  (Westenbroek et al., Tomlin/Sastry lab), **not a reachability paper**. The conceptual echo is real
  (A(x)⁻¹ is literally "map desired outputs back through the inverse of the local linear map") but it
  contains no target sets and no set propagation. **Open item: confirm with the PI whether this ID was
  intended or a typo** for a set-propagation survey.

**The nearest neighbors, and why the result is not already claimed** (verified by search):

| Paper | What it has | What it lacks |
|---|---|---|
| "Steered LLM Activations are Non-Surjective" (arXiv:2604.09839) | an existence proof of unreachable activations | no certificate machinery, no per-statement budgets |
| "When Truthful Representations Flip" (arXiv:2507.22149) | the correlational version, via prompting | no causal intervention, no mechanism decomposition |
| Tan et al., steering reliability (arXiv:2407.12404) | a base rate for steering failure | no mechanism |
| LiSeCo (arXiv:2405.15454) | linear-probe halfspace constraints on gemma-2-2b, closed-form projection | *forward* safe-set control, not backward reachability |

**No published work reports a certified-reachable-but-behaviorally-inert dissociation with mechanism.**
And the D2 prefix-vs-statement probe finding — the truth probe reading "deep FALSE" (g ≈ −52) on prompts
the model completes truthfully, at balanced accuracy 0.500 — appears **unclaimed**.

**Supporting context on why truth may be a structurally hard target:** 2-D truth subspace
(arXiv:2407.12831), task-orthogonal truth geometries, output-accessibility framework (arXiv:2604.15557),
Anthropic's honesty-elicitation work. **On SAE robustness duties:** GemmaScope (arXiv:2408.05147), the
steering-vector-encoding caveat (arXiv:2411.08790), absorption (arXiv:2409.14507), non-canonicity
(arXiv:2502.04878), dark matter (arXiv:2410.14670), and AxBench (arXiv:2501.17148 — SAEs underperform on
Gemma-2 specifically, which is why SAE *steering* is deprioritized while SAE *forensics* is not).

**Tooling landscape, surveyed.** Tier 0 (plain linear algebra for a single linearized hop) is where we
are and is sufficient. Tier 1 (INVPROP in α,β-CROWN; PREMAP) offers certified preimages without
linearizing, but **nothing in that literature has been run on a multi-block transformer with attention**
— a risk and an opportunity. Grid-based Hamilton-Jacobi is exact and dies at ~6 dimensions, so it can
only enter via learned value functions, never directly at d = 2304. Explicitly deprioritized with
reasons: HJ/NN-verification at our scale, crosscoder training (no pretrained checkpoint for 2b),
gemma-2-9b (invalidates every cached artifact), and all-position token-resolved probing (~24 GB
uncached).

---

## 13. The quantified ledger — wins, stated as numbers

**Scientific results established:**

1. Truth is linearly decodable at **0.990** on templated statements; the linear story degrades to
   **0.706** with **+0.082** non-linear headroom on heterogeneous claims. Gap grows monotonically across
   four datasets: 0.003 → 0.028 → 0.038 → 0.082.
2. The readable truth direction is **not** among the model's 512 most causally-salient directions:
   max cosine **1.1–1.2× random**, subspace containment **1.0–1.4× chance**, tested at the layer where
   truth is *most* cleanly defined (supervised agreement 0.413).
3. No top DCT vector flips an established fact: **0/20 pooled**, 95% upper bound **~14%**.
4. Steering the truth axis has **no lie-asymmetry**: Δ = **−0.010**, 95% CI **[−0.042, +0.023]**, TOST
   equivalence passing on every slice. This is a *bounded* null, not a failure to find an effect.
5. The one real steering effect is **symmetric degradation**: z = **−5.06**, p < 1e-6 for magnitude;
   χ² = 4.49, p = 0.34 for direction. Truth is a degradation lever, not a truth lever.
6. The non-linear escape hatch is closed: XGBoost's **+0.057** advantage on the full residual stream
   collapses to **−0.01** inside DCT's causal subspace.
7. A second, unrelated miner (MAG) reaches the same conclusion, and adds: the model **never says
   "false"** (0/354, 0/1496, 0/1200, 1/4450) and its verdict position reads truth **at chance**
   (0.48–0.62).
8. **No λ** on the entire path from the supervised truth axis (cos 0.95) to DCT's free search (cos 0.03)
   produces a truth lever; FALSE stays pinned at 0.03–0.09 in all 11 panels on both datasets.
9. The FALSE halfspace **is** linearly reachable, and that reachability is **behaviorally inert**:
   lie rate **3.5%** at maximum push versus a **3%** baseline, with ε\* **inside** the trust radius
   (0.94×).
10. The certificate is **arithmetically exact at its own linearization point** — calibration **0.9997**
    (IQR 0.9910–1.0117, n=199). The failure is transfer, not math.
11. **One word of context** collapses gain **13.7×** (cities), rotates the actuating direction to
    cosine **0.305**, and replaces **29 of 32** SAE features (Jaccard **0.049**).
12. The truth probe has **balanced accuracy exactly 0.500** on generation prefixes, and a **non-linear**
    refit recovers nothing (XGBoost gap +0.013 / 0.000). Absence of signal, not a threshold shift.
13. The Jacobian is **structured, not isotropic** (eff_rank 21–23%, s64/s1 0.11–0.16) and its channel is
    **downstream-truth-carrying** (74% of `mean_diff_tgt` energy in top-64 U, 17–125× random) while
    **upstream concept-blind** (1.8–7× random). This is a structural ceiling for *any* linear
    concept-steering method in this band.
14. Controllability is **front-loaded** (peaks L0–8, 65–87% lost by the target layers); the behavioral
    verdict readout has **2–7× less margin** than the truth probe everywhere; the best window is
    **L5–9**, 1.5–3× better-conditioned than what we used.

**Methodological wins (the ones that make the above trustworthy):**

15. An open, citable, validated judge: OLMo-3-7B-Instruct at **0.970** agreement (gate ≥0.85).
16. A measurement-validity bug found **before** reporting, with a quantified fix: baseline TRUE
    **0.50 → 0.81**.
17. Judge hardening drives the cities FALSE floor from **0.031 → 0.000** — every FALSE verdict in that
    arm was a judge artifact.
18. A negative control that behaved exactly as predicted (Newton: readout to |g| ≈ 5e-5, 0/32 capped,
    direction barely rotating at cos 0.971 — and behavior still inert).
19. Two conclusion-inverting bugs caught in Chapter 1 (left-padding; max-gap vs best-layer gap), one
    data-leakage bug caught in MAG's review (E1 fit its scaler before cross-validation), and **six
    wrong-conclusion defect classes** caught in the Horizon-1 review before any of it ran.
20. **299 tests**, all green, with per-task reviewer gates and a documented STOP condition on the
    cluster runbook.

**Engineering throughput:** 145 commits in 5.5 weeks; peak weeks at 43–45 commits; 11,489 LOC of source
against 3,765 LOC of tests; 38 SLURM scripts and runbooks; every experiment reproducible from a
committed runbook.

---

## 14. What is *not* established (the honest column)

Keeping this explicit is what makes the rest credible.

- **Everything so far is a null.** The positive control has not run. Until it does, "the instrument
  works and truth isn't actuatable" and "the instrument doesn't work" are not separated. This is the
  single biggest open item and it is correctly gated.
- **Single model, two hops, one probe family.** The dissociation is *demonstrated*, not
  *characterized* across layers, probes, or models. gemma-2-9b would be a generality check, and it
  invalidates every cached artifact.
- **The D2 balanced-accuracy result rests on ~9–10 FALSE examples** per dataset (base rate 0.94). The
  0.500 is a strong signal with low power.
- **P1↔P3 population mismatch.** ε\* is computed on full-statement activations; the steering arm reads
  the probe on generation prefixes. The behavioral null is population-independent and stands alone, but
  the exact ε\*-versus-crossing comparison should be read qualitatively.
- **common_claim's mean-arm readout is numerically unstable at large scale** — trust the per-statement
  arm and the behavioral fractions there.
- **MAG E4 used a weaker evaluator** than the OLMo pipeline (single verdict token, 24 statements,
  19–41% incoherence). Its one antisymmetric-looking result is treated as suggestive only.
- **The warm-DCT sweep tops out at ±1·input_scale at a single injection layer.** A stronger or
  multi-layer push might eventually force falsehoods, at the cost of ever more incoherence. Within the
  coherent regime the answer is unambiguous; outside it, untested.
- **SAE forensics is cities-only, one width**, without the absorption / non-canonicity / dark-matter
  robustness checks the literature requires.
- **Human-vs-judge agreement (κ) on the actual steering completions** is still on the to-do list. The
  0.970 gate certified clean statements, not messy generations — the exact gap that produced the
  measurement bug in the first place.
- **arXiv:1910.13272 remains unconfirmed** with the PI (intended, or a typo for a set-propagation
  survey?).

---

## 15. Where it goes next

**Immediately (the gate):** run Track A, the refusal positive control. Fully implemented, backward-compatible,
~3–6 GPU-hr of the ~475 remaining. `deltaai/REFUSAL_RUN.md`, hard STOP at Phase 2 for the model screen.
Outcome determines the paper's venue tier and whether anything downstream is worth running.

**Then, the two natural A-LQR handoffs into arXiv:2604.19018:**
(a) **receding-horizon / MPC re-planning at each generated token** — the structurally correct fix for
D1's context-transfer failure, and exactly the open-loop → closed-loop upgrade A-LQR formalizes;
(b) **LQR with a verdict-logit-margin cost** — `w_verdict` is already built for free in `reach_jlens.py`.
Combine with **source at layers 5–8** (D4's best-conditioned window) for the maximum-contrast
configuration: early window × verdict direction × closed-loop.

**The paper.** Primary framing: *dissociation with mechanism* — per-statement certificates + behavioral
falsification + the D1–D4 decomposition, with the refusal contrast as instrument validation and the D2
prefix-vs-statement finding as a supporting figure. **cities is the cleanest case to hand to A-LQR**:
readout-controllable, behavior-inert, certificate exact at its own point.

**One design principle the audit earned, worth carrying forward:** a reachability certificate should
only be reported as "reachable" when ε\* is inside the linearization trust radius **and** a behavioral
spot-check moves. cities passes the first and fails the second — which is exactly why it is the
interesting scientific case rather than a bug report.
