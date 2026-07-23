# What We Built and Found Since the Last Meeting

*A from-scratch, plain-language walkthrough of the machinery we stood up since the last PI meeting —
the unsupervised **feature-discovery pipeline**, the open **LLM-as-a-judge (OLMo-3)**, and the two
**GH200 cluster runs** — and what they let us conclude. Written to be read top-to-bottom by someone
who saw the last talk but hasn't followed the day-to-day. Last updated: 2026-07-16 — added §8, the
MAG battery (a second unsupervised miner, built and merged; experiments not yet run).*

**➡️ Continued in `RESULTS_SINCE_LAST_MEETING_PART2.md`** — the MAG battery (§8) has since been **run**,
and a new **warm-started DCT** experiment was run on the GH200. Part 2 covers both, top to bottom.

**Companion docs** (this doc is the narrative; these are the receipts):
`OLMO_JUDGE_IMPLEMENTATION_PLAN.md` (the judge build, task-by-task),
`INVESTIGATION_steering_validity.md` (the full statistics + the corrected rerun, §6),
`DCT_VS_XGBOOST_FINDINGS.md` (the non-linear extension),
`PI_MEETING_RESULTS.md` (the master funnel writeup — last meeting's deliverable).

---

## 0. Where we left off, in one paragraph

At the last meeting the story was: **truth is easy to *read* out of gemma-2-2b's activations but is
not one of the directions the model is most *causally* driven by.** A linear probe decodes true-vs-false
at ~99% on clean data, yet when we asked an unsupervised method (DCT) to hand us the model's strongest
causal levers, the supervised truth direction was not among them, and directly steering that truth
direction barely moved the model's factual behavior. The slogan was **"decodable ≠ causal."**

That story rested on **two pieces of evidence that were read by hand** — I eyeballed the DCT vectors'
completions and I eyeballed the steering completions — plus some geometry math. A hand-read is exactly
the thing a PI should push on: *how do you know you're not seeing what you want to see?* **Everything
below is the work of turning those two hand-reads into numbers**, plus one extension that closes a
loophole. Nothing in the conclusion reversed; it got quantitative, and in one place it got *sharper*
(see §5, "the correction").

---

## 1. The three things we built (and why each one)

Think of it as a small assembly line. Each stage feeds the next.

```
   (1) FEATURE-DISCOVERY PIPELINE            (2) LLM-AS-A-JUDGE                (3) CLUSTER RUNS
   DCT finds the model's causal      →   OLMo-3 reads each completion   →   run it all at scale
   levers + we steer the truth axis      and scores it TRUE/FALSE/           on a GH200 GPU, twice
   → produces text completions            INCOHERENT → produces numbers       (found a bug, fixed it)
```

- **The pipeline** produces the raw behavioral evidence — text the model generates when we perturb it.
- **The judge** converts that text into scores, so we're counting instead of squinting.
- **The cluster** is just where it runs, but the *two* runs are scientifically important: the first
  run surfaced a measurement bug, and the second run fixed it. That story (§4) is the part that shows
  the result is trustworthy.

The rest of this section explains each stage in turn.

---

## 2. Stage 1 — the feature-discovery pipeline (what DCT is and why we use it)

**The question DCT answers.** A probe (logistic regression) is *supervised*: we hand it labels and it
finds the direction that best separates true from false. That tells us truth is **decodable**, but it
can't tell us whether the model *itself* cares about that direction. To ask what the model is causally
driven by, we need a method that gets **no labels** and instead discovers, unsupervised, "which
directions, when I nudge them, change the output the most."

**DCT = Deep Causal Transcoding.** It scans the residual stream and returns a bank of **512 candidate
directions**, each as a pair:
- **V** — the *input* direction: the thing you add to the activations to steer with.
- **U** — the *output* direction: the downstream effect that steering produces.
- Each direction has a **potency** `‖U‖` — how big an effect it has. Rank by potency and the top of
  the list is *the model's strongest causal levers*, discovered without ever being told about truth.

Our config: 512 factors, injected at layer 13 and read at layer 22 (`src/dct_train.py`).

**The key test this sets up.** If truth were one of the model's dominant causal directions, the
supervised truth direction should show up *inside* the top DCT directions. It doesn't — that was the
last meeting's headline. This cycle we pushed on the two behavioral follow-ups the geometry couldn't
settle:

- **Test 1 (interpret):** take the **top-10** DCT directions (the 10 most potent causal levers) and,
  for each, generate steered-vs-unsteered completions on a set of probe prompts. Then *ask what each
  lever does.* If the model's top levers were about truth, some of them should flip facts. (`src/interpret_top10.py`.)
- **Test 2 (steer):** take the **supervised truth direction itself** (computed two ways — the
  difference of class means, `mean_diff`, and the probe's gradient, `grad`) and inject it at the
  truth-peak layer across a signed sweep of strengths, −120 … +120. Negative pushes *away* from truth,
  positive *toward* it. Then read whether the factual completions actually flip. (`src/steer_supervised.py`.)

Both tests produce **text**. The old bottleneck: someone had to read that text and decide "did this
flip a fact or not?" That's Stage 2's job.

---

## 3. Stage 2 — the LLM-as-a-judge (OLMo-3), and why this specific model

**Why we need a judge at all.** Test 1's output is 20 vectors' worth of completions; Test 2's is
hundreds of completions across the strength sweep. Scoring those by hand is slow, unblinded, and
un-auditable — the exact weakness a reviewer targets. An **LLM-as-a-judge** reads each completion and
emits a label, so the result is a *count* anyone can reproduce, not my reading.

Two judging modes, matched to the two tests:
- **steer mode** → label each completion **TRUE / FALSE / INCOHERENT.** This is the important one,
  because it separates two very different things the old keyword scoring couldn't tell apart:
  **FALSE** = the model was steered into *lying* (a genuine causal truth effect) versus
  **INCOHERENT** = the model was just steered into *gibberish* (mere degradation). That distinction is
  the whole ballgame for Test 2.
- **interpret mode** → for each top-10 DCT vector, decide "does this vector *manipulate truthfulness*?"
  turning Test 1's hand-read into a 0-to-10 count.

**Why OLMo-3-7B-Instruct specifically.** We had three options and picked OLMo for concrete reasons:

| Judge option | Why we rejected / chose it |
|---|---|
| Anthropic API (Claude) | Needs an API key we don't have; it's billed, and a closed judge is harder to cite in a paper. |
| TruthfulQA-7B (a narrow yes/no head) | Only speaks TruthfulQA's Q/A format — a mismatch with our free-form completions — and **can't do interpret mode** at all. |
| **OLMo-3-7B-Instruct** ✅ | **Fully open** — public weights, training data, and recipe, so it's reproducible and citable. **Free and local** (no key). And it's a **general chat model**, so the *same* judge handles **both** steer and interpret modes with a rubric prompt. |

Mechanically it's small: a ~40-line module (`src/judges/olmo_judge.py`) that loads the 7B model and
exposes a `.chat(system, user)` call, wired into the existing `judge_results.py` behind a
`--backend olmo` flag. It reuses the rubric prompts we'd already written, so nothing else in the
pipeline changed.

**The honesty guard — validating the judge before believing it.** A judge is only worth using if it
agrees with ground truth on cases where we *know* the answer. So before trusting any judged number we
run `src/validate_judge.py`: it feeds the judge clean statements from `cities.csv` (which have gold
true/false labels) and measures agreement. **Gate: ≥0.85 or we don't trust it.** It passed at **0.970**
(97 of 100 gold labels). This is the guardrail the PI will (correctly) ask for.

---

## 4. Stage 3 — the cluster runs, and the bug that made them a two-act story

**Why the cluster.** A 7B judge swap-thrashed the 24 GB laptop under load, so the judging moved to the
**DeltaAI GH200** GPU cluster. (Small logistical wrinkle worth knowing: the judge needs a newer
`transformers` than DCT's pinned version, so they live in two separate virtualenvs, `.venv-dct-gpu` and
`.venv-judge-gpu`.) The runs are submitted with SLURM and *chained* — the judge job auto-starts only if
the steering job succeeds (`--dependency=afterok`, see `deltaai/submit_rerun.sh`) — so it's one submit
and walk away.

Now the important part: **we ran it twice, and the reason we ran it twice is the most scientifically
valuable thing in this cycle.**

### Act 1 — the first run, and the trap we caught

The first run produced judged curves and a headline interpret count of **3/10** vectors "manipulating
truthfulness." Before reporting either, we **drilled in by hand** on those flagged vectors and on the
unsteered (zero-strength) completions — and found the instrument was lying to us:

> At **zero steering**, the prompt "Two plus two equals" produced "**four.** That's the message from a
> new study that found people who eat…" and the judge marked it **FALSE**. Its reason: "completion is
> unrelated to the stem."

The direct answer — "four" — was *correct*. gemma-2-2b is a 2B **base** model: it doesn't stop at the
answer, it **rambles** for the full generation length, and the judge was scoring **the whole rambling
paragraph** instead of the answer. So correct answers were being failed on their tails. This is a
**measurement-validity bug**: the numbers were real, but they were measuring the wrong thing.

**Why the 0.970 gate didn't catch it:** the gate validated the judge on *clean, single-claim*
statements ("The city of Krasnodar is in Russia."). Judging a long, half-rambling *generation* is a
harder, different task the gate never exercised. **A passing gate on clean statements does not certify
valid labels on messy completions** — a genuinely useful lesson about judge validation.

Catching this *before* reporting is the point. The first run's real deliverable wasn't the 3/10; it was
"don't trust the 3/10, here's why, here's the fix."

### Act 2 — the corrected rerun (the design decisions)

We fixed the *instrument first*, then added statistical power. Four changes, in priority order:

1. **Short completions.** Cut generation from 24 tokens to ~8, so the completion *is* the answer, with
   no rambling tail to misjudge.
2. **Answer-only rubric.** Tightened the judge's instructions to "judge ONLY the direct answer to the
   stem; ignore trailing sentences." (Belt-and-suspenders with #1.)
3. **Factual-only prompts.** Dropped open-ended stems with no truth value ("I think that…") and kept
   clean factual ones ("The capital of X is").
4. **More prompts, for power.** Went from 8 prompts/strength to **32**, chosen from a power analysis so
   we could not just fail to find a lie-effect but *put an upper bound on how big one could be* and hide.

Result: **576 judged completions per dataset** (2 directions × 9 strengths × 32 prompts), 1,152 total,
through a now-trustworthy instrument. The proof the fix worked: the unsteered baseline TRUE rate rose
from **0.50 → 0.81** — the rambling artifact is gone. (The residual 0.19 isn't noise; it's the 2B base
model genuinely flubbing ~5 of the 32 prompts, e.g. answering "the capital of Japan is" with "a city
that has been around for over…". Crucially, that flubbing is *identical* across datasets and
*symmetric* across steering sign, so it can't fake a directional effect.)

---

## 5. What the corrected run actually showed

Two behavioral tests, now with numbers instead of hand-reads.

### Test 1 (interpret): 0/10 vectors flip a fact — on both datasets

Under a strict "does this NEGATE a verifiable claim?" rubric, the judge flags **0 of 10** top DCT
vectors on cities and **0 of 10** on common_claim (**0/20 pooled**). With the rule-of-three, that puts
a **95% upper bound of ~14%** on the fraction of top DCT vectors that *could* flip an established fact —
and the observed count is zero.

This **exactly reproduces the hand drill-in, without a human in the loop.** The first run's inflated
3/10 came from a *loose* rubric that counted the wrong things: those vectors confabulate invented
details ("Tokyo founded by JFK"), shift register (into code or Vietnamese), or degrade into gibberish —
but the *target fact* (Paris→France, Japan→Tokyo, 2+2→four) survives in every single one. The top
causal levers are about **geography, format, and tone**, never a clean truth switch.

### Test 2 (steer): no lie-asymmetry — but a real, symmetric degradation effect

This is where the result got **sharper than the last meeting**, and it's worth stating the correction
plainly:

- **Last meeting's hand-read said:** steering hard against truth produces "a few real falsehoods."
- **The judge says:** those falsehoods are **not sign-dependent** — steering hard in *either*
  direction produces them at the same rate. **The truth direction is a *degradation* lever, not a
  *truth* lever.**

The numbers behind that:

- **No lie-asymmetry, and now *bounded*.** Is steering *away* from truth more likely to make the model
  say something FALSE than steering *toward* it? **No.** Pooled difference **Δ = −0.010, 95% CI
  [−0.042, +0.023]** — statistically **equivalent to zero** (a TOST equivalence test passes on every
  slice; a prompt-clustered bootstrap agrees). The last run could only *assert* "about zero"; this run
  *proves* there is no lie-asymmetry bigger than ~0.04. This is the upgrade the rerun existed to
  deliver: **a bounded null, not just a failure to find an effect.**
- **The one real effect is symmetric degradation.** Push the truth axis hard in *either* direction and
  the model's TRUE rate falls — the ~0.81 plateau in the middle of the curve dips to ~0.55–0.65 at both
  ±120 ends, the same on both sides (**Cochran-Armitage z = −5.06, p < 1e-6**). But the direction of
  the push (sign) carries **no** information — the verdict×sign omnibus is not significant (χ²=4.49,
  p=0.34). **Only *magnitude* matters, not *direction*.** That is the fingerprint of a direction that,
  when amplified, **breaks** the model rather than **flipping its truth value.**

**Why we ran these particular tests (in plain terms).** There were three competing explanations for what
steering does, and each test rules one in or out:
- "Steering away from truth makes it lie" (a real truth lever) → tested by the **lie-asymmetry** check
  (paired −vs+ comparison). **Ruled out** — bounded to zero.
- "Steering just breaks the model, regardless of direction" (degradation) → tested by the **trend across
  strength** (Cochran-Armitage). **Ruled in** — significant and symmetric.
- "It's all noise" → tested by the **omnibus** χ². The *directional* part is noise; the *magnitude* part
  is not.

The full statistical battery — McNemar pairing, TOST equivalence, clustered bootstrap, Cochran-Armitage
trend, Benjamini-Hochberg multiple-comparison control — lives in `INVESTIGATION_steering_validity.md` §6.
The point of that many tests is not to pile on; it's that **a null result needs more defense than a
positive one** — you have to show the effect isn't hiding in any single prompt, any single dataset, or
any slicing of the data.

---

## 6. The loophole we also closed (non-linear truth)

One objection to "decodable ≠ causal" is: *maybe truth IS causally real, but encoded non-linearly, so
no single direction lines up.* We closed that hatch separately (`DCT_VS_XGBOOST_FINDINGS.md`):

- On the messiest dataset (common_claim), a **non-linear** model (XGBoost) reads truth **+0.057** better
  than a linear probe from the full residual stream — so non-linear truth structure genuinely exists.
- But project onto DCT's top causal directions and that non-linear gap **collapses to ≈0** (−0.01, no
  better than random projections).

So DCT misses truth **non-linearly as well as linearly.** It doesn't miss truth *because* truth is
non-linear; it misses truth because **truth isn't causally salient.** The 2×2 is now complete: truth is
present in the full activations (linear *and* non-linear) and absent from DCT's causal subspace (linear
*and* non-linear).

---

## 7. What it all means — the bottom line for the PI

> **"Decodable ≠ causal" is now quantitative, not hand-read.** We built an open, validated LLM-judge
> (OLMo-3, 0.970 gate) and ran the whole pipeline on the GH200 — twice, because the first run exposed a
> measurement bug that we caught and fixed before reporting.
>
> The corrected result: **no top DCT vector flips an established fact** (0/10 on both datasets, ≤14%
> upper bound), and **steering the supervised truth direction shows no lie-asymmetry** (Δ = −0.010,
> equivalent to zero) — the only real effect is **symmetric degradation** (steering hard in either
> direction breaks the model, z = −5.06). Truth is a **degradation lever, not a truth lever.**
>
> This is *sharper* than what I presented last time: the "a few real falsehoods" I read by hand turn out
> to be symmetric byproducts of degradation, not a directional truth effect. And the last non-linear
> escape hatch is closed too (XGBoost's +0.057 truth gap vanishes inside DCT's subspace).

**One caveat we keep honest:** the 0.970 gate validated the judge on clean single-claim statements, not
on the completions themselves. The instrument fix (baseline 0.50→0.81) is strong evidence the judge is
now reading completions correctly, but a human-vs-judge agreement check on a sample of the *actual*
steering completions is still on the to-do list.

---

## 8. New since 2026-07-16 — the MAG battery (built and merged; not yet run)

**The gap it fills.** Every null above is a null *about DCT*. A skeptic's easiest reply is: "maybe
truth just isn't the kind of thing *DCT's* objective finds — try a different unsupervised miner."
So we built one. **MAG (Mining via Activation Geometry)** mines features not from a steering
objective but from the **prefix-induced activation shift**: ask the model a question about its own
input ("Is this statement true?"), and measure how the activations *move* when the question is
prepended — Δ<sup>Q</sup>(p) = m(Q‖p) − m(p). If the model internally "knows" the answer, that shift
should carry it. Running MAG head-to-head with DCT on the same truth datasets tests whether the
funnel's null is a fact about *truth in gemma-2-2b* or an artifact of one mining method.

**What we built** (14 commits, merged to main at `f19771b`; 68 tests green): a `src/mag/` package +
CLI that implements the full battery —

- **8 feature operators** (Direct, Prefixed, Answered, Verdict, InputDelta, QuestionDelta,
  Interaction, FewShot) — different ways of reading the question-conditioned activations.
- **A self-verdict `y^M`** — the model's *own* first-token yes/no answer to "Is this true?", read
  from the logits. This gives an unsupervised label to compare against gold: does the geometry track
  what the model *says*, or what is *actually true*?
- **Directions** — the prefix-shift direction `v_Q`, and class-contrast directions `u_Q` built from
  gold labels and from `y^M`; plus a calibration constant `A_prefix_norm` so steering strength is
  expressed in units of "one natural question-shift" (α(τ) = τ·‖Δ‖) instead of arbitrary magnitudes.
- **Five probes:** E1 *readability* (can a linear probe read truth from each operator's features —
  including inside DCT's top-k subspace and random-k controls), E2 *disagreement* (on statements
  where `y^M` ≠ gold, which does the geometry follow?), E3 *linearity* (how much of the prefix shift
  does a single direction explain, ε_Q), the §4-style *transfer/rank* test (does MAG's direction
  rank highly among DCT's levers?), and E4 *calibrated steering* (inject `u_Q` at natural-shift
  scale, generate, count verdict flips) — E4's completions feed the **same OLMo judge and the same
  statistical battery** (`investigate_steer.py`, now prefix-parameterized) as §5, so the DCT and
  MAG results will be directly comparable numbers.

**The process kept us honest again.** The subagent-driven build had a reviewer gate per task plus a
final whole-branch review, and it caught three real defects before any experiment ran — the one worth
recording: E1's readability probe **fit its feature scaler on the whole dataset before
cross-validation** (a classic leak that would have inflated the exact accuracies this battery exists
to report). Fixed with per-fold scaling. Also caught: a tokenization bug in the empty-input baseline,
and a wrong-path guard that would have silently reported the MAG-vs-supervised cosines as NaN.

> **Update (2026-07-22): the MAG battery has been run** on all four datasets. Findings are written up in
> `RESULTS_SINCE_LAST_MEETING_PART2.md` (Part A). Short version: MAG reaches the *same* conclusion as
> DCT — truth is linearly readable from the activations (including the question-shift), the model's own
> verdict is a stuck "yes" that reads truth at chance, and the question-shift direction is orthogonal to
> the truth axis. The paragraph below describes the pre-run state.

**Status: no findings yet.** The pipeline is smoke-verified end-to-end on 20 statements only (at that
toy scale the sanity signals point the right way — `u_Q` from gold labels aligns with the supervised
mean-diff direction at cos ≈ 0.94 and is near-orthogonal to DCT's top vector — but n=20 numbers are
not results and are not reported as such). The actual experiments are queued:

1. **Full MAG extraction** on all 4 datasets (laptop, MPS) — note the activation caches are large
   (extrapolating from the smoke file: ~1.5 GB for cities, ~4 GB for common_claim, ~7 GB for all
   four; the disk budget needs a look before launch).
2. **All probes** per dataset (`run_mag.py --probe all` + `--probe transfer`) — E1/E2/E3 + rank.
3. **E4 steering → OLMo judge on the GH200 → statistical battery** — the head-to-head behavioral
   comparison with §5. (One known plumbing item: `judge_results.run_steer` hardcodes its output
   prefix, so the judged MAG file needs a rename or a small `--out-prefix` flag.)
4. **Write `DCT_VS_MAG_ON_TRUTH.md`** — the findings doc this section is a placeholder for.

---

## 9. What's next

1. **Run the MAG battery** (§8) — the queued extraction → probes → E4-plus-judge sequence. This is
   the next block of actual findings.
2. **Human-vs-judge agreement (κ)** on a sample of the real steering completions — to certify the judge
   on the messy generations, not just the clean statements the gate used.
3. **Dose-response** on the degradation effect — an odds-ratio for how fast TRUE decays with strength —
   and a cross-dataset homogeneity check (is the null identical in both datasets?).
4. **The positive control.** Everything so far is a *null* — truth is the concept DCT *fails* to
   recover. The next move is **refusal**, a concept we expect DCT to recover *well*, to show the method
   works when the concept really is causally salient. This turns the binary null into a **spectrum**:
   *DCT recovers a concept in proportion to how causally load-bearing it is.* The judge we just built and
   validated is the prerequisite that unblocks measuring that spectrum's behavioral axis.
