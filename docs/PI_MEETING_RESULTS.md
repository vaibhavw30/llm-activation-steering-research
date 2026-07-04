# DCT vs. Truth — Detailed Meeting Explainer (with evidence)

*A self-contained, teach-it-from-scratch writeup: the question, the methods, every result with a
plain-English "what this means," the statistical baselines, and the mechanistic interpretation.
Pitched so you can read it once and explain it to a grad-level ML student. Companion docs:
`FUNNEL_RESULTS.md` (terser), `DCT_METHODOLOGY.md` (how DCT works internally).*

---

## 0. The one-paragraph story

Prior interpretability work (Marks & Tegmark 2023) shows a **linear probe** can read "is this
statement true?" off a language model's hidden activations with ~99% accuracy — so truth is
**linearly decodable**. But "a probe can read it" only says the information is *present* and
linearly separable; it does **not** say the model *uses* that direction to compute anything. We
tested the stronger, causal claim with **DCT** (Deep Causal Transcoding), an unsupervised method
that finds the directions the model is most *causally sensitive* to — the ones that, when you push
on them, most change what the model does next. **The truth direction is not among them.** Neither
as a single direction, nor as a combination, nor as something that generalizes across datasets;
and when we steer the probe's truth direction directly, it changes the model's factuality only
weakly. **Conclusion: truth is easy to read but hard to push — decodable ≠ causally dominant.**

Why a grad student should care: this is a concrete, quantified instance of the
**representation-vs-mechanism gap** that underlies a lot of debate in interpretability. Probes
measure *correlation* between a direction and a label; they are routinely over-interpreted as
finding "the feature the model uses." Our funnel shows a case where the two come apart cleanly.

---

## 1. Concepts you need (grad-ML level, but precise)

**Residual stream & hidden states.** gemma-2-2b maps each token, at each of its 26 layers, to a
vector `h ∈ ℝ^2304`. These vectors (the "residual stream") are what both methods operate on.

**A "direction."** A unit vector `u ∈ ℝ^2304`. Two things you can do with a direction:
- **Read** it: project activations onto it (`h·u`) and see if the projection separates true/false.
  This is what a **linear probe** does — it *learns* the `u` that best separates the classes.
- **Write** it (steer): add `α·u` to `h` mid-forward-pass and see how the model's output changes.
  This is what **steering / DCT** cares about.

The whole project is about the difference between "a direction you can *read* truth from" and "a
direction that *writes* truth into behavior."

**Two ways to get a supervised truth direction** (both used as our reference "truth axis"):
- **Mean-difference** `v_mean = mean(h | true) − mean(h | false)`, normalized. Simple, robust.
- **Logistic-gradient** `v_grad = w / σ` where `w` is the logistic-regression weight vector and
  `σ` the feature std. This is the direction the linear probe's decision boundary is normal to.
- On clean data these should agree; on `cities` at its best layer they have cosine **0.413** —
  correlated but not identical, already a hint that "the" truth direction is a bit fuzzy.

**DCT in one line** (full detail in `DCT_METHODOLOGY.md`): it isolates a slice of layers
`source→target` as a function `f`, and searches for input perturbations `θ` that maximize the
downstream change `‖f(x+θ) − f(x)‖`, returning ~512 orthogonal such directions `V`. These are the
model's **highest-gain knobs** — the directions it's most causally sensitive to — found with **no
labels**. We compare them to the supervised truth axis.

**Decodable vs. causally-salient — the punchline distinction.**
- *Decodable*: ∃ a linear read-out (a probe) that recovers the label. ⇒ the info is present.
- *Causally-salient*: perturbing along the direction strongly changes downstream computation.
- These can diverge because a model can *represent* a feature (store it linearly) without that
  feature being a *high-gain lever* on the particular slice/behavior you're steering. Superposition
  makes this common: many features are linearly present; only some are "load-bearing" for a given
  computation.

---

## 2. The experimental logic: why a "funnel"

Our first result was a **null**: DCT's single most-aligned direction had cosine ≈ 0.07–0.10 with
the truth axis — about **1.2× a random baseline** (essentially no alignment). A null is only
interesting if you rule out the boring reasons for it. There are three:

1. **Truth genuinely isn't a top causal direction.** (The interesting scientific claim.)
2. **Our supervised "truth direction" isn't actually causal** — it's a decodable *correlate*, so
   there was never a real causal target for DCT to match. (Then the comparison is meaningless.)
3. **Truth is a combination of several DCT directions**, so no *single* one aligns, and a
   single-vector cosine is the wrong instrument. (Then we just measured it wrong.)

The funnel is five tests designed to separate these. Preview of how they map:
- **Test 1 (interpret)** and **Test 3 (subspace)** attack explanation #3 and characterize what DCT
  *did* find.
- **Test 2 (steering)** attacks explanation #2 (is the probe direction causal at all?).
- **Test 4 (cross-dataset)** checks whether either method's "truth direction" is a *general* truth
  feature or a dataset artifact.
- Together they leave only explanation #1 standing.

All analysis is at each dataset's **best (truth-peak) layer**: `cities` layer 11, `common_claim`
layer 13 — the layers where the supervised probe is strongest, i.e., where we give the truth
direction its best possible chance.

---

## 3. Results, one test at a time

Each subsection: **what we did → the number → what it means → intuition/why it's the right test.**

### Test 1 — What are DCT's top directions actually about?

**What we did.** Rank DCT's 512 directions by downstream-effect magnitude `‖U_i‖` (how much each
one changes the target layer). Take the top 10. For each, add it to the residual stream during
generation and read the steered completions vs. the unsteered ones. Label what changed.
(`src/interpret_top10.py` → `interpret_top10_<ds>.md`.)

**What we found.** On `cities` the top directions are about **geography** (they inject
cities/countries/coordinates), plus **format** (one switches the output to French, one to code) and
**encyclopedic register**. On `common_claim` (steered at higher magnitude) they're about **tone**
(dark/negative) and mostly **incoherence**. *None* is a clean "make the statement true/false" knob.

**What it means.** DCT's biggest levers on this model are **content, format, and style** — not a
dedicated truthfulness switch. When falsehoods appear (e.g. "the capital of Japan is in Germany"),
they're a **side effect** of a *geography* vector scrambling which entities get named, not a vector
that dials truth up or down.

**Intuition / why the right test.** This is the test the PI insisted on: before asking "does DCT
find truth?", ask "what does DCT find *at all*?" A subtle, important observation: DCT was *trained
on the cities prompts*, so the directions it flags as high-gain are about the **content of those
prompts** (places). This tells you DCT's discoveries are **input-distribution-dependent** — it
surfaces "what varies / what the model is sensitive to *on the inputs you gave it*," which for
cities is geography, not an abstract truth concept.

### Test 2 — Is the supervised truth direction even causal? (± steering sweep)

**What we did.** Take the probe's truth axis (both `v_mean` and `v_grad`) and *inject* it at the
best layer with strength `α ∈ {−120,…,0,…,+120}` during generation on 8 factual prompts. "+" pushes
toward truth, "−" away. Read whether completions become false or just degrade.
(`src/steer_supervised.py` → `steer_supervised_<ds>.md/.csv`; chart `plot_findings_steering.png`.)

**What we found.**
- **Real falsehoods at strong negative gradient steering** (cities): `α=−120` gives "two plus two
  equals **three**," "the sun rises in the **west**," "Tokyo, the largest city in **Canada**."
- **A mild "+ → more factual" nudge** (common_claim): "the capital of Japan is *a city…*" →
  "**Tokyo**, 13M people" as α rises.
- **But the dominant effect is magnitude, not sign:** factual accuracy falls at *both* extremes
  (cities/gradient hits ~chance at −120 **and** +120). `v_mean` is nearly flat.

**What it means.** The probe direction **is causal** — you can push the model into specific
falsehoods, so it is *not* a dead correlate (this **rules out explanation #2**). But the effect is
**weak and confounded with degradation**: hard steering mostly just breaks the output rather than
cleanly setting a truth value. So the supervised truth direction is a **real but minor** causal
lever.

**Intuition / why the right test.** Probing tells you truth is *there*; steering tells you whether
moving along it *does* anything. A clean causal truth axis would look like a monotonic S-curve
(more "−" → more false, more "+" → more true). We got a weak, roughly-symmetric dome (both extremes
degrade) with only a few genuine sign-dependent flips — the signature of a low-gain direction, not
a control dial. **Caveat:** at large `|α|` you leave the model's normal operating regime, so "false
vs. just broken" is genuinely hard to separate (we address this with the LLM-judge in §7).

### Test 3 — Is truth a *combination* of the top DCT directions?

**What we did.** Two sub-tests, restricted to the **top-k** DCT directions (k = 10, 20, 50):
- **(A) In-span:** what fraction of the truth axis's length lies inside the subspace spanned by the
  top-k DCT vectors? Compare to the chance level for a random vector, which is exactly **k/d**
  (k dimensions out of 2304).
- **(B) Classify-from-DCT-features:** project activations onto the top-k DCT directions and train a
  probe on just those k numbers; compare its accuracy to using k *random* directions.
(`src/subspace_top_k.py`.)

**What we found.**
- (A) top-10 in-span ≈ **0.01** vs chance **0.004**; top-50 ≈ **0.03** vs **0.02**. Barely above
  chance.
- (B) top-10 DCT features classify truth at **0.94 (cities)** — but **random** k features get
  **0.97**. DCT features are *no better than random* (slightly worse on cities).

**What it means.** Truth is **not** reconstructable from DCT's top directions, even as a linear
combination — this **rules out explanation #3**. The "0.94 accuracy" in (B) looks high but is a
trap: because truth is *so* linearly decodable, **almost any** low-dimensional projection preserves
enough of it to classify (a Johnson–Lindenstrauss effect — random projections approximately
preserve linear separability). The meaningful comparison is **DCT vs. random**, and DCT carries
**no special truth signal**.

**Intuition / why the right test.** The single-cosine null could have missed truth hiding as a mix
of a few DCT vectors. (A) asks "can you *build* the truth arrow out of DCT arrows?" — no. (B) asks
"do DCT directions *predict* truth better than nothing?" — no. Two angles, same answer.

### Test 4 — Do DCT directions generalize as a truth detector?

**What we did.** Train a truth classifier on dataset A, test its accuracy on dataset B (all pairs),
for two feature sets: (A) the raw activations (supervised probe), (B) the top-20 DCT directions
from A used as a fixed basis. A *real* truth feature should transfer; a dataset artifact won't.
(`src/cross_dataset.py`.)

**What we found (off-diagonal = cross-dataset transfer):**
- **Supervised probe:** cities→common_claim **0.60**, common_claim→cities **0.76** (both well above
  the 0.50 chance line).
- **DCT directions:** both off-diagonals **0.50 = chance**.

**What it means.** The **supervised** truth direction encodes something *general* about truth that
partially transfers across very different datasets. The **DCT** directions encode nothing about
truth that transfers — consistent with Test 1 (they're about dataset-specific content like
geography). So DCT is **not** a model-intrinsic truth carrier; if anything the supervised direction
is the more robust notion of truth.

**Intuition / why the right test.** Transfer is the classic test for "did you find a real feature
or overfit the dataset?" DCT collapsing to chance off-diagonal is strong evidence its directions
are tied to the input distribution it was trained on, not to an abstract truth concept.

### The single-vector view (the "money" figure)

`plot_findings_decode_vs_causal.png` puts the core contrast in one chart — *how well can a single
direction classify truth?*

| | supervised probe (full) | best single DCT direction | random direction |
|---|---|---|---|
| cities @ L11 | **0.99** | 0.72 | 0.70 |
| common_claim @ L13 | **0.72** | 0.61 | 0.55 |

The best of DCT's 512 directions reads truth at ~the level of a **random** direction, and far below
the supervised probe. That single row is the whole finding.

---

## 4. Putting it together — what the convergent null means

Five independent measurements point the same way:

| Question | Answer | Rules out |
|---|---|---|
| Are DCT's top directions about truth? (Test 1) | No — geography/format/tone | — |
| Is the supervised direction causal? (Test 2) | Yes, but weakly | explanation #2 |
| Is truth a combination of top DCT dirs? (Test 3) | No (≈ chance) | explanation #3 |
| Do DCT dirs generalize as truth? (Test 4) | No (≈ chance) | — |
| Best single DCT dir vs truth (money fig) | ≈ random | — |

Only **explanation #1** survives: **truth is causally real but not a dominant causal direction** in
gemma-2-2b at these layers.

**The interpretation, three ways to say it:**
1. *Plain:* Truth is easy to read out of the model but hard to push on — reading and driving are
   different things.
2. *Mechanistic:* Linear decodability certifies that truth is **linearly represented**; it does not
   certify that truth is a **high-gain functional direction**. DCT measures the latter, and truth
   scores low. The model *stores* truth without *routing much behavior through it* — at least in a
   base model whose "behavior" is next-token completion.
3. *Why it matters:* Interpretability constantly infers "the model uses feature X" from "a probe
   finds X." This is a clean counterexample: a strongly-decodable feature that is not causally
   dominant. It's direct evidence for the **non-identifiability** worry (many directions can be
   read as "truth"; being readable doesn't make one *the* causal one) and motivates causal/
   behavioral evaluation over probe-only claims.

**One caveat to state honestly:** "causally dominant" is relative to *what DCT optimizes* (max
change to a mid-layer slice) and *what we steer* (base-model completions). Truth might be more
causally load-bearing for an instruction-tuned model, a later layer, or a task that actually
requires asserting facts. That's exactly what the next steps probe.

---

## 5. Evidence table (verbatim generated text)

*Every quote is copied from the run outputs. `⚠️ false` = factual error; `✓` = correct.*

### Test 1 — DCT top vectors = topic/format/tone, not truth

| Claim | Where | Prompt → steered output |
|---|---|---|
| Top vectors inject **geography**, and scramble it into false geography | `interpret_top10_cities.md`, vec #375 | "The capital of Japan is" → "a city located in the **country of Germany** … within Frankfurt" ⚠️ false |
| " (false entity attribution) | vec #62 | "The capital of Japan is" → "…'Tokyo' was **founded by John F Kennedy**" ⚠️ false |
| Some are **encyclopedic** (often accurate) | vec #139 | "The capital of Japan is" → "**Tokyo**, and it's the country's most populous city…" ✓ |
| Others are pure **format** (language switch) | vec #114 | "The city of Paris is in the country of" → "France. **French: Paris est une ville…**" |
| On common_claim, top vectors are **tone/incoherence** | `interpret_top10_common_claim_true_false.md`, vec #355 | "Two plus two equals" → "four, but I also **ride horses and eat babies…**" (dark word-salad) |

### Test 2 — Steering the supervised truth direction (± sweep)

| Claim | Where | Prompt → steered output |
|---|---|---|
| Strong **−** gradient steering induces real falsehoods | `steer_supervised_cities.md`, grad, `α=−120` | "Two plus two equals" → "**three**" ⚠️ false |
| " | grad, −120 | "The sun rises in the" → "**west**, and sets on a lake" ⚠️ false |
| " | grad, −120 | "The capital of Japan is" → "Tokyo, the largest city in **Canada**" ⚠️ false |
| mean-diff also errs at the extreme (weaker) | mean_diff, −120 | "The first president…was" → "a man named **John Adams**" ⚠️ false |
| **+** steering sharpens facts | `steer_supervised_common_claim_true_false.md`, grad, `α=+20` | "The capital of Japan is" → "**Tokyo**…13 million people!" ✓ |
| Dominant pattern = degradation, not a sign flip | `plot_findings_steering.png` | cities/grad accuracy: −120 → **0.50**, +120 → **0.50** (both extremes → chance) |

### Tests 3 & 4 — corroborating numbers (math on saved vectors)

| Claim | Numbers |
|---|---|
| Truth isn't a combination of top DCT vectors | in-span: top-10 **0.01** vs chance 0.004; top-50 **0.03** vs 0.02 |
| DCT features don't beat random at reading truth | cities top-10 DCT **0.94** vs random **0.97** |
| DCT dirs don't generalize across datasets | DCT transfer **0.50** (chance) vs supervised **0.60–0.76** |

---

## 6. Expectations vs. reality

| We expected | We found | Takeaway |
|---|---|---|
| DCT (unsupervised + causal) rediscovers truth → truth is a primary causal axis | DCT's top dirs are geography/format/tone; truth absent as single / combination / transferable feature | Causal salience ≠ decodability |
| Steering the truth axis cleanly flips truth↔false | Weak/noisy: a few real falsehoods at extreme "−", mild "+→factual", mostly degradation | Truth is a *minor* causal lever |
| Clean (cities) vs messy (common_claim) behave differently | Both the same story: readable everywhere, causally dominant nowhere | The gap is general, not a data quirk |

---

## 7. Limitations (state these to the PI up front)

- **Interpretation is currently hand-read** (my read of the completions; the `Label:` lines in
  `interpret_top10_*.md` are for you to confirm). We built an **LLM-judge** (`src/judge_results.py`)
  to score every completion TRUE/FALSE/INCOHERENT and auto-label the top-10 vectors — not yet run
  (needs `ANTHROPIC_API_KEY`). That upgrades Tests 1–2 from qualitative to quantitative.
- **Step-2 factual score is a coarse keyword heuristic over 8 prompts** (1/8 granularity → noisy).
  The judge + more prompts fix this.
- **Degradation confound:** at large `|α|`, false vs. incoherent is genuinely ambiguous; the judge's
  explicit INCOHERENT class disentangles this.
- **DCT config is one point in a space:** 512 factors, source→target = 11→20 / 13→22. A truth lever
  could appear with more factors or a different/deeper slice.
- **Geometry, not behavior:** everything here is representation-level. The gold standard for a
  causal truth direction is a behavioral eval (**A-LQR**), which we haven't run.

---

## 8. How to explain it (layered, so you can scale to your audience)

**30-second version:** "A linear probe reads 'is this true?' off gemma-2-2b at 99%. We used an
unsupervised method, DCT, to find the model's most causally powerful directions and checked if
truth was one of them. It isn't — its top directions are about geography and formatting. Steering
the probe's truth direction only weakly changes factuality. So truth is decodable but not a
dominant causal lever: reading a feature and driving behavior with it are different."

**2-minute version:** add the funnel logic ("a null has three explanations; we ran five tests to
isolate the interesting one"), the money figure (best DCT dir ≈ random for reading truth), and the
"±steering gives a few real falsehoods but mostly degrades" nuance.

**Whiteboard version:** draw ℝ^2304; a probe finds the hyperplane normal `v_truth` (reading);
DCT finds the top eigen-directions of the slice Jacobian (driving); show they're nearly orthogonal
(cos ≈ random); note the subspace test (project `v_truth` onto span(top-k DCT) ≈ k/d) and transfer
(DCT 0.50 vs supervised 0.6–0.76). Land on decodable ≠ causally-salient and the non-identifiability
point.

---

## 9. Open questions / next steps for the PI

- Run the **LLM-judge** to make Tests 1–2 quantitative (does 0/10 top vectors manipulate truth? does
  the "−" side raise the FALSE rate vs. the INCOHERENT rate?).
- **Scale DCT:** `--num-factors 1024`, deeper/other target layers — does a truth lever appear lower
  in the ranking or in a different slice?
- **Instruction-tuned model / fact-requiring task:** is truth more causally load-bearing when the
  behavior actually depends on asserting facts?
- **Next concepts:** toxicity, sycophancy — same decodable-vs-causal question.
- **A-LQR access** for a proper behavioral (not geometric) evaluation.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Residual stream / hidden state | the ℝ^2304 vector representing a token at a layer |
| Linear probe | logistic regression on hidden states; finds the read-out direction for a label |
| Decodable | a linear probe can recover the label (info is linearly present) |
| Causally-salient / high-gain | perturbing the direction strongly changes downstream computation |
| DCT | unsupervised search for the highest-gain steering directions (label-free) |
| `V` / top-k by `‖U‖` | DCT's input directions / ranked by downstream-effect magnitude |
| mean-diff / gradient direction | two supervised estimates of the truth axis |
| cosine baseline (~0.06) | max \|cos\| of ~512 random unit vectors in ℝ^2304 ≈ chance "match" |
| in-span / k/d chance | fraction of a vector inside a k-dim subspace; random ≈ k/d |
| transfer (0.50 = chance) | train-on-A test-on-B accuracy; 0.50 = no generalizable signal |
| non-identifiability | many directions read as "truth"; readability doesn't make one causal |
| superposition | models store many features linearly in shared dimensions; not all are load-bearing |
