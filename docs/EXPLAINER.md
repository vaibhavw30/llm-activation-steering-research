# Geometry of Truth: Full Explainer

A plain-English walkthrough of the experiment — why we ran it, how it works,
and what we found.

---

## The big question

When a language model "knows" that a statement is true, where does that knowledge
live, and what shape does it have?

One influential answer comes from Marks & Tegmark (2023): truth is encoded as a
**linear direction** in the model's internal activations. That means if you drew
all the model's internal representations of true statements vs false statements as
points in high-dimensional space, a single straight line through that space would
cleanly separate them. No curves, no clusters — just a line.

That would be a remarkable result. It would mean truth is simple and geometric
inside the model — which matters enormously for how we try to read out, steer, and
ultimately trust model behavior.

This experiment asks: **is that actually true — and does it depend on what kind of
truth you're testing?**

---

## Why Gemma-2-2b?

We used `google/gemma-2-2b`, a 2-billion-parameter language model from Google.
Three reasons:

1. **It matches Julian's A-LQR paper.** This experiment is meant to feed directly
   into that work, so using the same model keeps results comparable.
2. **Bigger = cleaner signal.** Larger models develop more structured internal
   representations, which makes the "is this linear?" question more meaningful.
   A tiny model might fail to encode truth in any organized way.
3. **26 layers is the right scale for CPU.** Running the full activation extraction
   on all 26 transformer layers (+ the embedding layer = 27 total), across ~7,500
   statements, fits in a 5-hour overnight run on a MacBook Air with no GPU.
   Bigger models (7B, 13B) would have required a GPU or days of compute.

The model stores weights as ~9.7 GB on disk (float32, not compressed). It has a
hidden dimension of 2,304 — meaning each token's internal representation is a
vector of 2,304 numbers.

---

## Why these four datasets?

The core hypothesis is: **the answer depends on what kind of truth you're asking
about.** To test that, we need datasets that vary from "very clean and narrow" to
"broad and heterogeneous."

| Dataset | Statements | Character |
|---|---|---|
| **cities.csv** | 1,496 | "The city of X is in country Y." — perfectly templated, one domain, no ambiguity |
| **sp_en_trans.csv** | 354 | Spanish–English translation judgments — clean, different domain |
| **companies_true_false.csv** | 1,199 | Company descriptions — real-world but a single topic |
| **common_claim_true_false.csv** | 4,450 | General world claims — mixed topics, heterogeneous, noisiest |

Each statement is labeled `1` (true) or `0` (false), with a 50/50 balance.

The prediction going in: **cities should be linear** (narrow, clean, one template),
and **common_claim should show non-linear structure** (many kinds of truth tangled
together). The other two fill in the gradient between those extremes.

---

## How the experiment works, step by step

### Step 1 — Run statements through Gemma and record activations

For every statement in every dataset, we feed the text into Gemma and record its
internal state at **the last word token** (the period at the end of each
statement), at **every one of its 27 layers**.

Why the last token? In a causal (left-to-right) language model, each token's
internal representation has "seen" everything before it. By the final token,
the model has processed the whole statement — so its activations there are the
richest summary of what the model "thinks" about that statement.

Why every layer? Representations evolve as information passes through the network.
Early layers do basic processing (syntax, word identity); later layers do more
abstract reasoning. We want to know *which layers* encode truth, not just whether
any layer does.

The result for each dataset is a tensor of shape **(27 layers × N statements × 2304
numbers)**. For `common_claim` that's 27 × 4,450 × 2,304 ≈ 277 million numbers,
stored in a ~944 MB `.npz` file.

### Step 2 — Train two probes at every layer

For each layer, we take the N activation vectors (one per statement) and try to
predict the label (true/false) from them. We do this twice with two very different
classifiers:

**Linear probe (logistic regression)**
A logistic regression learns a single direction in the 2,304-dimensional space that
best separates true from false. It can *only* draw a straight-line boundary. If
truth is linearly encoded, this should work almost perfectly.

**Non-linear probe (XGBoost)**
XGBoost is a powerful gradient-boosted tree model. It can find curved, jagged,
arbitrarily complex decision boundaries. If there's *any* predictable pattern in
the activations — linear or not — XGBoost will find it.

Both are trained on 80% of statements and evaluated on the held-out 20%.

**The key comparison:** `xgb_accuracy − linear_accuracy` = the **non-linear gap**.

- Gap ≈ 0 → truth is linearly encoded (XGBoost finds nothing extra)
- Gap > 0.02 → non-linear headroom exists (XGBoost sees structure the linear probe missed)

### Step 3 — Find the best layer

We do this for all 27 layers and pick the **best layer** — the one where XGBoost
accuracy peaks. That's the layer where truth is most strongly encoded in the
activations, regardless of shape.

### Step 4 — Compare two "truth directions"

On clean datasets, a linear probe should find *the* truth direction. But there are
actually two natural ways to estimate that direction:

- **Mean-difference (v_mean):** take the average activation of all true statements,
  subtract the average of all false statements. The resulting vector points "toward
  truth."
- **Classifier gradient (v_grad):** take the weights learned by the logistic
  regression. These represent the direction the linear probe uses to separate
  true from false.

If truth is cleanly linear, these two should point in the same direction. We
measure their agreement with **cosine similarity** (1.0 = identical, 0.0 =
unrelated, −1.0 = opposite). We do this at every layer too.

---

## What we found

### Probe accuracy: the gap grows with messiness

| Dataset | Best layer | Linear acc | XGBoost acc | Gap | Verdict |
|---|---:|---:|---:|---:|---|
| cities | 11 | 0.990 | 0.993 | **+0.003** | Linear |
| sp_en_trans | 7 | 0.972 | 1.000 | +0.028 | ~Linear |
| companies_true_false | 14 | 0.917 | 0.954 | +0.038 | Mild non-linear |
| common_claim_true_false | 13 | 0.706 | 0.788 | **+0.082** | Non-linear headroom |

**cities:** A linear probe hits 99% accuracy. XGBoost adds 0.3 of a percentage
point. Truth for "The city of X is in country Y" is essentially perfectly linearly
encoded in Gemma's activations at layer 11. This replicates Marks & Tegmark.

**common_claim:** The linear probe drops to 71%. XGBoost pulls it up to 79% — an
8-point gap. The model is encoding *something* about truth in these claims, but
it's tangled up in a way a straight line can't fully capture.

**The pattern is clean:** gap grows monotonically as data gets messier (0.003 →
0.028 → 0.038 → 0.082), and overall decodability drops (0.99 → 0.97 → 0.92 →
0.71). The two things move together: messier concepts are both harder to decode
*and* encoded more non-linearly.

### Direction agreement: same pattern

| Dataset | Cosine @ best layer | Peak cosine | Mean cosine |
|---|---:|---:|---:|
| sp_en_trans | 0.563 | 0.725 (L9) | 0.540 |
| cities | 0.413 | 0.495 (L24) | 0.324 |
| companies_true_false | 0.204 | 0.338 (L24) | 0.214 |
| common_claim_true_false | 0.085 | 0.101 (L0) | 0.073 |

On clean datasets (cities, sp_en_trans), the mean-difference direction and the
classifier-gradient direction point in roughly the same direction, and they agree
*more* as you go deeper in the network (early layers: cosine ~0.2–0.4; late layers:
cosine ~0.4–0.7). This is consistent with truth becoming more organized as it's
processed through the network.

On common_claim, the two directions are essentially unrelated at every layer
(cosine ~0.07 throughout). The model doesn't have a coherent "truth direction"
for messy claims — it's doing something more complicated.

### The important caveat

High cosine agreement sounds like proof that we found *the* truth direction. It
isn't. The **non-identifiability** problem: different directions in activation space
can produce equivalent steering behavior. Two probes agreeing on a vector tells you
something, but not that that vector is what the model is *actually using* to
represent truth causally. Even our best cosines (~0.5–0.7) are well below 1.0 —
the two estimates are correlated but genuinely different vectors. Geometry is not
enough. You need behavioral testing.

---

## The one-sentence version

> On clean, narrow truth ("The city of X is in Y"), Gemma encodes truth linearly —
> a straight line separates true from false activations with 99% accuracy, and
> XGBoost adds nothing. On messy, heterogeneous world claims, truth becomes both
> harder to decode (71%) and partly non-linear (XGBoost gains 8 points), and the
> two "truth direction" estimates stop agreeing — suggesting truth is entangled
> across multiple directions for complex concepts.

---

## What comes next

This is a geometry result. The real question is causal: if you *steer* the model
along one of these candidate truth directions, does its behavior change in the
expected way? That requires **A-LQR** (behavioral evaluation). The questions for
Julian:

- Which concepts should we probe next — **toxicity? sycophancy?**
- Can we get access to the A-LQR code to test whether these directions actually
  control model behavior?

---

## Files on disk

```
got_datasets/          ← raw CSVs (input)
src/
  extract.py           ← activation extraction (Step 1)
  analyze.py           ← probe training + direction analysis (Steps 2–4)
  summary.py           ← cross-dataset summary plot (Step 5)
activations/
  acts_<dataset>.npz   ← extracted activations (27 × N × 2304)
results/
  csvs/                ← probe accuracy CSVs, direction CSVs, summary
  plots/               ← 12 per-dataset plots + 1 summary plot
docs/
  EXPLAINER.md         ← this file
  MEETING_SUMMARY.md   ← concise results for the meeting
  DIRECTIONS_FINDINGS.md ← full direction-agreement analysis
```
