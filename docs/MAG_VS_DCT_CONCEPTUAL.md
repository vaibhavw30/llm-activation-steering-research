# MAG vs. DCT — The Concepts Behind the Numbers

*A conceptual companion to `DCT_VS_MAG_ON_TRUTH.md` (the results) and
`PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` §8 (the plain-language build note). This doc has no new
numbers — it explains **what the two methods actually ask**, **why they answer differently**, and
**why building the second one strengthens the whole project's claim**. Read this if you want the
"why," not the "what."*

---

## 0. The one-paragraph idea

We have a claim about gemma-2-2b: **truth is easy to read out of the activations, but the model is not
strongly *driven* by a truth direction.** "Decodable ≠ causal." That claim was built entirely on one
unsupervised method, DCT. The obvious objection is that the null might be a property of DCT's
particular objective rather than of the model. So we built a **second, mechanically unrelated** method
— MAG — that arrives at the truth direction from a completely different starting point (watching how
activations *move* when you ask the model a question, rather than searching for high-impact steering
levers). MAG **finds** the truth direction cleanly, confirming it is really there in the geometry — and
that direction turns out to be **orthogonal** to what DCT flags as causally important. Two methods that
share no machinery agree, so the null is about the model, not the tool.

---

## 1. Two different questions you can ask about a representation

When someone says "the model represents truth," they can mean two genuinely different things, and the
difference is the whole story here.

**Question A — Representational (is it *there*, and readable?).**
Do the activations for true statements sit in a different place than the activations for false ones?
If a simple readout can separate them, the information is *present and linearly accessible*. A
supervised probe answers this — but a probe is *told* the labels, so it can only confirm the
information exists, not tell us whether the model uses it.

**Question B — Causal (does the model *act* on it?).**
If you reach into the residual stream and push along a direction, does the model's *behavior* change a
lot? A direction can be readable yet causally inert — present in the activations but not something the
downstream computation leans on. This is the question that matters for control, safety, and "does the
model really know."

These come apart. A concept can be **highly readable but causally weak** (our finding for truth), or
**causally strong but not cleanly readable**. The entire "decodable ≠ causal" thesis is the claim that
*truth lives at (high A, low B)* in gemma-2-2b.

**DCT is a Question-B instrument. MAG is a Question-A instrument built without labels.** That is the
cleanest way to hold them in your head.

---

## 2. How DCT sets itself up (the interventional route)

DCT — Deep Causal Transcoding — is an **intervention search**. It ignores labels entirely and asks:
*of all the directions I could inject into the residual stream, which ones change the model's output
the most?*

- It returns 512 candidate directions, each a pair: **V** (the input direction you add at layer 13)
  and **U** (the downstream effect it produces by layer 22).
- Each direction has a **potency** `‖U‖` — the size of the effect. Sort by potency and the top of the
  list *is*, by construction, "the model's strongest causal levers, discovered without ever being told
  about truth."

The logic of the funnel test: if truth were one of the things the model is most causally driven by,
the supervised truth direction should appear *inside* those top DCT levers. It doesn't. DCT's answer to
Question B for truth is **no**.

The defining features of DCT's setup: it is **causal** (it perturbs and watches the output), **global**
(it searches the whole intervention space, agnostic to any concept), and **output-anchored** (potency
is measured in downstream effect). Its natural failure mode is exactly the thing we're studying: a
readable-but-inert concept is *invisible* to DCT, because DCT only sees things that move the output.

---

## 3. How MAG sets itself up (the observational route)

MAG — Mining via Activation Geometry — never intervenes on the model and never optimizes anything. It
runs the model **forward** and watches a natural computation the model already performs.

**The core move: the prefix-induced shift.** Take a statement `p`. Run it through the model and read
the residual stream: call that `m(p)`. Now prepend a question — "Is this statement true?" — to get
`Q‖p`, run *that* through, and read `m(Q‖p)`. The **shift** is

> Δ<sup>Q</sup>(p) = m(Q‖p) − m(p)

This is the change in the model's internal state caused by *being asked to consider the property*. The
premise: if the model internally represents whether `p` is true, then asking it to think about truth
should move its activations along the axis that encodes truth. We never tell MAG any labels — it reads
the geometry of that movement.

**Eight operators** are just eight different ways of reading that movement (raw activations, the shift,
the question-conditioned activations, a few-shot-primed version, an interaction term, etc.). They are
different lenses on the same underlying computation; **InputDelta** — the bare shift Δ<sup>Q</sup> — is
the most direct one, and it is the one that carries essentially the full truth signal.

**The subtle part — average vs. contrast.** The *average* shift direction (`v_Q`) is boring: it's just
"the model is now in question-answering mode," generic across true and false statements alike — and
indeed `v_Q` is orthogonal to the truth direction. Truth lives in the **class contrast**: the
*difference* between how true statements shift and how false statements shift,
`u_Q = shift(false) − shift(true)`. That contrast direction is where the truth signal concentrates.
Conceptually: asking the question moves everything; asking it moves *true* and *false* statements in
subtly different directions, and that difference is the readout.

**A bonus channel DCT structurally cannot have — the self-verdict `y^M`.** Because MAG literally asks
the model a yes/no question, it can also read the model's **own answer** from the first-token logits.
This is a second, behavioral notion of "does the model represent truth": not "do the activations
separate" but "does the model *say* the right thing." DCT has no analogue — it never asks the model
anything.

**Natural-unit calibration.** Because MAG knows the size of a real question-shift (‖Δ<sup>Q</sup>‖), it
can express steering strength in *natural units*: α(τ) = τ · ‖prefix-shift‖. τ = 1 means "push as hard
as actually asking the question does." DCT's steering magnitudes, by contrast, are in arbitrary units
you have to sweep and eyeball. This matters for the behavioral test (§7).

The defining features of MAG's setup: it is **observational** (no intervention), **concept-directed**
(it targets truth by construction, via the question it asks), and **representation-anchored** (it reads
the geometry of a forward pass, not a downstream effect). Its natural blind spot is the mirror image of
DCT's: MAG will happily find a readable-but-inert concept — which is precisely why it's the right second
instrument here.

---

## 4. The setups, side by side

| | **DCT** | **MAG** |
|---|---|---|
| Underlying question | B — is the direction **causally** load-bearing? | A — is the concept **there and readable**, unsupervised? |
| Mechanism | Search for high-impact **interventions** | Read the **activation shift** from a forward pass |
| Uses labels? | No | No (gold used only for the semi-supervised contrast) |
| Concept-agnostic or -directed? | Agnostic (finds whatever moves output) | Directed (asks about truth specifically) |
| Anchored on | Downstream **output** effect (potency ‖U‖) | Internal **representation** geometry (Δ<sup>Q</sup>) |
| Natural blind spot | Readable-but-inert concepts are **invisible** | Says nothing about causal power on its own |
| Extra channel | — | The model's **self-verdict** y<sup>M</sup> |
| Steering units | Arbitrary magnitude sweep | **Calibrated** natural units α(τ)=τ·‖Δ<sup>Q</sup>‖ |

The two methods are almost perfectly complementary: each one's blind spot is the other's specialty.
That is *why* running both is worth the effort — not for a second opinion on the same measurement, but
for coverage of both halves of "does the model represent truth."

---

## 5. What the findings mean, conceptually

**Finding 1 — the truth direction is really there (MAG answers Question A: yes).**
MAG, with no labels, reconstructs a direction that agrees with the supervised truth direction at cosine
≈ 0.98–1.00 on all four datasets, and reading the prefix shift classifies truth as well as reading the
raw activations. This is not a foregone conclusion — it says the model *spontaneously reorganizes its
activations along a truth axis when asked to consider truth.* The representation is genuine and
accessible, not an artifact of a probe being handed the answers.

**Finding 2 — that direction is orthogonal to DCT's causal levers (A and B come apart, confirmed
twice).**
The very direction MAG recovers so cleanly has cosine ≈ 0 with DCT's most potent causal vector, and
projecting activations onto DCT's causal subspace reads truth no better than random directions. So the
thing that *encodes* truth and the thing the model is *driven by* are different objects. This is exactly
the funnel's null — but now reached from the representational side instead of the causal side. **The
epistemic upgrade is the point:** before, "truth is decodable but not causal" rested on one method, so a
skeptic could say "you just didn't find it because DCT's objective doesn't look for it." Now a method
whose entire job is to *find readable concept directions* finds this one immediately and still lands
orthogonal to DCT. The null survives the strongest available objection.

**Finding 3 — the model represents truth but won't *say* it (representation ≠ expression).**
MAG's self-verdict channel is degenerate on the base model: gemma-2-2b answers "yes" to essentially
every statement, and the activations at the verdict position separate true from false only at chance.
Put beside Finding 1, this is striking: **the geometry cleanly separates true from false, yet the
model's own yes/no behavior does not.** The information is *in there* (Question A: yes) but the base
model has no reliable machinery to *express* it as an answer. This is the "knows but won't tell" gap,
and it is specifically a **base-model** phenomenon — an instruction-tuned model, trained to answer, is
where the self-verdict channel would be expected to come alive. It also cleanly scopes MAG's fully
unsupervised promise: the *geometry* arm works on a base model; the *self-labeling* arm needs an instruct
model.

**A supporting texture — the shift is a clean line that frays with messiness.**
The prefix shift is nearly one-dimensional on clean datasets and progressively less so on messy ones
(ε_Q rising from ~0.18 to ~0.55). Conceptually: for a crisp concept like "which country is this city
in," asking the question moves the state along essentially a single axis; for a messy grab-bag like
common-sense claims, the "consider the truth of this" operation is spread across more directions,
because the concept itself is more entangled. The geometry's tidiness tracks the concept's tidiness.

---

## 6. Why the second method matters more than "another data point"

It is tempting to read MAG as a replication — "we checked the null twice." It is stronger than that,
for a specific reason: **the two methods can only agree by accident if they share a bias, and they share
almost no machinery.** DCT is gradient-based intervention optimization anchored on output effects; MAG
is a subtraction of two forward passes anchored on internal geometry. There is no common objective, no
shared hyperparameter, no shared notion of "important" that could produce the same artifact in both. So
when both say "the truth direction is not the causal direction," the most economical explanation is that
it's a fact about gemma-2-2b — which is exactly what triangulation is supposed to buy.

Equally, MAG contributes something DCT *cannot*: a clean, unsupervised, positive demonstration that the
truth direction exists and is readable. DCT can only ever tell you a concept is *absent* from its causal
levers; it has no way to affirm "the concept is present and here it is." MAG closes that gap from the
other side. Together they let us say the strong version: **not "we couldn't find truth among the causal
levers," but "we *did* find truth, cleanly, and it is provably elsewhere than the causal levers."**

---

## 7. How this sets up the behavioral test differently

Everything above is geometry. The remaining question is behavioral: if you actually inject MAG's truth
direction and let the model generate, does it change what the model *says*? Two setup differences from
the DCT steering test are worth stating in advance:

1. **Calibrated strength.** MAG steers in natural units — τ = 1 is "as hard as asking the question."
   The DCT-side test had to sweep arbitrary magnitudes (−120 … +120) and locate the interesting range
   by hand. MAG's sweep (signed τ) is interpretable from the first run: we know what "one unit" means.
2. **A direction we already know is the right one.** The DCT steering test injected the *supervised*
   truth direction (built from labels). MAG's `u_Q` is an *unsupervised* reconstruction of that same
   direction (cosine ≈ 0.99), so the behavioral test asks a subtly sharper question: does a direction
   the model itself reveals — by how it reorganizes under the question — behave like a truth lever, or,
   like the supervised direction before it, merely a **degradation** lever that breaks the model
   symmetrically without inducing directional lying?

The prior (from the DCT steering result: bounded lie-asymmetry, symmetric degradation) is that it will
be a degradation lever. If MAG's calibrated, model-revealed direction reproduces that, the "truth is a
degradation axis, not a truth axis" story holds across *both* how you obtain the direction (supervised
vs. unsupervised) and *both* mining philosophies. That is the E4 run, and it is the one piece still
outstanding.

---

## 8. What this does *not* show (honest limits)

- **One model.** Everything is gemma-2-2b base. The clean recovery, the orthogonality, and especially
  the dead self-verdict are all statements about this model. The self-verdict result in particular is
  expected to differ on an instruction-tuned model.
- **Geometry, not yet behavior, for MAG.** MAG's causal claim is still borrowed from the DCT-side
  steering result; MAG's *own* behavioral test (E4) has not been run. Findings 1–3 are representational.
- **Orthogonality is measured against DCT's top vector / subspace, not proof of zero causal role.**
  "Truth is orthogonal to the *most potent* causal directions" is what we show; it remains logically
  possible that truth has a small causal role spread thinly below DCT's detection floor. The behavioral
  tests are what bound that, not the geometry.
- **"Truth" is operationalized as these four datasets.** Clean factual recall (cities), translation,
  company facts, and common-sense claims. The monotonic ε_Q trend is suggestive that the story
  generalizes across concept "tidiness," but four datasets is four datasets.
