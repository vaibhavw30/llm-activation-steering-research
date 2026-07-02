# DCT vs. Truth — Meeting Results (with evidence)

*Self-contained results doc for the PI meeting: what we found, why, and the actual generated text
that backs each claim. Detailed version: `FUNNEL_RESULTS.md`. How DCT works: `DCT_METHODOLOGY.md`.*

---

## Headline

We tested whether **DCT** (an unsupervised method that finds the model's *most causally powerful*
steering directions) recovers the **"truth direction"** our supervised probes read at ~99%.
**It doesn't.** Across five independent tests, truth is **highly readable but not one of the
model's dominant causal directions.** When we steer the supervised truth direction directly, it
*can* make the model lie a little, but only weakly. **Verdict: truth is easy to read, hard to push.**

## What we did (in one breath)

Supervised probes locate a truth direction in gemma-2-2b. DCT, with no labels, finds ~512
directions the model is most causally sensitive to. We then asked, four ways, whether truth is
among them — and separately steered the supervised direction to see if it's causal at all.

## Results at a glance

| Test | Question | Answer |
|---|---|---|
| 1. Interpret top-10 DCT vectors | What are DCT's strongest directions about? | Topic / format / tone — **not truth** |
| 2. Steer the supervised direction (±) | Does the truth direction causally change behavior? | **Weakly** — a few falsehoods at strong "−", mostly just degradation |
| 3. Subspace | Is truth a *combination* of top DCT vectors? | No (≈ chance) |
| 4. Cross-dataset | Do DCT directions generalize as a truth detector? | No (≈ chance); supervised ones do |

**Ending A:** truth is causally real but **not a dominant causal lever**; DCT's biggest levers are
topic, format, and tone. → *decodable ≠ causally dominant.*

## Expectations vs. reality

| We expected | We found |
|---|---|
| DCT would rediscover truth → truth is a primary causal axis | DCT's top directions are topic/format/tone; truth absent (single, combination, and transfer) |
| Steering the truth direction would cleanly flip truth↔false | Weak, noisy — a few falsehoods at extreme "−", a mild "+→factual" nudge, mostly degradation |
| Clean (cities) vs messy (common_claim) data would differ | Both the same: readable everywhere, causally dominant nowhere |

---

## Evidence table (actual generated text)

*Every quote below is copied from the run outputs. "steered" = with the vector/direction injected.
`⚠️ false` marks a factual error; `✓` marks a correct/expected completion.*

### Step 1 — DCT's top vectors are about topic/format/tone (not truth)

| Claim | Where | Prompt → steered output |
|---|---|---|
| Top vectors inject **geography** (DCT was trained on the cities dataset) — and scramble it into **false geography** | `interpret_top10_cities.md`, vec #375 | "The capital of Japan is" → "a city located in the **country of Germany** … within Frankfurt" ⚠️ false |
| " (false entity attribution) | vec #62 | "The capital of Japan is" → "…'Tokyo' was **founded by John F Kennedy**, who died on November 23rd" ⚠️ false |
| Some directions read as **encyclopedic** (often accurate) | vec #139 | "The capital of Japan is" → "**Tokyo**, and it's the country's most populous city…" ✓ |
| Others are pure **format** (foreign language) | vec #114 | "The city of Paris is in the country of" → "France. **French: Paris est une ville de la région Île-de-France…**" (switches to French) |
| On common_claim (high scale), top vectors are **tone / incoherence**, not truth | `interpret_top10_common_claim_true_false.md`, vec #355 | "Two plus two equals" → "four, but I also **ride horses and eat babies. I am a gay alcoholic in the rain over kittens**…" (dark word-salad) |

**Takeaway:** none of the top-10 is a clean "truthfulness" knob. Falsehoods appear only as a
*byproduct* of entity/geography scrambling — not a controlled true↔false switch.

### Step 2 — Steering the supervised truth direction (± sweep)

| Claim | Where | Prompt → steered output |
|---|---|---|
| Strong **negative** gradient steering **induces real falsehoods** | `steer_supervised_cities.md`, grad, scale **−120** | "Two plus two equals" → "**three**" ⚠️ false |
| " | grad, −120 | "The sun rises in the" → "**west**, and sets on a lake" ⚠️ false |
| " | grad, −120 | "The capital of Japan is" → "Tokyo, the largest city in **Canada**" ⚠️ false |
| The mean-diff version also errs at the extreme (weaker) | mean_diff, −120 | "The first president of the United States was" → "a man named **John Adams**. I'm not kidding…" ⚠️ false |
| **Positive** steering slightly **sharpens facts** | `steer_supervised_common_claim_true_false.md`, grad, scale **+20** | "The capital of Japan is" → "**Tokyo**, and it's the largest city in Asia. …13 million people!" ✓ (vs unsteered "a city that has been around…") |
| But the dominant pattern is **magnitude-degradation**, not a clean sign flip | `plot_findings_steering.png` | cities/gradient factual accuracy: **−120 → 0.50** and **+120 → 0.50** (both extremes fall to chance) |

**Takeaway:** the supervised direction *is* causal (it can push the model into specific
falsehoods), but the effect is **weak and noisy** — closer to "hard steering degrades output" than
"a dial that sets truthfulness." So even the supervised truth direction is only a **minor** lever,
which is why DCT (which finds the *biggest* levers) misses it.

### Steps 3 & 4 — the corroborating numbers (no text; math on saved vectors)

| Claim | Numbers |
|---|---|
| Truth isn't a *combination* of the top DCT vectors | top-10 in-span ≈ 0.01 vs chance 0.004; top-50 ≈ 0.03 vs 0.02 — essentially chance |
| DCT directions don't **generalize** as truth detectors | DCT cross-dataset transfer **0.50 = chance**; supervised transfer **0.60–0.76** |
| Best single DCT direction ≈ a random direction for reading truth | cities 0.72 vs random 0.70 (probe 0.99); common_claim 0.61 vs random 0.55 (probe 0.72) |

---

## Figures to show

- **`plot_findings_decode_vs_causal.png`** — the money chart: probe reads truth at ~99% / 72%, but
  the best DCT direction ≈ a random direction. *Decodable ≠ causally dominant.*
- `plot_findings_steering.png` — Step 2 ± sweep: weak/noisy, mostly magnitude-degradation.
- `plot_findings_transfer.png` — supervised generalizes (green off-diagonals); DCT doesn't (red 0.50).
- `plot_findings_alignment.png`, `plot_findings_subspace.png` — the single-vector and subspace nulls.

## How to say it to the PI

> "We ran the DCT paper's method on gemma-2-2b and checked, five ways, whether truth is one of its
> top causal directions. It isn't — the top DCT vectors are about geography, formatting, and tone
> (geography because DCT was trained on the cities data). Directly steering our supervised truth
> direction confirms it's *weakly* causal — strong negative steering makes the model say things
> like 'two plus two equals three' and 'the sun rises in the west' — but the effect is small and
> mostly just degrades the output. So truth is highly decodable yet only a minor causal lever:
> decodable ≠ causally dominant."

## Honest limitations (say these too)

- The interpretation is from **reading the completions by hand** (Claude's read; the `Label:` lines
  in `interpret_top10_*.md` are for you to confirm). No LLM-judge was used yet.
- The Step-2 accuracy score is a **keyword heuristic over 8 prompts** (coarse); a larger prompt set
  + an LLM-judge would firm up the small trends.
- At large \|steering\| the output degrades, so "false vs. just broken" is partly confounded.
- This is **geometry + qualitative steering**, not a full behavioral eval (**A-LQR**).

## Open questions for the PI

- Want an **LLM-judge** truthfulness score to make Step 2 quantitative and airtight?
- Push further: `--num-factors 1024` / deeper target layer — does a truth lever appear lower in the
  ranking?
- Next concept beyond factual truth — **toxicity? sycophancy?** — and access to the **A-LQR** code
  for behavioral evaluation.
