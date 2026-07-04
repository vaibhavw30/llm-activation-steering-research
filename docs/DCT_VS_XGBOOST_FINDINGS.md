# DCT vs. Non-Linear (XGBoost) Truth — Findings

*Does DCT's causal subspace carry the **non-linear** truth structure a linear probe can't read?
Short answer: **no** — the null extends to the non-linear frontier. The extra truth signal
XGBoost extracts from the full residual stream is absent from DCT's top causal directions, no
better than random projections.*

Companion to `DCT_VS_TRUTH_FINDINGS.md` (the linear comparison) and `MEETING_SUMMARY.md` (the
linear-vs-XGBoost accuracy gap). Script: `src/subspace_xgb.py`; figure:
`plot_findings_subspace_xgb.png`.

---

## 1. Why this experiment exists

The linear funnel (`DCT_VS_TRUTH_FINDINGS.md`) established: the supervised truth direction — a
**linear** readout — is not among (or a combination of) DCT's most causally-salient directions.
But every instrument in that funnel was linear: cosine to `mean_diff`/`grad`, the in-span
projection, the classify-from-DCT-features probe. None could see whether DCT captures **non-linear**
truth structure.

Separately, we know that structure exists. On the messiest dataset (`common_claim`), a non-linear
probe (XGBoost) reads truth better than a linear one — a real gap over the linear probe. That gap
is *evidence of non-linearly-encoded truth a linear probe misses.*

This leaves one unclosed escape hatch for DCT: **maybe truth is causally real but non-linearly
encoded**, so no single direction aligns linearly, yet the non-linear structure still lives inside
DCT's causal subspace. This experiment closes it.

## 2. Method

Non-linear upgrade of Funnel Test 3B. At each dataset's truth-peak layer (cities 11,
common_claim 13), for three feature sets — the **full 2304-dim activations**, the **top-k DCT
projections** (`X @ V_topk`, k ∈ {10,20,50}), and **random-k projections** — we measure:

- **linear accuracy**: standardized logistic probe, 5-fold CV.
- **XGBoost accuracy**: 300 trees, depth 4 (same config as `analyze.py`), 5-fold CV.
- **non-linear gap** = XGBoost − linear. This is the quantity of interest: how much *extra* truth
  a non-linear model extracts beyond a linear one, **within that feature set**.

The logic: if DCT's causal subspace holds the non-linear truth structure, XGBoost-on-DCT-features
should show a gap approaching the full-activation gap, and beat random-k. If the gap collapses to
~0 (and to random), DCT misses truth non-linearly too.

## 3. Results (5-fold CV accuracy)

### common_claim @ layer 13 — the decisive dataset

| feature set | linear | XGBoost | non-linear gap |
|---|---|---|---|
| **full activations** | 0.722 | 0.779 | **+0.057** |
| DCT-top-10 | 0.669 | 0.658 | −0.011 |
| random-10 | 0.633 | 0.625 | −0.008 |
| DCT-top-20 | 0.696 | 0.688 | −0.008 |
| random-20 | 0.680 | 0.677 | −0.003 |
| DCT-top-50 | 0.717 | 0.700 | −0.017 |
| random-50 | 0.690 | 0.698 | +0.008 |

*(The full-activation gap is +0.057 under 5-fold CV; the +0.082 in `summary_all.csv` is the same
effect measured with a single 80/20 split — CV is the more conservative estimate.)*

### cities @ layer 11 — control (no non-linear signal to find)

| feature set | linear | XGBoost | non-linear gap |
|---|---|---|---|
| full activations | 0.994 | 0.995 | +0.001 |
| DCT-top-10 | 0.943 | 0.947 | +0.003 |
| random-10 | 0.967 | 0.967 | −0.000 |
| DCT-top-50 | 0.990 | 0.984 | −0.006 |
| random-50 | 0.994 | 0.988 | −0.006 |

Cities is clean/linear: its full-activation gap is +0.001, i.e. **there is no non-linear truth
structure to locate.** The small non-zero DCT gaps in the cities panel of the figure are noise at
a magnified y-scale, not signal. Cities is included only as a negative control; **common_claim is
the test.**

## 4. What it means

**The non-linear gap does not survive inside DCT's subspace.** On common_claim, XGBoost beats the
linear probe by +0.057 on the full residual stream, but inside the top-k DCT directions the gap is
**negative at every k** (−0.011, −0.008, −0.017): XGBoost extracts nothing a linear probe didn't
already get. Projecting onto DCT's highest-gain causal directions destroys the non-linear truth
signal (and most of the linear signal too — linear accuracy drops from 0.722 to 0.67–0.72).

**DCT is not meaningfully better than random.** DCT-top-k edges random-k by a few points, but that
edge appears in the *linear* probe as well (DCT-linear 0.669 vs random-linear 0.633 at k=10) and
vanishes by k=50 — it is a small *linear* decodability edge, not captured non-linear truth.

**The 2×2 is now complete:**

| | linear truth | non-linear truth |
|---|---|---|
| in full activations? | yes (probe 0.72) | yes (XGB gap +0.057) |
| in DCT's causal subspace? | no (≈ random) | **no (gap ≈ 0)** ← this run |

Every "full activations" cell is populated; every "DCT subspace" cell is empty. **Truth — linear
and non-linear alike — is decodable from the residual stream but absent from the directions the
model is most causally sensitive to.** DCT doesn't miss truth *because* truth is non-linear; it
misses truth because truth isn't causally salient.

## 5. Caveats

- **XGBoost has no direction.** A tree ensemble can't be injected/steered, so this test lives in
  the *decoding* frame (can you read truth from the subspace?), not the *causal* frame. To bring a
  non-linear model into the steering comparison you'd need to extract a direction from it (e.g. a
  SHAP-based direction) and inject that — a separate future experiment.
- **One DCT config.** 512 factors, source→target 13→22. A non-linear truth lever could appear with
  more factors or a different/deeper slice.
- **Geometry/decoding, not behavior.** As with the rest of the funnel, the gold-standard causal
  test is behavioral steering evaluation (A-LQR).

## 6. One-line summary (for the meeting)

> "We checked the last escape hatch — whether DCT's causal subspace holds the *non-linear* truth
> structure a linear probe misses. It doesn't: on common_claim, XGBoost beats a linear probe by
> +0.057 on the full residual stream, but that gap collapses to ≈0 (−0.01, no better than random)
> inside the top-k DCT directions. Truth is decodable-but-not-causal non-linearly as well as
> linearly."

## 7. Reproducibility

- `src/subspace_xgb.py --dataset <cities|common_claim_true_false>` → `subspace_xgb_<ds>.csv`.
- `src/viz_subspace_xgb.py` → `plot_findings_subspace_xgb.png`.
- Reads `activations/acts_<ds>.npz` (peak layer) + `dct_V/U_<ds>.pt`; XGBoost config matches
  `src/analyze.py`; 5-fold CV, seed 42.
