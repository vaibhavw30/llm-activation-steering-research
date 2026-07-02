# DCT vs. the Supervised Truth Direction — Findings

*Does unsupervised Deep Causal Transcoding (DCT) recover the same "truth" direction our
supervised probes found? Short answer: **no** — a robust null. Truth is linearly decodable
but is not among (or a combination of) the model's most causally-salient directions.*

Companion to `DCT_METHODOLOGY.md` (how DCT works) and the geometry-of-truth docs
(`EXPLAINER.md`, `MEETING_SUMMARY.md`, `DIRECTIONS_FINDINGS.md`).

---

## 1. The question

- **Supervised (geometry-of-truth):** probes trained on true/false labels find a "truth
  direction" in gemma-2-2b's residual stream (best layers: cities 11, common_claim 13).
- **Unsupervised (DCT):** with no labels, finds the directions the model is most *causally*
  sensitive to (the biggest steering levers), as 512 vectors `V` at a chosen source layer.

If an unsupervised, causal method independently lands on the supervised truth axis, that's
strong corroboration the direction is real and causal. We tested it.

## 2. Method

- Ran the paper's DCT (`run_dct_data.py`) on 64 balanced true/false statements per dataset,
  `num_factors=512`, `num_iters=30`, on a GH200, with the DCT **source layer set to each
  dataset's truth-peak layer** (cities 11→target 20, common_claim 13→target 22).
- Compared (`compare_directions.py`) the 512 DCT vectors against two supervised truth
  directions computed at the same layer — **mean-difference** and **logistic-gradient** —
  using three tests:
  1. **Max single-vector cosine** vs a random-vector baseline ("is truth *one* DCT vector?")
  2. **Subspace projection**: fraction of the truth direction's variance inside the span of
     all 512 DCT vectors, vs chance ≈ k/d = 512/2304 ≈ 0.22 ("is truth a *combination*?")
  3. **Layer-6 control**: the same comparison at an early layer where truth is weakly encoded.

## 3. Results

### Single-vector alignment (best DCT vector vs truth direction)

| Layer | Dataset | vs mean-diff | vs gradient | supervised agreement (ref) |
|---|---|---|---|---|
| 6 (control) | cities | 0.101 (1.5× rand) | 0.054 (0.8×) | cos(mean,grad)=0.161 |
| 6 (control) | common_claim | 0.106 (1.8×) | 0.055 (0.6×) | 0.066 |
| **11 (peak)** | **cities** | **0.074 (1.2×)** | **0.075 (0.8×)** | **0.413** |
| **13 (peak)** | **common_claim** | **0.094 (1.1×)** | **0.057 (0.8×)** | **0.085** |

### Subspace test (truth's variance inside span of 512 DCT vectors; chance ≈ 0.22)

| Dataset (peak layer) | mean-diff | gradient |
|---|---|---|
| cities @ 11 | 0.288 (1.3× chance) | 0.253 (1.1×) |
| common_claim @ 13 | 0.316 (1.4× chance) | 0.223 (1.0×) |

## 4. What it means

**Robust null.** No DCT vector aligns with the truth direction beyond ~1.2× the random
baseline, and gradient alignment is *at* chance. Truth is barely above chance even in the
512-dimensional DCT span (1.0–1.4×). DCT does not recover the truth direction as a single
lever or a combination.

**The layer-mismatch escape hatch is closed.** At layer 11 the two *supervised* directions
agree at cos 0.413 (vs 0.161 at layer 6) — truth is cleanly, linearly defined there. So the
null isn't because we looked at a bad layer; we looked exactly where truth is strongest and
DCT still missed it.

**Interpretation: decodability ≠ causal salience.** Truth is highly *decodable* (probes hit
~99% on cities) yet is **not among the directions the model is most causally sensitive to**.
The truth axis is largely orthogonal to the subspace of dominant causal levers DCT surfaces.
This is direct, causal evidence for the **non-identifiability** point raised in the geometry
docs: a direction can be strongly predictive without being a primary causal driver.

**Clean-vs-messy aside.** Even the supervised methods agree far more on cities (0.413) than
common_claim (0.085) at their peak layers — consistent with the geometry finding that clean
truth is a cleaner linear direction than messy/heterogeneous truth. DCT misses both.

## 5. Honest caveats — what could still change this

This is a strong preliminary null, not the last word. Legitimate sensitivities to check:

1. **Number of factors.** We used 512. More factors could surface truth lower in DCT's
   ranking. *Caveat on the subspace test:* as k→d the span trivially saturates (chance→1), so
   for large k the **single-vector cosine** is the trustworthy metric — and it's null.
2. **Target layer / slice depth.** We used source→source+9. Truth may exert its effect on
   *later* layers (toward the output); extending the target toward the final layer is worth
   testing.
3. **More statements.** 64 balanced statements per run; more data might sharpen subtle
   directions (though unlikely to turn ~1× chance into strong alignment).
4. **This is geometry, not behavior.** Cosine/subspace alignment is correlational. The real
   test of whether *any* direction is the causal truth direction is **behavioral steering
   evaluation (A-LQR)** — the appropriate next step regardless of this result.

## 6. One-line summary (for the meeting)

> "Running the DCT paper's own method on gemma-2-2b, we find that truth — though linearly
> decodable at ~99% by supervised probes — is **not recovered by unsupervised DCT**: none of
> its 512 most causally-salient directions align with the supervised truth direction above
> chance, even at the layer where truth is most cleanly encoded (where two independent probes
> agree at cos 0.41). Linear decodability does not imply causal salience — concrete evidence
> for non-identifiability, and motivation for behavioral (A-LQR) evaluation."

## 7. Reproducibility

- Model `google/gemma-2-2b`, fp32, eager attention; DeltaAI GH200, `transformers==4.51.3`.
- DCT: `run_dct_data.py --dataset <ds> --source-layer <11|13> --target-layer <20|22>
  --num-factors 512 --num-iters 30 --num-samples 64 --balanced`
- Compare: `compare_directions.py --dataset <ds>` (reads source layer from `dct_meta_<ds>.json`;
  computes supervised directions from `activations/acts_<ds>.npz` at that layer).
- Per-vector cosines saved to `compare_<ds>.csv`. Each DCT run ≈ 9 min / ≈0.15 GPU-hr.
