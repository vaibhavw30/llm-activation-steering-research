# Truth-Direction Agreement: mean-diff vs gradient (cos_mean_vs_grad)

**Experiment:** For each of 27 layers (0 = embeddings, 1–26 = transformer blocks) of
`google/gemma-2-2b`, we measured the cosine similarity between two candidate "truth
directions": `v_mean` (contrastive mean-difference of true vs false activations) and
`v_grad` (logistic-regression gradient direction). Source files:
`directions_<dataset>.csv`.

## Per-dataset cosine table

| Dataset                   | Character          | Best layer | Cosine @ best layer | Max cosine (layer) | Mean cosine (all layers) | Rises with depth? |
| ------------------------- | ------------------ | ---------- | ------------------- | ------------------ | ------------------------ | ----------------- |
| cities                    | clean, curated     | 11         | 0.413               | 0.495 (L24)        | 0.324                    | Yes               |
| sp_en_trans               | clean translations | 7          | 0.563               | 0.725 (L9)         | 0.540                    | Yes               |
| companies_true_false      | messier            | 14         | 0.204               | 0.338 (L24)        | 0.214                    | Yes (weak)        |
| common_claim_true_false   | messiest           | 13         | 0.085               | 0.101 (L0)         | 0.073                    | Flat / negligible |

"Rises with depth" compares the mean cosine of layers 1–8 against layers 19–26:

| Dataset                   | mean cos L1–8 | mean cos L19–26 |
| ------------------------- | ------------- | --------------- |
| cities                    | 0.191         | 0.430           |
| sp_en_trans               | 0.423         | 0.637           |
| companies_true_false      | 0.163         | 0.284           |
| common_claim_true_false   | 0.053         | 0.077           |

## Clean vs messy comparison

The two direction estimates agree substantially more on clean data than on messy data.
The clean datasets reach mean cosines of 0.32 (cities) and 0.54 (sp_en_trans) and peaks
of 0.50–0.73, whereas the messy datasets sit at mean cosines of 0.21 (companies) and just
0.07 (common_claim), with common_claim never exceeding ~0.10 at any layer. Agreement also
strengthens with depth on the clean datasets (and weakly on companies), but stays flat and
near-zero throughout for common_claim. Note, however, that even on the cleanest data the
cosine is far from 1.0 — `v_mean` and `v_grad` are correlated but clearly distinct vectors,
not the same direction.

## Narrative

Across all four datasets the mean-difference direction and the logistic-regression gradient
direction are positively but only partially aligned, and the degree of alignment tracks
dataset cleanliness almost monotonically: sp_en_trans (mean cos 0.54, peak 0.73 at L9) and
cities (mean 0.32, peak 0.50 at L24) agree far more than companies (mean 0.21) and
common_claim (mean 0.07, essentially flat and never above ~0.10). Agreement also tends to
rise with network depth on the clean concepts, consistent with truth becoming more linearly
organized in mid-to-late residual streams, while it stays near zero at every layer for the
messiest claims. This is suggestive evidence that "truth" is more coherently and linearly
encoded for clean, narrow-domain statements than for heterogeneous world claims. Crucially,
though, high cosine agreement between two linear estimators is necessary-but-not-sufficient
evidence for a single causal truth direction: the non-identifiability literature shows that
different directions can steer model behavior equivalently, so two methods agreeing (or
disagreeing) on a vector does not establish that that vector is the one the model actually
uses. Even our best cosines top out around 0.5–0.7, meaning the two estimates are related
but genuinely different vectors — and the real test is behavioral, not geometric. The
appropriate next step is causal/behavioral evaluation (A-LQR) to check whether these
candidate directions, however well they correlate, actually control the model's truth-related
outputs.
