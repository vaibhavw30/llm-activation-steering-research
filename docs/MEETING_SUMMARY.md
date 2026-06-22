# Geometry of Truth: Linear vs Non-Linear Probes in gemma-2-2b

## 1. Headline finding

On **clean, curated truth statements (cities)**, a linear probe and a non-linear probe (XGBoost) are essentially equivalent — best-layer accuracy 0.990 vs 0.993, a gap of just **+0.003** — consistent with Marks & Tegmark's linear-truth result. On the **messiest dataset (common_claim)**, XGBoost opens a **+0.082** gap over the linear probe (0.706 → 0.788), suggesting non-linear structure where concepts are entangled. The gap grows monotonically with dataset "messiness," and messier datasets also show lower overall decodability.

## 2. Results table

| Dataset | Character | Best layer | Linear acc | XGBoost acc | Best-layer gap | Verdict |
|---|---|---:|---:|---:|---:|---|
| cities | clean / curated | 11 | 0.990 | 0.993 | +0.003 | **Linear** |
| sp_en_trans | clean / translation | 7 | 0.972 | 1.000 | +0.028 | Borderline |
| companies_true_false | messier | 14 | 0.917 | 0.954 | +0.038 | Mild non-linear headroom |
| common_claim_true_false | messiest | 13 | 0.706 | 0.788 | +0.082 | **Non-linear headroom** |

*Pattern: gap grows with messiness; clean curated truth is linearly decodable, messy claims show both non-linear headroom and lower overall decodability.*

## 3. Direction finding

We compared two estimates of the "truth direction" at each layer: the **mean difference** (true − false activations) and the **classifier gradient**. Their cosine similarity tracks dataset cleanliness — the same monotonic pattern as the accuracy gap, in reverse.

| Dataset | Cosine @ best layer | Peak cosine (layer) | Mean cosine |
|---|---:|---:|---:|
| sp_en_trans | 0.563 | 0.725 (L9) | 0.540 |
| cities | 0.413 | 0.495 (L24) | 0.324 |
| companies_true_false | 0.204 | 0.338 (L24) | 0.214 |
| common_claim_true_false | 0.085 | 0.101 (L0) | 0.073 |

- The two directions agree most on **clean** concepts (sp_en_trans, cities), and agreement **strengthens with network depth** there.
- On the **messiest** data (common_claim) agreement is **near-zero and flat** at every layer.
- **Caveat (carries into §4):** even peak cosines top out at ~0.5–0.7 — these are *correlated but genuinely different* vectors, not one shared direction.

*(Full per-layer analysis in DIRECTIONS_FINDINGS.md.)*

## 4. Maturity caveat

- Cosine agreement between directions is **not** the real test of whether a direction is meaningful.
- Non-identifiability work shows that **different directions can steer behavior equivalently** — high cosine similarity is neither necessary nor sufficient.
- The next step is **behavioral evaluation via A-LQR**, not just probe accuracy or direction geometry.

## 5. Open questions for Julian

- Which concepts should we target next — **toxicity? sycophancy?** — beyond factual truth?
- Can we get **access to the A-LQR code** for behavioral evaluation of these directions?

## 6. Methods footnote

- Model: `google/gemma-2-2b`, CPU, fp32.
- Features: last-token residual-stream activations extracted at **every layer**.
- Eval: 80/20 stratified train/test split, seed 42.
- Linear probe: LogisticRegression (max_iter=2000) on StandardScaler-normalized features.
- Non-linear probe: XGBoost (300 trees, max_depth 4) on raw features.
