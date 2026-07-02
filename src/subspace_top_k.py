"""
subspace_top_k.py — Funnel Step 3: is truth a COMBINATION of a few TOP causal directions?

Two tests, restricted to the top-k DCT vectors (ranked by ||U||), at the truth-peak layer:

  (A) Truth-in-span: fraction of the truth direction's variance inside the span of the top-k
      DCT vectors, vs chance ~ k/d. (The full-512 span saturates trivially; small k is the
      meaningful test.)
  (B) Classify-from-DCT-features: project activations onto the top-k DCT directions, train a
      logistic probe on those k numbers, and see how well it predicts true/false — vs a
      random-k-directions baseline and the full-activation probe. If the top-k causal
      directions classify truth well, truth lives in that causal subspace even if no single
      vector is "the" truth axis.

Local; uses activations/acts_<ds>.npz + dct_V/U_<ds>.pt. Run in the geometry .venv.
    .venv/bin/python subspace_top_k.py --dataset cities
"""

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import funnel_utils as fu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--ks", default="10,20,50", help="comma-separated top-k values to test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def proj_frac(Q, t):
    """Fraction of unit vector t's variance inside the span with orthonormal basis Q (d x r)."""
    return float(((Q.T @ t) ** 2).sum())


def classify_acc(feats, y, seed):
    """5-fold CV accuracy of a logistic probe on the given features."""
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, feats, y, cv=5).mean())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = args.dataset
    layer = fu.resolve_layer(ds, args.layer)
    ks = [int(x) for x in args.ks.split(",")]

    X, y = fu.load_acts(ds, layer)
    d = X.shape[1]
    V, U, meta = fu.load_dct(ds)
    V = fu.unit(V, axis=0)
    print(f"Dataset={ds} | layer={layer} | acts {X.shape} | DCT V {V.shape}")

    v_mean = fu.mean_diff_dir(X, y)
    v_grad = fu.grad_dir(X, y)

    # Baseline: full-activation probe accuracy (standardized) — the decodability ceiling.
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    full_acc = classify_acc(Xz, y, args.seed)
    print(f"\nFull-activation probe accuracy (decodability ceiling): {full_acc:.3f}")

    print("\n=== (A) Truth-in-span of top-k DCT vectors (vs k/d chance) ===")
    print(f"{'k':>4} {'chance':>8} {'mean_diff':>10} {'gradient':>10}")
    for k in ks:
        idx = fu.top_k_by_potency(V, U, k)
        Vk = V[:, idx]                       # (d, k)
        Q, _ = np.linalg.qr(Vk)              # orthonormal basis of the top-k span
        chance = k / d
        print(f"{k:>4} {chance:>8.3f} {proj_frac(Q, v_mean):>10.3f} {proj_frac(Q, v_grad):>10.3f}")

    print("\n=== (B) Classify true/false from top-k DCT projections ===")
    print(f"{'k':>4} {'DCT-feats':>10} {'random-k':>10}  (full={full_acc:.3f})")
    for k in ks:
        idx = fu.top_k_by_potency(V, U, k)
        Vk = V[:, idx]
        dct_feats = X @ Vk                   # (n, k) projections onto top-k DCT directions
        dct_acc = classify_acc(dct_feats, y, args.seed)
        # random-k baseline: k random unit directions
        R = fu.unit(rng.standard_normal((d, k)), axis=0)
        rand_acc = classify_acc(X @ R, y, args.seed)
        print(f"{k:>4} {dct_acc:>10.3f} {rand_acc:>10.3f}")

    print("\n=== Interpretation ===")
    print("- (A) >> k/d  → truth lies in the top-k causal subspace (multi-direction feature).")
    print("- (B) DCT-feats >> random-k and approaching full → top causal directions carry truth.")
    print("- Both near baseline → truth is genuinely outside DCT's top causal directions.")


if __name__ == "__main__":
    main()
