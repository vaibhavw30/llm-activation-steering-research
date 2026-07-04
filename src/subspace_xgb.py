"""
subspace_xgb.py — Funnel Step 3, non-linear upgrade: does DCT's top causal subspace carry the
*non-linear* truth structure that XGBoost exploits (but a linear probe can't read)?

Test 3B compared a LINEAR probe on top-k DCT projections vs random-k vs full activations, and
found DCT-features ~= random (truth is not in DCT's top causal directions, linearly). But XGBoost
opens a non-linear gap over the linear probe on the full activations (cities +0.003, common_claim
+0.082). This script asks the natural follow-up:

  For each feature set — top-k DCT projections, random-k projections, full activations — measure
  BOTH a linear probe and XGBoost, and report the non-linear gap (xgb - linear). The questions:

    Q1. Does XGBoost on DCT-features beat XGBoost on random-features?
        -> is there non-linearly-decodable truth in DCT's causal subspace that no single linear
           direction revealed?
    Q2. Is the DCT-subspace non-linear gap close to the full-activation gap?
        -> does DCT's causal subspace preserve the non-linear truth structure, or discard it?

  Interpretation:
    - DCT-xgb >> random-xgb, gap approaching full  -> DCT captures the non-linear truth signal
      (the linear null in 3B was hiding a non-linear hit). Strong new positive result.
    - DCT-xgb ~= random-xgb                         -> DCT misses truth non-linearly too; the
      decodable != causal null gets STRONGER, now on the non-linear frontier.

Local; mirrors subspace_top_k.py. Same acts + DCT artifacts, same 5-fold CV protocol.
    .venv/bin/python subspace_xgb.py --dataset cities
"""

import argparse
import csv
import numpy as np
import xgboost as xgb
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


def linear_acc(feats, y, seed):
    """5-fold CV accuracy of a standardized logistic probe."""
    fz = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    return float(cross_val_score(LogisticRegression(max_iter=2000), fz, y, cv=5).mean())


def xgb_acc(feats, y, seed):
    """5-fold CV accuracy of XGBoost (same config as analyze.py), no scaling."""
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed)
    return float(cross_val_score(clf, feats, y, cv=5).mean())


def row(name, feats, y, seed):
    lin = linear_acc(feats, y, seed)
    xg = xgb_acc(feats, y, seed)
    return name, lin, xg, xg - lin


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

    out_rows = [("feature_set", "k", "linear", "xgb", "gap")]

    # Full-activation reference: linear vs XGBoost on the raw residual stream (the ceiling, and
    # the non-linear gap we're trying to explain).
    name, lin, xg, gap = row("full-acts", X, y, args.seed)
    print(f"\nFull-activation reference: linear={lin:.3f}  xgb={xg:.3f}  gap={gap:+.3f}")
    full_gap = gap
    out_rows.append(("full-acts", d, f"{lin:.4f}", f"{xg:.4f}", f"{gap:.4f}"))

    print("\n=== Linear vs XGBoost on top-k DCT projections (vs random-k) ===")
    print(f"{'k':>4}  {'feature set':>12}  {'linear':>7} {'xgb':>7} {'gap':>7}")
    for k in ks:
        idx = fu.top_k_by_potency(V, U, k)
        Vk = V[:, idx]
        R = fu.unit(rng.standard_normal((d, k)), axis=0)
        for name, feats in [("DCT-top-k", X @ Vk), ("random-k", X @ R)]:
            _, lin, xg, gap = row(name, feats, y, args.seed)
            print(f"{k:>4}  {name:>12}  {lin:>7.3f} {xg:>7.3f} {gap:>+7.3f}")
            out_rows.append((name, k, f"{lin:.4f}", f"{xg:.4f}", f"{gap:.4f}"))
        print()

    out_path = f"subspace_xgb_{ds}.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print(f"saved {out_path}")

    print("=== Interpretation ===")
    print(f"- full-activation non-linear gap (xgb - linear) = {full_gap:+.3f} (the target signal).")
    print("- Q1: DCT-top-k xgb >> random-k xgb  -> DCT's causal subspace carries non-linear truth")
    print("      the linear 3B test missed (new positive result).")
    print("- Q2: DCT-top-k gap approaching full gap -> DCT preserves the non-linear structure;")
    print("      DCT-top-k xgb ~= random-k xgb      -> null gets stronger (decodable-not-causal,")
    print("      non-linearly too).")


if __name__ == "__main__":
    main()
