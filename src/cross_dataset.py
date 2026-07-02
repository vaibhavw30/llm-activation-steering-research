"""
cross_dataset.py — Funnel Step 4: which directions generalize, supervised or unsupervised?

Two transfer tests across datasets (at each dataset's truth-peak layer):

  (A) Supervised transfer matrix: train a linear truth probe on dataset A, test its
      true/false accuracy on dataset B, for all pairs. Supervised directions are expected to
      be somewhat dataset-specific.
  (B) DCT-as-features transfer: use dataset A's top-k DCT directions as a fixed feature basis,
      train a probe on A's projections, test on B's projections. DCT directions are
      unsupervised / model-intrinsic, so they may transfer differently.

Caveat: datasets are compared at their own best layers, and activations are matched by layer
index (same residual stream). Local; run in the geometry .venv.
    .venv/bin/python cross_dataset.py
"""

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import funnel_utils as fu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cities,common_claim_true_false",
                   help="comma-separated; needs acts_<ds>.npz (and dct_V_<ds>.pt for part B)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return sc, clf


def acc(sc, clf, X, y):
    return float(clf.score(sc.transform(X), y))


def main():
    args = parse_args()
    names = args.datasets.split(",")
    # Load each dataset's activations at its own truth-peak layer.
    data = {}
    for ds in names:
        layer = fu.resolve_layer(ds)
        X, y = fu.load_acts(ds, layer)
        data[ds] = {"X": X, "y": y, "layer": layer}
        print(f"{ds}: layer {layer}, acts {X.shape}")

    # ---- (A) Supervised transfer matrix (rows=train, cols=test) ----
    print("\n=== (A) Supervised probe transfer accuracy (train row -> test col) ===")
    hdr = "train\\test   " + " ".join(f"{n[:12]:>13}" for n in names)
    print(hdr)
    for a in names:
        sc, clf = fit_probe(data[a]["X"], data[a]["y"])
        row = f"{a[:12]:<12}"
        for b in names:
            row += f" {acc(sc, clf, data[b]['X'], data[b]['y']):>13.3f}"
        print(row)
    print("(diagonal = in-distribution; off-diagonal = cross-dataset transfer)")

    # ---- (B) DCT-as-features transfer ----
    print(f"\n=== (B) DCT-features transfer (top-{args.top_k} dirs from train row) ===")
    print(hdr)
    for a in names:
        try:
            Va, Ua, _ = fu.load_dct(a)
        except FileNotFoundError:
            print(f"{a[:12]:<12}  (no dct_V_{a}.pt — skipped)")
            continue
        Va = fu.unit(Va, axis=0)
        idx = fu.top_k_by_potency(Va, Ua, args.top_k)
        basis = Va[:, idx]                              # (d, k) — A's top causal directions
        # train probe on A's projections onto its own DCT basis
        fa = data[a]["X"] @ basis
        sc, clf = fit_probe(fa, data[a]["y"])
        row = f"{a[:12]:<12}"
        for b in names:
            fb = data[b]["X"] @ basis                   # project B onto A's basis
            row += f" {acc(sc, clf, fb, data[b]['y']):>13.3f}"
        print(row)
    print("(uses dataset A's unsupervised DCT directions as a fixed basis for all datasets)")

    print("\n=== Interpretation ===")
    print("- If DCT-feature transfer (B off-diagonal) holds up better than supervised (A off-")
    print("  diagonal), DCT found model-intrinsic directions. If worse, supervised directions")
    print("  are more robust. Compare each off-diagonal cell between the two matrices.")


if __name__ == "__main__":
    main()
