"""
viz_subspace_xgb.py — chart the non-linear-gap result from subspace_xgb.py.

Reads subspace_xgb_<ds>.csv (feature_set, k, linear, xgb, gap) and plots the NON-LINEAR GAP
(xgb - linear accuracy) per feature set: the full residual stream vs. the top-k DCT subspace vs.
random-k projections. The point of the figure: the non-linear truth gap that exists on the full
activations COLLAPSES to ~0 once you restrict to DCT's causal directions (no better than random)
— i.e. DCT's causal subspace does not carry the non-linear truth structure XGBoost exploits.

    .venv/bin/python viz_subspace_xgb.py
Output: plot_findings_subspace_xgb.png
"""

import csv
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"full-acts": "#228833", "DCT-top-k": "#4477aa", "random-k": "#ccbb44"}


def load(ds):
    rows = list(csv.DictReader(open(f"subspace_xgb_{ds}.csv")))
    full = next(r for r in rows if r["feature_set"] == "full-acts")
    ks = sorted({int(r["k"]) for r in rows if r["feature_set"] != "full-acts"})
    dct = {int(r["k"]): float(r["gap"]) for r in rows if r["feature_set"] == "DCT-top-k"}
    rnd = {int(r["k"]): float(r["gap"]) for r in rows if r["feature_set"] == "random-k"}
    return float(full["gap"]), ks, dct, rnd


def main():
    datasets = [os.path.basename(p).replace("subspace_xgb_", "").replace(".csv", "")
                for p in sorted(glob.glob("subspace_xgb_*.csv"))]
    if not datasets:
        print("No subspace_xgb_*.csv found — run subspace_xgb.py first.")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        full_gap, ks, dct, rnd = load(ds)
        # x layout: full-acts bar, then a (DCT, random) pair per k
        labels = ["full\nacts"] + [f"DCT\ntop-{k}" for k in ks] + [f"rand\n{k}" for k in ks]
        # interleave DCT/random per k for readability
        labels = ["full\nacts"]
        vals = [full_gap]
        colors = [COLORS["full-acts"]]
        for k in ks:
            labels += [f"DCT\ntop-{k}", f"rand\n{k}"]
            vals += [dct[k], rnd[k]]
            colors += [COLORS["DCT-top-k"], COLORS["random-k"]]
        xpos = np.arange(len(vals))
        ax.bar(xpos, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("non-linear gap  (XGBoost − linear accuracy)")
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)
        for x, v in zip(xpos, vals):
            ax.annotate(f"{v:+.3f}", (x, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7,
                        xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")

    fig.suptitle("Does DCT's causal subspace carry the NON-LINEAR truth signal?\n"
                 "Full-activation gap is real; inside the top-k DCT directions it collapses to "
                 "~0 (no better than random) → DCT misses truth non-linearly too.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("plot_findings_subspace_xgb.png", dpi=150)
    print("saved plot_findings_subspace_xgb.png")


if __name__ == "__main__":
    main()
