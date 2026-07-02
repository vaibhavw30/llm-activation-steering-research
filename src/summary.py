"""
summary.py — cross-dataset summary (the money plot).
Run after analyze.py has produced results_<dataset>.csv files.

Usage:  python summary.py
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for path in sorted(glob.glob("results_*.csv")):
    ds = os.path.basename(path).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(path)
    best = df.loc[df["xgb_acc"].idxmax()]
    rows.append({
        "dataset": ds,
        "best_layer": int(best["layer"]),
        "linear_acc": round(float(best["linear_acc"]), 3),
        "xgb_acc": round(float(best["xgb_acc"]), 3),
        # Headline metric per CLAUDE.md PROMPT 5: gap AT the best layer
        # (xgb_acc - linear_acc where xgb_acc peaks), NOT the max over all layers
        # (which can come from an early layer where the linear probe merely underfits).
        "best_layer_gap": round(float(best["xgb_acc"] - best["linear_acc"]), 3),
        "max_gap_anylayer": round(float(df["gap"].max()), 3),  # secondary, for transparency
    })

if not rows:
    print("No results_*.csv files found. Run analyze.py first.")
    raise SystemExit

summary = pd.DataFrame(rows)
print("\n=== Cross-dataset summary ===")
print(summary.to_string(index=False))
summary.to_csv("summary_all.csv", index=False)

# money plot: best-layer non-linear gap per dataset (ordered clean -> messy)
summary_sorted = summary.sort_values("best_layer_gap")
plt.figure(figsize=(8, 5))
plt.bar(summary_sorted["dataset"], summary_sorted["best_layer_gap"])
plt.axhline(0.02, color="r", ls="--", alpha=0.6, label="0.02 threshold")
plt.ylabel("Non-linear gap at best layer (XGBoost − Linear)")
plt.title("Where is truth non-linearly encoded?")
plt.xticks(rotation=20, ha="right")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_summary_maxgap.png", dpi=150, bbox_inches="tight")
print("Saved plot_summary_maxgap.png")

# auto-generated interpretation (uses best-layer gap = decodability-peak gap)
print("\n=== Interpretation ===")
for r in rows:
    g = r["best_layer_gap"]
    verdict = ("appears LINEARLY encoded (gap < 0.02)"
               if g < 0.02 else
               f"shows NON-LINEAR headroom (gap {g:+.3f})")
    print(f"- {r['dataset']}: {verdict}; "
          f"best linear {r['linear_acc']}, xgb {r['xgb_acc']} at layer {r['best_layer']}.")