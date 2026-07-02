"""
analyze.py — Steps 2-4: probe vs XGBoost, direction comparison, plots.
Pure CPU tabular ML. Run after extract.py.

Usage:
    python analyze.py cities
    python analyze.py common_claim_true_false

Reads:  acts_<dataset>.npz
Writes: results_<dataset>.csv, directions_<dataset>.csv,
        plot_<dataset>_accuracy.png, plot_<dataset>_gap.png,
        plot_<dataset>_directions.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

SEED = 42


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def main(dataset_name):
    data = np.load(f"acts_{dataset_name}.npz", allow_pickle=True)
    acts = data["activations"]   # (L+1, n, d)
    labels = data["labels"]
    n_layers = acts.shape[0]
    print(f"{dataset_name}: {acts.shape[1]} examples, {n_layers} layers, "
          f"d={acts.shape[2]}")

    acc_rows, dir_rows = [], []

    for layer in range(n_layers):
        X = acts[layer]
        y = labels
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        # --- linear probe ---
        lin = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr_s, y_tr)
        lin_acc = lin.score(X_te_s, y_te)

        # --- xgboost (non-linear) ---
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=SEED)
        clf.fit(X_tr, y_tr)
        xgb_acc = clf.score(X_te, y_te)

        acc_rows.append({"layer": layer, "linear_acc": lin_acc,
                         "xgb_acc": xgb_acc, "gap": xgb_acc - lin_acc})

        # --- direction comparison (on full data) ---
        v_mean = unit(X[y == 1].mean(0) - X[y == 0].mean(0))
        scaler_full = StandardScaler().fit(X)
        lin_full = LogisticRegression(max_iter=2000).fit(
            scaler_full.transform(X), y)
        v_grad = unit(lin_full.coef_[0] / scaler_full.scale_)
        cos = float(np.dot(v_mean, v_grad))
        dir_rows.append({"layer": layer, "cos_mean_vs_grad": cos})

        print(f"  L{layer:2d}  lin={lin_acc:.3f}  xgb={xgb_acc:.3f}  "
              f"gap={xgb_acc-lin_acc:+.3f}  cos={cos:+.3f}")

    acc_df = pd.DataFrame(acc_rows)
    dir_df = pd.DataFrame(dir_rows)
    acc_df.to_csv(f"results_{dataset_name}.csv", index=False)
    dir_df.to_csv(f"directions_{dataset_name}.csv", index=False)

    # summary
    best = acc_df.loc[acc_df["xgb_acc"].idxmax()]
    print(f"\nBest layer: {int(best.layer)} "
          f"(linear {best.linear_acc:.3f}, xgb {best.xgb_acc:.3f}, "
          f"gap {best.gap:+.3f})")
    print(f"Max non-linear gap across layers: {acc_df['gap'].max():+.3f}")

    # --- plots ---
    plt.figure(figsize=(8, 5))
    plt.plot(acc_df.layer, acc_df.linear_acc, "o-", label="Linear probe")
    plt.plot(acc_df.layer, acc_df.xgb_acc, "s-", label="XGBoost")
    plt.xlabel("Layer"); plt.ylabel("Test accuracy")
    plt.title(f"Truth decodability: linear vs non-linear ({dataset_name})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"plot_{dataset_name}_accuracy.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.bar(acc_df.layer, acc_df.gap)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Layer"); plt.ylabel("XGBoost − Linear accuracy")
    plt.title(f"Non-linear headroom by layer ({dataset_name})")
    plt.grid(alpha=0.3)
    plt.savefig(f"plot_{dataset_name}_gap.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(dir_df.layer, dir_df.cos_mean_vs_grad, "o-")
    plt.axhline(1.0, color="g", ls="--", alpha=0.5, label="identical")
    plt.xlabel("Layer"); plt.ylabel("cos(mean-diff, classifier-gradient)")
    plt.title(f"Direction agreement ({dataset_name})")
    plt.ylim(-1, 1.05); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"plot_{dataset_name}_directions.png", dpi=150, bbox_inches="tight")

    print(f"Saved results + 3 plots for {dataset_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <dataset_name>  (no .csv)")
        print("e.g.:  python analyze.py cities")
        sys.exit(1)
    main(sys.argv[1])