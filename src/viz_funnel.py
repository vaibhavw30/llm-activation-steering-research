"""
viz_funnel.py — visualizations for the DCT-interpretation funnel (local).

Produces two plots per dataset (the others — autointerp table, steering-vs-magnitude — come
from the cluster generation steps):

  1. plot_funnel_cosine_<ds>.png   — heatmap: top-K DCT vectors x {mean-diff, gradient}, |cos|.
     One glance shows whether any top DCT vector aligns with a supervised truth direction.
  2. plot_funnel_dctclass_<ds>.png — bars: how well projecting onto each top DCT vector alone
     classifies true/false, vs the supervised single-direction reference and a random-dir band.

Run in the geometry .venv:  .venv/bin/python viz_funnel.py --dataset cities
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import funnel_utils as fu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def one_dir_acc(X, v, y):
    """5-fold CV accuracy of a 1-feature logistic probe on the projection X@v."""
    feats = (X @ v).reshape(-1, 1)
    return float(cross_val_score(LogisticRegression(max_iter=2000), feats, y, cv=5).mean())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds, k = args.dataset, args.top_k
    layer = fu.resolve_layer(ds, args.layer)
    X, y = fu.load_acts(ds, layer)
    d = X.shape[1]
    V, U, _ = fu.load_dct(ds)
    V = fu.unit(V, axis=0)
    idx = fu.top_k_by_potency(V, U, k)
    v_mean, v_grad = fu.mean_diff_dir(X, y), fu.grad_dir(X, y)
    print(f"{ds}: layer {layer}, top-{k} DCT vectors {idx}")

    # ---- Plot 1: cosine heatmap ----
    M = np.array([[abs(V[:, i] @ v_mean), abs(V[:, i] @ v_grad)] for i in idx])  # (k, 2)
    fig, ax = plt.subplots(figsize=(4, max(4, 0.4 * k)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.2, M.max()))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["mean-diff", "gradient"])
    ax.set_yticks(range(k)); ax.set_yticklabels([f"#{i}" for i in idx])
    ax.set_ylabel("top DCT vectors (by ‖U‖)")
    ax.set_title(f"|cos(DCT vector, truth direction)|\n{ds} @ layer {layer}")
    for r in range(k):
        for c in range(2):
            ax.text(c, r, f"{M[r,c]:.02f}", ha="center", va="center",
                    color="w" if M[r, c] < 0.15 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="|cosine|")
    fig.tight_layout(); fig.savefig(f"plot_funnel_cosine_{ds}.png", dpi=150)
    print(f"  saved plot_funnel_cosine_{ds}.png")

    # ---- Plot 2: per-vector single-direction truth-classification ----
    dct_accs = [one_dir_acc(X, V[:, i], y) for i in idx]
    mean_acc = one_dir_acc(X, v_mean, y)
    grad_acc = one_dir_acc(X, v_grad, y)
    rand_accs = [one_dir_acc(X, fu.unit(rng.standard_normal(d)), y) for _ in range(20)]
    rand_mu, rand_sd = float(np.mean(rand_accs)), float(np.std(rand_accs))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(k), dct_accs, color="#4477aa", label="top DCT vectors")
    ax.axhline(mean_acc, color="green", ls="-", lw=2, label=f"mean-diff dir ({mean_acc:.2f})")
    ax.axhline(grad_acc, color="darkgreen", ls="--", lw=1.5, label=f"gradient dir ({grad_acc:.2f})")
    ax.axhspan(rand_mu - rand_sd, rand_mu + rand_sd, color="gray", alpha=0.3,
               label=f"random dir ±σ ({rand_mu:.2f})")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"#{i}" for i in idx], rotation=45, ha="right")
    ax.set_ylabel("1-direction truth-classification accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(f"Does any single top DCT direction carry truth signal?\n{ds} @ layer {layer}")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"plot_funnel_dctclass_{ds}.png", dpi=150)
    print(f"  saved plot_funnel_dctclass_{ds}.png")
    print(f"  (DCT best {max(dct_accs):.3f} vs supervised mean-diff {mean_acc:.3f}; "
          f"random {rand_mu:.3f}±{rand_sd:.3f})")


if __name__ == "__main__":
    main()
