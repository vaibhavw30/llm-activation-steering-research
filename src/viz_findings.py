"""
viz_findings.py — summary figures that tell the funnel story (local, run in geometry .venv).

Produces four publication-style charts from the activations + DCT vectors you already have:

  plot_findings_alignment.png       — best DCT vector vs truth direction, against a random baseline
                                       (the null: no DCT vector aligns with truth)
  plot_findings_subspace.png        — truth-in-span of the top-k DCT vectors vs chance
                                       (truth isn't even a COMBINATION of top DCT directions)
  plot_findings_transfer.png        — supervised vs DCT cross-dataset transfer heatmaps
                                       (supervised generalizes; DCT doesn't)
  plot_findings_decode_vs_causal.png— the thesis in one picture: truth is readable (high probe
                                       accuracy) but DCT's directions carry ~no truth signal

Usage:  .venv/bin/python viz_findings.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import funnel_utils as fu

DATASETS = ["cities", "common_claim_true_false"]
LABELS = {"cities": "cities (clean)", "common_claim_true_false": "common_claim (messy)"}
SEED = 42
BLUE, GRAY, GREEN, RED = "#4477aa", "#999999", "#228833", "#cc3311"


def load_all():
    """Compute everything the figures need, once."""
    rng = np.random.default_rng(SEED)
    D = {}
    for ds in DATASETS:
        layer = fu.resolve_layer(ds)
        X, y = fu.load_acts(ds, layer)
        d = X.shape[1]
        V, U, _ = fu.load_dct(ds)
        V = fu.unit(V, axis=0)
        nf = V.shape[1]
        vm, vg = fu.mean_diff_dir(X, y), fu.grad_dir(X, y)
        R = fu.unit(rng.standard_normal((d, nf)), axis=0)   # random directions, same count

        best = {"mean_diff": float(np.abs(V.T @ vm).max()),
                "grad": float(np.abs(V.T @ vg).max())}
        rand = {"mean_diff": float(np.abs(R.T @ vm).max()),
                "grad": float(np.abs(R.T @ vg).max())}

        # subspace in-span for top-k
        spans, chance = {}, {}
        for k in (10, 20, 50):
            idx = fu.top_k_by_potency(V, U, k)
            Q, _ = np.linalg.qr(V[:, idx])
            spans[k] = float(((Q.T @ vm) ** 2).sum())
            chance[k] = k / d

        # decodability vs causal signal
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        full_acc = float(cross_val_score(LogisticRegression(max_iter=2000), Xz, y, cv=5).mean())
        def one_dir(v):
            return float(cross_val_score(LogisticRegression(max_iter=2000),
                                         (X @ v).reshape(-1, 1), y, cv=5).mean())
        idx10 = fu.top_k_by_potency(V, U, 10)
        best_dct_acc = max(one_dir(V[:, i]) for i in idx10)
        rand_acc = float(np.mean([one_dir(fu.unit(rng.standard_normal(d))) for _ in range(15)]))

        D[ds] = dict(X=X, y=y, layer=layer, V=V, U=U, vm=vm, vg=vg, d=d, nf=nf,
                     best=best, rand=rand, spans=spans, chance=chance,
                     full_acc=full_acc, best_dct_acc=best_dct_acc, rand_acc=rand_acc)
        print(f"{ds}: layer {layer}, best|cos|(mean-diff)={best['mean_diff']:.3f} "
              f"(rand {rand['mean_diff']:.3f}), probe={full_acc:.3f}, "
              f"best DCT-dir acc={best_dct_acc:.3f} (rand {rand_acc:.3f})")
    return D


def fig_alignment(D):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    groups, xt = [], []
    x = 0
    for ds in DATASETS:
        for dirn, name in [("mean_diff", "mean-diff"), ("grad", "gradient")]:
            ax.bar(x, D[ds]["best"][dirn], width=0.6, color=BLUE,
                   label="best DCT vector" if x == 0 else None)
            ax.bar(x, 0, width=0.6)  # spacer noop
            ax.hlines(D[ds]["rand"][dirn], x - 0.35, x + 0.35, color=RED, lw=2,
                      label="random baseline" if x == 0 else None)
            xt.append((x, f"{LABELS[ds].split()[0]}\n{name}"))
            x += 1
        x += 0.5
    ax.set_xticks([t[0] for t in xt]); ax.set_xticklabels([t[1] for t in xt], fontsize=8)
    ax.set_ylabel("|cosine| with truth direction")
    ax.set_title("No DCT vector aligns with the truth direction\n(best of 512 ≈ random baseline)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_findings_alignment.png", dpi=150); plt.close(fig)
    print("saved plot_findings_alignment.png")


def fig_subspace(D):
    ks = [10, 20, 50]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.35
    xpos = np.arange(len(ks))
    for i, ds in enumerate(DATASETS):
        vals = [D[ds]["spans"][k] for k in ks]
        ax.bar(xpos + (i - 0.5) * w, vals, w, color=[BLUE, GREEN][i], label=LABELS[ds])
    ch = [D[DATASETS[0]]["chance"][k] for k in ks]
    for j, c in enumerate(ch):
        ax.hlines(c, xpos[j] - 0.5, xpos[j] + 0.5, color=RED, lw=2,
                  label="chance (k/d)" if j == 0 else None)
    ax.set_xticks(xpos); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("truth variance inside top-k DCT span")
    ax.set_title("Truth is not a combination of the top DCT directions\n(in-span ≈ chance)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_findings_subspace.png", dpi=150); plt.close(fig)
    print("saved plot_findings_subspace.png")


def transfer_matrices(D):
    names = DATASETS
    sup = np.zeros((2, 2)); dct = np.zeros((2, 2))
    for i, a in enumerate(names):
        # supervised probe trained on A
        sc = StandardScaler().fit(D[a]["X"]); clf = LogisticRegression(max_iter=2000).fit(
            sc.transform(D[a]["X"]), D[a]["y"])
        # DCT top-20 basis from A
        idx = fu.top_k_by_potency(D[a]["V"], D[a]["U"], 20); basis = D[a]["V"][:, idx]
        fa = D[a]["X"] @ basis
        scd = StandardScaler().fit(fa); clfd = LogisticRegression(max_iter=2000).fit(
            scd.transform(fa), D[a]["y"])
        for j, b in enumerate(names):
            sup[i, j] = clf.score(sc.transform(D[b]["X"]), D[b]["y"])
            dct[i, j] = clfd.score(scd.transform(D[b]["X"] @ basis), D[b]["y"])
    return names, sup, dct


def fig_transfer(D):
    names, sup, dct = transfer_matrices(D)
    short = [LABELS[n].split()[0] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, M, title in [(axes[0], sup, "Supervised probe"), (axes[1], dct, "DCT directions")]:
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels(short); ax.set_yticks([0, 1]); ax.set_yticklabels(short)
        ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
        ax.set_title(f"{title}\ntransfer accuracy")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{M[r,c]:.2f}", ha="center", va="center",
                        color="k", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Supervised truth directions generalize across datasets; DCT directions don't "
                 "(off-diagonal ≈ 0.5 chance)", fontsize=10)
    fig.tight_layout(); fig.savefig("plot_findings_transfer.png", dpi=150); plt.close(fig)
    print("saved plot_findings_transfer.png")


def fig_decode_vs_causal(D):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xpos = np.arange(len(DATASETS)); w = 0.25
    probe = [D[ds]["full_acc"] for ds in DATASETS]
    bestd = [D[ds]["best_dct_acc"] for ds in DATASETS]
    randd = [D[ds]["rand_acc"] for ds in DATASETS]
    ax.bar(xpos - w, probe, w, color=GREEN, label="supervised probe (decodability)")
    ax.bar(xpos, bestd, w, color=BLUE, label="best top-10 DCT direction")
    ax.bar(xpos + w, randd, w, color=GRAY, label="random direction")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xticks(xpos); ax.set_xticklabels([LABELS[d] for d in DATASETS])
    ax.set_ylabel("truth-classification accuracy"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Decodable ≠ causally dominant\ntruth is highly readable, but DCT's causal "
                 "directions carry ~no truth signal")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_findings_decode_vs_causal.png", dpi=150); plt.close(fig)
    print("saved plot_findings_decode_vs_causal.png")


if __name__ == "__main__":
    D = load_all()
    fig_alignment(D)
    fig_subspace(D)
    fig_transfer(D)
    fig_decode_vs_causal(D)
    print("\nDone — 4 findings figures written (plot_findings_*.png).")
