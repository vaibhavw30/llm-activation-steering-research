"""Arm A1 figures: label-split margin-vs-tau curves and verdict accuracy-vs-tau.

    python -m mag.viz_verdict --dataset cities"""
import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_COLOR = {1: "#228833", 0: "#cc3311"}     # green = gold-true, red = gold-false


def load_rows(ds):
    out = []
    for r in csv.DictReader(open(f"mag_verdict_logits_{ds}.csv")):
        out.append({"direction": r["direction"], "tau": float(r["tau"]),
                    "label": int(r["label"]), "margin": float(r["margin"])})
    return out


def margin_stats(rows):
    """(direction, tau, label) -> (mean margin, sem)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["tau"], r["label"]), []).append(r["margin"])
    return {k: (float(np.mean(v)), float(np.std(v) / max(1, len(v)) ** 0.5))
            for k, v in groups.items()}


def accuracy(rows):
    """(direction, tau) -> fraction where (margin > 0) == gold label."""
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["tau"]), []).append(
            int((r["margin"] > 0) == bool(r["label"])))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def _directions_and_taus(rows):
    dirs = sorted({r["direction"] for r in rows})
    taus = sorted({r["tau"] for r in rows})
    return dirs, taus


def plot_margin(ds):
    rows = load_rows(ds)
    dirs, taus = _directions_and_taus(rows)
    ms = margin_stats(rows)
    ncols = 3
    nrows_ = int(np.ceil(len(dirs) / ncols))
    fig, axes = plt.subplots(nrows_, ncols, figsize=(4.6 * ncols, 3.6 * nrows_),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, d in zip(flat, dirs):
        for lab in (1, 0):
            ys = [ms.get((d, t, lab), (np.nan, 0.0)) for t in taus]
            mean = [y[0] for y in ys]; sem = [y[1] for y in ys]
            ax.errorbar(taus, mean, yerr=sem, fmt="o-", color=LABEL_COLOR[lab],
                        label=f"gold {'true' if lab else 'false'}", lw=2, ms=4)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_title(d, fontsize=10); ax.grid(alpha=0.25)
    for ax in flat[len(dirs):]:
        ax.axis("off")
    flat[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tau  (-=push FALSE, +=push TRUE)")
    for r in range(nrows_):
        axes[r][0].set_ylabel("mean p_yes - p_no")
    fig.suptitle(f"{ds}: verdict margin vs tau, split by gold label\n"
                 f"(differential movement = latent truth knowledge; parallel = content-blind)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"plot_mag_verdict_margin_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_verdict_margin_{ds}.png")


def plot_accuracy(ds):
    rows = load_rows(ds)
    dirs, taus = _directions_and_taus(rows)
    acc = accuracy(rows)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for d in dirs:
        ax.plot(taus, [acc.get((d, t), np.nan) for t in taus], "o-", label=d, lw=2, ms=4)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel("tau"); ax.set_ylabel("verdict accuracy vs gold label")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{ds}: does steering make the model's verdict more accurate?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(f"plot_mag_verdict_acc_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_verdict_acc_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_margin(a.dataset)
    plot_accuracy(a.dataset)


if __name__ == "__main__":
    main()
