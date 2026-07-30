"""viz_sae.py — Horizon-1 1.2 figures.

  plot_sae_explained_<ds>.png  cumulative explained variance vs OMP rank, one line per
                               decomposed direction (how sparse is each in SAE feature
                               space?)
  plot_sae_overlap_<ds>.png    pairwise Jaccard of the top-k feature supports, with the
                               full-context vs stem-context pair highlighted — the D2
                               mechanism figure.

    PYTHONPATH=src .venv/bin/python src/viz_sae.py --dataset cities
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from sae_decompose import D2_PAIR         # noqa: E402

HIGHLIGHT, BASE = "#cc3311", "#4477aa"


def _series(ds):
    by = {}
    with open(f"sae_features_{ds}.csv", newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["vector"], []).append(
                (int(r["rank"]), float(r["cumulative_explained"])))
    return {k: [c for _, c in sorted(v)] for k, v in by.items()}


def fig_explained(ds):
    path = f"sae_features_{ds}.csv"
    if not os.path.exists(path):
        print(f"skip fig_explained: {path} missing")
        return
    series = _series(ds)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, cum in sorted(series.items()):
        ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3, lw=1.4, label=name)
    ax.set_xlabel("OMP rank (number of SAE features)")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{ds}: sparsity of reach directions in GemmaScope features",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"plot_sae_explained_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_sae_explained_{ds}.png")


def fig_overlap(ds):
    path = f"sae_overlap_{ds}.json"
    if not os.path.exists(path):
        print(f"skip fig_overlap: {path} missing")
        return
    ov = json.load(open(path))
    pairs = sorted(ov["jaccard"].items(), key=lambda kv: -kv[1])
    labels = [k.replace("|", "\nvs ") for k, _ in pairs]
    vals = [v for _, v in pairs]
    colors = [HIGHLIGHT if k == D2_PAIR else BASE for k, _ in pairs]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(pairs)), 4.4))
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel(f"Jaccard of top-{ov['k']} feature supports")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"{ds}: shared SAE features between directions "
                 f"(red = full vs stem context)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"plot_sae_overlap_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_sae_overlap_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    fig_explained(ds)
    fig_overlap(ds)


if __name__ == "__main__":
    main()
