# src/viz_mag.py
"""viz_mag.py — plots for the MAG battery (mirrors viz_steer/viz_funnel). Guards missing CSVs.

    python src/viz_mag.py --dataset cities
"""
import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def plot_readability(ds):
    rows = _read(f"mag_readability_{ds}.csv")
    if not rows:
        return
    gold = [r for r in rows if r["target"] == "gold"]
    ops = [r["operator"] for r in gold]
    acc = [float(r["acc"]) for r in gold]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(ops)), acc, color="#4477aa")
    ax.axhline(0.5, color="k", ls="--", lw=0.7, label="chance")
    ax.set_xticks(range(len(ops))); ax.set_xticklabels(ops, rotation=45, ha="right")
    ax.set_ylabel("CV accuracy (gold)"); ax.set_ylim(0, 1.0)
    ax.set_title(f"MAG E1 readability — {ds}"); ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_mag_readability_{ds}.png", dpi=150)
    print(f"wrote plot_mag_readability_{ds}.png")


def plot_linearity(ds):
    rows = _read(f"mag_linearity_{ds}.csv")
    if not rows:
        return
    labels = [f'{r["direction"]}/{r["mode"]}' for r in rows]
    eps = [float(r["eps_Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), eps, color="#ee6677")
    ax.axhline(1.0, color="k", ls="--", lw=0.7, label="no better than not steering")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("reconstruction error eps_Q")
    ax.set_title(f"MAG E3 linearity — {ds}"); ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_mag_linearity_{ds}.png", dpi=150)
    print(f"wrote plot_mag_linearity_{ds}.png")


def plot_flips(ds):
    rows = _read(f"mag_verdict_flips_{ds}.csv")
    if not rows:
        return
    dirs = sorted({r["direction"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for dn in dirs:
        sub = sorted([r for r in rows if r["direction"] == dn], key=lambda r: float(r["tau"]))
        ax.plot([float(r["tau"]) for r in sub], [float(r["flip_rate"]) for r in sub],
                "o-", label=dn)
    ax.set_xlabel("tau (calibrated)"); ax.set_ylabel("verdict-flip rate"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"MAG E4 verdict flips — {ds}"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"plot_mag_flips_{ds}.png", dpi=150)
    print(f"wrote plot_mag_flips_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_readability(a.dataset); plot_linearity(a.dataset); plot_flips(a.dataset)


if __name__ == "__main__":
    main()
