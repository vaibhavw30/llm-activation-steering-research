"""Arm A2 figure: gate0 vs gate1 verdict-vs-tau panels from the judged CSV.

    python -m mag.viz_conditional --dataset cities"""
import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_dct_warm import verdict_fractions, _sorted_taus, _curve, VERDICT_COLOR

PANELS = [("meandiff_gate0", "mean_diff alone (gate off) — known inert"),
          ("meandiff_gate1", "v_Q + mean_diff (gate on) — conditional test")]


def plot_conditional(ds):
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_mag_conditional_{ds}.csv"))))
    tk = _sorted_taus(fr)
    taus = [float(t) for t in tk]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, (d, title) in zip(axes, PANELS):
        for v in ("TRUE", "FALSE", "INCOHERENT"):
            ax.plot(taus, _curve(fr, d, tk, v), "o-", color=VERDICT_COLOR[v],
                    label=v.title(), lw=2, ms=4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
        ax.set_xlabel("tau  (-=push FALSE, +=push TRUE)")
    axes[0].set_ylabel("verdict fraction")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{ds}: does the question-mode gate unlock the truth axis?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_mag_conditional_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_conditional_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_conditional(a.dataset)


if __name__ == "__main__":
    main()
