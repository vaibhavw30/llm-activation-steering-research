"""U-anchored warm-DCT figures: verdict-vs-tau panels and the U/V geometry ladder.

    python viz_dct_uwarm.py --dataset cities"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_dct_warm import verdict_fractions, _sorted_taus, _curve, VERDICT_COLOR
from dct_warm import lam_tag

PANELS = [
    ("raw_mean_diff", "supervised axis (mean_diff)"),
    ("cold_top", "cold DCT top factor (free)"),
    ("uwarm_mean_diff_lam0p3", "U-anchored λ=0.3"),
    ("uwarm_mean_diff_lam1", "U-anchored λ=1"),
    ("uwarm_mean_diff_lam3", "U-anchored λ=3"),
]
UWARM_LAMS = [0.3, 1.0, 3.0]


def plot_curves(ds):
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_dct_uwarm_steer_{ds}.csv"))))
    tk = _sorted_taus(fr)
    taus = [float(t) for t in tk]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (d, title) in zip(flat, PANELS):
        for v in ("TRUE", "FALSE", "INCOHERENT"):
            ax.plot(taus, _curve(fr, d, tk, v), "o-", color=VERDICT_COLOR[v],
                    label=v.title(), lw=2, ms=4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    for ax in flat[len(PANELS):]:
        ax.axis("off")
    flat[0].legend(fontsize=9, loc="center left")
    for ax in axes[1]:
        ax.set_xlabel("τ  (−=push FALSE, +=push TRUE)")
    for r in (0, 1):
        axes[r][0].set_ylabel("verdict fraction")
    fig.suptitle(f"{ds}: U-anchored DCT — verdict vs steering strength τ\n"
                 f"(input free, effect anchored toward the truth readout)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"plot_dct_uwarm_curves_{ds}.png", dpi=140)
    print(f"[viz] wrote plot_dct_uwarm_curves_{ds}.png")


def plot_geometry(ds):
    rows = list(csv.DictReader(open(f"dct_uwarm_geometry_{ds}.csv")))
    by = {r["direction"]: r for r in rows}
    cos_u, cos_v = [], []
    for lam in UWARM_LAMS:
        r = by.get(f"uwarm_mean_diff_{lam_tag(lam)}")
        cos_u.append(abs(float(r["cos_U0_mdtgt"])) if r else np.nan)
        cos_v.append(abs(float(r["cos_V0_mdsrc"])) if r else np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(UWARM_LAMS, cos_u, "o-", color="#aa3377",
            label="|cos(U₀, mean_diff@target)| — does the anchor knob work?")
    ax.plot(UWARM_LAMS, cos_v, "s-", color="#4477aa",
            label="|cos(V₀, mean_diff@source)| — what input gets chosen?")
    ax.set_xlabel("U-anchor λ"); ax.set_ylabel("|cos|"); ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{ds}: effect-space anchoring geometry")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"plot_dct_uwarm_geometry_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_uwarm_geometry_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_curves(a.dataset)
    plot_geometry(a.dataset)


if __name__ == "__main__":
    main()
