"""
viz_mag_linearity.py — the better E3 "linearity cosine shift score" figure (PI request, 2026-07-23).

The old plot_mag_linearity.png showed only eps_Q(v_Q) per dataset. It buried the actual finding:
the question-shift is owned by ONE direction (cos(shift, v_Q) ~ 0.84-0.98) while the supervised
truth axis and DCT's causal lever are BOTH orthogonal to it (cos ~ 0), and their eps_Q > 1 means
they reconstruct the shift worse than predicting zero.

Reads mag_linearity_<ds>.csv (columns: direction, mode, eps_Q, cos; some datasets only have the
v_Q row) and renders one two-panel figure: left = the cosine ladder per dataset, right = the
unexplained-fraction eps_Q with the "explains nothing" line at 1.0.

Usage:  PYTHONPATH=src .venv/bin/python src/viz_mag_linearity.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GRAY, GREEN, RED = "#4477aa", "#999999", "#228833", "#cc3311"

DATASETS = ["cities", "sp_en_trans", "companies_true_false", "common_claim_true_false"]
SHORT = {"cities": "cities", "sp_en_trans": "sp_en_trans",
         "companies_true_false": "companies", "common_claim_true_false": "common_claim"}
DIRS = [("v_Q", "MAG shift direction v$_Q$", GREEN),
        ("mean_diff", "supervised truth axis", BLUE),
        ("dct", "DCT top causal lever", RED)]


def read_linearity(ds):
    out = {}
    with open(f"mag_linearity_{ds}.csv") as f:
        for r in csv.DictReader(f):
            if r["mode"] == "final":
                out[r["direction"]] = (float(r["eps_Q"]), float(r["cos"]))
    return out


def main():
    data = {ds: read_linearity(ds) for ds in DATASETS}
    # order datasets clean -> messy by measured eps_Q(v_Q), same convention as viz_mag_findings
    order = sorted(DATASETS, key=lambda ds: data[ds]["v_Q"][0])

    fig, (axC, axE) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    xpos = np.arange(len(order))
    w = 0.26

    # ---- left panel: cosine of each candidate direction to the prefix shift
    for j, (key, label, color) in enumerate(DIRS):
        off = (j - 1) * w
        labeled = False
        for i, ds in enumerate(order):
            if key in data[ds]:
                c = data[ds][key][1]
                axC.bar(xpos[i] + off, c, w, color=color, label=None if labeled else label)
                labeled = True
                axC.text(xpos[i] + off, c + (0.02 if c >= 0 else -0.055),
                         f"{c:.2f}", ha="center", fontsize=7)
            else:
                axC.text(xpos[i] + off, 0.03, "n/a", ha="center", va="bottom",
                         fontsize=7, color=GRAY, rotation=90)
    axC.axhline(1.0, color="k", lw=1.0, ls="--")
    axC.text(len(order) - 0.5, 1.015, "shift perfectly one-dimensional", ha="right", fontsize=8)
    axC.axhspan(-0.1, 0.1, color=GRAY, alpha=0.18, zorder=0)
    axC.text(-0.55, 0.115, "orthogonal band (|cos| < 0.1)", fontsize=8, color="#555555")
    axC.axhline(0, color="k", lw=0.6)
    axC.set_xticks(xpos); axC.set_xticklabels([SHORT[d] for d in order], fontsize=9)
    axC.set_ylabel("cos(direction, prefix shift Δ$^Q$)")
    axC.set_ylim(-0.15, 1.1)
    axC.set_title("One direction owns the question-shift —\nand it is not the truth axis or the causal lever",
                  fontsize=10)
    axC.legend(fontsize=8, loc="center right"); axC.grid(axis="y", alpha=0.3)

    # ---- right panel: unexplained fraction eps_Q (lower = direction captures the shift)
    for j, (key, label, color) in enumerate(DIRS):
        off = (j - 1) * w
        labeled = False
        for i, ds in enumerate(order):
            if key in data[ds]:
                e = data[ds][key][0]
                axE.bar(xpos[i] + off, e, w, color=color, label=None if labeled else label)
                labeled = True
                axE.text(xpos[i] + off, e + 0.02, f"{e:.2f}", ha="center", fontsize=7)
            else:
                axE.text(xpos[i] + off, 0.03, "n/a", ha="center", va="bottom",
                         fontsize=7, color=GRAY, rotation=90)
    axE.axhline(1.0, color=RED, lw=1.2, ls=":")
    axE.text(-0.45, 1.03, "eps$_Q$ = 1: direction explains nothing", ha="left",
             fontsize=8, color=RED)
    axE.set_xticks(xpos); axE.set_xticklabels([SHORT[d] for d in order], fontsize=9)
    axE.set_ylabel("eps$_Q$  (unexplained fraction of the shift;  0 = perfect)")
    axE.set_ylim(0, 1.55)
    axE.set_title("v$_Q$ leaves 18–55% unexplained (rising with messiness);\n"
                  "truth axis and DCT lever do worse than predicting zero", fontsize=10)
    axE.legend(fontsize=8, loc="upper left"); axE.grid(axis="y", alpha=0.3)

    fig.suptitle("E3 linearity of the question-shift: cosine score and residual, all candidate directions",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("plot_mag_linearity_v2.png", dpi=150)
    print("saved plot_mag_linearity_v2.png")


if __name__ == "__main__":
    main()
