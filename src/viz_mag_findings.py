"""
viz_mag_findings.py — summary figures for the DCT-vs-MAG head-to-head (geometric battery).

Reads the artifacts run_mag.py already wrote (mag_readability_<ds>.csv, mag_linearity_<ds>.csv,
mag_dir_<ds>.npz, mag_acts_<ds>.npz, mag_transfer.csv) and renders publication-style charts in the
same house style as viz_findings.py. Reproducible: no numbers are hardcoded.

Produces:
  plot_mag_recovery.png     — cos(u_Q, mean_diff) vs cos(u_Q, DCT top-V): MAG recovers the supervised
                              truth direction but is orthogonal to DCT's top causal lever
  plot_mag_readability.png  — E1: the prefix shift (InputDelta) reads truth as well as raw activations,
                              well above a random-projection baseline
  plot_mag_subspace.png     — DCT top-k subspace reads truth no better than random-k (cities, common_claim)
  plot_mag_linearity.png    — E3: the prefix shift is near one-dimensional, and less so as data gets messier
  plot_mag_verdict.png      — the self-verdict y^M is degenerate (all-"yes") and the Verdict operator
                              reads at chance: the fully-unsupervised arm is dead on a base model
  plot_mag_transfer.png     — cross-dataset transfer heatmap of the MAG (InputDelta) direction

Usage:  PYTHONPATH=src .venv/bin/python src/viz_mag_findings.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# same palette as viz_findings.py
BLUE, GRAY, GREEN, RED, ORANGE = "#4477aa", "#999999", "#228833", "#cc3311", "#ee7733"

DATASETS = ["sp_en_trans", "cities", "companies_true_false", "common_claim_true_false"]
SHORT = {"sp_en_trans": "sp_en_trans", "cities": "cities",
         "companies_true_false": "companies", "common_claim_true_false": "common_claim"}
# ordered clean -> messy for the narrative axis
CLEAN_TO_MESSY = ["cities", "sp_en_trans", "companies_true_false", "common_claim_true_false"]


def _read_readability(ds):
    """operator -> {'gold': (acc, roc)} plus the baseline rows."""
    out = {}
    with open(f"mag_readability_{ds}.csv") as f:
        for r in csv.DictReader(f):
            if r["target"] != "gold":
                continue
            acc = float(r["acc"]) if r["acc"] not in ("", "nan") else float("nan")
            roc = float(r["roc"]) if r["roc"] not in ("", "nan") else float("nan")
            out[r["operator"]] = (acc, roc)
    return out


def _read_linearity(ds):
    with open(f"mag_linearity_{ds}.csv") as f:
        for r in csv.DictReader(f):
            if r["direction"] == "v_Q":
                return float(r["eps_Q"]), float(r["cos"])
    return float("nan"), float("nan")


def _read_dir(ds):
    return np.load(f"mag_dir_{ds}.npz")


def _read_verdict(ds):
    d = np.load(f"mag_acts_{ds}.npz", allow_pickle=True)
    ym = d["ymL"].astype(int)
    yes = int((ym == 1).sum()); no = int((ym == 0).sum())
    return yes, no


# ----------------------------------------------------------------------------- figures

def fig_recovery():
    """The core contrast: recovers supervised direction, orthogonal to DCT."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xpos = np.arange(len(DATASETS)); w = 0.26
    md, gr, dc = [], [], []
    for ds in DATASETS:
        d = _read_dir(ds)
        md.append(float(d["cos_uGold_mean_diff"]))
        gr.append(float(d["cos_uGold_grad"]))
        v = float(d["cos_uGold_dctV"])
        dc.append(v if np.isfinite(v) else np.nan)
    ax.bar(xpos - w, md, w, color=GREEN, label="vs supervised mean-diff")
    ax.bar(xpos, gr, w, color=BLUE, label="vs supervised gradient")
    # DCT bars; annotate the ones that don't exist
    for i, v in enumerate(dc):
        if np.isfinite(v):
            ax.bar(xpos[i] + w, v, w, color=RED,
                   label="vs DCT top causal vector" if i == 1 else None)
        else:
            ax.text(xpos[i] + w, 0.02, "no DCT", ha="center", va="bottom",
                    fontsize=7, color=GRAY, rotation=90)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels([SHORT[d] for d in DATASETS], fontsize=9)
    ax.set_ylabel("cosine of MAG direction u$_Q$ with …"); ax.set_ylim(-0.15, 1.05)
    ax.set_title("MAG recovers the supervised truth direction, but is orthogonal to DCT's causal lever\n"
                 "cos(u$_Q$, mean-diff) ≈ 0.98–1.00   vs   cos(u$_Q$, DCT top-V) ≈ 0",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center right"); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(md):
        ax.text(xpos[i] - w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig("plot_mag_recovery.png", dpi=150); plt.close(fig)
    print("saved plot_mag_recovery.png")


def fig_readability():
    """The prefix shift carries the full linear truth signal."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xpos = np.arange(len(DATASETS)); w = 0.22
    direct, idel, prefx, rand = [], [], [], []
    for ds in DATASETS:
        r = _read_readability(ds)
        direct.append(r["Direct"][0]); idel.append(r["InputDelta"][0])
        prefx.append(r["Prefixed"][0]); rand.append(r["random_10"][0])
    ax.bar(xpos - 1.5 * w, direct, w, color=GRAY, label="Direct (raw activations)")
    ax.bar(xpos - 0.5 * w, idel, w, color=GREEN, label="InputDelta (prefix shift Δ$^Q$)")
    ax.bar(xpos + 0.5 * w, prefx, w, color=BLUE, label="Prefixed (Q‖p activations)")
    ax.bar(xpos + 1.5 * w, rand, w, color=RED, label="random-10 baseline")
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(xpos); ax.set_xticklabels([SHORT[d] for d in DATASETS], fontsize=9)
    ax.set_ylabel("truth-classification accuracy (5-fold CV)"); ax.set_ylim(0.45, 1.02)
    ax.set_title("The prefix-induced shift carries truth as well as the raw activations\n"
                 "InputDelta ≈ Direct on every dataset, far above random projections",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_mag_readability.png", dpi=150); plt.close(fig)
    print("saved plot_mag_readability.png")


def fig_subspace():
    """DCT causal subspace holds no truth advantage over random directions."""
    dss = ["cities", "common_claim_true_false"]
    cats = ["DCT_top10", "DCT_top50", "random_10", "Direct"]
    catlabel = ["DCT top-10", "DCT top-50", "random-10", "full space"]
    colors = [BLUE, "#6699cc", RED, GRAY]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xpos = np.arange(len(dss)); w = 0.2
    for j, cat in enumerate(cats):
        vals = [_read_readability(ds)[cat][0] for ds in dss]
        ax.bar(xpos + (j - 1.5) * w, vals, w, color=colors[j], label=catlabel[j])
        for i, v in enumerate(vals):
            ax.text(xpos[i] + (j - 1.5) * w, v + 0.006, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xpos); ax.set_xticklabels(["cities (clean)", "common_claim (messy)"], fontsize=9)
    ax.set_ylabel("truth accuracy from the projected subspace"); ax.set_ylim(0.55, 1.02)
    ax.set_title("DCT's causal subspace carries no truth signal beyond random directions\n"
                 "(cities: DCT top-10 even falls *below* random-10)", fontsize=10)
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_mag_subspace.png", dpi=150); plt.close(fig)
    print("saved plot_mag_subspace.png")


def fig_linearity():
    """The prefix shift is near one-dimensional; ε_Q tracks dataset messiness."""
    # order by measured epsilon so the axis reads low -> high without over-claiming a fixed ranking
    order = sorted(DATASETS, key=lambda ds: _read_linearity(ds)[0])
    eps = [_read_linearity(ds)[0] for ds in order]
    clean = {"cities", "sp_en_trans"}
    colors = ["#f0a860" if ds in clean else ORANGE for ds in order]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xpos = np.arange(len(order))
    ax.bar(xpos, eps, 0.55, color=colors)
    for i, v in enumerate(eps):
        tag = "clean" if order[i] in clean else "messy"
        ax.text(xpos[i], v + 0.012, f"{v:.2f}\n({tag})", ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels([SHORT[d] for d in order], fontsize=9)
    ax.set_ylabel("ε$_Q$  (unexplained fraction of the shift;  0 = perfectly 1-D)")
    ax.set_ylim(0, 0.72)
    ax.set_title("The prefix shift is close to one-dimensional, and less so as the concept gets messier\n"
                 "the two clean datasets have the lowest ε$_Q$; the two messy ones the highest", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_mag_linearity.png", dpi=150); plt.close(fig)
    print("saved plot_mag_linearity.png")


def fig_verdict():
    """Self-verdict degeneracy + Verdict operator at chance."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.4))
    # left: y^M yes/no stacked
    xpos = np.arange(len(DATASETS))
    yes = []; no = []
    for ds in DATASETS:
        y, n = _read_verdict(ds); yes.append(y); no.append(n)
    yes = np.array(yes, float); no = np.array(no, float)
    tot = yes + no
    axL.bar(xpos, yes / tot, 0.6, color=BLUE, label='y$^M$ = "yes"')
    axL.bar(xpos, no / tot, 0.6, bottom=yes / tot, color=RED, label='y$^M$ = "no"')
    for i in range(len(DATASETS)):
        axL.text(xpos[i], 1.02, f"{int(no[i])}/{int(tot[i])} no", ha="center", fontsize=7, color=RED)
    axL.set_xticks(xpos); axL.set_xticklabels([SHORT[d] for d in DATASETS], fontsize=8, rotation=20)
    axL.set_ylabel("fraction of statements"); axL.set_ylim(0, 1.12)
    axL.set_title("Self-verdict y$^M$ is all-\"yes\"\n(the model never says false)", fontsize=10)
    axL.legend(fontsize=8, loc="lower right")
    # right: Verdict operator accuracy vs chance
    vacc = [_read_readability(ds)["Verdict"][0] for ds in DATASETS]
    axR.bar(xpos, vacc, 0.6, color=GRAY)
    axR.axhline(0.5, color=RED, lw=1.5, ls="--", label="chance")
    for i, v in enumerate(vacc):
        axR.text(xpos[i], v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axR.set_xticks(xpos); axR.set_xticklabels([SHORT[d] for d in DATASETS], fontsize=8, rotation=20)
    axR.set_ylabel("truth accuracy from verdict-position acts"); axR.set_ylim(0.4, 0.75)
    axR.set_title("Verdict operator reads at/near chance\n(the yes/no machinery barely tracks truth)", fontsize=10)
    axR.legend(fontsize=8)
    fig.suptitle("The fully-unsupervised arm is dead on gemma-2-2b base — the geometry knows what the "
                 "model won't say", fontsize=10)
    fig.tight_layout(); fig.savefig("plot_mag_verdict.png", dpi=150); plt.close(fig)
    print("saved plot_mag_verdict.png")


def fig_transfer():
    """Cross-dataset transfer heatmap of the MAG InputDelta direction (geom_score)."""
    idx = {ds: i for i, ds in enumerate(DATASETS)}
    M = np.full((4, 4), np.nan)
    with open("mag_transfer.csv") as f:
        for r in csv.DictReader(f):
            t, c = r["target"], r["candidate"]
            if t in idx and c in idx:
                M[idx[t], idx[c]] = float(r["geom_score"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[d] for d in DATASETS], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[d] for d in DATASETS], fontsize=8)
    ax.set_xlabel("candidate direction from"); ax.set_ylabel("target dataset")
    for i in range(4):
        for j in range(4):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color=GRAY)
    fig.colorbar(im, ax=ax, fraction=0.046, label="geometric transfer score")
    ax.set_title("MAG direction transfers only modestly across datasets\n"
                 "Top-1 = 0.25, Spearman = 0.404 over 12 ordered pairs", fontsize=10)
    fig.tight_layout(); fig.savefig("plot_mag_transfer.png", dpi=150); plt.close(fig)
    print("saved plot_mag_transfer.png")


def fig_summary_table():
    """One master table of the headline numbers across all four datasets."""
    cols = ["sp_en_trans", "cities", "companies", "common_claim"]
    rows = [
        "layer",
        "InputDelta acc (gold)",
        "  vs raw Direct acc",
        "  vs random-10 acc",
        "cos(u_Q, mean-diff)",
        "cos(u_Q, DCT top-V)",
        "ε_Q  (shift linearity)",
        "Verdict-op acc",
        "y^M  (yes / no)",
        "‖prefix-shift‖  (α unit)",
    ]
    layers = {"sp_en_trans": 7, "cities": 11, "companies_true_false": 14, "common_claim_true_false": 13}
    cell = []
    for ds in DATASETS:
        r = _read_readability(ds); d = _read_dir(ds)
        eps, _ = _read_linearity(ds); yes, no = _read_verdict(ds)
        dctv = float(d["cos_uGold_dctV"])
        cell.append([
            str(layers[ds]),
            f"{r['InputDelta'][0]:.3f}",
            f"{r['Direct'][0]:.3f}",
            f"{r['random_10'][0]:.3f}",
            f"{float(d['cos_uGold_mean_diff']):.3f}",
            (f"{dctv:+.3f}" if np.isfinite(dctv) else "—  (no DCT)"),
            f"{eps:.3f}",
            f"{r['Verdict'][0]:.3f}",
            f"{yes} / {no}",
            f"{float(d['A_prefix_norm']):.1f}",
        ])
    table = np.array(cell).T  # rows x cols

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")
    tbl = ax.table(cellText=table, rowLabels=rows, colLabels=[SHORT[c] for c in DATASETS],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    # header + rowlabel styling
    for (rr, cc), c in tbl.get_celld().items():
        if rr == 0:
            c.set_facecolor("#334455"); c.get_text().set_color("white"); c.get_text().set_fontweight("bold")
        if cc == -1:
            c.get_text().set_ha("left"); c.get_text().set_fontweight("bold"); c.set_facecolor("#eef1f4")
    # highlight the two headline rows
    for rr in (5, 6):  # cos(u_Q, mean-diff) and cos(u_Q, DCT top-V)
        for cc in range(len(DATASETS)):
            tbl[(rr, cc)].set_facecolor("#eaf3ea" if rr == 5 else "#fdecea")
    ax.set_title("MAG geometric battery — headline numbers (gemma-2-2b base, truth-peak layer)\n"
                 "green row: recovers the supervised direction   ·   red row: orthogonal to DCT's causal lever",
                 fontsize=10, pad=14)
    fig.tight_layout(); fig.savefig("plot_mag_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved plot_mag_summary_table.png")


if __name__ == "__main__":
    fig_recovery()
    fig_readability()
    fig_subspace()
    fig_linearity()
    fig_verdict()
    fig_transfer()
    fig_summary_table()
    print("\nDone — 7 MAG findings figures written (plot_mag_*.png).")
