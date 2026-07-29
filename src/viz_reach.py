"""viz_reach.py — figures for the backward-reachability audit.

Phase 1: margin distributions, reachability curves (with Phase-5 trust region
overlaid when reach_linerr_summary_<ds>.csv exists), and J^T w source geometry.
Later tasks extend this module with SVD (Phase 2) and J-lens (Phase 4) figures.

    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GRAY, GREEN, RED, ORANGE = "#4477aa", "#999999", "#228833", "#cc3311", "#ee7733"


def _load(ds):
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    dz = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    with open(f"reach_summary_{ds}.json") as f:
        summ = json.load(f)
    names = [str(x) for x in mz["names"]]
    groups = [str(x) for x in mz["groups"]]
    return mz, dz, summ, names, groups


def fig_margins(ds):
    mz, dz, summ, names, groups = _load(ds)
    m = np.asarray(mz["margins"], np.float64)
    sel = {"truth readouts": [k for k, g in enumerate(groups) if g == "truth"],
           "truth subspace": [k for k, g in enumerate(groups) if g == "truth_sub"],
           "DCT U top (control)": [k for k, g in enumerate(groups) if g == "dct_u"],
           "random null (64)": [k for k, g in enumerate(groups) if g == "rand"]}
    colors = [BLUE, GREEN, RED, GRAY]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    data = [np.log10(np.clip(m[:, ks].ravel(), 1e-12, None)) for ks in sel.values()]
    parts = ax.violinplot(data, showmedians=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(sel) + 1)); ax.set_xticklabels(sel.keys(), fontsize=9)
    ax.set_ylabel("log10 controllability margin  ||J^T w||")
    ax.axhline(np.log10(max(summ["rand_null"]["median"], 1e-12)), color=GRAY,
               ls="--", lw=1, label="random-null median")
    ax.set_title(f"{ds}: controllability margins per readout group "
                 f"(verdict: {summ['verdict']})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"plot_reach_margins_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_margins_{ds}.png")


def _read_curves(ds):
    curves = {}
    with open(f"reach_curve_{ds}.csv") as f:
        for r in csv.DictReader(f):
            curves.setdefault(r["direction"], []).append(
                (float(r["eps"]), float(r["frac_reachable"])))
    return {k: np.array(v) for k, v in curves.items()}


def fig_curves(ds):
    with open(f"reach_summary_{ds}.json") as f:
        summ = json.load(f)
    curves = _read_curves(ds)
    scale = summ["input_scale"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    palette = [BLUE, GREEN, ORANGE, RED, GRAY, "#aa3377", "#66ccee"]
    show = ["mean_diff_tgt", "probe_grad_tgt", "truth_sub_best"]
    show += [k for k in curves if k.startswith("dct_u")]
    show += [k for k in curves if k not in show][:2]
    for i, nm in enumerate([s for s in show if s in curves]):
        c = curves[nm]
        lw = 2.4 if nm == "truth_sub_best" else 1.4
        ax.plot(c[:, 0], c[:, 1], label=nm, color=palette[i % len(palette)], lw=lw)
    ax.axvline(scale, color="k", ls="--", lw=1)
    ax.text(scale, 1.02, "input_scale", ha="center", fontsize=8)
    lin = f"reach_linerr_summary_{ds}.csv"
    if os.path.exists(lin):
        with open(lin) as f:
            rows = {r["direction"]: float(r["eps20"]) for r in csv.DictReader(f)}
        e20 = rows.get("jtw_mean_diff_tgt")
        if e20 is not None:
            ax.axvspan(e20, ax.get_xlim()[1], color=GRAY, alpha=0.15)
            ax.text(e20, 0.5, " beyond linear validity (rel err > 20%)",
                    fontsize=8, color="#555555", rotation=90, va="center")
    ax.set_xlabel("perturbation budget eps (activation norm units)")
    ax.set_ylabel("fraction of TRUE statements reachable into FALSE halfspace")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(f"{ds}: first-order reachability of the FALSE-probe halfspace",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"plot_reach_curves_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_curves_{ds}.png")


def fig_geometry(ds):
    mz, dz, summ, names, groups = _load(ds)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    k_md = names.index("mean_diff_tgt")
    picks = [("mean_diff_tgt", k_md, BLUE)]
    if "dct_u_0" in names:
        picks.append(("dct_u_0", names.index("dct_u_0"), RED))
    panels = [(axes[0], "cos_md_src", "cos(J^T w, mean_diff@src)")]
    if "cos_vq" in mz.files:
        panels.append((axes[1], "cos_vq", "cos(J^T w, v_Q)"))
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "no mag_dir landmark", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9)
    for ax, key, lab in panels:
        arr = np.asarray(mz[key], np.float64)
        for nm, k, c in picks:
            ax.hist(arr[:, k], bins=40, alpha=0.55, color=c, label=f"w={nm}")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab); ax.set_xlim(-1, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylabel("statements")
    fig.suptitle(f"{ds}: where J^T w points at the source layer", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_reach_geometry_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_geometry_{ds}.png")


def fig_svd(ds):
    path = f"reach_svd_energy_{ds}.csv"
    if not os.path.exists(path):
        print(f"skip fig_svd: {path} missing")
        return
    series = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            series.setdefault(r["quantity"], []).append(
                (int(r["k"]), float(r["mean_energy"])))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    style = {"w_mean_diff_tgt_in_U": (BLUE, "truth mean_diff (left/U)"),
             "w_probe_grad_tgt_in_U": (GREEN, "truth probe grad (left/U)"),
             "md_src_in_V": (ORANGE, "mean_diff@src (right/V)"),
             "rand_in_U": (GRAY, "random readout (left/U)")}
    for q, pts in series.items():
        pts = np.array(sorted(pts))
        c, lab = style.get(q, (RED, q))
        ax.plot(pts[:, 0], pts[:, 1], color=c, label=lab)
    ax.set_xlabel("top-k singular subspace"); ax.set_ylabel("captured energy of w")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{ds}: does truth live in the singular tail of the hop?",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"plot_reach_svd_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_svd_{ds}.png")


def fig_jlens(ds):
    path = f"reach_jlens_{ds}.csv"
    if not os.path.exists(path):
        print(f"skip fig_jlens: {path} missing")
        return
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            data.setdefault((r["mode"], r["w_name"]), []).append(
                (int(r["layer"]), float(r["margin_mean"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    style = {"verdict": (RED, "verdict (yes-no unembed)"),
             "truth_final": (BLUE, "truth-content probe"),
             "v_q_final": (GREEN, "question-mode v_Q")}
    for ax, mode, title in ((axes[0], "decl", "declarative input"),
                            (axes[1], "quest", "question-prefixed input")):
        for wn, (c, lab) in style.items():
            pts = np.array(sorted(data.get((mode, wn), [(0, np.nan)])))
            ax.semilogy(pts[:, 0], pts[:, 1], color=c, label=lab)
        ax.set_xlabel("source layer l"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean ||J_l^T w||")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ds}: per-layer controllability of final-basis readouts "
                 f"(workspace selectivity)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_reach_jlens_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_jlens_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    fig_margins(ds); fig_curves(ds); fig_geometry(ds); fig_svd(ds); fig_jlens(ds)


if __name__ == "__main__":
    main()
