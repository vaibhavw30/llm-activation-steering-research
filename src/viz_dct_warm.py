"""Warm-DCT figures: drift-vs-lambda, verdict head-to-head, and the note-#2 cold-factor audit.

    python viz_dct_warm.py --dataset cities"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

LAMS = [0.0, 0.3, 1.0, 3.0]


def verdict_fractions(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], str(float(r["scale"]))), []).append(r["verdict"])
    out = {}
    for key, vs in groups.items():
        n = len(vs) or 1
        out[key] = {v: sum(x == v for x in vs) / n for v in ("TRUE", "FALSE", "INCOHERENT")}
    return out


def plot_drift(ds):
    rows = list(csv.DictReader(open(f"dct_warm_geometry_{ds}.csv")))
    drift = {r["direction"]: float(r["drift"]) for r in rows}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for seed, color in [("mean_diff", "#228833"), ("grad", "#4477aa")]:
        ys = [drift.get(f"warm_{seed}_{lam_tag(l)}", np.nan) for l in LAMS]
        ax.plot(LAMS, ys, "o-", color=color, label=seed)
    ax.set_xlabel("anchor λ"); ax.set_ylabel("drift (1 − cos to seed)")
    ax.set_title(f"{ds}: how far the causal search wanders vs anchor strength")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_drift_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_drift_{ds}.png")


def plot_verdict(ds):
    rows = list(csv.DictReader(open(f"judge_dct_warm_steer_{ds}.csv")))
    fr = verdict_fractions(rows)
    taus = sorted({k[1] for k in fr}, key=float)
    # directions point toward TRUE; the causal truth->false flip lives at the most-negative tau
    strongest = taus[0]
    names = sorted({k[0] for k in fr})
    false_v = [fr.get((n, strongest), {}).get("FALSE", 0.0) for n in names]
    incoh_v = [fr.get((n, strongest), {}).get("INCOHERENT", 0.0) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.6))
    ax.bar(x - 0.2, false_v, 0.4, color="#cc3311", label="FALSE (causal)")
    ax.bar(x + 0.2, incoh_v, 0.4, color="#999999", label="INCOHERENT (degrade)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1.0)
    ax.set_title(f"{ds}: causal lever vs degrader at τ={strongest}")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_verdict_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_verdict_{ds}.png")


CURVE_PANELS = [
    ("raw_mean_diff", "supervised axis (mean_diff)"),
    ("raw_grad", "supervised axis (grad)"),
    ("cold_top", "cold DCT top factor (free)"),
    ("warm_mean_diff_lam0", "warm mean_diff λ=0"),
    ("warm_mean_diff_lam0p3", "warm mean_diff λ=0.3"),
    ("warm_mean_diff_lam1", "warm mean_diff λ=1"),
    ("warm_mean_diff_lam3", "warm mean_diff λ=3"),
    ("warm_grad_lam0", "warm grad λ=0"),
    ("warm_grad_lam0p3", "warm grad λ=0.3"),
    ("warm_grad_lam1", "warm grad λ=1"),
    ("warm_grad_lam3", "warm grad λ=3"),
]
VERDICT_COLOR = {"TRUE": "#228833", "FALSE": "#cc3311", "INCOHERENT": "#999999"}


def _sorted_taus(fr):
    return sorted({k[1] for k in fr}, key=float)


def _curve(fr, d, tau_keys, v):
    return [fr.get((d, t), {}).get(v, np.nan) for t in tau_keys]


def plot_curves(ds):
    """Full verdict-vs-τ sweep per direction: a truth lever = FALSE rises at −τ;
    a degrader = INCOH rises; inert = flat."""
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_dct_warm_steer_{ds}.csv"))))
    tk = _sorted_taus(fr)
    taus = [float(t) for t in tk]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (d, title) in zip(axes, CURVE_PANELS):
        for v in ("TRUE", "FALSE", "INCOHERENT"):
            ax.plot(taus, _curve(fr, d, tk, v), "o-", color=VERDICT_COLOR[v],
                    label=v.title(), lw=2, ms=4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    for ax in axes[len(CURVE_PANELS):]:
        ax.axis("off")
    axes[0].legend(fontsize=9, loc="center left")
    for ax in axes[8:12]:
        ax.set_xlabel("τ  (−=push FALSE, +=push TRUE)")
    for r in (0, 1, 2):
        axes[r * 4].set_ylabel("verdict fraction")
    fig.suptitle(f"{ds}: verdict vs steering strength τ, per direction\n"
                 f"(a truth lever = FALSE rises at −τ; a degrader = INCOH rises; inert = flat)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"plot_dct_warm_curves_{ds}.png", dpi=130)
    print(f"[viz] wrote plot_dct_warm_curves_{ds}.png")


def plot_effect(ds):
    """Per direction: max rise above the τ=0 baseline in FALSE (would-be lie lever)
    vs INCOHERENT (degradation lever). Shows steering degrades, never lies."""
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_dct_warm_steer_{ds}.csv"))))
    tk = _sorted_taus(fr)
    zero = str(float(0.0))
    b_false = fr.get((CURVE_PANELS[0][0], zero), {}).get("FALSE", 0.0)
    b_incoh = fr.get((CURVE_PANELS[0][0], zero), {}).get("INCOHERENT", 0.0)
    names, dF, dI = [], [], []
    for d, title in CURVE_PANELS:
        names.append(title)
        dF.append(np.nanmax(_curve(fr, d, tk, "FALSE")) - b_false)
        dI.append(np.nanmax(_curve(fr, d, tk, "INCOHERENT")) - b_incoh)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, dF, 0.4, color="#cc3311", label="max ΔFALSE (would-be lie lever)")
    ax.bar(x + 0.2, dI, 0.4, color="#999999", label="max ΔINCOH (degradation lever)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("max rise above τ=0 baseline")
    ax.set_title(f"{ds}: what does steering actually do?  "
                 f"(baseline FALSE={b_false:.2f}, INCOH={b_incoh:.2f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_effect_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_effect_{ds}.png")


def plot_audit(ds):
    V, U, _ = fu.load_dct(ds)
    md = unit(np.asarray(np.load(f"truth_dir_{ds}.npz")["mean_diff"], np.float64))
    potency = np.linalg.norm(U, axis=0)
    order = np.argsort(potency)[::-1][:30]
    cos = np.array([abs(float(unit(V[:, i].astype(float)) @ md)) for i in order])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(range(len(order)), potency[order], c=cos, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("cold DCT factor (ranked by ‖U‖ potency)")
    ax.set_ylabel("‖U‖ potency")
    ax.set_title(f"{ds}: is the top cold DCT factor truth-aligned? (color = |cos| to mean_diff)")
    fig.colorbar(sc, label="|cos| to mean_diff"); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_audit_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_audit_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_drift(a.dataset)
    plot_verdict(a.dataset)
    plot_curves(a.dataset)
    plot_effect(a.dataset)
    plot_audit(a.dataset)


if __name__ == "__main__":
    main()
