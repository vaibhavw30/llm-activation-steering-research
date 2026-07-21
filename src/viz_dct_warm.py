"""Warm-DCT figures: drift-vs-lambda, verdict head-to-head, and the note-#2 cold-factor audit.

    python viz_dct_warm.py --dataset cities"""
import argparse
import csv
import numpy as np
import torch
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
    plot_audit(a.dataset)


if __name__ == "__main__":
    main()
