# src/viz_length.py
"""Length-steering figures: coherence trajectory (intrinsic) + verdict-vs-cutoff (judge).

    python viz_length.py --dataset cities
Reads length_steer_<ds>.csv and judge_length_<ds>.csv; writes two PNGs."""
import argparse
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {"mean_diff": "#228833", "resid_pc1": "#cc3311"}
_STYLES = {"0.0": ":", "0.3": "--", "1.0": "-"}


def mean_trajectory(rows, signal):
    groups = {}
    for r in rows:
        arr = json.loads(r[signal])
        groups.setdefault((r["direction"], str(float(r["tau"]))), []).append(arr)
    out = {}
    for key, arrs in groups.items():
        m = min(len(a) for a in arrs)
        out[key] = [sum(a[i] for a in arrs) / len(arrs) for i in range(m)]
    return out


def plot_coherence(ds):
    rows = list(csv.DictReader(open(f"length_steer_{ds}.csv")))
    traj = mean_trajectory(rows, "entropy")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dirn, tau), ys in sorted(traj.items()):
        ax.plot(range(1, len(ys) + 1), ys, _STYLES.get(tau, "-"),
                color=_COLORS.get(dirn, "#333333"), label=f"{dirn} τ={tau}")
    ax.set_xlabel("generated token position")
    ax.set_ylabel("mean next-token entropy (nats) — higher = degrading")
    ax.set_title(f"{ds}: coherence trajectory under steering")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plot_length_coherence_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_length_coherence_{ds}.png")


def plot_verdict(ds):
    rows = list(csv.DictReader(open(f"judge_length_{ds}.csv")))
    # direction column is "<dir>_tau<tau>", scale column is the cutoff
    cutoffs = sorted({int(float(r["scale"])) for r in rows})
    dirs = sorted({r["direction"] for r in rows})
    fig, axes = plt.subplots(1, len(dirs), figsize=(4.2 * len(dirs), 4.2), squeeze=False)
    for ax, dirn in zip(axes[0], dirs):
        fr = {"TRUE": [], "FALSE": [], "INCOHERENT": []}
        for k in cutoffs:
            sub = [r for r in rows if r["direction"] == dirn and int(float(r["scale"])) == k]
            n = len(sub) or 1
            for v in fr:
                fr[v].append(sum(r["verdict"] == v for r in sub) / n)
        ax.plot(cutoffs, fr["TRUE"], "o-", color="#228833", label="TRUE")
        ax.plot(cutoffs, fr["FALSE"], "s-", color="#cc3311", label="FALSE")
        ax.plot(cutoffs, fr["INCOHERENT"], "^--", color="#999999", label="INCOHERENT")
        ax.set_title(dirn); ax.set_xlabel("cutoff (tokens)"); ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("fraction"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"{ds}: verdict vs generation length", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"plot_length_verdict_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_length_verdict_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_coherence(a.dataset)
    plot_verdict(a.dataset)


if __name__ == "__main__":
    main()
