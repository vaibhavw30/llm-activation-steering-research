"""viz_token.py — figures for the token-space program.

Five figures, each answering one question a reader will ask:

  plot_token_steer_<ds>.png    THE RESULT. Hit rate on the target token vs budget,
                               one panel per arm. `oracle` is the ceiling, not a
                               competitor: it reaching 1.0 is what licenses reading
                               every other curve as a fraction of what was provably
                               available. Panels are restricted to the statements
                               every direction in that arm actually ran.

  plot_token_budget_<ds>.png   How expensive is a token flip, and how much more
                               expensive along the directions we actually used?
                               The bar for eps_req vs delta_cone IS the diagnosis.
  plot_token_alpha_<ds>.png    Distribution of alpha per candidate direction against
                               the chance level for a random direction in d=2304,
                               sqrt(2/(pi d)) = 0.0166. A direction sitting at chance
                               is not "weakly aligned", it is uninformative.
  plot_token_layers_<ds>.png   Certified budget per layer, both injection conventions,
                               with the layers we actually used marked. This is the
                               figure that decides whether our layer choice was wrong.
  plot_signed_steer_<ds>.png   Per-statement signed steerability, split into inert /
                               churn / mover. Never show the mean alone here.

    PYTHONPATH=src python src/viz_token.py --dataset cities
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import token_steer as ts                 # noqa: E402

CHANCE = None                            # filled from d at runtime


def fig_budget(ds, g, out):
    ok = g[g.cone_solved]
    if not len(ok):
        print("[viz] no solved rows; skipping budget figure")
        return
    names = [c[len("eps_req_"):] for c in g.columns if c.startswith("eps_req_")]
    vals = [ok["delta_cone"].median()] + [ok[f"eps_req_{n}"].replace(
        [np.inf, -np.inf], np.nan).median() for n in names]
    labs = ["optimal\n(oracle)"] + names
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(range(len(vals)), vals,
              color=["#2a9d8f"] + ["#e76f51"] * len(names))
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(labs)))
    ax[0].set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("budget to flip one token (log)")
    ax[0].set_title(f"{ds}: cost of a flip, by actuator")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax[0].text(i, v, f"{v / vals[0]:.0f}x" if i else "1x",
                       ha="center", va="bottom", fontsize=8)

    ax[1].hist(ok["delta_rel_z"], bins=30, color="#264653")
    ax[1].axvline(ok["delta_rel_z"].median(), color="#e9c46a", lw=2,
                  label=f"median {ok['delta_rel_z'].median():.3f}")
    ax[1].set_xlabel("optimal budget / ||z||")
    ax[1].set_ylabel("statements")
    ax[1].set_title("a flip is a small fraction of the activation")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_alpha(ds, g, out, d=2304):
    names = [c[len("alpha_"):] for c in g.columns if c.startswith("alpha_")]
    if not names:
        return
    chance = np.sqrt(2.0 / (np.pi * d))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.boxplot([g[f"alpha_{n}"].dropna() for n in names], tick_labels=names,
               showfliers=False)
    ax.axhline(chance, color="#e76f51", ls="--", lw=1.5,
               label=f"random direction in d={d}: {chance:.4f}")
    ax.axhline(1.0, color="#2a9d8f", ls=":", lw=1.5, label="perfect (oracle)")
    ax.set_yscale("log")
    ax.set_ylabel("alpha = |cos(actuator, token-difference direction)|")
    ax.set_title(f"{ds}: how much of each actuator points at the decision")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_layers(ds, j, out, used=()):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(j.layer, j.eps_all_median, "-o", ms=3, color="#264653",
            label="broadcast to all positions")
    ax.plot(j.layer, j.eps_last_median, "-s", ms=3, color="#e76f51",
            label="last position only (ActAdd convention)")
    ax.fill_between(j.layer, j.eps_all_p10, j.eps_all_p90, alpha=0.15,
                    color="#264653")
    for lay, lab in used:
        ax.axvline(lay, color="#8a817c", ls="--", lw=1)
        ax.text(lay, ax.get_ylim()[1], f" {lab}", rotation=90, va="top",
                fontsize=7, color="#8a817c")
    best = int(j.loc[j.eps_all_median.idxmin(), "layer"])
    ax.axvline(best, color="#2a9d8f", lw=2, alpha=.6)
    ax.text(best, ax.get_ylim()[1], f" cheapest: {best}", rotation=90, va="top",
            fontsize=8, color="#2a9d8f")
    ax.set_yscale("log")
    ax.set_xlabel("injection layer")
    ax.set_ylabel("certified budget eps* to flip the token (log)")
    ax.set_title(f"{ds}: control authority by layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_signed(ds, s, out):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    order = ["inert", "churn", "mover"]
    cnt = [int((s.population == p).sum()) for p in order]
    ax[0].bar(order, cnt, color=["#8a817c", "#e9c46a", "#2a9d8f"])
    for i, c in enumerate(cnt):
        ax[0].text(i, c, f"{c}\n({c / len(s):.0%})", ha="center", va="bottom",
                   fontsize=8)
    ax[0].set_ylabel("statements")
    ax[0].set_title(f"{ds}: what actually moved")

    nz = s[s.signed != 0]
    ax[1].hist(nz.signed, bins=np.linspace(-2, 2, 17), color="#264653")
    ax[1].axvline(0, color="k", lw=1)
    pos, neg = int((nz.signed > 0).sum()), int((nz.signed < 0).sum())
    ax[1].set_xlabel("signed steerability  v(+eps) - v(-eps)")
    ax[1].set_ylabel("statements")
    ax[1].set_title(f"movers only: {pos} with the direction, {neg} against")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_sens(ds, s, out, site=""):
    """E6. Left: the spread of realized gain across directions, which is claim 4 as a
    picture. Right: gain against budget per direction, which separates 'anisotropic but
    linear' (flat lines at different heights) from 'chaotic' (lines that bend)."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fracs = sorted(s.frac.unique())

    per = [s[s.frac == f].groupby("direction").gain.median().values for f in fracs]
    ax[0].boxplot(per, tick_labels=[f"{f:g}" for f in fracs])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("budget, as a fraction of the certified delta_cone")
    ax[0].set_ylabel("realized gain  ||dz|| / eps")
    ax[0].set_title(f"{ds} {site}: spread of gain across directions")

    # Random directions are the reference: what a direction that knows nothing about
    # this decision achieves. Named directions are drawn individually against it.
    for nm, grp in s.groupby("direction"):
        m = grp.groupby("frac").gain.median()
        rnd = str(nm).startswith("random")
        ax[1].plot(m.index, m.values, marker="o", ms=3,
                   color="#bfc0c0" if rnd else None, lw=1 if rnd else 1.8,
                   label=None if rnd else nm, zorder=1 if rnd else 2)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("budget fraction")
    ax[1].set_ylabel("realized gain (median over statements)")
    ax[1].set_title("flat = linear at any budget; bent = nonlinear")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_steer(ds, arms, out):
    """E1/E3/E4/E7. The behavioral table as a picture: hit rate on the target token
    against the budget, one panel per arm, one line per direction.

    `oracle` is drawn heavy because it is the CEILING, not a competitor. At the
    post-norm site the flip it buys is a closed-form theorem, so its curve reaching
    1.0 is what makes every other curve on the panel readable as a fraction of what
    was provably available. A panel whose oracle does not reach 1.0 is a panel whose
    other lines mean nothing, and the dashed guide at 1.0 is there to make that
    obvious at a glance rather than on a second reading.

    Directions do NOT all cover the same statements: `jtw_legacy` is skipped wherever
    a statement falls outside the reach_margins subsample, which on common_claim is
    110 of 200. Averaging a 90-statement arm against a 200-statement arm compares two
    different populations, so every panel is restricted to the statements every one of
    its directions actually ran, and the title says how many that is."""
    n = len(arms)
    if not n:
        print("[viz] no steer arms; skipping the steer figure")
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.9 * rows),
                           squeeze=False)
    for k, (tag, d) in enumerate(arms):
        a = ax[k // cols][k % cols]

        # Restrict to the statements common to every direction in THIS arm.
        sets = [set(g.stmt.unique()) for _, g in d.groupby("direction")]
        shared = set.intersection(*sets) if sets else set()
        full = max((len(s) for s in sets), default=0)
        d = d[d.stmt.isin(shared)]

        for nm, grp in d.groupby("direction"):
            m = grp.groupby("frac").hit_target.mean()
            orc = str(nm) == "oracle"
            a.plot(m.index, m.values, marker="o", ms=3.5,
                   color="#2a9d8f" if orc else None, lw=2.6 if orc else 1.6,
                   zorder=3 if orc else 2, label=str(nm))
        a.axhline(1.0, ls="--", lw=1, color="#bfc0c0")
        a.axvline(0.0, ls=":", lw=1, color="#bfc0c0")
        a.set_ylim(-0.03, 1.05)
        a.set_xlabel("budget, as a fraction of the certified delta_cone")
        a.set_ylabel("hit rate on the target token")
        note = f"n={len(shared)}" + (f" shared of {full}" if len(shared) < full else "")
        a.set_title(f"{ds} {tag}  ({note})", fontsize=9)
        a.legend(fontsize=7)
    for k in range(n, rows * cols):
        ax[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--sens-tag", default="",
                   help="which token_sens_<ds>_<tag>.csv to plot; default: all of them")
    a = p.parse_args()
    ds, sfx = a.dataset, (f"_{a.tag}" if a.tag else "")

    f = f"token_geom_{ds}{sfx}.csv"
    if os.path.exists(f):
        g = pd.read_csv(f)
        fig_budget(ds, g, f"plot_token_budget_{ds}{sfx}.png")
        fig_alpha(ds, g, f"plot_token_alpha_{ds}{sfx}.png")
    else:
        print(f"[viz] {f} missing; run token_geom.py")

    f = f"token_jac_{ds}.csv"
    if os.path.exists(f):
        used = []
        try:
            import json
            m = json.load(open(f"dct_meta_{ds}.json"))
            used = [(int(m["source_layer"]), "src (DCT)"),
                    (int(m["target_layer"]), "tgt (probe)")]
        except Exception:                                    # noqa: BLE001
            pass
        fig_layers(ds, pd.read_csv(f), f"plot_token_layers_{ds}.png", used)
    else:
        print(f"[viz] {f} missing; run token_jac.py")

    f = f"signed_steer_{ds}.csv"
    if os.path.exists(f):
        fig_signed(ds, pd.read_csv(f), f"plot_signed_steer_{ds}.png")
    else:
        print(f"[viz] {f} missing; run signed_steer_audit.py")

    # One figure per site: a postnorm spectrum and a layer:13 spectrum are different
    # experiments and averaging them would hide exactly the contrast E6 is after.
    import glob

    # All arms on ONE figure: the comparisons that matter are between arms (post-norm
    # vs prenorm vs layer, broadcast vs per-position, rp=1.0 vs rp=1.3), so splitting
    # them into separate files would hide exactly what the run was for.
    pre = f"token_steer_{ds}_"
    # load_steer_csv, not read_csv: the arms on disk span two header generations
    # (`readout_delta` before 2026-08-04, `tgt_minus_top_delta` after) and the old files
    # are deliberately not regenerated.
    arms = [(f[len(pre):-len(".csv")], ts.load_steer_csv(f))
            for f in sorted(glob.glob(f"{pre}*.csv"))]
    if arms:
        fig_steer(ds, arms, f"plot_token_steer_{ds}.png")
    else:
        print(f"[viz] no {pre}*.csv; run token_steer.py")

    pat = (f"token_sens_{ds}_{a.sens_tag}.csv" if a.sens_tag
           else f"token_sens_{ds}_*.csv")
    hits = sorted(glob.glob(pat))
    for f in hits:
        tag = f[len(f"token_sens_{ds}_"):-len(".csv")]
        fig_sens(ds, pd.read_csv(f), f"plot_token_sens_{ds}_{tag}.png", site=tag)
    if not hits:
        print(f"[viz] no {pat}; run token_sens.py")


if __name__ == "__main__":
    main()
