"""D1 analysis: does a clean window exist? (design spec section 7)

Reads dose_yesno_<ds>.csv, dose_meta_<ds>.json and, when it exists, judge_dose_<ds>.csv.
Writes dose_summary_<ds>.csv, dose_window_<ds>.csv, plot_d1_dose_<ds>.png,
plot_d1_margin_<ds>.png. CPU only, seconds to run.

    ./.venv/bin/python src/dose_analyze.py --dataset cities

THE PRE-REGISTERED DECISION RULE
--------------------------------
The spec commits to two branches in advance. Operationalised here, and fixed before any D1
data existed:

  A dose is a CLEAN WINDOW for a direction when all three hold at that dose:
    1. behaviour moved   : the direction's flip count (or judged FALSE count) is significantly
                           above rand_ctrl's at the SAME dose. Random is the control because
                           it holds the perturbation norm fixed and removes only the semantics.
    2. nothing broke     : the direction's INCOHERENT count is not significantly above the
                           unsteered baseline's.
    3. it survives Holm  : over every (direction, dose) cell in the family.

  PI IS RIGHT     if at least one clean window exists for at least one truth direction.
  NULL IS REAL    if none does, and flip and incoherence instead rise together.

Everything is tested against rand_ctrl rather than against zero, because S1's whole lesson is
that a bare rate with n = 24 or n = 32 cannot separate "the direction did something" from
"a vector of that size did something".

THE MARGIN TEST IS THE ONE WITH POWER
-------------------------------------
Flip rate at n = 24 cannot resolve an effect below 0.375 (S1). The paired margin can: d_margin
is a within-statement difference, so the Wilcoxon signed-rank test on 24 paired values detects
a consistent shift far smaller than one argmax flip. A direction that moves the verdict readout
monotonically with dose but never crosses the argmax boundary is a REAL finding, not a null,
and it is invisible to every measurement this project has made until now.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

ALPHA = 0.05
CONTROL = "rand_ctrl"
BASELINE = "baseline"
# S3's bracket on the behavioural transition: below 0.29 the naive directions are already at
# baseline, above 0.78 the committed arm had already saturated. Shaded on every plot.
UNSAMPLED_BAND = (0.29, 0.78)

COL_FLIP = "#4477aa"
COL_INCOH = "#cc6677"
COL_FALSE = "#ddaa33"
COL_BAND = "#cccccc"
COL_BUDGET = "#117733"
# Paul Tol muted, colourblind-safe, matching the palette the rest of the audit plots use.
# The two controls are deliberately grey and dashed: a control should not read as a series.
DIR_STYLE = {
    "sup_grad":          ("#4477aa", "-"),
    "sup_mean_diff":     ("#cc6677", "-"),
    "mag_u_gold":        ("#ddaa33", "-"),
    "jtw_mean_diff_tgt": ("#117733", "-"),
    "mag_resid_pc1":     ("#888888", "--"),
    "rand_ctrl":         ("#444444", ":"),
}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ------------------------------------------------------------------ per-cell counts
def yesno_counts(ds):
    """Per (direction, signed dose): flip count out of n, the paired margin shift, and the
    decomposition of the flip count that tells you what it actually measured.

    `flipped` (from src/mag/steer.py:88) is "the steered answer parses as yes/no AND differs
    from the baseline answer". When the unsteered answer did not parse, recorded as "?", that
    condition is satisfied by ANY parseable steered answer, including "yes" on a true
    statement. So the same integer counts two different events:

        k_parse_gain : baseline "?" -> steered "yes" or "no".  The model began answering.
        k_flip_true  : baseline "yes" -> steered "no", or "no" -> "yes".  The verdict reversed.

    Only the second is a truthfulness effect. The first is a formatting or compliance effect
    and would look identical in every flip rate this project has published, because no run
    before D1 logged the per-statement baseline answer needed to separate them."""
    df = pd.read_csv(f"dose_yesno_{ds}.csv")
    df["scale"] = df["scale"].astype(float)
    yn = {"yes", "no"}
    rows = []
    for (d, s), g in df.groupby(["direction", "scale"]):
        dm = g["d_margin"].astype(float).to_numpy()
        # Wilcoxon needs at least one nonzero difference; the baseline cell is all zeros.
        if np.any(dm != 0) and len(dm) >= 6:
            w_p = float(wilcoxon(dm, zero_method="wilcox", alternative="two-sided").pvalue)
        else:
            w_p = 1.0
        a = g["answer"].astype(str)
        b = g["base_answer"].astype(str)
        parse_gain = int(((~b.isin(yn)) & a.isin(yn)).sum())
        flip_true = int((b.isin(yn) & a.isin(yn) & (a != b)).sum())
        rows.append({"direction": d, "scale": s, "n_yesno": len(g),
                     "k_flip": int(g["flipped"].sum()),
                     "flip_rate": float(g["flipped"].mean()),
                     "k_parse_gain": parse_gain,
                     "k_flip_true": flip_true,
                     "base_parse_rate": float(b.isin(yn).mean()),
                     "mean_d_margin": float(dm.mean()),
                     "sd_d_margin": float(dm.std(ddof=1)) if len(dm) > 1 else 0.0,
                     "wilcoxon_p_raw": w_p})
    return pd.DataFrame(rows)


def judged_counts(ds):
    """Per (direction, signed dose): FALSE / INCOHERENT / TRUE counts out of the 32 free-form
    completions. Absent until judge_results.py has run, in which case D1 is analysable on the
    yes/no probe alone and the judged columns are simply missing."""
    path = f"judge_dose_{ds}.csv"
    if not os.path.exists(path):
        print(f"[D1] {path} not found; analysing the yes/no probe only. "
              f"Run judge_results.py --mode steer --steer-input dose_{ds}.csv "
              f"--steer-output {path} to add the free-form arm.")
        return None
    df = pd.read_csv(path)
    df["scale"] = df["scale"].astype(float)
    rows = []
    for (d, s), g in df.groupby(["direction", "scale"]):
        v = g["verdict"]
        rows.append({"direction": d, "scale": s, "n_judged": len(g),
                     "k_false": int((v == "FALSE").sum()),
                     "k_incoh": int((v == "INCOHERENT").sum()),
                     "false_rate": float((v == "FALSE").mean()),
                     "incoh_rate": float((v == "INCOHERENT").mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the decision rule
def two_by_two(k1, n1, k2, n2, alternative="greater"):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=alternative).pvalue)


def window_tests(sm, has_judged):
    """Condition 1 (moved vs the same-dose random control) and condition 2 (did not break vs
    the unsteered baseline), one row per (direction, dose). Holm is applied across every cell
    of each family separately: the two conditions ask different questions and pooling them
    would penalise both."""
    ctrl = sm[sm["direction"] == CONTROL].set_index("scale")
    base = sm[sm["direction"] == BASELINE]
    if base.empty:
        raise SystemExit("[D1] no baseline row in the dose files; cannot test condition 2.")
    b = base.iloc[0]

    rows = []
    for _, r in sm.iterrows():
        if r["direction"] in (CONTROL, BASELINE):
            continue
        c = ctrl.loc[r["scale"]] if r["scale"] in ctrl.index else None
        rec = {"direction": r["direction"], "scale": r["scale"]}

        # 1. behaviour moved, against the norm-matched random vector at the same dose
        rec["p_flip_vs_rand"] = (
            two_by_two(r["k_flip"], r["n_yesno"], c["k_flip"], c["n_yesno"])
            if c is not None else np.nan)
        if has_judged and not np.isnan(r.get("k_false", np.nan)):
            rec["p_false_vs_rand"] = (
                two_by_two(r["k_false"], r["n_judged"], c["k_false"], c["n_judged"])
                if c is not None else np.nan)
            # 2. nothing broke, against the unsteered baseline
            rec["p_incoh_vs_base"] = two_by_two(r["k_incoh"], r["n_judged"],
                                                b["k_incoh"], b["n_judged"])
        rows.append(rec)

    w = pd.DataFrame(rows)
    for col in ("p_flip_vs_rand", "p_false_vs_rand", "p_incoh_vs_base"):
        if col in w.columns:
            adj = "padj_" + col[2:]        # not str.replace: "flip_vs" contains "p_" too
            ok = w[col].notna()
            w[adj] = np.nan
            w.loc[ok, adj] = holm(w.loc[ok, col].to_numpy())
    return w


def classify_windows(w, has_judged):
    """A cell is a clean window when behaviour moved above the random control and coherence
    did not degrade below the unsteered baseline, both after Holm."""
    moved = w["padj_flip_vs_rand"] < ALPHA
    if has_judged and "padj_false_vs_rand" in w.columns:
        moved = moved | (w["padj_false_vs_rand"] < ALPHA)
        intact = ~(w["padj_incoh_vs_base"] < ALPHA)
    else:
        # Without the judged arm there is no coherence measurement, so condition 2 is
        # unevaluable rather than satisfied. Recorded honestly as such.
        intact = pd.Series(np.nan, index=w.index)
    w["moved"] = moved
    w["coherent"] = intact
    w["clean_window"] = moved & (intact == True)  # noqa: E712 - NaN must not count as clean
    return w


# ------------------------------------------------------------------ plots
def plot_dose(ds, sm, meta, path):
    """Small multiples, one per direction: flip rate and (when judged) FALSE and INCOHERENT
    rates against absolute relative magnitude. All three are rates in [0, 1], so they share
    one axis; no second y-scale is ever introduced."""
    dirs = [d for d in meta["directions"] if d in set(sm["direction"])]
    budget = meta["eps_star_mean_diff_tgt"] / meta["h_src_median"]
    ncol = 3
    nrow = int(np.ceil(len(dirs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    base = sm[sm["direction"] == BASELINE]
    has_j = "incoh_rate" in sm.columns and sm["incoh_rate"].notna().any()

    for ax, d in zip(axes, dirs):
        sub = sm[(sm["direction"] == d) & (sm["scale"] != 0)].copy()
        sub["rel"] = sub["scale"].abs()
        ax.axvspan(*UNSAMPLED_BAND, color=COL_BAND, alpha=0.45, zorder=0, lw=0)
        ax.axvline(budget, color=COL_BUDGET, ls=":", lw=1.4, zorder=1)
        for sign, ls, mk in ((+1, "-", "o"), (-1, "--", "s")):
            s = sub[np.sign(sub["scale"]) == sign].sort_values("rel")
            if s.empty:
                continue
            ax.plot(s["rel"], s["flip_rate"], ls, marker=mk, ms=4, lw=2,
                    color=COL_FLIP, label=f"flip ({'+' if sign > 0 else '-'})")
            if has_j:
                ax.plot(s["rel"], s["false_rate"], ls, marker=mk, ms=4, lw=2,
                        color=COL_FALSE, label=f"FALSE ({'+' if sign > 0 else '-'})")
                ax.plot(s["rel"], s["incoh_rate"], ls, marker=mk, ms=4, lw=2,
                        color=COL_INCOH, label=f"INCOHERENT ({'+' if sign > 0 else '-'})")
        if has_j and not base.empty:
            ax.axhline(float(base["incoh_rate"].iloc[0]), color=COL_INCOH, lw=1, alpha=0.55)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(d, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes[len(dirs):]:
        ax.set_visible(False)
    # x label only on the bottom row: sharex already hides the interior ticks
    for ax in axes[max(0, len(dirs) - ncol):len(dirs)]:
        ax.set_xlabel("relative magnitude  |alpha| / median ||h_src||")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("rate")
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(f"D1 dose-response: {ds} (layer {meta['layer']})\n"
                 f"shaded: band no experiment had sampled   dotted: certified budget "
                 f"rel={budget:.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[D1] wrote {path}")


def plot_margin(ds, sm, meta, path):
    """The high-power view: mean paired shift in the yes/no verdict margin against SIGNED dose.
    A truth direction should trace a monotone line through the origin with opposite signs on
    the two halves; a norm effect should be symmetric or flat."""
    dirs = [d for d in meta["directions"] if d in set(sm["direction"])]
    budget = meta["eps_star_mean_diff_tgt"] / meta["h_src_median"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0, color="#666666", lw=1)
    for r in (+1, -1):
        ax.axvspan(r * UNSAMPLED_BAND[0], r * UNSAMPLED_BAND[1],
                   color=COL_BAND, alpha=0.45, zorder=0, lw=0)
        ax.axvline(r * budget, color=COL_BUDGET, ls=":", lw=1.3, zorder=1)
    for d in dirs:
        sub = sm[sm["direction"] == d].sort_values("scale")
        if sub.empty:
            continue
        colour, ls = DIR_STYLE.get(d, ("#4477aa", "-"))
        se = sub["sd_d_margin"] / np.sqrt(sub["n_yesno"].clip(lower=1))
        ax.errorbar(sub["scale"], sub["mean_d_margin"], yerr=1.96 * se, ls=ls,
                    marker="o", ms=4, lw=1.8, capsize=2, color=colour, label=d)
    ax.set_xlabel("signed relative magnitude  alpha / median ||h_src||")
    ax.set_ylabel("mean paired shift in p(yes) - p(no)")
    ax.set_title(f"D1 verdict-margin dose-response: {ds}\n"
                 f"paired within statement, n = {int(sm['n_yesno'].max())}; "
                 f"resolves shifts far below one argmax flip", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[D1] wrote {path}")


# ------------------------------------------------------------------ main
def run(ds):
    meta = json.load(open(f"dose_meta_{ds}.json"))
    sm = yesno_counts(ds)
    j = judged_counts(ds)
    has_judged = j is not None
    if has_judged:
        sm = sm.merge(j, on=["direction", "scale"], how="outer")
    sm = sm.sort_values(["direction", "scale"]).reset_index(drop=True)
    sm.to_csv(f"dose_summary_{ds}.csv", index=False)
    print(f"[D1] wrote dose_summary_{ds}.csv ({len(sm)} cells)")

    w = classify_windows(window_tests(sm, has_judged), has_judged)
    # carry the rates alongside the p-values so the window file is readable on its own
    rate_cols = [c for c in ("flip_rate", "k_flip_true", "k_parse_gain", "false_rate",
                             "incoh_rate", "mean_d_margin") if c in sm.columns]
    w = w.merge(sm[["direction", "scale"] + rate_cols], on=["direction", "scale"], how="left")
    w.to_csv(f"dose_window_{ds}.csv", index=False)
    print(f"[D1] wrote dose_window_{ds}.csv")

    plot_dose(ds, sm, meta, f"plot_d1_dose_{ds}.png")
    plot_margin(ds, sm, meta, f"plot_d1_margin_{ds}.png")

    # What the flip count actually counted. Printed before the verdict, because it decides
    # how the verdict may be read.
    base_pr = float(sm["base_parse_rate"].max())
    tot_flip = int(sm.loc[sm["direction"] != BASELINE, "k_flip"].sum())
    tot_gain = int(sm.loc[sm["direction"] != BASELINE, "k_parse_gain"].sum())
    tot_true = int(sm.loc[sm["direction"] != BASELINE, "k_flip_true"].sum())
    print(f"\n[D1] flip decomposition: {tot_flip} flips = {tot_gain} parse gains "
          f"(baseline did not answer) + {tot_true} genuine yes/no reversals; "
          f"unsteered parse rate {base_pr:.0%}")
    if tot_flip and tot_gain > tot_true:
        print("     WARNING: the flip metric here is majority parseability, not truthfulness.")
        print("     A window that fires on flips but NOT on judged FALSE is a formatting")
        print("     effect. Read k_flip_true and the verdict margin instead. This applies to")
        print("     the committed mag_verdict_flips_*.csv, which uses the same definition and")
        print("     never logged the baseline answer needed to see it.")

    clean = w[w["clean_window"]]
    print(f"\n[D1] {ds}: {len(clean)} clean windows out of {len(w)} cells")
    if len(clean):
        print("     PI IS RIGHT branch: behaviour moved above the norm-matched control at")
        print("     doses where coherence was intact.")
        cols = [c for c in ("direction", "scale", "flip_rate", "k_flip_true", "k_parse_gain",
                            "false_rate", "incoh_rate", "padj_flip_vs_rand",
                            "padj_incoh_vs_base") if c in clean.columns]
        print(clean[cols].to_string(index=False))
    else:
        print("     NULL IS REAL branch on the discrete outcomes: no dose moved behaviour")
        print("     above a random vector of the same norm while coherence held.")

    # The margin arm is reported separately: it can be positive when the discrete arm is null,
    # and that combination is the most informative outcome D1 can produce.
    mv = sm[(sm["direction"] != BASELINE) & (sm["wilcoxon_p_raw"] < ALPHA)]
    print(f"\n[D1] verdict margin moved (raw p < {ALPHA}) in {len(mv)} of "
          f"{len(sm) - 1} cells")
    if len(mv):
        top = mv.reindex(mv["mean_d_margin"].abs().sort_values(ascending=False).index).head(8)
        print(top[["direction", "scale", "mean_d_margin", "flip_rate",
                   "wilcoxon_p_raw"]].to_string(index=False))
    return sm, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
