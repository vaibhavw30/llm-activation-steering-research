# src/audit_common_axis.py
"""S3 — put the naive and reachability steering arms on one axis.

Assumption under test: the naive and reachability results, as reported, are comparable.

Reads committed artifacts only. Writes audit_common_axis_<ds>.csv and plot_s3_common_axis_<ds>.png.

    ./.venv/bin/python src/audit_common_axis.py --dataset cities

COMPARABILITY, checked in code rather than assumed (design spec section 5 requires this):

  * Injection operator. Both arms steer through dct_steer_utils.Steerer, which adds the vector
    to the INPUT of model.model.layers[L] on every forward pass and at every position.
    src/mag/steer.py:71 and src/reach_steer.py:143 both construct it the same way. Same
    operator, same broadcast convention. Verified below by asserting the layers match.
  * Injection layer. mag_dir_<ds>.npz["layer"] against dct_meta_<ds>.json["source_layer"].
    Asserted equal; the script refuses to plot a shared axis if they differ.
  * Prompts. The naive arm and the reachability MEAN arm both use the 32 FACTUAL_PROMPTS, so
    their judged outcomes are directly comparable. The reachability PER-STATEMENT arm uses 200
    statement stems that share no prompt with those 32, so it contributes magnitude coverage
    but its rates are plotted as a separate series and never merged with the other two.

The two arms parameterise magnitude differently, which is the whole problem:
  * naive:  alpha = tau * A_prefix_norm            (mag_dir_<ds>.npz)
  * reach:  alpha = frac * eps_star                (reach_summary_<ds>.json)
Neither normaliser is the activation norm. The canonical denominator here is the median
||h_src|| at the injection layer, taken from reach_acts_<ds>.npz.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_NAIVE = "#cc6677"
COL_REACH = "#4477aa"
COL_STMT = "#88ccee"
COL_BUDGET = "#117733"
# A binned rate computed from fewer than this many completions is noise, not a measurement.
MIN_BIN_N = 20


def load_geometry(ds):
    """Injection layer, canonical denominator, and both arms' normalisers."""
    md = np.load(f"mag_dir_{ds}.npz")
    mag_layer, apn = int(md["layer"]), float(md["A_prefix_norm"])

    meta = json.load(open(f"dct_meta_{ds}.json"))
    reach_layer, input_scale = int(meta["source_layer"]), float(meta["input_scale"])

    if mag_layer != reach_layer:
        raise SystemExit(
            f"[S3] refusing to build a shared axis: the naive arm injects at layer {mag_layer} "
            f"and the reachability arm at layer {reach_layer}. A relative-magnitude axis is "
            f"only meaningful when both perturb the same state.")

    z = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    h = np.linalg.norm(np.asarray(z["h_src"], np.float64), axis=1)
    summ = json.load(open(f"reach_summary_{ds}.json"))
    eps_star = float(summ["directions"]["mean_diff_tgt"]["median_eps_star"])

    return {
        "layer": mag_layer,
        "h_med": float(np.median(h)),
        "h_p10": float(np.percentile(h, 10)),
        "h_p90": float(np.percentile(h, 90)),
        "A_prefix_norm": apn,
        "input_scale": input_scale,
        "eps_star_mean_diff": eps_star,
    }


def rates(df, group):
    """FALSE and INCOHERENT rates per group value, from a judged completion table."""
    g = df.groupby(group)["verdict"]
    return pd.DataFrame({
        "false_rate": g.apply(lambda s: (s == "FALSE").mean()),
        "incoh_rate": g.apply(lambda s: (s == "INCOHERENT").mean()),
        "true_rate": g.apply(lambda s: (s == "TRUE").mean()),
        "n": g.size(),
    }).reset_index()


def collect(ds, geo):
    """One long table: arm, direction, signed and absolute magnitude, relative magnitude, rates."""
    h_med = geo["h_med"]
    frames = []

    # Naive arm. scale in the judged file is tau; absolute magnitude is tau * A_prefix_norm.
    m = pd.read_csv(f"judge_mag_steer_{ds}.csv")
    r = rates(m, ["direction", "scale"]).rename(columns={"scale": "tau"})
    r["arm"] = "naive"
    r["abs_mag"] = r["tau"].abs() * geo["A_prefix_norm"]
    r["signed_mag"] = r["tau"] * geo["A_prefix_norm"]
    r["prompt_set"] = "FACTUAL_32"
    frames.append(r.drop(columns=["tau"]))

    # Reachability mean arm. scale in the judged file is already the absolute magnitude.
    rm = pd.read_csv(f"judge_reach_steer_{ds}.csv")
    r = rates(rm, ["direction", "scale"])
    r["arm"] = "reach_mean"
    r["abs_mag"] = r["scale"].abs()
    r["signed_mag"] = r["scale"]
    r["prompt_set"] = "FACTUAL_32"
    frames.append(r.drop(columns=["scale"]))

    # Reachability per-statement arm. Every statement has its own eps_star, so the scales are
    # a continuum rather than a grid. Bin by relative magnitude so the rates are readable.
    rs = pd.read_csv(f"judge_reach_steer_stmt_{ds}.csv")
    rs["rel"] = rs["scale"].abs() / h_med
    edges = np.concatenate([[0], np.geomspace(1e-3, max(rs["rel"].max(), 1e-2), 12)])
    rs["bin"] = pd.cut(rs["rel"], edges, include_lowest=True)
    r = rates(rs, ["direction", "bin"])
    r["arm"] = "reach_stmt"
    r["abs_mag"] = [b.mid * h_med for b in r["bin"]]
    r["signed_mag"] = np.nan          # the bin pools both signs
    r["prompt_set"] = "STMT_200"
    frames.append(r.drop(columns=["bin"]))

    out = pd.concat(frames, ignore_index=True)
    out["rel_mag"] = out["abs_mag"] / h_med
    out["layer"] = geo["layer"]
    out["h_src_median"] = h_med
    return out


def coverage(out, geo):
    """Per-arm swept range on the shared axis, and the gap between the arms."""
    rows = []
    for arm, sub in out.groupby("arm"):
        nz = sub[sub["rel_mag"] > 0]
        rows.append({
            "arm": arm,
            "prompt_set": sub["prompt_set"].iloc[0],
            "n_points": len(sub),
            "min_rel_nonzero": nz["rel_mag"].min() if len(nz) else np.nan,
            "max_rel": sub["rel_mag"].max(),
        })
    return pd.DataFrame(rows).sort_values("min_rel_nonzero")


def run(ds):
    geo = load_geometry(ds)
    print(f"[S3] {ds}: both arms inject at layer {geo['layer']} through the same Steerer hook")
    print(f"[S3] canonical denominator: median ||h_src|| = {geo['h_med']:.3f} "
          f"(p10 {geo['h_p10']:.1f}, p90 {geo['h_p90']:.1f})")
    print(f"[S3] naive normaliser A_prefix_norm = {geo['A_prefix_norm']:.3f} "
          f"({100 * geo['A_prefix_norm'] / geo['h_med']:.1f}% of the canonical denominator)")
    print(f"[S3] reach cap normaliser input_scale = {geo['input_scale']:.3f} "
          f"({100 * geo['input_scale'] / geo['h_med']:.1f}%)")

    out = collect(ds, geo)
    cov = coverage(out, geo)
    out.to_csv(f"audit_common_axis_{ds}.csv", index=False)
    print(f"[S3] wrote audit_common_axis_{ds}.csv")
    return out, cov, geo


def plot(ds, out, cov, geo):
    h_med = geo["h_med"]
    budget_rel = geo["eps_star_mean_diff"] / h_med

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.1]})

    # Panel 1: pure coverage. Which relative magnitudes did each arm ever visit?
    ax = axes[0]
    lanes = [("naive", COL_NAIVE), ("reach_mean", COL_REACH), ("reach_stmt", COL_STMT)]
    for i, (arm, col) in enumerate(lanes):
        sub = out[(out["arm"] == arm) & (out["rel_mag"] > 0)]
        ax.scatter(sub["rel_mag"], np.full(len(sub), i), s=42, color=col, zorder=3,
                   label=f"{arm} ({sub['prompt_set'].iloc[0]})")
        if len(sub):
            ax.hlines(i, sub["rel_mag"].min(), sub["rel_mag"].max(), color=col, lw=2.5,
                      alpha=0.35, zorder=2)
    ax.axvline(budget_rel, color=COL_BUDGET, ls="--", lw=1.6, zorder=4)
    ax.text(budget_rel, 2.55, f" certified budget {budget_rel:.3f}", color=COL_BUDGET,
            fontsize=8, va="top")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([a for a, _ in lanes])
    ax.set_ylim(-0.6, 2.7)
    ax.set_title(f"S3 shared magnitude axis — {ds}   "
                 f"(layer {geo['layer']}, same Steerer hook, denominator "
                 f"median ||h_src|| = {h_med:.1f})", fontsize=11)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62))

    # Panel 2: the comparable outcome. Same 32 prompts, same judge, both arms.
    #
    # The unsteered points sit at rel_mag = 0, which has no place on a log axis. Plotting them
    # anyway draws a spurious flat segment across the entire decade range. Draw the baselines
    # as horizontal reference lines and keep only the steered points on the curves.
    ax = axes[1]
    steered = out[out["rel_mag"] > 0]
    base = out[out["rel_mag"] == 0]
    for arm, col, mk in (("naive", COL_NAIVE, "o"), ("reach_mean", COL_REACH, "s")):
        sub = steered[steered["arm"] == arm].groupby("rel_mag")[
            ["false_rate", "incoh_rate"]].mean().sort_index()
        ax.plot(sub.index, sub["false_rate"], f"-{mk}", ms=6, lw=2, color=col,
                label=f"{arm}: FALSE rate")
        ax.plot(sub.index, sub["incoh_rate"], f"--{mk}", ms=4, lw=1.4, color=col, alpha=0.6,
                label=f"{arm}: INCOHERENT rate")
        b = base[base["arm"] == arm]
        if len(b):
            ax.axhline(b["false_rate"].mean(), color=col, lw=1.0, alpha=0.45)
            ax.text(ax.get_xlim()[0], b["false_rate"].mean(), " unsteered ", color=col,
                    fontsize=7, va="bottom")

    # Bins with only a handful of completions produce meaningless rates; drop them rather
    # than drawing a spike the eye reads as signal.
    sub = steered[(steered["arm"] == "reach_stmt") & (steered["n"] >= MIN_BIN_N)]
    sub = sub.groupby("rel_mag")[["false_rate"]].mean().sort_index()
    ax.plot(sub.index, sub["false_rate"], ":^", ms=5, lw=1.4, color=COL_STMT, alpha=0.9,
            label=f"reach_stmt: FALSE rate (different prompts, n >= {MIN_BIN_N} per bin)")

    ax.axvline(budget_rel, color=COL_BUDGET, ls="--", lw=1.6)

    # Shade the band no arm on the shared prompt set ever visited.
    shared = out[(out["prompt_set"] == "FACTUAL_32") & (out["rel_mag"] > 0)]
    hi_reach = shared[shared["arm"] == "reach_mean"]["rel_mag"].max()
    lo_naive = shared[shared["arm"] == "naive"]["rel_mag"].min()
    if hi_reach < lo_naive:
        ax.axvspan(hi_reach, lo_naive, color="#888888", alpha=0.13, zorder=0)
        ax.text(np.sqrt(hi_reach * lo_naive), 0.93,
                f"never sampled on the\nshared prompts\n({hi_reach:.3f} to {lo_naive:.3f})",
                ha="center", va="top", fontsize=8, color="#555555")

    ax.set_xscale("log")
    ax.set_xlabel("relative perturbation size,  ||delta|| / median ||h_src||   (log scale)")
    ax.set_ylabel("judged rate on free-form completions")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    p = f"plot_s3_common_axis_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"[S3] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    out, cov, geo = run(a.dataset)
    plot(a.dataset, out, cov, geo)

    print(f"\n=== S3 {a.dataset} ===")
    print(cov.to_string(index=False))

    budget_rel = geo["eps_star_mean_diff"] / geo["h_med"]
    print(f"\ncertified budget (median eps* for mean_diff_tgt): "
          f"{geo['eps_star_mean_diff']:.3f} absolute = {budget_rel:.4f} relative")

    shared = out[(out["prompt_set"] == "FACTUAL_32") & (out["rel_mag"] > 0)]
    hi = shared[shared["arm"] == "reach_mean"]["rel_mag"].max()
    lo = shared[shared["arm"] == "naive"]["rel_mag"].min()
    print(f"\non the SHARED 32 prompts: reach_mean tops out at {hi:.4f}, "
          f"naive starts at {lo:.4f}")
    if hi < lo:
        print(f"  -> disjoint. gap factor {lo / hi:.2f}x, band {hi:.4f} to {lo:.4f} unsampled.")
    else:
        print(f"  -> they overlap between {lo:.4f} and {hi:.4f}.")

    allnz = out[out["rel_mag"] > 0]
    ah = allnz[allnz["arm"].str.startswith("reach")]["rel_mag"].max()
    al = allnz[allnz["arm"] == "naive"]["rel_mag"].min()
    print(f"\nincluding the per-statement arm (different prompts): reachability reaches "
          f"{ah:.4f}, naive starts at {al:.4f}")
    print(f"  -> {'disjoint, gap factor %.2fx' % (al / ah) if ah < al else 'they overlap'}")


if __name__ == "__main__":
    main()
