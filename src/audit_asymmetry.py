# src/audit_asymmetry.py
"""S1 — direction asymmetry audit.

Assumption under test: behavioral flips attributed to steering are directional effects of a
truth-carrying vector, not magnitude effects of an arbitrary perturbation.

Reads committed artifacts only. Writes audit_asymmetry_<ds>.csv and plot_s1_asymmetry_<ds>.png.

    ./.venv/bin/python src/audit_asymmetry.py --dataset cities

Two prompt sets are involved and they are NOT the same set:
  * flip_rate in mag_verdict_flips_<ds>.csv is measured on the 24 matched-format yes/no
    statements (src/mag/steer.py:23), scored as "answer differs from the tau=0 baseline and is
    not unparseable". It is direction-agnostic about semantics: it counts a changed answer,
    not a turn toward falsehood.
  * verdict in judge_mag_steer_<ds>.csv is measured on the 32 free-form factual stems
    (src/steer_supervised.py:31), judged TRUE / FALSE / INCOHERENT.
Incoherence is therefore a companion measurement, not a decomposition of the flip rate.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# Number of yes/no statements behind every flip_rate. Defined at src/mag/steer.py:23.
# Asserted against the data in load_flips() so the constant cannot silently drift.
N_YESNO = 24

# Pre-registered classification thresholds (section 3 of the design spec).
INERT_MAX_FLIP = 0.10   # peak flip rate at or below this over all nonzero tau => inert
ALPHA = 0.05            # family-wise error rate, Holm-corrected within a dataset

COL_POS = "#4477aa"
COL_NEG = "#cc6677"
COL_INCOH = "#999933"


def load_flips(ds):
    """flip_rate table as direction x tau, with counts recovered and integrality asserted."""
    df = pd.read_csv(f"mag_verdict_flips_{ds}.csv")
    counts = df["flip_rate"].to_numpy() * N_YESNO
    off = np.abs(counts - np.round(counts))
    if off.max() > 1e-6:
        raise ValueError(
            f"flip_rate values are not multiples of 1/{N_YESNO} "
            f"(max residual {off.max():.3g}); N_YESNO is wrong for {ds}"
        )
    df["k"] = np.round(counts).astype(int)
    return df


def load_incoherence(ds):
    """Incoherence and FALSE rate per (direction, tau) from the judged free-form completions."""
    path = f"judge_mag_steer_{ds}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    g = df.groupby(["direction", "scale"])["verdict"]
    out = pd.DataFrame({
        "incoh": g.apply(lambda s: (s == "INCOHERENT").mean()),
        "false": g.apply(lambda s: (s == "FALSE").mean()),
        "n_judged": g.size(),
    }).reset_index().rename(columns={"scale": "tau"})
    return out


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def classify(sub):
    """Direction-level verdict from its per-magnitude rows.

    inert       peak flip rate over all nonzero tau <= INERT_MAX_FLIP
    directional not inert, and at least one magnitude shows Holm-significant +tau vs -tau
    symmetric   not inert, no magnitude shows a significant asymmetry
    """
    peak = max(sub["flip_pos"].max(), sub["flip_neg"].max())
    if peak <= INERT_MAX_FLIP:
        return "inert"
    if bool(sub["sig_holm"].any()):
        return "directional"
    return "symmetric"


def run(ds):
    flips = load_flips(ds)
    incoh = load_incoherence(ds)

    taus = sorted(flips["tau"].unique())
    mags = sorted({abs(t) for t in taus if t != 0})
    piv = flips.pivot(index="direction", columns="tau", values="flip_rate")
    kpiv = flips.pivot(index="direction", columns="tau", values="k")

    rows = []
    for d in piv.index:
        for m in mags:
            if m not in piv.columns or -m not in piv.columns:
                continue
            kp, kn = int(kpiv.loc[d, m]), int(kpiv.loc[d, -m])
            # Unpaired Fisher exact on [[flip+, no-flip+], [flip-, no-flip-]].
            # The per-statement outcomes were not persisted (only the rate), so the paired
            # McNemar test is unavailable. Fisher on the marginals is the conservative
            # substitute; it under-detects real asymmetry rather than inventing it.
            _, p = fisher_exact([[kp, N_YESNO - kp], [kn, N_YESNO - kn]], alternative="two-sided")
            rows.append({
                "direction": d,
                "abs_tau": m,
                "flip_pos": piv.loc[d, m],
                "flip_neg": piv.loc[d, -m],
                "k_pos": kp,
                "k_neg": kn,
                "n_yesno": N_YESNO,
                "asym": abs(piv.loc[d, m] - piv.loc[d, -m]),
                "fisher_p": p,
            })
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out["fisher_p"].to_numpy())
    out["sig_holm"] = out["holm_p"] < ALPHA

    # Non-monotonicity: within one sign, does the flip rate fall as |tau| grows?
    # A dose-response that decreases with dose is crossing a regime change, not dosing.
    lo, hi = mags[0], mags[-1]
    for sign, col in ((1, "flip_pos"), (-1, "flip_neg")):
        key = "nonmono_pos" if sign > 0 else "nonmono_neg"
        flag = {d: bool(piv.loc[d, sign * hi] < piv.loc[d, sign * lo]) for d in piv.index}
        out[key] = out["direction"].map(flag)

    if incoh is not None:
        base = incoh[incoh["tau"] == 0].set_index("direction")["incoh"]
        for sign, name in ((1, "incoh_pos"), (-1, "incoh_neg")):
            lut = {(r.direction, abs(r.tau)): r.incoh
                   for r in incoh.itertuples() if np.sign(r.tau) == sign}
            out[name] = [lut.get((r.direction, r.abs_tau), np.nan) for r in out.itertuples()]
        out["incoh_base"] = out["direction"].map(base)

    cls = {d: classify(out[out["direction"] == d]) for d in out["direction"].unique()}
    out["classification"] = out["direction"].map(cls)

    # Post-hoc diagnostic, NOT part of the pre-registered classification rule.
    # A direction whose only significant asymmetry sits at the largest magnitude is asymmetric
    # in a saturated regime, where flip rates are pinned near 0 or 1 and the measurement cannot
    # distinguish a truth effect from a regime change. Significance at the smallest swept
    # magnitude is the stronger evidence, and it is what D1 needs.
    at_min = out[out["abs_tau"] == lo].set_index("direction")["sig_holm"]
    out["sig_at_min_mag"] = out["direction"].map(at_min)
    sat = out.groupby("direction").apply(
        lambda s: bool(((s["flip_pos"] >= 0.95) | (s["flip_pos"] <= 0.05)).all()
                       and ((s["flip_neg"] >= 0.95) | (s["flip_neg"] <= 0.05)).all()),
        include_groups=False,
    )
    out["saturated"] = out["direction"].map(sat)

    path = f"audit_asymmetry_{ds}.csv"
    out.to_csv(path, index=False)
    print(f"[S1] wrote {path}")
    return out, piv, incoh, cls


def plot(ds, out, piv, incoh, cls):
    dirs = list(piv.index)
    mags = sorted(out["abs_tau"].unique())
    hi = mags[-1]
    y = np.arange(len(dirs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(dirs) + 3.2),
                             gridspec_kw={"width_ratios": [2.4, 2.4, 1.8]})

    tag = {"directional": "D", "symmetric": "S", "inert": "I"}
    labels = [f"{d}  [{tag[cls[d]]}]" for d in dirs]

    for ax, m in zip(axes[:2], mags):
        sub = out[out["abs_tau"] == m].set_index("direction").loc[dirs]
        ax.barh(y - 0.19, sub["flip_pos"], height=0.36, color=COL_POS, label=f"tau = +{m}")
        ax.barh(y + 0.19, sub["flip_neg"], height=0.36, color=COL_NEG, label=f"tau = -{m}")
        for i, r in enumerate(sub.itertuples()):
            if r.sig_holm:
                ax.text(max(r.flip_pos, r.flip_neg) + 0.03, i, "*", va="center",
                        fontsize=13, color="k")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("yes/no flip rate (n=24)")
        ax.set_title(f"|tau| = {m}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    ax = axes[2]
    if incoh is not None:
        # The tau=0 completions are generated with the steering hook disabled, so the baseline
        # is one number shared by every direction. Draw it once as a reference line rather than
        # repeating an identical bar nine times.
        b = incoh[incoh["tau"] == 0]["incoh"]
        h = incoh[incoh["tau"].abs() == hi].groupby("direction")["incoh"].mean().reindex(dirs)
        ax.barh(y, h, height=0.5, color=COL_INCOH, label=f"|tau| = {hi}")
        ax.axvline(b.mean(), color="#333333", ls="--", lw=1.0,
                   label=f"tau = 0 baseline ({b.mean():.3f})")
        ax.set_xlabel("INCOHERENT rate (n=32 stems)")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_title("degradation")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(f"S1 direction asymmetry — {ds}   (D directional, S symmetric, I inert; "
                 f"* Holm p < {ALPHA})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"plot_s1_asymmetry_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"[S1] wrote {p}")


def detection_floor(n_tests):
    """Smallest flip-count difference against 0 that survives Holm at ALPHA with n_tests tests.

    States the power of the design up front so a null asymmetry is not read as evidence of
    symmetry when the test could not have detected one.
    """
    for k in range(1, N_YESNO + 1):
        _, p = fisher_exact([[k, N_YESNO - k], [0, N_YESNO]], alternative="two-sided")
        if p * n_tests < ALPHA:
            return k, p
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    out, piv, incoh, cls = run(a.dataset)
    plot(a.dataset, out, piv, incoh, cls)

    print(f"\n=== S1 {a.dataset} ===")
    show = ["direction", "abs_tau", "flip_pos", "flip_neg", "asym", "fisher_p",
            "holm_p", "sig_holm", "nonmono_pos", "nonmono_neg", "classification"]
    print(out[show].to_string(index=False))
    tally = {k: sum(1 for v in cls.values() if v == k) for k in ("directional", "symmetric", "inert")}
    print(f"\nclassification tally: {tally}")

    k, p = detection_floor(len(out))
    print(f"detection floor: {len(out)} tests, Holm at {ALPHA}; smallest detectable asymmetry "
          f"vs 0 is {k}/{N_YESNO} = {k / N_YESNO:.3f} (raw p={p:.2e}). "
          f"A null below that is underpowered, not symmetric.")

    surv = sorted(d for d, v in cls.items() if v == "directional")
    print(f"\ndirections classified directional: {surv if surv else 'NONE'}")
    strong = sorted(out.loc[out["sig_at_min_mag"].fillna(False), "direction"].unique())
    print(f"of those, significant at the smallest magnitude (|tau|={out.abs_tau.min()}): "
          f"{strong if strong else 'NONE'}")
    satd = sorted(out.loc[out["saturated"], "direction"].unique())
    print(f"saturated at every swept magnitude (flip rate pinned near 0 or 1): "
          f"{satd if satd else 'NONE'}")


if __name__ == "__main__":
    main()
