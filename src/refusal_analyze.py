"""refusal_analyze.py: Horizon-1 Track A: the paired statistics behind the
per-statement refusal control's `actuatable` verdict.

`reach_control.py` names the verdict cell. It does that from BUCKET MEANS, which is
the right call for a verdict but the wrong basis for an effect size, because the
frac buckets in the per-statement arm hold DIFFERENT STATEMENTS: `scale_grid`'s
`min(f*eps_i, 1.5*input_scale)` clamp (reach_steer.py:41) means only the statements
with a small enough eps_i ever reach |frac| = 2. Comparing the frac -2 bucket (n=64)
against the frac 0 bucket (n=199) therefore compares 64 near-boundary statements
against a 199-statement population that mostly is not them, and the resulting odds
ratio is a mixture of the dose effect and that selection.

Everything here is COMPLETE-CASE PAIRED instead: restrict to the statements observed
at every frac in `CORE_FRACS`, and compare a statement against itself. That makes
three things available that the bucket table cannot express:

  1. a sign control: the +2 arm is the SAME statements at the SAME |perturbation|
     in the opposite direction, so it holds the prompt set, the norm, the layer, and
     the direction family fixed and varies only the sign;
  2. McNemar: how many statements were flipped INTO refusal from a compliant
     baseline, versus how many were flipped out, which is the quantity the positive
     control is actually about;
  3. a mechanism test: Mantel-Haenszel on `crossed the certificate boundary` while
     STRATIFYING on perturbation magnitude, which separates "the readout crossed"
     from "the activation was pushed hard".

Reads only. Writes refusal_control_stats_<ds>.json.

    PYTHONPATH=src .venv/bin/python src/refusal_analyze.py --dataset refusal
"""
import argparse
import csv
import json

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm

from reach_control import align_stmt_rows

CORE_FRACS = (-2.0, -1.0, 0.0, 1.0, 2.0)
BASELINE_FRAC = 0.0
FRAC_ROUND = 2
# Splits the steered rows into "about one eps*" and "about two eps*". The point is to
# hold perturbation magnitude roughly fixed WITHIN a stratum so the Mantel-Haenszel OR
# for crossing is not just re-measuring dose. The cut sits at 1.2 rather than 1.5
# because the clamp produces a cluster of idiosyncratic fracs in [1.4, 2.0] that
# belong with the two-eps* group.
MAG_STRATA = ((0.0, 1.2), (1.2, 2.1))


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paired_frame(rows, fracs=CORE_FRACS, ndigits=FRAC_ROUND):
    """Complete-case pivot: {stmt_index: {frac: (g_read, refused)}} keeping only the
    statements observed at EVERY frac in `fracs`. Dropping the incomplete statements
    is what makes every comparison below within-statement; it is also why n here
    (64) is smaller than any single bucket's n in reach_control's table."""
    want = {round(float(f), ndigits) for f in fracs}
    by_stmt = {}
    for r in rows:
        f = round(float(r["frac"]), ndigits)
        if f in want:
            by_stmt.setdefault(int(r["stmt_index"]), {})[f] = (float(r["g_read"]),
                                                               float(r["refused"]))
    return {i: d for i, d in by_stmt.items() if want <= set(d)}


def dose_table(frame, fracs=CORE_FRACS):
    """One row per frac over the SAME statements: mean g, how many individually
    crossed into the target halfspace, and the refusal count."""
    out = []
    for f in fracs:
        g = [frame[i][f][0] for i in frame]
        r = [frame[i][f][1] for i in frame]
        out.append({"frac": float(f), "n": len(g), "mean_g_read": float(np.mean(g)),
                    "n_crossed": int(sum(1 for v in g if v <= 0.0)),
                    "n_refused": int(sum(r)),
                    "frac_refused": float(np.mean(r)) if r else 0.0})
    return out


def mcnemar(frame, f_from, f_to):
    """Paired discordance between two fracs. `gained` = compliant at f_from and
    refusing at f_to; `lost` = the reverse. The exact binomial on the discordant
    pairs is the whole test: concordant pairs carry no information about a change,
    and with `lost` at 0 no unpaired test can express how one-directional the
    movement is."""
    gained = sum(1 for i in frame
                 if frame[i][f_from][1] == 0 and frame[i][f_to][1] == 1)
    lost = sum(1 for i in frame
               if frame[i][f_from][1] == 1 and frame[i][f_to][1] == 0)
    n = gained + lost
    p = float(binomtest(gained, n, 0.5).pvalue) if n else 1.0
    return {"from_frac": float(f_from), "to_frac": float(f_to), "gained": gained,
            "lost": lost, "n_discordant": n, "p_exact": p,
            "n_eligible_to_gain": sum(1 for i in frame if frame[i][f_from][1] == 0),
            "n_eligible_to_lose": sum(1 for i in frame if frame[i][f_from][1] == 1)}


def two_by_two(frame, f_a, f_b):
    """Unpaired Fisher between two fracs of the same statement set. Reported
    alongside McNemar because it is the effect size a reader expects; McNemar is
    the test that respects the pairing."""
    a = int(sum(frame[i][f_a][1] for i in frame))
    b = int(sum(frame[i][f_b][1] for i in frame))
    n = len(frame)
    odds, p = fisher_exact([[a, n - a], [b, n - b]])
    return {"frac_a": float(f_a), "frac_b": float(f_b), "n": n, "n_refused_a": a,
            "n_refused_b": b, "odds_ratio": float(odds), "p_fisher": float(p)}


def cochran_armitage(rows, fracs=CORE_FRACS, ndigits=FRAC_ROUND):
    """Trend of refusal across dose on the FULL pooled sample (unbalanced n per
    frac). Scores are the frac values themselves, so the slope is per unit eps*.
    Kept on the pooled sample rather than the paired frame because a trend test is
    the one place the extra statements at |frac| <= 1 add power without the
    selection worry: the question is monotonicity, not a between-bucket contrast."""
    want = {round(float(f), ndigits) for f in fracs}
    x, y = [], []
    for r in rows:
        f = round(float(r["frac"]), ndigits)
        if f in want:
            x.append(f)
            y.append(float(r["refused"]))
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or y.std() == 0 or x.std() == 0:
        return {"n": int(len(x)), "z": 0.0, "p": 1.0, "slope_per_eps_star": 0.0}
    xb, yb = x.mean(), y.mean()
    num = float(((x - xb) * (y - yb)).sum())
    var = float(yb * (1 - yb) * ((x - xb) ** 2).sum())
    z = num / np.sqrt(var)
    slope = num / float(((x - xb) ** 2).sum())
    return {"n": int(len(x)), "z": float(z), "p": float(2 * norm.sf(abs(z))),
            "slope_per_eps_star": float(slope)}


def crossing_vs_magnitude(rows, strata=MAG_STRATA):
    """Mantel-Haenszel OR for `g_read <= 0` (the readout crossed the certificate
    boundary) as a predictor of refusal, stratified by |frac| so perturbation
    magnitude is held roughly fixed within each stratum.

    This is the mechanism test. The certificate's claim is not "a large enough push
    changes behaviour" but "pushing PAST THIS BOUNDARY changes behaviour". Without
    the stratification the two are confounded, because bigger pushes cross more
    often. Baseline (unsteered) rows are excluded: their g is positive by
    construction for every statement that has a boundary at all, so including them
    would load the not-crossed cell with rows that were never perturbed."""
    per, num, den = [], 0.0, 0.0
    for lo, hi in strata:
        cell = [r for r in rows
                if abs(float(r["frac"])) > lo and abs(float(r["frac"])) <= hi
                and abs(float(r["frac"])) > 1e-12]
        a = sum(1 for r in cell if r["g_read"] <= 0 and r["refused"] == 1)
        b = sum(1 for r in cell if r["g_read"] <= 0 and r["refused"] == 0)
        c = sum(1 for r in cell if r["g_read"] > 0 and r["refused"] == 1)
        d = sum(1 for r in cell if r["g_read"] > 0 and r["refused"] == 0)
        n = a + b + c + d
        if n == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]])
        per.append({"abs_frac_lo": lo, "abs_frac_hi": hi, "n": n,
                    "crossed_refused": a, "crossed_total": a + b,
                    "not_crossed_refused": c, "not_crossed_total": c + d,
                    "odds_ratio": float(odds), "p_fisher": float(p)})
        num += a * d / n
        den += b * c / n
    return {"strata": per,
            "mh_odds_ratio": float(num / den) if den else float("inf")}


def run(ds):
    judged = _read(f"judge_refusal_{ds}_stmt.csv")
    meta = _read(f"reach_steer_stmt_meta_{ds}.csv")
    rows, n_never = align_stmt_rows(judged, meta)
    frame = paired_frame(rows)
    if not frame:
        raise SystemExit(f"[refusal] no statement in {ds} was observed at every frac "
                         f"in {CORE_FRACS}, so nothing is paired: none of these "
                         f"statistics are defined")
    lo, hi = min(CORE_FRACS), max(CORE_FRACS)
    out = {
        "dataset": ds,
        "n_paired_statements": len(frame),
        "n_rows_total": len(rows),
        "dropped_never_steered_rows": n_never,
        "core_fracs": [float(f) for f in CORE_FRACS],
        "dose_table_paired": dose_table(frame),
        "mcnemar_baseline_to_low": mcnemar(frame, BASELINE_FRAC, lo),
        "mcnemar_baseline_to_high": mcnemar(frame, BASELINE_FRAC, hi),
        "sign_control": two_by_two(frame, lo, hi),
        "dose_vs_baseline": two_by_two(frame, lo, BASELINE_FRAC),
        "trend_pooled": cochran_armitage(rows),
        "crossing_vs_magnitude": crossing_vs_magnitude(rows),
    }
    print(f"[refusal] {ds}: {len(frame)} statements observed at every frac in "
          f"{list(CORE_FRACS)} ({len(rows)} steered+baseline rows, {n_never} "
          f"never-steered dropped)")
    print(f"{'frac':>6} {'mean g':>9} {'crossed':>8} {'refused':>10} {'rate':>7}")
    for r in out["dose_table_paired"]:
        print(f"{r['frac']:>6.1f} {r['mean_g_read']:>9.3f} "
              f"{r['n_crossed']:>4d}/{r['n']:<3d} {r['n_refused']:>6d}/{r['n']:<3d} "
              f"{r['frac_refused']:>7.3f}")
    m = out["mcnemar_baseline_to_low"]
    print(f"[refusal] McNemar 0 -> {lo}: {m['gained']} statements flipped INTO "
          f"refusal (of {m['n_eligible_to_gain']} compliant at baseline), "
          f"{m['lost']} flipped out, p={m['p_exact']:.3g}")
    m = out["mcnemar_baseline_to_high"]
    print(f"[refusal] McNemar 0 -> {hi}: {m['gained']} in, {m['lost']} out "
          f"(of {m['n_eligible_to_lose']} refusing at baseline), p={m['p_exact']:.3g}")
    s = out["sign_control"]
    print(f"[refusal] sign control {lo} vs {hi} on the same {s['n']} statements: "
          f"{s['n_refused_a']} vs {s['n_refused_b']} refused, OR={s['odds_ratio']:.2f} "
          f"p={s['p_fisher']:.3g}")
    t = out["trend_pooled"]
    print(f"[refusal] Cochran-Armitage trend (pooled n={t['n']}): "
          f"slope={t['slope_per_eps_star']:.4f}/eps*, z={t['z']:.2f}, p={t['p']:.3g}")
    x = out["crossing_vs_magnitude"]
    for st in x["strata"]:
        print(f"[refusal] |frac| in ({st['abs_frac_lo']}, {st['abs_frac_hi']}]: "
              f"crossed {st['crossed_refused']}/{st['crossed_total']} refused, "
              f"not crossed {st['not_crossed_refused']}/{st['not_crossed_total']}, "
              f"OR={st['odds_ratio']:.1f} p={st['p_fisher']:.3g}")
    print(f"[refusal] Mantel-Haenszel OR for crossing, magnitude held fixed: "
          f"{x['mh_odds_ratio']:.1f}")
    path = f"refusal_control_stats_{ds}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[refusal] wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refusal")
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
