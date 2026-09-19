"""tqa_form.py — C2: is the TruthfulQA gain still there once answer length is held fixed?

docs/PLAN_PI_FEEDBACK_2026-09-18.md section 9, the form-vs-content row of the dataset card.
C1 (the logistic regression) lives in tqa_q2_analyze.length_adjusted_p and C3 (truncation)
in judge_audit. This is the third, stratified reading.

THE QUESTION. Steering along the truth direction lengthens answers (4.7 -> 18.6 words at
frac -2) and TruthfulQA's judge rewards fuller answers. Within answers of the SAME length,
is the steered answer still more often truthful than the unsteered one?

THE BINS ARE FIXED BY THE BASELINE ALONE. Edges are the quartiles of the unsteered (frac 0)
word counts, closed at their maximum. Nothing about the steered answers or the outcome
chooses them, so the bins cannot be tuned toward a result. Steered answers longer than
any unsteered answer land in an "outside" stratum: there is nothing to compare them with,
so they are counted and reported, and never enter the pooled test.

THE TEST. Cochran-Mantel-Haenszel over the in-support strata, with the Mantel-Haenszel
common odds ratio. It treats the two doses as independent samples within a stratum. They
are the same 64 questions, and a question's truthfulness is positively correlated across
doses, which makes the unpaired test conservative, not anti-conservative.

READING IT. A pooled OR near 1 with most of the gain in "outside" says the gain is in the
extra length (form). A pooled OR clearly above 1 says steered answers are more truthful
even at matched length (some content). Length is a mediator (see tqa_q2_analyze's
docstring), so neither reading is causal on its own. C3's truncation test is the causal one.

    PYTHONPATH=src python src/tqa_form.py
    PYTHONPATH=src python src/tqa_form.py --judged-pattern 'judge_v2_{ds}_{arm}.csv'
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

import tqa_q2_analyze as q2

TARGET = q2.TARGET


def baseline_edges(d, direction):
    """Upper edges of the in-support bins: the frac-0 quartiles, closed at the frac-0 max."""
    w = d[(d["direction"] == direction) & (d["frac"] == 0)]["words"]
    qs = [int(np.floor(w.quantile(p))) for p in (0.25, 0.5, 0.75)]
    return sorted(set(qs + [int(w.max())]))


def length_strata(d, direction, frac, score_col="truthful"):
    """Per-bin counts at `frac` vs frac 0 for one direction, plus the outside stratum."""
    g = d[d["direction"] == direction]
    edges = baseline_edges(d, direction)
    labels, lo = [], 1
    for hi in edges:
        labels.append((f"{lo}-{hi}", lo, hi))
        lo = hi + 1
    labels.append(("outside", edges[-1] + 1, math.inf))
    rows = []
    for name, lo, hi in labels:
        def cell(f):
            x = g[(g["frac"] == f) & (g["words"] >= lo) & (g["words"] <= hi)]
            return int(x[score_col].sum()), len(x)
        ks, ns = cell(frac)
        kb, nb = cell(0.0)
        rows.append({"direction": direction, "frac": frac, "stratum": name,
                     "k_steer": ks, "n_steer": ns, "k_base": kb, "n_base": nb,
                     "rate_steer": round(ks / ns, 4) if ns else "",
                     "rate_base": round(kb / nb, 4) if nb else ""})
    return pd.DataFrame(rows)


def cmh(strata):
    """Mantel-Haenszel common OR and the continuity-corrected CMH p, over strata both
    doses reach. Strata with an empty side carry no within-stratum comparison."""
    s = strata[(strata["n_steer"] > 0) & (strata["n_base"] > 0)]
    num = den = sum_a = sum_e = sum_v = 0.0
    for r in s.itertuples():
        a, b = r.k_steer, r.n_steer - r.k_steer        # steered: truthful, not
        c, dd = r.k_base, r.n_base - r.k_base          # baseline: truthful, not
        n = a + b + c + dd
        num += a * dd / n
        den += b * c / n
        m1, n1 = a + c, a + b
        sum_a += a
        sum_e += n1 * m1 / n
        if n > 1:
            sum_v += n1 * (n - n1) * m1 * (n - m1) / (n * n * (n - 1))
    or_mh = num / den if den else (math.inf if num else 1.0)
    if sum_v == 0:
        return {"or_mh": or_mh, "chi2": 0.0, "p": 1.0, "n_strata": len(s)}
    chi2 = (max(abs(sum_a - sum_e) - 0.5, 0.0)) ** 2 / sum_v
    return {"or_mh": or_mh, "chi2": chi2, "p": math.erfc(math.sqrt(chi2 / 2.0)),
            "n_strata": len(s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--arms", nargs="+", default=["mean", "randctrl"])
    ap.add_argument("--frac", type=float, default=-2.0)
    ap.add_argument("--score-col", default="truthful",
                    choices=["truthful", "truthful_and_informative"])
    ap.add_argument("--judged-pattern", default=q2.ARM_JUDGED)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    q2.SCORE_COL = a.score_col
    summ = json.load(open(f"reach_summary_{a.dataset}.json"))
    d = pd.concat([q2.load_arm(a.dataset, arm, summ, a.judged_pattern) for arm in a.arms],
                  ignore_index=True)
    directions = [TARGET] + sorted(x for x in d["direction"].unique()
                                   if x.startswith(q2.RAND_PREFIX))
    tables, summary = [], []
    for direction in directions:
        s = length_strata(d, direction, a.frac, a.score_col)
        r = cmh(s)
        out = s[s["stratum"] == "outside"].iloc[0]
        base_n = int(s["n_base"].sum())
        gain = s["k_steer"].sum() - s["k_base"].sum()
        tables.append(s)
        summary.append({"direction": direction, "frac": a.frac,
                        "rate_steer": round(s["k_steer"].sum() / s["n_steer"].sum(), 4),
                        "rate_base": round(s["k_base"].sum() / base_n, 4),
                        "n_outside": int(out["n_steer"]), "k_outside": int(out["k_steer"]),
                        "gain_total": int(gain),
                        "or_mh_in_support": round(r["or_mh"], 3),
                        "cmh_p_in_support": float(f"{r['p']:.3g}"),
                        "n_strata": r["n_strata"]})
        print(f"\n--- {direction}, frac {a.frac} vs 0, score {a.score_col} ---")
        print(s[["stratum", "k_steer", "n_steer", "rate_steer",
                 "k_base", "n_base", "rate_base"]].to_string(index=False))
        print(f"  in-support MH OR = {r['or_mh']:.3g}, CMH p = {r['p']:.3g} "
              f"over {r['n_strata']} strata; {int(out['n_steer'])} steered answers "
              f"({int(out['k_steer'])} truthful) are longer than any unsteered answer")
    tag = "" if a.judged_pattern == q2.ARM_JUDGED else "_" + a.judged_pattern.split("_{ds}")[0]
    out = a.out or f"tqa_form_c2_{a.dataset}{tag}_{a.score_col}.csv"
    pd.concat(tables).to_csv(out, index=False)
    pd.DataFrame(summary).to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print(f"\n[c2] wrote {out} and its _summary")


if __name__ == "__main__":
    main()
