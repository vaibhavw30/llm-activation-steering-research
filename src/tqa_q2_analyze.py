"""tqa_q2_analyze.py — the Q2 numbers, recomputed from the judged CSVs.

Every figure docs/Q2_TRUTHFULQA_STEERING.md quotes comes from here, so the doc can
name one artifact (tqa_q2_summary.csv) instead of citing a session transcript.

WHAT Q2 ASKED. Q1 established the gate: unsteered gemma-2-2b is truthful+informative
on 0.266 of the 64 held-out TruthfulQA questions (job 3081925), so unlike cities there
is headroom. Q2 steers along J^T w at +/-{0.5,1,1.5,2} x eps* and asks whether that
rate moves.

THE THREE THINGS THIS SCRIPT REPORTS, AND WHY EACH IS NEEDED.

1. The per-frac rate table. reach_control.py already adjudicates a verdict off this,
   but it thresholds a point estimate against MIN_DELTA and reports no paired test.
   The mean arm's buckets are the SAME 64 questions at every frac, so the unpaired
   Wilson intervals reach_control prints are far wider than the design supports.

2. Exact McNemar against the frac=0 baseline, per frac. This is the test the design
   earns: n=64 unpaired needs the rate to reach 0.51 to detect anything at 80% power,
   whereas paired needs only 6 one-directional flips for p=0.031. It is how the
   refusal positive control was read (14 flips against 0).

3. Answer length per frac, and truthfulness adjusted for it. On truthfulqa the
   steering changes how LONG the answer is, and TruthfulQA rewards hedged,
   qualified answers. CAVEAT, and it is not a small one: length is on the causal
   path (steer -> longer answer -> judged truthful), so adjusting for it is
   adjusting for a MEDIATOR. A null adjusted term does NOT show the steering had no
   effect. It shows the direction is not specifically about truth: whatever it does,
   it does through length. The experiment that separates a truth direction from a
   generic large-perturbation effect is the norm-matched random control
   (reach_steer.arm_rand_ctrl, --arm randctrl here), not this adjustment.

    PYTHONPATH=src python src/tqa_q2_analyze.py --dataset truthfulqa
    PYTHONPATH=src python src/tqa_q2_analyze.py --dataset truthfulqa --arms mean randctrl
"""
import argparse
import csv
import json
import math

import pandas as pd

from reach_control import wilson_interval

ARM_JUDGED = "judge_refusal_{ds}_{arm}.csv"
SCORE_COL = "truthful_and_informative"
# reach_steer.arm_rand_ctrl deliberately reuses mean_diff_tgt's scale grid, so its
# rows must be divided by mean_diff_tgt's eps* to land on the same frac axis. Without
# this they would get their own, and the control would not be comparable to the thing
# it controls.
RAND_PREFIX = "rand_ctrl"
RAND_EPS_FROM = "mean_diff_tgt"


def eps_star_of(direction, summ):
    """The eps* whose multiples the `scale` column counts, for one direction name.

    `direction` is what reach_steer wrote: "jtw_<w-name>" for a real direction, or
    "rand_ctrl_<i>" for a control vector, which has no eps* of its own."""
    name = (RAND_EPS_FROM if str(direction).startswith(RAND_PREFIX)
            else str(direction).removeprefix("jtw_"))
    try:
        return float(summ["directions"][name]["median_eps_star"])
    except KeyError:
        raise SystemExit(f"[q2] no median_eps_star for {name!r} (direction "
                         f"{direction!r}) in the reach summary")


def load_arm(ds, arm, summ):
    """One judged arm, with `frac` and `words` added. SystemExit on a missing file so
    a partial run says which stage never produced its artifact."""
    path = ARM_JUDGED.format(ds=ds, arm=arm)
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"[q2] {path} not found: arm {arm!r} has not been judged")
    for col in ("direction", "scale", "prompt", "answer", SCORE_COL):
        if col not in d.columns:
            raise SystemExit(f"[q2] {path} has no {col!r} column")
    eps = d["direction"].map(lambda x: eps_star_of(x, summ))
    d = d.assign(arm=arm,
                 frac=(d["scale"] / eps).round(2),
                 words=d["answer"].astype(str).str.split().str.len())
    return d


def mcnemar_exact(base, arm):
    """(gained, lost, two-sided exact p) for two aligned 0/1 Series.

    The exact binomial on the discordant pairs, not the chi-square approximation:
    the counts here are single digits, where the continuity-corrected chi-square is
    unreliable. Returns p=1.0 when nothing is discordant."""
    j = pd.concat([base.rename("b"), arm.rename("a")], axis=1).dropna()
    gained = int(((j["b"] == 0) & (j["a"] == 1)).sum())
    lost = int(((j["b"] == 1) & (j["a"] == 0)).sum())
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0, len(j)
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1))
    return gained, lost, min(2.0 * tail / 2 ** n, 1.0), len(j)


def length_adjusted_p(d):
    """Two-sided p for `-frac` added to a logistic model already carrying word count.

    Returns None when scikit-learn is absent, so the rest of the report still runs.
    See the module docstring: length is a MEDIATOR, and a large p here means "not
    specifically about truth", never "no effect"."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    y = d[SCORE_COL].to_numpy()
    if len(set(y.tolist())) < 2:
        return None
    w = d["words"].to_numpy(dtype=float)
    x1 = w.reshape(-1, 1)
    x2 = np.column_stack([w, -d["frac"].to_numpy(dtype=float)])

    def loglik(x):
        p = LogisticRegression(max_iter=2000).fit(x, y).predict_proba(x)[:, 1]
        p = p.clip(1e-12, 1 - 1e-12)
        return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    stat = 2.0 * (loglik(x2) - loglik(x1))
    # Survival function of chi-square with 1 df, in closed form, so this file needs
    # no scipy (reach_control avoids it for the same reason).
    return math.erfc(math.sqrt(max(stat, 0.0) / 2.0)), stat


TARGET = "jtw_mean_diff_tgt"


def control_contrast(frames, frac=-2.0, target=TARGET):
    """The REGISTERED test: does `target` beat each norm-matched control at one dose?

    Everything else in this module tests an arm against its OWN frac-0 baseline, which
    answers "did it move" and not "did it move for a reason". The bar Q2 was registered
    against (PLAN_ADVISOR_NOTES_2026-09.md section 3) is beating a norm-matched random
    control at the same dose, and that is this comparison: the same 64 questions steered
    two ways, paired question by question.

    Pairing matters more here than against the baseline. The control and the target share
    an identical unsteered block (the slurm job asserts it byte for byte), so a question
    the model gets right unsteered tends to stay right under both, and those concordant
    pairs carry no information about which direction is better. Returns [] when the arms
    needed are not loaded, so a one-arm run still reports.
    """
    d = pd.concat(frames, ignore_index=True)
    at = d[d["frac"] == frac]
    tgt = at[at["direction"] == target]
    if tgt.empty:
        return []
    t = tgt.set_index("prompt")[SCORE_COL]
    rows = []
    for name, g in at[at["direction"].str.startswith(RAND_PREFIX)].groupby("direction"):
        c = g.set_index("prompt")[SCORE_COL]
        hit = t.index.intersection(c.index)
        won, lost, p, n = mcnemar_exact(c.reindex(hit), t.reindex(hit))
        rows.append({"frac": frac, "target": target, "control": name, "n_paired": n,
                     "target_rate": round(float(t.reindex(hit).mean()), 4),
                     "control_rate": round(float(c.reindex(hit).mean()), 4),
                     "target_words": round(float(tgt["words"].mean()), 2),
                     "control_words": round(float(g["words"].mean()), 2),
                     "target_wins": won, "control_wins": lost,
                     "mcnemar_p": float(f"{p:.3g}")})
    return rows


def analyze(ds, arms):
    summ = json.load(open(f"reach_summary_{ds}.json"))
    frames = [load_arm(ds, a, summ) for a in arms]
    rows = []
    for d in frames:
        for direction, g in d.groupby("direction"):
            base = g[g["frac"] == 0].set_index("prompt")[SCORE_COL]
            adj = length_adjusted_p(g)
            for frac, b in sorted(g.groupby("frac"), key=lambda kv: kv[0]):
                n, k = len(b), int(b[SCORE_COL].sum())
                lo, hi = wilson_interval(k, n)
                gained = lost = 0
                p = 1.0
                if frac != 0:
                    gained, lost, p, _ = mcnemar_exact(
                        base, b.set_index("prompt")[SCORE_COL])
                rows.append({
                    "arm": b["arm"].iloc[0], "direction": direction, "frac": frac,
                    "n": n, "n_truthful_informative": k, "rate": round(k / n, 4),
                    "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                    "rate_truthful": round(float(b["truthful"].mean()), 4)
                    if "truthful" in b else "",
                    "rate_informative": round(float(b["informative"].mean()), 4)
                    if "informative" in b else "",
                    "mean_words": round(float(b["words"].mean()), 2),
                    "mcnemar_gained": gained, "mcnemar_lost": lost,
                    # 3 significant figures, not round(p, 6): a decisive
                    # p of 1e-7 would round to 0.0 and read as a formatting bug.
                    "mcnemar_p": float(f"{p:.3g}"),
                    "length_adjusted_frac_p": round(adj[0], 4) if adj else "",
                })
    return rows


def report_contrast(rows):
    """The registered comparison, printed with its own Bonferroni threshold.

    The threshold is stated in the output rather than left to the reader: three controls
    were run, so the honest bar is 0.05/3, and a result that only clears 0.05 should be
    read as such."""
    if not rows:
        return
    alpha = 0.05 / len(rows)
    print(f"\n=== registered test: {rows[0]['target']} vs each norm-matched control "
          f"at frac {rows[0]['frac']}, paired on the same questions ===")
    print(pd.DataFrame(rows)[["control", "n_paired", "control_rate", "target_rate",
                              "control_words", "target_words", "target_wins",
                              "control_wins", "mcnemar_p"]].to_string(index=False))
    worst = max(r["mcnemar_p"] for r in rows)
    print(f"  Bonferroni for {len(rows)} controls: alpha = {alpha:.4f}; "
          f"largest p = {worst:.3g} -> "
          f"{'ALL CLEAR' if worst < alpha else 'NOT all clear'}")


def report(rows, out):
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    for (arm, direction), g in df.groupby(["arm", "direction"], sort=False):
        print(f"\n--- {arm} / {direction} ---")
        print(g[["frac", "n", "n_truthful_informative", "rate", "wilson_lo",
                 "wilson_hi", "mean_words", "mcnemar_gained", "mcnemar_lost",
                 "mcnemar_p"]].to_string(index=False))
        p = g["length_adjusted_frac_p"].iloc[0]
        if p != "":
            print(f"  frac added to a model already carrying word count: p={p} "
                  f"(mediator adjustment, see the module docstring)")
    print(f"\n[q2] wrote {out} ({len(rows)} rows)")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--arms", nargs="+", default=["mean"],
                    help="judged arms to include, e.g. --arms mean randctrl")
    ap.add_argument("--out", default="")
    ap.add_argument("--frac", type=float, default=-2.0,
                    help="dose at which to run the registered control contrast")
    return ap


def main():
    a = build_parser().parse_args()
    out = a.out or f"tqa_q2_summary_{a.dataset}.csv"
    summ = json.load(open(f"reach_summary_{a.dataset}.json"))
    frames = [load_arm(a.dataset, arm, summ) for arm in a.arms]
    report(analyze(a.dataset, a.arms), out)
    report_contrast(control_contrast(frames, frac=a.frac))


if __name__ == "__main__":
    main()
