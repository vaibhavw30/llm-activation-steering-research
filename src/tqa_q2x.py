"""tqa_q2x.py: push Q2's direction past the certificate's boundary. Does content move? (Q2x)

    PYTHONPATH=src python src/tqa_q2x.py --stage all --device cuda                  # CLUSTER
    PYTHONPATH=src python src/tqa_q2x.py --stage all --device cuda \\
        --limit 2 --prefix smoke_                                                  # its smoke
    PYTHONPATH=src python src/tqa_q2x.py --stage summary                            # LAPTOP

Spec: docs/superpowers/specs/2026-09-30-round4-q2x-v1-design.md. Q2 never crossed the
certificate's boundary: the layer-20 readout g fell 37.88 -> 15.05 by frac -2, and the
linear fit puts g = 0 at 3.49 eps*. J-A's C3 found the -2 gain is form: cut to the
unsteered length, it is gone. This job pushes past the crossing and scores every answer
twice, full and cut to the same question's unsteered length, to ask whether content moves
once the boundary is crossed (refusal did; truth on cities did not).

It imports tqa_discovery, tqa_confirm, judge_audit, reach_steer and reach_hop and edits
none of them: queued cluster jobs read those files at run time.

DOSES are in Q2 eps* units and point TOWARD TRUTHFUL (tqa_discovery.q2_vector is already
signed), so dose 2 here is Q2's frac -2: the anchor that must reproduce Q2 (reading R0).

STAGES
  steer     one (direction, dose) block of 64 holdout questions at a time: the answer, and
            the layer-20 readout g with the hook still set, in one pass. Resumes per block.
  judge     the v2 allenai judges on every row. Resumes per 64 rows.
  truncate  every steered answer cut to its question's baseline word count (C3 per dose)
            and re-judged. Resumes per 64 rows.
  summary   rates, McNemar, permutation p, the form columns, and the registered readings
            R0-R2 written to q2x_outcome_truthfulqa.json.
"""
import argparse
import csv
import json
import os
import statistics

import numpy as np

import tqa_discovery as td

DS = td.DS
DOSES = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)        # x Q2's eps*, toward truthful; 0 is `baseline`
ANCHOR = 2.0                                   # Q2's frac -2
PREDICTED_CROSSING = 3.49                      # 37.88 / 10.84, Q2 doc section 4
N_RAND, Q2X_RAND_SEED = 8, 31                  # J-C draws with 13; this is a fresh null
Q2_DIR = "q2_mean_diff"
READ_DIRECTION = "mean_diff_tgt"               # the readout Q2's certificate was issued on
CAP_X_INPUT_SCALE = 1.5                        # reach_steer.scale_grid's cap
ALPHA = 0.05
FORM_MARGIN = 0.10                             # full gain over cut gain that reads as form
Q2_V2_TRUTHFUL = "tqa_q2_summary_truthfulqa_judge_v2_truthful.csv"
SCORE_COLS = ("truthful", "truthful_and_informative")
PREFIX = ""                                    # set by --prefix; smoke runs write smoke_*


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure
def check_doses(eps, input_scale, doses=DOSES):
    """The largest push, refused if it is past reach_steer's cap of 1.5 x input_scale."""
    cap = CAP_X_INPUT_SCALE * float(input_scale)
    top = max(doses) * float(eps)
    if top > cap:
        raise SystemExit(f"[q2x] dose {max(doses):g} x eps* = {top:.1f} is past the cap "
                         f"{cap:.1f} (1.5 x input_scale)")
    return top


def q2x_directions(q2_vec, n_rand=N_RAND, seed=Q2X_RAND_SEED):
    """[(name, unit vector)]: Q2's direction (already truthful-signed) and n_rand
    norm-matched random ones."""
    d = len(q2_vec)
    rng = np.random.default_rng(seed)
    return ([(Q2_DIR, td.unit(np.asarray(q2_vec, np.float64)))]
            + [(f"rand_{j}", td.unit(rng.standard_normal(d))) for j in range(n_rand)])


def budget_hit(raw):
    """True when the answer never ended. Every TruthfulQA generation runs to the budget
    by inventing a next turn (Q1, job 3081925), so only a newline marks a finished
    answer; a raw completion with none was cut by the 48-token budget mid-answer."""
    return "\n" not in str(raw).strip()


def distinct_ratio(answer):
    w = str(answer).lower().split()
    return len(set(w)) / len(w) if w else 0.0


def truncated_row(row, base_words):
    """C3 for one row: the answer cut to its question's baseline word count (at least one
    word), with the full text kept beside it."""
    from judge_audit import truncate_words
    text, was_cut = truncate_words(row["answer"], base_words)
    return dict(row, answer_full=row["answer"], answer=text, cut=int(was_cut),
                words_base=int(base_words))


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def crossing_dose(median_g):
    """{dose: median g} -> the smallest dose with median g < 0, or None."""
    crossed = [d for d in sorted(median_g) if median_g[d] < 0]
    return crossed[0] if crossed else None


def form_stats(rows):
    """Per (direction, frac): the readout and the form columns."""
    by = {}
    for r in rows:
        by.setdefault((r["direction"], float(r["frac"])), []).append(r)
    out = {}
    for k, g in by.items():
        n = len(g)
        out[k] = {"median_g": statistics.median([float(r["g"]) for r in g]),
                  "crossed_share": sum(int(r["crossed"]) for r in g) / n,
                  "mean_words": sum(int(r["words"]) for r in g) / n,
                  "budget_hit_share": sum(int(r["budget_hit"]) for r in g) / n,
                  "median_distinct_ratio":
                      statistics.median([float(r["distinct_ratio"]) for r in g])}
    return out


_STAT_FIELDS = ("rate", "lo", "hi", "gain", "gained", "lost", "mcnemar_p", "perm_p")


def summary_rows(full, cut, form):
    """One row per (direction, frac). full/cut map a score column to
    tqa_confirm.summarize's rows. truthful columns are full_* / cut_*, the
    truthful-and-informative ones full_ti_* / cut_ti_*."""
    def idx(rows):
        return {(r["direction"], float(r["frac"])): r for r in rows}

    tabs = {(tag, col): idx(src[col]) for tag, src in (("full", full), ("cut", cut))
            for col in SCORE_COLS}
    out = []
    for k in sorted(tabs[("full", "truthful")]):
        row = {"direction": k[0], "frac": k[1]}
        for (tag, col), t in tabs.items():
            pre = f"{tag}_" if col == "truthful" else f"{tag}_ti_"
            r = t.get(k)
            for f in _STAT_FIELDS:
                row[pre + f] = r[f] if r else ""
        row.update(form.get(k, {}))
        out.append(row)
    return out


def classify(r, cut_p_adj, full_p_adj, anchor_rate, n_rand=N_RAND):
    """R2's outcome labels for one dose of Q2's direction. Beating all n_rand random
    directions means perm p = 1/(n_rand + 1): a tie counts against."""
    beats_all = 1.0 / (n_rand + 1) + 1e-9
    labels = []
    if cut_p_adj < ALPHA and r["cut_gain"] > 0 and r["cut_perm_p"] <= beats_all:
        labels.append("a_content")
    else:
        full_bar = (full_p_adj < ALPHA and r["full_gain"] > 0
                    and r["full_perm_p"] <= beats_all)
        if full_bar or r["full_gain"] - r["cut_gain"] >= FORM_MARGIN:
            labels.append("b_form")
    if r["full_rate"] < anchor_rate and (r["budget_hit_share"] >= 0.5
                                         or r["median_distinct_ratio"] < 0.5):
        labels.append("c_degeneration")
    return labels or ["none"]


def read_outcome(rows, q2_lo, q2_hi, n_rand=N_RAND):
    """R0 (the anchor gate), R1 (the realized crossing) and R2 (per-dose outcomes, Holm
    across the doses read), from summary_rows' Q2 rows."""
    q2 = {float(r["frac"]): r for r in rows if r["direction"] == Q2_DIR}
    anchor = q2[ANCHOR]["full_rate"]
    gate = bool(q2_lo <= anchor <= q2_hi)
    med = {f: r["median_g"] for f, r in q2.items()}
    dx = crossing_dose(med)
    read_at = [f for f in sorted(q2) if dx is not None and f >= dx] or [max(q2)]
    cut_adj = holm([q2[f]["cut_mcnemar_p"] for f in read_at])
    full_adj = holm([q2[f]["full_mcnemar_p"] for f in read_at])
    per = {str(f): {"labels": classify(q2[f], c, u, anchor, n_rand),
                    "cut_p_holm": c, "full_p_holm": u}
           for f, c, u in zip(read_at, cut_adj, full_adj)}
    return {"gate": {"anchor_rate": anchor, "q2_interval": [q2_lo, q2_hi], "passed": gate},
            "crossing": {"realized": dx, "predicted": PREDICTED_CROSSING,
                         "median_g_by_dose": {str(f): med[f] for f in sorted(med)}},
            "uncrossed": dx is None, "read_at": read_at, "per_dose": per,
            "n_rand": n_rand,
            "headline": per[str(read_at[0])]["labels"] if gate else ["GATE FAILED"]}


def q2_interval(p=Q2_V2_TRUTHFUL):
    """(wilson_lo, wilson_hi) of Q2's mean arm at frac -2, truthful, v2 judges."""
    for r in td.read_csv(p):
        if (r["arm"], r["direction"], float(r["frac"])) == \
                ("mean", "jtw_mean_diff_tgt", -2.0):
            return float(r["wilson_lo"]), float(r["wilson_hi"])
    raise SystemExit(f"[q2x] no frac -2 row for jtw_mean_diff_tgt in {p}: R0 has no "
                     "interval to check the anchor against")
