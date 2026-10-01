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
    answer; a raw completion with none was cut by the 48-token budget mid-answer. Only
    leading whitespace is dropped: a trailing newline is an answer that ended."""
    return "\n" not in str(raw).lstrip()


def distinct_ratio(answer):
    w = str(answer).lower().split()
    return len(set(w)) / len(w) if w else 0.0


def truncated_row(row, base_words):
    """C3 for one row: the answer cut to its question's baseline word count (at least one
    word), with the full text kept beside it. `words` and `distinct_ratio` describe the cut
    text; the full answer's word count stays as `words_full`."""
    from judge_audit import truncate_words
    text, was_cut = truncate_words(row["answer"], base_words)
    return dict(row, answer_full=row["answer"], answer=text, cut=int(was_cut),
                words_base=int(base_words), words_full=len(str(row["answer"]).split()),
                words=len(text.split()), distinct_ratio=distinct_ratio(text))


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
    out = {"gate": {"anchor_rate": anchor, "q2_interval": [q2_lo, q2_hi], "passed": gate},
           "crossing": {"realized": dx, "predicted": PREDICTED_CROSSING,
                        "median_g_by_dose": {str(f): med[f] for f in sorted(med)}},
           "uncrossed": dx is None, "read_at": read_at, "per_dose": per,
           "n_rand": n_rand,
           "headline": per[str(read_at[0])]["labels"] if gate else ["GATE FAILED"]}
    if not gate:
        # The spec: a failed anchor means no R1/R2 reading is made. The numbers are kept,
        # out of the reading's keys, so a label never sits next to a failed gate.
        out["diagnostic"] = {"crossing_realized": dx, "read_at": read_at, "per_dose": per}
        out.update(per_dose={}, read_at=[], uncrossed="not read")
        out["crossing"]["realized"] = "not read"
    return out


def q2_interval(p=Q2_V2_TRUTHFUL):
    """(wilson_lo, wilson_hi) of Q2's mean arm at frac -2, truthful, v2 judges."""
    for r in td.read_csv(p):
        if (r["arm"], r["direction"], float(r["frac"])) == \
                ("mean", "jtw_mean_diff_tgt", -2.0):
            return float(r["wilson_lo"]), float(r["wilson_hi"])
    raise SystemExit(f"[q2x] no frac -2 row for jtw_mean_diff_tgt in {p}: R0 has no "
                     "interval to check the anchor against")


# ------------------------------------------------------------------ io
def append_rows(p, rows):
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def steer_rows(model, tok, st, name, vec, frac, eps, recs, read):
    """One (direction, dose) block. The same primitives as tqa_discovery.steer_block
    (prompt_of, generate_raw at Q2's 48 tokens, first_answer), in a loop of its own so
    the raw completion's newline is seen before it is flattened. `read(prompt)` returns
    the layer-20 readout g with the hook still set to this block's vector."""
    import torch

    import dct_steer_utils as su
    from prep_truthfulqa import prompt_of
    from tqa_baseline import first_answer

    scale = 0.0 if vec is None else frac * eps
    st.set(None if vec is None or scale == 0 else
           torch.tensor(scale * np.asarray(vec), dtype=torch.float32))
    rows = []
    for r in recs:
        p = prompt_of(r["question"])
        raw = su.generate_raw(model, tok, p, td.MAX_NEW_TOKENS)
        ans = first_answer(raw)
        g = float(read(p))
        rows.append({"direction": name, "scale": scale, "frac": frac,
                     "question": r["question"], "prompt": p,
                     "completion": raw.replace("\n", " ").strip(), "answer": ans,
                     "g": g, "crossed": int(g < 0), "budget_hit": int(budget_hit(raw)),
                     "words": len(ans.split()), "distinct_ratio": distinct_ratio(ans)})
    return rows


def blocks_todo(done, blocks, n):
    """The (name, vec, frac) blocks still to run. A block is written in one append, so a
    count strictly between 0 and n means a killed write: refuse rather than duplicate."""
    todo = []
    for name, vec, f in blocks:
        k = done.get((name, f), 0)
        if 0 < k != n:
            raise SystemExit(f"[q2x] {name} dose {f:g} has {k} of {n} rows: a partial or "
                             "duplicated block. Remove its rows from the steer CSV and "
                             "resume.")
        if k < n:
            todo.append((name, vec, f))
    return todo


def steer_counts(p):
    """{(direction, frac): rows} in the steer CSV, refusing a file whose last write was
    killed: no final newline, or a row with missing fields."""
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        text = f.read()
    if text and not text.endswith("\n"):
        raise SystemExit(f"[q2x] {p} does not end in a newline: its last write was killed. "
                         "Remove the last block's rows and resume.")
    done = {}
    for r in td.read_csv(p):
        if None in r or None in r.values():
            raise SystemExit(f"[q2x] {p} has a torn row ({r}): remove the last block's rows "
                             "and resume.")
        k = (r["direction"], float(r["frac"]))
        done[k] = done.get(k, 0) + 1
    return done


def check_judged(full, cut):
    """Refuse to summarize an incomplete run (a partial pull, a judge still going): every
    (direction, dose) needs the baseline's n, the truncated file one row per steered row,
    and at least one random direction for the permutation test."""
    n = sum(r["direction"] == "baseline" for r in full)
    if not n:
        raise SystemExit("[q2x] no baseline rows in the judged file")
    counts = {}
    for r in full:
        k = (r["direction"], float(r["frac"]))
        counts[k] = counts.get(k, 0) + 1
    for (name, f), k in sorted(counts.items()):
        if k != n:
            raise SystemExit(f"[q2x] {name} dose {f:g} has {k} judged rows, the baseline "
                             f"{n}: the judged file is incomplete")
    steered = len(full) - n
    if len(cut) != steered:
        raise SystemExit(f"[q2x] the truncated file has {len(cut)} rows for {steered} "
                         "steered rows: the truncate stage is incomplete")
    if not any(name.startswith("rand_") for name, _ in counts):
        raise SystemExit("[q2x] no random direction in the judged file: the permutation "
                         "test has nothing to compare against")


# ------------------------------------------------------------------ stages
def stage_steer(device, limit=0):
    import pandas as pd
    import torch

    import dct_steer_utils as su
    from reach_hop import load_meta
    from reach_steer import read_g

    out = path(f"q2x_steer_{DS}.csv")
    src, tgt, input_scale, model_name = load_meta(DS)
    eps = td.q2_eps_star()
    check_doses(eps, input_scale)
    dirs = q2x_directions(td.q2_vector())
    recs = [{"question": q} for q in
            pd.read_csv(td.HOLDOUT)["question"].astype(str).str.strip()]
    if limit:
        recs, dirs = recs[:limit], dirs[:2]                 # Q2's direction and rand_0
    done = steer_counts(out)
    blocks = [("baseline", None, 0.0)] + [(n, v, f) for n, v in dirs for f in DOSES]
    todo = blocks_todo(done, blocks, len(recs))
    rd = np.load(f"reach_dirs_{DS}.npz", allow_pickle=True)
    k = [str(x) for x in rd["names"]].index(READ_DIRECTION)
    t02 = float(rd["thresh02"][k])
    tok, model, dev = su.load_model(device, model_name=model_name)
    w = torch.tensor(np.asarray(rd["W"][k], np.float32)).to(dev)

    def read(p):
        return read_g(model, tok, p, tgt, w, t02, dev)

    print(f"[q2x] {len(todo)} of {len(blocks)} blocks to run, {len(recs)} questions, "
          f"eps* {eps:.4g}, doses {DOSES}, readout {READ_DIRECTION} at layer {tgt}",
          flush=True)
    with su.Steerer(model, src) as st:
        for name, vec, f in todo:
            append_rows(out, steer_rows(model, tok, st, name, vec, f, eps, recs, read))
            print(f"[q2x] {name} dose {f:g} done", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def _judge():
    from judges.local_hf import TruthJudge
    return TruthJudge


def _judged_all(rows, out):
    """out already holds a verdict for every row: a resubmit skips the stage without
    loading a 7B judge to find nothing to do."""
    if os.path.exists(out) and len(td.read_csv(out)) >= len(rows):
        print(f"[q2x] {out} complete, skipping", flush=True)
        return True
    return False


def stage_judge(device):
    from judge_audit import judge_resumable
    rows, out = td.read_csv(path(f"q2x_steer_{DS}.csv")), path(f"q2x_judged_{DS}.csv")
    if _judged_all(rows, out):
        return
    tj = _judge()(device)
    judge_resumable(rows, tj.score, out)


def stage_truncate(device):
    from judge_audit import judge_resumable
    steer = td.read_csv(path(f"q2x_steer_{DS}.csv"))
    base = {r["question"]: len(r["answer"].split()) for r in steer
            if r["direction"] == "baseline"}
    cut = [truncated_row(r, base[r["question"]]) for r in steer
           if r["direction"] != "baseline"]
    print(f"[q2x] truncate: {sum(c['cut'] for c in cut)} of {len(cut)} answers cut to "
          "their question's baseline length", flush=True)
    out = path(f"q2x_trunc_judged_{DS}.csv")
    if _judged_all(cut, out):
        return
    tj = _judge()(device)
    judge_resumable(cut, tj.score, out)


def stage_summary():
    import tqa_confirm as tc

    # Everything is computed before anything is written, so a stop leaves no file behind
    # and a rerun is never skipped on the strength of a half-written summary.
    sp, op = path(f"q2x_summary_{DS}.csv"), path(f"q2x_outcome_{DS}.json")
    for p in (sp, op):
        if os.path.exists(p):
            raise SystemExit(f"[q2x] {p} exists; move {sp} and {op} aside to rerun")
    lo, hi = q2_interval()
    full = td.read_csv(path(f"q2x_judged_{DS}.csv"))
    cut = td.read_csv(path(f"q2x_trunc_judged_{DS}.csv"))
    check_judged(full, cut)
    base = [r for r in full if r["direction"] == "baseline"]
    n_rand = len({r["direction"] for r in full if r["direction"].startswith("rand_")})
    ftabs = {c: tc.summarize(full, c) for c in SCORE_COLS}
    ctabs = {c: tc.summarize(cut + base, c) for c in SCORE_COLS}
    form = form_stats(full)
    rows = summary_rows(ftabs, ctabs, form)
    out = read_outcome(rows, lo, hi, n_rand)
    out.update(baseline=form.get(("baseline", 0.0), {}), score_col="truthful")
    td.write_new(sp, rows)
    with open(op, "w") as f:                  # written last: main skips on this file
        json.dump(out, f, indent=2)
    print(f"[q2x] scored on `truthful`; doses toward truthful; {n_rand} random directions",
          flush=True)
    for r in rows:
        if r["direction"] != Q2_DIR:
            continue
        print(f"[q2x] dose {r['frac']:>4g}  g {r['median_g']:+8.2f} crossed "
              f"{r['crossed_share']:.2f}  words {r['mean_words']:5.1f}  budget "
              f"{r['budget_hit_share']:.2f}  full {r['full_rate']:.3f} (+{r['full_gained']}"
              f"/-{r['full_lost']}, perm {r['full_perm_p']:.3g})  cut {r['cut_rate']:.3f} "
              f"(+{r['cut_gained']}/-{r['cut_lost']}, perm {r['cut_perm_p']:.3g})",
              flush=True)
    g = out["gate"]
    print(f"[q2x] R0 anchor at dose {ANCHOR:g}: {g['anchor_rate']:.3f} in "
          f"[{lo:.4f}, {hi:.4f}]? {'PASS' if g['passed'] else 'GATE FAILED'}", flush=True)
    if not g["passed"]:
        print("[q2x] R1/R2 not read: GATE FAILED (the numbers are under `diagnostic` in "
              f"{op})", flush=True)
    else:
        c = out["crossing"]
        print(f"[q2x] R1 crossing: realized {c['realized']}, predicted {c['predicted']}",
              flush=True)
    for f, v in out["per_dose"].items():
        print(f"[q2x] R2 dose {f}: {v['labels']}  (Holm p cut {v['cut_p_holm']:.3g}, "
              f"full {v['full_p_holm']:.3g})", flush=True)
    print(f"[q2x] HEADLINE: {out['headline']}"
          + ("  (no crossing by dose 5: read as uncrossed)" if out["uncrossed"] is True
             else ""),
          flush=True)


STAGES = ("steer", "judge", "truncate", "summary")


def main(argv=None):
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="questions (smoke)")
    ap.add_argument("--prefix", default="", help="prefix every output (smoke)")
    a = ap.parse_args(argv)
    PREFIX = a.prefix
    if a.limit and not a.prefix:
        raise SystemExit("--limit writes partial outputs; give it a --prefix")
    for st in STAGES if a.stage == "all" else (a.stage,):
        print(f"=== tqa_q2x: {st} ===", flush=True)
        if st == "summary" and os.path.exists(path(f"q2x_outcome_{DS}.json")):
            print("[q2x] summary exists, skipping", flush=True)
            continue
        if st == "steer":
            stage_steer(a.device, a.limit)
        elif st == "judge":
            stage_judge(a.device)
        elif st == "truncate":
            stage_truncate(a.device)
        else:
            stage_summary()


if __name__ == "__main__":
    main()
