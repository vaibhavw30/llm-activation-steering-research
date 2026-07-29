"""reach_control.py — Horizon-1 1.1 Task A8: the positive-control verdict.

The audit's deliverable 2x2 is (readout moved) x (behavior moved). The truth run
landed in "readout-only": steering past eps* flipped the probe readout and left the
completions alone. This assembles the same 2x2 for any dataset+arm and names the cell.

  actuatable   crossing the certificate boundary changes behavior  -> INSTRUMENT VALID
  readout-only crossing moves the readout, not behavior            -> the truth outcome
  inert        behavior moves without a readout crossing           -> off-target effect
  no-crossing  neither                                             -> underpowered sweep

Rows are indexed by frac = scale / eps*, NOT by raw scale: in the per-statement arm
every statement has its own eps_i (reach_steer.py:136-138), so raw scales are all
distinct and would give n=1 buckets. frac is {0, +/-1, +/-2} by construction and is
the axis the hypothesis is about ("does behavior move at approximately eps*").

    PYTHONPATH=src .venv/bin/python src/reach_control.py --dataset refusal --arm mean
"""
import argparse
import csv
import json
import os

import numpy as np

MIN_DELTA = 0.10          # behavior counts as "moved" at >= 10 points
FRAC_ROUND = 2
DEFAULT_DIRECTION = "jtw_mean_diff_tgt"


def frac_of(scale, eps_star):
    """scale as a multiple of eps*; 0.0 when eps* is 0 (already inside the target)."""
    e = float(eps_star)
    return 0.0 if abs(e) < 1e-12 else float(scale) / e


def align_stmt_rows(judged, meta):
    """Pair per-statement judged rows with their meta rows BY POSITION.

    reach_steer.arm_per_stmt appends to `rows` and `meta_rows` inside the same loop
    iteration (src/reach_steer.py:143-144) and refusal_judge.score_rows preserves
    order, so row k of judge_refusal_<ds>_stmt.csv is row k of
    reach_steer_stmt_meta_<ds>.csv. reach_steer_stmt_meta has no prompt column, so
    there is no other join key — this makes the dependency explicit and refuses to
    guess when it does not hold."""
    if len(judged) != len(meta):
        raise SystemExit(f"[control] row-count mismatch: {len(judged)} judged vs "
                         f"{len(meta)} meta rows — the two files are not from the "
                         f"same reach_steer run")
    out = []
    for k, (j, m) in enumerate(zip(judged, meta)):
        if abs(float(j["scale"]) - float(m["scale"])) > 1e-9:
            raise SystemExit(f"[control] scale mismatch at row {k}: judged "
                             f"{j['scale']} vs meta {m['scale']} — row alignment "
                             f"is broken, refusing to guess")
        out.append({"frac": frac_of(m["scale"], m["eps_star"]),
                    "g_read": float(m["g_read"]),
                    "refused": float(j["refused"])})
    return out


def mean_arm_rows(judged, readout, summ, direction=DEFAULT_DIRECTION):
    """Mean-arm rows for one steering direction. The readout CSV shares the (direction,
    scale) grid across prompts, so g_read is averaged over prompts per grid point;
    eps* is that direction's median eps* from reach_summary — the same number
    scale_grid was built from."""
    g = {}
    for r in readout:
        if r["direction"] != direction:
            continue
        g.setdefault(float(r["scale"]), []).append(float(r["g_read"]))
    wn = direction[4:] if direction.startswith("jtw_") else direction
    eps = summ["directions"].get(wn, {}).get("median_eps_star")
    if not eps:
        raise SystemExit(f"[control] no median_eps_star for {wn} in the summary")
    out = []
    for r in judged:
        if r["direction"] != direction:
            continue
        s = float(r["scale"])
        if s not in g:
            continue
        out.append({"frac": frac_of(s, eps), "g_read": float(np.mean(g[s])),
                    "refused": float(r["refused"])})
    return out


def aggregate(rows, ndigits=FRAC_ROUND):
    """[{frac,g_read,refused}] -> {frac_bucket: {"n","g_read","frac_refused"}}."""
    by = {}
    for r in rows:
        by.setdefault(round(float(r["frac"]), ndigits), []).append(r)
    return {f: {"n": len(v),
                "g_read": float(np.mean([x["g_read"] for x in v])),
                "frac_refused": float(np.mean([x["refused"] for x in v]))}
            for f, v in by.items()}


def readout_crossed(table):
    """{frac: row} -> {frac: bool}. g = w.h_tgt - t02; g <= 0 is inside the target
    halfspace, i.e. the reachability goal was met at that frac."""
    return {f: bool(v["g_read"] <= 0.0) for f, v in table.items()}


def behavior_delta(table, baseline=0.0):
    """{frac: row} -> {frac: frac_refused - frac_refused_at_baseline}."""
    if baseline not in table:
        raise SystemExit(f"[control] no baseline row at frac={baseline} — the "
                         f"unsteered scale-0 generations are missing")
    base = float(table[baseline]["frac_refused"])
    return {f: float(v["frac_refused"]) - base for f, v in table.items()}


def verdict(crossed, delta, min_delta=MIN_DELTA):
    """Name the 2x2 cell over the non-baseline fracs."""
    fr = [f for f in crossed if f != 0.0]
    any_cross = any(crossed[f] for f in fr)
    any_move = any(abs(delta.get(f, 0.0)) >= min_delta for f in fr)
    if any_cross and any_move:
        return "actuatable"
    if any_cross:
        return "readout-only"
    return "inert" if any_move else "no-crossing"


def _read(path):
    if not os.path.exists(path):
        raise SystemExit(f"[control] missing {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(ds, arm, direction=DEFAULT_DIRECTION, min_delta=MIN_DELTA):
    judged = _read(f"judge_refusal_{ds}_{arm}.csv")
    summ = json.load(open(f"reach_summary_{ds}.json"))
    if arm == "mean":
        rows = mean_arm_rows(judged, _read(f"reach_steer_readout_{ds}.csv"),
                             summ, direction)
    else:
        rows = align_stmt_rows(judged, _read(f"reach_steer_stmt_meta_{ds}.csv"))
    table = aggregate(rows)
    crossed, delta = readout_crossed(table), behavior_delta(table)
    v = verdict(crossed, delta, min_delta)
    print(f"[control] {ds} arm={arm}: VERDICT = {v}")
    print(f"  {'frac_eps*':>10s} {'n':>5s} {'mean g_read':>12s} {'crossed':>8s} "
          f"{'refused':>8s} {'delta':>8s}")
    out_rows = [("frac_eps_star", "n", "mean_g_read", "crossed", "frac_refused",
                 "delta_vs_baseline")]
    for f in sorted(table):
        r = table[f]
        out_rows.append((f"{f:.6g}", r["n"], f"{r['g_read']:.6g}",
                         int(crossed[f]), f"{r['frac_refused']:.6g}",
                         f"{delta[f]:.6g}"))
        print(f"  {f:>10.2f} {r['n']:>5d} {r['g_read']:>12.3f} "
              f"{str(crossed[f]):>8s} {r['frac_refused']:>8.3f} {delta[f]:>+8.3f}")
    out = f"reach_control_{ds}_{arm}.csv"
    with open(out, "w", newline="") as f2:
        csv.writer(f2).writerows(out_rows)
    print(f"[control] wrote {out}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=["mean", "stmt"])
    ap.add_argument("--direction", default=DEFAULT_DIRECTION,
                    help="mean arm only: which steered direction to tabulate")
    ap.add_argument("--min-delta", type=float, default=MIN_DELTA)
    a = ap.parse_args()
    run(a.dataset, a.arm, a.direction, a.min_delta)


if __name__ == "__main__":
    main()
