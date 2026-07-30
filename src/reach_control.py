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
    try:
        e = float(eps_star)
        s = float(scale)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[control] frac_of: cannot convert scale={scale!r} / "
                         f"eps_star={eps_star!r} to float ({exc})")
    return 0.0 if abs(e) < 1e-12 else s / e


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
    scale_grid was built from.

    Returns `(rows, dropped)`. A judged row whose scale has no matching readout
    entry (e.g. a truncated readout CSV) is dropped rather than guessed — a
    dangerous silent failure, since it can delete exactly the crossing fracs and
    turn `actuatable` into `no-crossing`. `dropped` is returned rather than
    recomputed by the caller so the log warning here and any downstream artifact
    (e.g. the sidecar JSON) are guaranteed to agree, by construction, on the same
    number."""
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
    dropped = 0
    for r in judged:
        if r["direction"] != direction:
            continue
        s = float(r["scale"])
        if s not in g:
            dropped += 1
            continue
        out.append({"frac": frac_of(s, eps), "g_read": float(np.mean(g[s])),
                    "refused": float(r["refused"])})
    if dropped:
        print(f"[control] mean_arm_rows: dropped {dropped} judged row(s) for "
              f"direction={direction!r} with no matching readout scale — the "
              f"readout CSV may be truncated; this can silently delete crossing "
              f"fracs and understate the verdict")
    return out, dropped


def aggregate(rows, ndigits=FRAC_ROUND):
    """[{frac,g_read,refused}] -> {frac_bucket: {"n","g_read","frac_refused"}}."""
    by = {}
    for r in rows:
        by.setdefault(round(float(r["frac"]), ndigits), []).append(r)
    return {f: {"n": len(v),
                "g_read": float(np.mean([x["g_read"] for x in v])),
                "frac_refused": float(np.mean([x["refused"] for x in v]))}
            for f, v in by.items()}


def readout_crossed(table, baseline=0.0):
    """{frac: row} -> {frac: bool}: did steering CROSS the boundary at that frac?

    g = w.h_tgt - t02, and g <= 0 is inside the target halfspace. A crossing is a
    SIGN CHANGE relative to the unsteered baseline, not mere membership: the
    baseline must start outside (g > 0) and the steered point must land inside.
    Membership alone is a false-positive path to `actuatable` — eps* is the
    dataset-wide median_eps_star from reach_summary (reach_steer.py:123) while the
    baseline g_read is measured on a DIFFERENT population (FACTUAL_PROMPTS or the
    refusal holdout, reach_steer.py:111,133) that t02 was never fit on. If that
    population already sits inside the halfspace, `g <= 0` is trivially true at
    every frac and the co-occurrence gate degenerates into "did behaviour move
    anywhere". run() additionally fails loud in that case: a vacuous crossing test
    cannot support any verdict.
    """
    g0 = table[baseline]["g_read"] if baseline in table else None
    if g0 is None or g0 <= 0.0:
        return {f: False for f in table}
    return {f: bool(f != baseline and v["g_read"] <= 0.0) for f, v in table.items()}


def behavior_delta(table, baseline=0.0):
    """{frac: row} -> {frac: frac_refused - frac_refused_at_baseline}."""
    if baseline not in table:
        raise SystemExit(f"[control] no baseline row at frac={baseline} — the "
                         f"unsteered scale-0 generations are missing")
    base = float(table[baseline]["frac_refused"])
    return {f: float(v["frac_refused"]) - base for f, v in table.items()}


def verdict(crossed, delta, min_delta=MIN_DELTA, baseline=0.0):
    """Name the 2x2 cell over the non-baseline fracs.

    `actuatable` requires the crossing and the behaviour change to happen at the
    SAME frac. Independent quantifiers ("crossed somewhere" and "moved somewhere")
    would label a crossing at frac -1 plus degradation at frac +2 as actuatable,
    which is the expected shape of an OFF-TARGET result: eps* is positive only
    where g > 0 (reach_steer.py:167), so crossings occur at negative fracs, while
    +2*eps* is a large residual-stream shift that can wreck generations without
    crossing anything. That pattern is the `inert` cell, not the `actuatable` one.

    Refuses a table with no non-baseline frac outright (rather than degrading to
    a plausible-looking `no-crossing`): `require_signal` guards the `run()` path,
    but a direct or future caller of `verdict` alone must not be able to get a
    label out of a table that never had any steered data in it.
    """
    fr = [f for f in crossed if f != baseline]
    if not fr:
        raise SystemExit(f"[control] verdict: no non-baseline frac in `crossed` "
                         f"(only frac={baseline} present) — refusing to report a "
                         f"verdict with no steered data")
    moved = {f: abs(delta.get(f, 0.0)) >= min_delta for f in fr}
    if any(crossed[f] and moved[f] for f in fr):
        return "actuatable"
    if any(crossed[f] for f in fr):
        return "readout-only"   # crossed; any movement was at a non-crossing frac
    return "inert" if any(moved[f] for f in fr) else "no-crossing"


def require_signal(table, baseline=0.0):
    """Raise if `table` has no bucket besides the baseline.

    Two different causes give this identical symptom, so name both: (1) scale_grid
    clamps magnitudes to `1.5 * input_scale` (reach_steer.py) — when eps* far
    exceeds that cap, every steered frac (scale / eps*) rounds to ~0 at
    FRAC_ROUND digits and the treatment rows get silently averaged INTO the
    baseline; (2) in the per-statement arm, eps_i = 0 whenever a statement's own
    g_i <= 0 already (reach_steer.py:167), and scale_grid then returns [0.0] only
    — every sampled statement already sat inside the target, nothing to do with
    input_scale at all. Reporting `no-crossing` in either case would silently
    read a contaminated or vacuous baseline as merely "underpowered"."""
    fr = [f for f in table if f != baseline]
    if not fr:
        raise SystemExit(f"[control] no non-baseline frac bucket found (every row "
                         f"rounded into frac={baseline}) — either the scale grid "
                         f"was fully clamped (eps* far exceeds 1.5*input_scale) or, "
                         f"in the per-statement arm, every sampled statement already "
                         f"had g<=0 (eps_i=0, reach_steer.py:167) so scale_grid "
                         f"returned only the baseline; check input_scale AND the "
                         f"per-statement eps_i values before assuming a clamp — "
                         f"refusing to report a verdict off a contaminated baseline")


def _read(path):
    if not os.path.exists(path):
        raise SystemExit(f"[control] missing {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path):
    if not os.path.exists(path):
        raise SystemExit(f"[control] missing {path}")
    with open(path) as f:
        return json.load(f)


COLUMN_SEMANTICS = {
    "frac_eps_star": "scale as a multiple of eps* (0 = unsteered baseline; "
                     "+/-1, +/-2 by construction up to the 1.5*input_scale cap)",
    "n": "number of judged rows aggregated into this frac bucket",
    "mean_g_read": "mean readout g = w.h_tgt - t02 over the bucket; g<=0 means the "
                   "activation is inside the target halfspace (certificate met)",
    "crossed": "1 if this frac is a genuine sign change from the baseline (baseline "
              "g_read>0 AND this bucket's mean_g_read<=0), else 0 -- the baseline "
              "frac itself is always 0",
    "frac_refused": "fraction of completions in this bucket judged as refused",
    "delta_vs_baseline": "frac_refused minus frac_refused at the frac=0 baseline; "
                         "movement is judged on magnitude (see min_delta), so a "
                         "negative delta counts as moved just like a positive one",
}


def _sixsig(x):
    """Round to 6 significant figures, matching the CSV's `{:.6g}` formatting, so
    the JSON sidecar and the CSV never disagree on precision."""
    return float(f"{x:.6g}")


def run(ds, arm, direction=DEFAULT_DIRECTION, min_delta=MIN_DELTA):
    judged = _read(f"judge_refusal_{ds}_{arm}.csv")
    summ = _read_json(f"reach_summary_{ds}.json")
    n_empty_completion = sum(1 for r in judged if not r.get("completion", "").strip())
    if arm == "mean":
        readout = _read(f"reach_steer_readout_{ds}.csv")
        rows, dropped = mean_arm_rows(judged, readout, summ, direction)
        direction_used = direction
    else:
        rows = align_stmt_rows(judged, _read(f"reach_steer_stmt_meta_{ds}.csv"))
        dropped = 0
        direction_used = judged[0]["direction"] if judged else None
    table = aggregate(rows)
    require_signal(table)
    if 0.0 in table and table[0.0]["g_read"] <= 0.0:
        raise SystemExit(f"[control] baseline readout is already inside the target "
                         f"halfspace (g_read={table[0.0]['g_read']:.6g} <= 0) — the "
                         f"crossing test is vacuous: this prompt population does not "
                         f"sit where t02 was fit, so `readout_crossed` cannot support "
                         f"any verdict here (it will report every frac as not-crossed "
                         f"by construction; do not read that as `no-crossing`)")
    crossed, delta = readout_crossed(table), behavior_delta(table)
    v = verdict(crossed, delta, min_delta)
    print(f"[control] {ds} arm={arm}: VERDICT = {v}")
    print(f"  {'frac_eps*':>10s} {'n':>5s} {'mean g_read':>12s} {'crossed':>8s} "
          f"{'refused':>8s} {'delta':>8s}")
    out_rows = [("frac_eps_star", "n", "mean_g_read", "crossed", "frac_refused",
                 "delta_vs_baseline")]
    json_table = {}
    for f in sorted(table):
        r = table[f]
        out_rows.append((f"{f:.6g}", r["n"], f"{r['g_read']:.6g}",
                         int(crossed[f]), f"{r['frac_refused']:.6g}",
                         f"{delta[f]:.6g}"))
        json_table[f"{f:.6g}"] = {
            "n": r["n"], "mean_g_read": _sixsig(r["g_read"]),
            "crossed": int(crossed[f]), "frac_refused": _sixsig(r["frac_refused"]),
            "delta_vs_baseline": _sixsig(delta[f]),
        }
        print(f"  {f:>10.2f} {r['n']:>5d} {r['g_read']:>12.3f} "
              f"{str(crossed[f]):>8s} {r['frac_refused']:>8.3f} {delta[f]:>+8.3f}")

    off_target = [f for f in sorted(table) if f != 0.0 and not crossed[f]
                  and abs(delta.get(f, 0.0)) >= min_delta]
    if off_target and v == "readout-only":
        print(f"[control] WARNING: behavior moved at non-crossing frac(s) "
              f"{off_target} — this is off-target degradation, not evidence "
              f"of actuation at the certificate boundary")
    elif off_target and v == "actuatable":
        print(f"[control] WARNING: verdict is actuatable, but behavior ALSO moved "
              f"at non-crossing frac(s) {off_target} — broad-perturbation, "
              f"off-target movement, not just movement at the certificate "
              f"boundary; the actuatable verdict still stands, but interpret it "
              f"with this in mind")

    csv_out = f"reach_control_{ds}_{arm}.csv"
    with open(csv_out, "w", newline="") as f2:
        csv.writer(f2).writerows(out_rows)
    print(f"[control] wrote {csv_out}")

    json_out = f"reach_control_{ds}_{arm}.json"
    with open(json_out, "w") as f3:
        json.dump({
            "dataset": ds,
            "arm": arm,
            "direction": direction_used,
            "min_delta": min_delta,
            "verdict": v,
            "dropped_unmatched_rows": dropped,
            "empty_completion_rows": n_empty_completion,
            "table": json_table,
            "column_semantics": COLUMN_SEMANTICS,
        }, f3, indent=2)
    print(f"[control] wrote {json_out}")
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
