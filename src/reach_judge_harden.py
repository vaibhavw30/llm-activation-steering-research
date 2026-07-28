"""reach_judge_harden.py — Horizon-0 item 0.3: judge hardening (CPU, free).

The deep audit found the FALSE/INCOHERENT verdicts concentrate on a handful of
prompts that fail identically at scale 0 (MCQ-style completions, judge factual
errors) — baseline judge noise, not steering effects. Hardening = drop every
prompt whose scale-0 completion is judged anything but TRUE, then recompute the
verdict fractions and the FALSE-vs-|scale| trend. Both arms, both datasets,
local CSVs only.

    PYTHONPATH=src python src/reach_judge_harden.py --dataset cities
"""
import argparse
import csv
import os

import numpy as np

ARM_FILES = [("mean", "judge_reach_steer_{ds}.csv"),
             ("stmt", "judge_reach_steer_stmt_{ds}.csv")]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def problem_prompts(rows):
    """Prompts whose scale-0 completion is judged anything but TRUE."""
    return {r["prompt"] for r in rows
            if float(r["scale"]) == 0.0 and r["verdict"] != "TRUE"}


def fractions(rows, exclude=frozenset()):
    """{(direction, scale): (n, frac_true, frac_false, frac_incoherent)}."""
    by = {}
    for r in rows:
        if r["prompt"] in exclude:
            continue
        by.setdefault((r["direction"], float(r["scale"])), []).append(r["verdict"])
    return {k: (len(v),
                sum(x == "TRUE" for x in v) / len(v),
                sum(x == "FALSE" for x in v) / len(v),
                sum(x == "INCOHERENT" for x in v) / len(v))
            for k, v in by.items()}


def _ranks(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    """Rank correlation without scipy; nan if either side is constant."""
    rx, ry = _ranks(x), _ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _trend(fr, direction):
    pts = sorted((abs(s), v[2]) for (d, s), v in fr.items() if d == direction)
    if len(pts) < 3:
        return float("nan")
    return spearman([p[0] for p in pts], [p[1] for p in pts])


def harden(ds):
    out = [("arm", "direction", "scale", "n", "frac_true", "frac_false",
            "frac_incoherent", "hardened")]
    for arm, tmpl in ARM_FILES:
        path = tmpl.format(ds=ds)
        if not os.path.exists(path):
            print(f"[harden] {path} missing — skipping {arm} arm")
            continue
        rows = load_rows(path)
        bad = problem_prompts(rows)
        raw, hard = fractions(rows), fractions(rows, exclude=bad)
        n_drop = sum(r["prompt"] in bad for r in rows)
        print(f"[harden] {ds} {arm}: {len(bad)} problem prompts "
              f"({n_drop}/{len(rows)} rows dropped)")
        for tag, fr in (("0", raw), ("1", hard)):
            for (d, s), (n, ft, ff, fi) in sorted(fr.items()):
                out.append((arm, d, s, n, f"{ft:.4f}", f"{ff:.4f}", f"{fi:.4f}", tag))
        for d in sorted({dd for dd, _ in raw}):
            print(f"  {d}: FALSE-vs-|scale| spearman raw {_trend(raw, d):+.3f} "
                  f"-> hardened {_trend(hard, d):+.3f}")
    with open(f"judge_hardened_fracs_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"[harden] wrote judge_hardened_fracs_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    harden(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
