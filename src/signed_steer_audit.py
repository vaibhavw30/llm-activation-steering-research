"""signed_steer_audit.py — 0a: per-statement SIGNED steerability of the existing runs.

Tan et al. (arXiv:2407.12404) report that in many datasets nearly half the examples
are ANTI-steerable: the same vector pushes them the wrong way. A population like that
produces a flat aggregate while every individual statement is moving. Our headline
null was an aggregate FALSE rate, so the objection is live and has to be answered
from data we already own.

The per-statement arm already swept SIGNED scales (+-1x and +-2x each statement's own
eps*), so the test needs no GPU. For each statement,

    v(s) in {+1 TRUE, 0 INCOHERENT, -1 FALSE}
    signed_i = mean v(s>0) - mean v(s<0)

and w = mean_diff points toward TRUE, so a working actuator gives signed_i > 0.
Three populations get counted separately, because they are three different failures:

    inert     completion byte-identical at every scale  (nothing moved at all)
    churn     text moved, verdict did not               (moved, not toward truth)
    mover     verdict moved                             (signed_i != 0)

Anti-steerability can only explain a null inside `mover`. If most statements are
`inert`, cancellation is not the mechanism and the budget is.

    PYTHONPATH=src python src/signed_steer_audit.py --dataset cities
"""
import argparse

import numpy as np
import pandas as pd

VMAP = {"TRUE": 1.0, "INCOHERENT": 0.0, "FALSE": -1.0}


def audit(path):
    d = pd.read_csv(path)
    missing = sorted(set(d["verdict"].astype(str)) - set(VMAP))
    if missing:
        raise SystemExit(f"unmapped verdicts in {path}: {missing}")
    d["v"] = d["verdict"].map(VMAP)
    rows = []
    for prompt, g in d.groupby("prompt"):
        base = g[g["scale"] == 0.0]
        pos, neg = g[g["scale"] > 0], g[g["scale"] < 0]
        if len(base) != 1 or not len(pos) or not len(neg):
            continue
        b = base.iloc[0]
        n_txt = int((g["completion"] != b["completion"]).sum())
        signed = float(pos["v"].mean() - neg["v"].mean())
        rows.append(dict(
            prompt=prompt, n_rows=len(g), n_text_changed=n_txt,
            v0=float(b["v"]), v_pos=float(pos["v"].mean()),
            v_neg=float(neg["v"].mean()), signed=signed,
            verdict_changed=int((g["v"] != b["v"]).any()),
            population=("inert" if n_txt == 0 else
                        "mover" if signed != 0.0 else "churn")))
    return pd.DataFrame(rows)


def sign_test(x):
    """Two-sided exact binomial on the nonzero signs. scipy if present, else an
    exact sum over the binomial pmf — this must not depend on an optional import."""
    nz = x[x != 0]
    k, n = int((nz > 0).sum()), int(len(nz))
    if n == 0:
        return k, n, float("nan")
    try:
        from scipy.stats import binomtest
        return k, n, float(binomtest(k, n, 0.5).pvalue)
    except ImportError:
        from math import comb
        lo = min(k, n - k)
        tail = sum(comb(n, j) for j in range(lo + 1)) / 2 ** n
        return k, n, float(min(1.0, 2 * tail))


def report(r, label):
    n = len(r)
    print(f"\n=== signed steerability: {label} ===")
    print(f"statements                         {n}")
    for pop in ("inert", "churn", "mover"):
        m = int((r["population"] == pop).sum())
        print(f"  {pop:6s}                           {m:4d}  ({m / n:5.1%})")
    k, nn, p = sign_test(r["signed"].values)
    print(f"signed_i = v(+eps) - v(-eps)")
    print(f"  mean {r['signed'].mean():+.4f}   sd {r['signed'].std():.4f}")
    print(f"  movers {nn}:  {k} with the direction, {nn - k} against it")
    print(f"  exact two-sided sign test        p = {p:.5f}")
    if nn and p < 0.05 and k > nn - k:
        print("  -> effect is DIRECTIONALLY CORRECT and significant. The aggregate "
              "null is NOT anti-steerability cancellation; it is that almost nothing "
              "moves at all.")
    elif nn and (nn - k) / nn > 0.35:
        print(f"  -> {(nn - k) / nn:.0%} of movers go the WRONG way: partial "
              "anti-steerability, consistent with Tan et al. Report the distribution, "
              "never the mean.")
    else:
        print("  -> no significant signed effect in either direction.")
    return dict(n=n, inert=int((r.population == 'inert').sum()),
                churn=int((r.population == 'churn').sum()),
                mover=int((r.population == 'mover').sum()),
                signed_mean=float(r.signed.mean()), pos=k, neg=nn - k, p=p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--input", default=None)
    a = p.parse_args()
    src = a.input or f"judge_reach_steer_stmt_{a.dataset}.csv"
    r = audit(src)
    s = report(r, f"{a.dataset} ({src})")
    r.to_csv(f"signed_steer_{a.dataset}.csv", index=False)
    pd.DataFrame([s]).to_csv(f"signed_steer_summary_{a.dataset}.csv", index=False)
    print(f"\nwrote signed_steer_{a.dataset}.csv and signed_steer_summary_{a.dataset}.csv")


if __name__ == "__main__":
    main()
