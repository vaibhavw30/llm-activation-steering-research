"""
investigate_steer.py — the full statistical investigation of the judged steering sweep.

Reads judge_steer_<ds>.csv (+ judge_interpret_<ds>.csv) and runs, over both datasets:

  Q1  lie-asymmetry            paired McNemar on FALSE, - vs +          (is '-' more FALSE than '+'?)
  EQ  equivalence / bound      CI + TOST on the FALSE-asymmetry         (rule OUT an effect > margin)
  BOOT prompt-clustered CI     bootstrap resampling PROMPTS             (honest CI; prompts repeat)
  Q2  degradation trend        Cochran-Armitage across |scale|          (INCOH up / TRUE down?)
  Q3  signed FALSE trend       Cochran-Armitage across signed scale     (any lie direction?)
  Q4  omnibus                  verdict x sign chi-square
  FDR per-prompt heterogeneity BH-corrected per-prompt asymmetry        (does ONE prompt hide an effect?)
  VAL baseline validity        binomial: scale-0 TRUE rate ~ 1.0        (is the instrument clean now?)
  INT interpret bound          rule-of-three on n/10 truth-flips        (bound the 0/10)
  PWR power / sample size      prompts needed for a target effect

Pure stat helpers (no I/O) are unit-tested in tests/test_investigate_steer.py.
Run:  python src/investigate_steer.py            # both datasets, whatever CSVs are present
"""
import csv
import math
import argparse
from scipy import stats

VERDS = ("TRUE", "FALSE", "INCOHERENT")
DATASETS_DEFAULT = ["cities", "common_claim_true_false"]
INPUT_PREFIX = "judge_steer"   # overridable so the MAG arm can point at judge_mag_steer_<ds>.csv


# ----------------------------------------------------------------- pure helpers (unit-tested)
def proportion_diff_ci(p1, n1, p2, n2, conf=0.95):
    """Wald CI for (p1 - p2). Returns (diff, lo, hi)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 and n2 else 0.0
    return diff, diff - z * se, diff + z * se


def tost_equivalent(p1, n1, p2, n2, margin, alpha=0.05):
    """Two One-Sided Tests: is |p1 - p2| statistically BELOW `margin`? Returns (is_equivalent, p_tost).
    Equivalent iff the (1-2*alpha) CI for the diff lies entirely within (-margin, +margin)."""
    diff, lo, hi = proportion_diff_ci(p1, n1, p2, n2, conf=1 - 2 * alpha)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 and n2 else 0.0
    if se == 0:
        return bool(abs(diff) < margin), 0.0
    z_lo = (diff - (-margin)) / se   # H0: diff <= -margin
    z_hi = ((margin) - diff) / se    # H0: diff >=  margin
    p_tost = max(1 - stats.norm.cdf(z_lo), 1 - stats.norm.cdf(z_hi))
    return bool(lo > -margin and hi < margin), float(p_tost)


def benjamini_hochberg(pvals, q=0.05):
    """Return a boolean list: which p-values are rejected at FDR level q (BH step-up)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rej = [False] * m
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            rej[i] = True
    return rej


def binom_upper(k, n, conf=0.95):
    """One-sided upper bound on a proportion given k successes in n (Clopper-Pearson).
    For k=0 this is the 'rule of three'-style bound (0/10 -> ~0.259)."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return stats.beta.ppf(conf, k + 1, n - k)


def n_per_group(p1, p2, alpha=0.05, power=0.80):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    return ((za * math.sqrt(2 * pbar * (1 - pbar))
             + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p1 - p2) ** 2)


def cochran_armitage(levels_counts):
    """levels_counts: dict level(float) -> [n_outcome, n_total]. Trend z and two-sided p."""
    levels = sorted(levels_counts)
    N = sum(v[1] for v in levels_counts.values())
    R = sum(v[0] for v in levels_counts.values())
    if R == 0 or R == N or N == 0:
        return 0.0, 1.0
    pbar = R / N
    num = sum(L * (levels_counts[L][0] - levels_counts[L][1] * pbar) for L in levels)
    var = pbar * (1 - pbar) * (sum(levels_counts[L][1] * L * L for L in levels)
                               - (sum(levels_counts[L][1] * L for L in levels) ** 2) / N)
    if var <= 0:
        return 0.0, 1.0
    z = num / math.sqrt(var)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# ----------------------------------------------------------------- data + analyses
def load(ds):
    try:
        rows = list(csv.DictReader(open(f"{INPUT_PREFIX}_{ds}.csv")))
    except FileNotFoundError:
        return []
    for r in rows:
        r["scale"] = float(r["scale"]); r["abs"] = abs(r["scale"]); r["ds"] = ds
    return rows


def false_pairs(rows):
    """Matched -s/+s per (prompt,direction,|scale|). Returns (b,c,pairs): b=FALSE only on -, c only on +."""
    by = {}
    for r in rows:
        if r["scale"] == 0:
            continue
        by.setdefault((r["direction"], r["abs"], r["prompt"]), {})[
            "neg" if r["scale"] < 0 else "pos"] = (r["verdict"] == "FALSE")
    b = c = 0
    for d in by.values():
        if "neg" in d and "pos" in d:
            b += d["neg"] and not d["pos"]
            c += d["pos"] and not d["neg"]
    n = sum(1 for d in by.values() if "neg" in d and "pos" in d)
    return b, c, n


def cluster_bootstrap_asym(rows, iters=2000, seed_base=0):
    """Prompt-clustered bootstrap CI for (FALSE rate on - side) - (on + side).
    Resamples PROMPTS with replacement (the unit of dependence). Deterministic given seed_base."""
    prompts = sorted({r["prompt"] for r in rows})
    byp = {p: [r for r in rows if r["prompt"] == p] for p in prompts}
    def rate(sub, sign):
        s = [r for r in sub if (r["scale"] < 0 if sign == "neg" else r["scale"] > 0)]
        return (sum(x["verdict"] == "FALSE" for x in s) / len(s)) if s else None
    diffs = []
    m = len(prompts)
    for it in range(iters):
        rng = _lcg(seed_base * 100003 + it + 1)
        pick = [prompts[rng() % m] for _ in range(m)]
        pool = [r for p in pick for r in byp[p]]
        rn, rp = rate(pool, "neg"), rate(pool, "pos")
        if rn is not None and rp is not None:
            diffs.append(rn - rp)
    diffs.sort()
    if not diffs:
        return 0.0, 0.0, 0.0
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return sum(diffs) / len(diffs), lo, hi


def _lcg(seed):
    """Tiny deterministic PRNG (numpy RNG is fine too, but keep it dependency-light + seedable)."""
    state = {"s": (seed % (2**31 - 1)) or 1}
    def nxt():
        state["s"] = (state["s"] * 48271) % (2**31 - 1)
        return state["s"]
    return nxt


def report(ds_rows):
    ALL = [r for rows in ds_rows.values() for r in rows]
    print(f"loaded {len(ALL)} judged completions "
          + ", ".join(f"{ds}:{len(r)}" for ds, r in ds_rows.items()))
    if not ALL:
        print("no judge_steer_*.csv present — pull the results back first."); return

    groups = [("POOLED", ALL)] + [(ds, r) for ds, r in ds_rows.items() if r]

    print("\n=== Q1/EQ. Lie-asymmetry (FALSE, - vs +): test AND bound it ===")
    for label, rows in groups:
        b, c, n = false_pairs(rows)
        p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
        neg = [r for r in rows if r["scale"] < 0]; pos = [r for r in rows if r["scale"] > 0]
        pn = sum(r["verdict"] == "FALSE" for r in neg) / len(neg) if neg else 0
        pp = sum(r["verdict"] == "FALSE" for r in pos) / len(pos) if pos else 0
        diff, lo, hi = proportion_diff_ci(pn, len(neg), pp, len(pos))
        eq10, _ = tost_equivalent(pn, len(neg), pp, len(pos), margin=0.10)
        print(f"  {label:>22}: McNemar b={b} c={c} p={p:.3f} | "
              f"Δ(FALSE neg-pos)={diff:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] | "
              f"equiv@0.10: {'YES (bounded null)' if eq10 else 'no'}")

    print("\n=== BOOT. Prompt-clustered bootstrap CI on the asymmetry (honest, prompts repeat) ===")
    for label, rows in groups:
        m, lo, hi = cluster_bootstrap_asym(rows)
        print(f"  {label:>22}: Δ={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
              f"{'excludes 0? no -> null' if lo <= 0 <= hi else 'EXCLUDES 0'}")

    print("\n=== Q2. Degradation trend (Cochran-Armitage across |scale|) ===")
    for label, rows in groups:
        for outc in ("INCOHERENT", "TRUE"):
            lc = {}
            for r in rows:
                lc.setdefault(r["abs"], [0, 0]); lc[r["abs"]][1] += 1
                lc[r["abs"]][0] += (r["verdict"] == outc)
            z, p = cochran_armitage(lc)
            sig = "  *SIG*" if p < 0.05 else ""
            print(f"  {label:>22} P({outc:<10})~|scale|: z={z:+.2f} p={p:.3f}{sig}")

    print("\n=== Q4. Omnibus verdict x sign chi-square ===")
    for label, rows in groups:
        bucket = lambda s: "neg" if s < 0 else ("pos" if s > 0 else "zero")
        tab = [[sum(1 for r in rows if bucket(r["scale"]) == bk and r["verdict"] == v)
                for v in VERDS] for bk in ("neg", "zero", "pos")]
        tab = [row for row in tab if sum(row) > 0]
        try:
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            print(f"  {label:>22}: chi2={chi2:.2f} dof={dof} p={p:.3f} {'SIG' if p<0.05 else 'n.s.'}")
        except ValueError:
            print(f"  {label:>22}: (degenerate table)")

    print("\n=== FDR. Per-prompt lie-asymmetry (Benjamini-Hochberg q=0.05) ===")
    for label, rows in groups:
        prompts = sorted({r["prompt"] for r in rows})
        pv, labels = [], []
        for pr in prompts:
            b, c, n = false_pairs([r for r in rows if r["prompt"] == pr])
            if b + c > 0:
                pv.append(stats.binomtest(b, b + c, 0.5).pvalue); labels.append(pr)
        if pv:
            rej = benjamini_hochberg(pv, 0.05)
            hits = [labels[i] for i in range(len(pv)) if rej[i]]
            print(f"  {label:>22}: {sum(rej)}/{len(pv)} prompts significant after FDR"
                  + (f" -> {hits}" if hits else " (none — no single prompt hides an effect)"))

    print("\n=== VAL. Baseline validity: scale-0 TRUE rate (should be ~1.0 after the fix) ===")
    for label, rows in groups:
        base = [r for r in rows if r["scale"] == 0]
        if base:
            k = sum(r["verdict"] == "TRUE" for r in base)
            p = stats.binomtest(k, len(base), 0.5, alternative="greater").pvalue
            print(f"  {label:>22}: {k}/{len(base)} TRUE at scale 0 = {k/len(base):.2f}  "
                  f"(binom>0.5 p={p:.3g})  {'CLEAN' if k/len(base) >= 0.85 else 'STILL NOISY'}")

    # interpret bound
    print("\n=== INT. Interpret truth-flip bound (rule-of-three on n/10) ===")
    tot_k = tot_n = 0
    for ds in ds_rows:
        try:
            ir = list(csv.DictReader(open(f"judge_interpret_{ds}.csv")))
        except FileNotFoundError:
            continue
        k = sum(str(r.get("manipulates_truth", "")).lower() == "true" for r in ir)
        tot_k += k; tot_n += len(ir)
        print(f"  {ds:>22}: {k}/{len(ir)} flip truth  (95% upper bound {binom_upper(k, len(ir)):.2f})")
    if tot_n:
        print(f"  {'POOLED':>22}: {tot_k}/{tot_n} flip truth  (95% upper bound {binom_upper(tot_k, tot_n):.2f})")

    print("\n=== PWR. Prompts/scale needed to detect a true FALSE-asymmetry (80% power, a=0.05) ===")
    for delta in (0.10, 0.15, 0.20):
        need = n_per_group(0.20 + delta / 2, 0.20 - delta / 2)
        print(f"  Δ={delta:.2f}: ~{math.ceil(need)} completions/side -> ~{math.ceil(need/4)} prompts/scale")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="judge_steer",
                    help="CSV prefix: judge_steer (DCT arm) or judge_mag_steer (MAG arm)")
    ap.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    args = ap.parse_args()
    global INPUT_PREFIX
    INPUT_PREFIX = args.prefix
    dsets = [d for d in args.datasets.split(",") if d]
    report({ds: load(ds) for ds in dsets})


if __name__ == "__main__":
    main()
