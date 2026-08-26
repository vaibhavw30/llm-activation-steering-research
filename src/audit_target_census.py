# src/audit_target_census.py
"""S4 — target set census and semantic vacuity rate.

Assumption under test: the certified target set represents the behavior we care about, namely
the model asserting something false.

Reads committed artifacts only.

    ./.venv/bin/python src/audit_target_census.py --dataset cities

TWO MEASUREMENTS, one of which needs no judge:

  1. Census. What tokens did the min-norm solver actually pick as targets, and what fraction of
     them could possibly change a claim's truth value? Pure counting, fully deterministic.

  2. Vacuity. Of the certified argmax flips, what fraction leave the claim's truth value
     unchanged? On `cities` this is decidable WITHOUT a judge, because the dataset is
     templated: got_datasets/cities.csv carries `correct_country` per statement, so
     "does the steered completion still assert the true country?" is a string test against
     ground truth rather than a model's opinion. That is stricter and cheaper than judging it.

     On `common_claim_true_false` the claims are free-form and no such test exists. The census
     is reported and the vacuity rate is left to the judge step, which is flagged as pending.

`target_mode` differs between the datasets and this matters more than the spec recorded:
     cities        -> "countries"   (target drawn from country-name tokens)
     common_claim  -> "runnerup"    (target IS the second-most-likely token)
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from token_conclusions import load_arm

# Tokens that cannot carry the truth value of a "city X is in country Y" claim. Directional
# modifiers are the important entries: "North East China" is still China. Listed explicitly so
# the classification is auditable rather than a judgment call buried in code.
DIRECTIONAL = {"north", "south", "east", "west", "northern", "southern", "eastern", "western",
               "central", "upper", "lower"}
FUNCTION = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "is", "was", "its",
            "their", "his", "her", "it", "that", "this", "for", "with", "by", "from", "as",
            "not", "but", "be", "are", "were", "has", "have", "had", "he", "she", "they"}


def classify_token(tok):
    """content, directional, function, or punctuation."""
    t = tok.strip().lower()
    if not t or not re.search(r"[a-z0-9]", t):
        return "punctuation"
    if t in DIRECTIONAL:
        return "directional"
    if t in FUNCTION:
        return "function"
    return "content"


def wilson(k, n, conf=0.95):
    """Wilson score interval. Returns (lo, hi); (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    a = 1 - conf
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def census(ds):
    g = pd.read_csv(f"token_geom_{ds}.csv")
    mode = str(np.load(f"token_geom_{ds}.npz", allow_pickle=True)["target_mode"])
    g["tgt_class"] = g["tok_tgt"].map(classify_token)
    counts = g["tok_tgt"].value_counts()
    cls = g["tgt_class"].value_counts()
    return g, mode, counts, cls


def vacuity_cities(g, arm_path, frac=1.0, direction="oracle"):
    """Deterministic vacuity on the templated cities claims. No judge involved.

    A certified flip is VACUOUS if the steered completion still names the statement's true
    country. It is a SEMANTIC CHANGE if the true country has disappeared from the completion.
    """
    # Join path, each link verified: the arm's `stmt` is a row index equal to token_geom's
    # `idx` (set equality holds), token_geom's `statement` joins to got_datasets at 100%
    # coverage. Do NOT join on `stem`: only 192 of the 200 stems are unique.
    truth = pd.read_csv("got_datasets/cities.csv")[["statement", "correct_country"]]
    truth = truth.drop_duplicates("statement")
    idx2country = (g[["idx", "statement"]]
                   .merge(truth, on="statement", how="left")
                   .set_index("idx")["correct_country"].to_dict())

    arm = load_arm(arm_path)
    s = arm[(arm["direction"] == direction) & (np.isclose(arm["frac"], frac))].copy()
    base = arm[(arm["direction"] == direction) & (np.isclose(arm["frac"], 0.0))]
    base = base.set_index("stmt")["completion"].to_dict()

    s["correct_country"] = s["stmt"].map(idx2country)
    missing = int(s["correct_country"].isna().sum())

    def names_country(text, country):
        if not isinstance(country, str) or not country:
            return np.nan
        # Match the country name, or its first word, case-insensitively. The first-word fallback
        # catches truncated generations ("South Afr") and multiword names the 8-token budget
        # cuts short.
        t = str(text).lower()
        return bool(country.lower() in t or country.split()[0].lower() in t)

    s["steered_names_true_country"] = [
        names_country(r.completion, r.correct_country) for r in s.itertuples()]
    s["unsteered_completion"] = s["stmt"].map(base)
    s["unsteered_names_true_country"] = [
        names_country(r.unsteered_completion, r.correct_country) for r in s.itertuples()]

    # Losing the true country is not the same as asserting a false one. Inspection of the
    # first pass showed the "changed" bucket contains truncations: an 8-token completion that
    # says "the northern part of Hebei Province," never reaches "China" but has not become
    # false. Split three ways so the ambiguous case is visible instead of being silently
    # counted as a success.
    countries = pd.read_csv("got_datasets/cities.csv")["country"].dropna().unique()

    def names_other_country(text, true_country):
        t = str(text).lower()
        tc = str(true_country).lower()
        for c in countries:
            cl = str(c).lower()
            if cl != tc and cl in t:
                return True
        return False

    s["names_other_country"] = [
        names_other_country(r.completion, r.correct_country) for r in s.itertuples()]

    def bucket(r):
        if r.steered_names_true_country is True:
            return "vacuous"
        if r.names_other_country:
            return "changed"
        return "indeterminate"

    s["outcome"] = [bucket(r) for r in s.itertuples()]

    # Only rows where the model asserted the true country BEFORE steering can demonstrate a
    # steering-induced loss of it. Rows that never named it are uninformative about vacuity.
    informative = s[s["unsteered_names_true_country"] == True].copy()  # noqa: E712
    counts = informative["outcome"].value_counts()
    vac = int(counts.get("vacuous", 0))
    changed = int(counts.get("changed", 0))
    indet = int(counts.get("indeterminate", 0))
    n = len(informative)
    return s, n, vac, changed, indet, missing


def plot(ds, g, mode, counts, cls, vac_summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1.1, 1.1]})

    ax = axes[0]
    top = counts.head(10)[::-1]
    colmap = {"content": "#117733", "directional": "#cc6677",
              "function": "#999933", "punctuation": "#888888"}
    ax.barh(range(len(top)), top.values,
            color=[colmap[classify_token(t)] for t in top.index])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([repr(t) for t in top.index], fontsize=8)
    ax.set_xlabel(f"times chosen as the certified target (of {len(g)})")
    ax.set_title(f"target tokens, target_mode = {mode!r}", fontsize=10)
    ax.grid(axis="x", lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    order = ["content", "directional", "function", "punctuation"]
    vals = [int(cls.get(k, 0)) for k in order]
    ax.bar(range(len(order)), vals, color=[colmap[k] for k in order])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v}\n{v / len(g):.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("targets")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("can this target carry a truth value?", fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[2]
    if vac_summary is None:
        ax.text(0.5, 0.5, "vacuity test not available\nfor this dataset\n(claims are not "
                          "templated;\nneeds the judge step)",
                ha="center", va="center", fontsize=9, color="#555555")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        n, vac, changed, indet, lo, hi = vac_summary
        ax.bar([0, 1, 2], [vac, indet, changed],
               color=["#cc6677", "#999933", "#117733"])
        for i, v in enumerate([vac, indet, changed]):
            ax.text(i, v + 1, f"{v}\n{v / n:.0%}", ha="center", fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["still names\nthe true country\n(VACUOUS)",
                            "names no country\n(truncated,\nindeterminate)",
                            "names a DIFFERENT\ncountry\n(real change)"], fontsize=7)
        ax.set_ylabel(f"certified flips (n = {n})")
        ax.set_ylim(0, max(vac, indet, changed) * 1.3)
        ax.set_title(f"vacuity {vac / n:.3f} to {(vac + indet) / n:.3f}", fontsize=10)
        ax.grid(axis="y", lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    fig.suptitle(f"S4 target set census — {ds}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f"plot_s4_target_census_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"[S4] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", default=None,
                    help="token_steer_*.csv to read completions from; "
                         "defaults to the postnorm all-position rp1 arm")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--direction", default="oracle")
    a = ap.parse_args()
    ds = a.dataset
    arm = a.arm or f"token_steer_{ds}_postnorm_all_rp1.csv"

    g, mode, counts, cls = census(ds)
    print(f"\n=== S4 {ds} ===")
    print(f"target_mode = {mode!r}, {len(g)} certified statements, "
          f"{counts.size} distinct target tokens")
    print(f"top targets: {dict(counts.head(6))}")
    print(f"concentration: the top 2 tokens are "
          f"{counts.head(2).sum() / len(g):.1%} of all targets")
    print("\ntarget class breakdown:")
    for k in ("content", "directional", "function", "punctuation"):
        v = int(cls.get(k, 0))
        print(f"  {k:12s} {v:4d}  {v / len(g):6.1%}")
    could = int(cls.get("content", 0))
    lo, hi = wilson(could, len(g))
    print(f"\ntargets that could possibly carry a truth value (content words): "
          f"{could}/{len(g)} = {could / len(g):.3f}  [{lo:.3f}, {hi:.3f}]")

    vac_summary = None
    if ds == "cities":
        s, n, vac, changed, indet, missing = vacuity_cities(g, arm, a.frac, a.direction)
        if missing:
            print(f"\n[warn] {missing} statements had no ground-truth country match; excluded")
        lo, hi = wilson(vac, n)
        hlo, hhi = wilson(vac + indet, n)
        vac_summary = (n, vac, changed, indet, lo, hi)
        print(f"\nDETERMINISTIC vacuity test ({a.direction}, frac={a.frac}), no judge:")
        print(f"  {n} of {len(s)} statements named the true country when unsteered")
        print(f"  vacuous       {vac:4d}  still names the true country")
        print(f"  changed       {changed:4d}  names a DIFFERENT country")
        print(f"  indeterminate {indet:4d}  names no country (usually truncation at 8 tokens)")
        print(f"\n  vacuity rate, lower bound = {vac / n:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
        print(f"  vacuity rate, upper bound = {(vac + indet) / n:.3f}  "
              f"[{hlo:.3f}, {hhi:.3f}]   (counting indeterminate as vacuous)")
        print(f"  certified flips that demonstrably made the claim FALSE: "
              f"{changed}/{n} = {changed / n:.3f}")
        s.to_csv(f"audit_vacuity_{ds}.csv", index=False)
        print(f"[S4] wrote audit_vacuity_{ds}.csv")
    else:
        print(f"\nvacuity test unavailable: {ds} claims are not templated, so there is no "
              f"ground-truth string to test against. The census stands on its own; a "
              f"judge-based vacuity rate is pending (needs ANTHROPIC_API_KEY).")

    out = pd.DataFrame({
        "token": counts.index, "count": counts.values,
        "frac": counts.values / len(g),
        "class": [classify_token(t) for t in counts.index],
    })
    out["target_mode"] = mode
    out["n_statements"] = len(g)
    out.to_csv(f"audit_target_census_{ds}.csv", index=False)
    print(f"[S4] wrote audit_target_census_{ds}.csv")

    plot(ds, g, mode, counts, cls, vac_summary)


if __name__ == "__main__":
    main()
