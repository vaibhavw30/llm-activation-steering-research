"""
validate_judge.py — the honest gate: does the LLM judge actually agree with ground truth?

Before we trust the FALSE-vs-INCOHERENT steering curves (judge_results.py --mode steer), we must
show the judge classifies KNOWN true/false factual statements correctly. This samples a balanced
set from a gold-labelled dataset (default got_datasets/cities.csv, which has clean 1/0 labels),
feeds each to the SAME judge + STEER_SYS prompt the pipeline uses, and compares its TRUE/FALSE
verdict to the gold label.

  PASS (accuracy >= threshold, default 0.85) -> trust the judge for this concept.
  FAIL                                        -> don't believe the curves; fall back / note it.

Exit code is 0 on PASS, 1 on FAIL, so it can gate a SLURM pipeline (with `set -e`).

Run (on the cluster, judge env):
    python3 src/validate_judge.py --backend olmo --dataset cities --device cuda --limit 100
"""

import argparse
import csv
import os
import sys

# Reuse the exact judge contract the pipeline uses (STEER_SYS, ask(), extract_json()).
sys.path.insert(0, os.path.dirname(__file__))
from judge_results import STEER_SYS, ask, extract_json, get_client, DEFAULT_MODEL


def row_to_judge_input(row):
    """Map a gold-labelled dataset row -> (stem, completion, gold_label:int).

    If the row has structured city/country columns (cities.csv), build the exact stem+completion
    the judge sees in steer mode, for format parity. Otherwise judge the whole `statement`.
    """
    gold = int(row["label"])
    if row.get("city") and row.get("country"):
        return f"The city of {row['city']} is in", row["country"], gold
    return "", row["statement"], gold


def balanced_sample(rows, n):
    """First n//2 label==1 rows + first n//2 label==0 rows. Balanced and deterministic (no RNG,
    so the gate is reproducible)."""
    k = n // 2
    pos = [r for r in rows if str(r["label"]) == "1"][:k]
    neg = [r for r in rows if str(r["label"]) == "0"][:k]
    return pos + neg


def score_agreement(pairs):
    """pairs: list of (gold:int in {0,1}, verdict:str). Agreement = TRUE on a true claim or FALSE
    on a false claim; INCOHERENT is never agreement. Returns n, agree, accuracy, confusion."""
    conf, agree = {}, 0
    for gold, v in pairs:
        conf[(gold, v)] = conf.get((gold, v), 0) + 1
        if (gold == 1 and v == "TRUE") or (gold == 0 and v == "FALSE"):
            agree += 1
    n = len(pairs)
    return {"n": n, "agree": agree, "accuracy": (agree / n if n else 0.0), "confusion": conf}


def passes_gate(accuracy, threshold=0.85):
    return accuracy >= threshold


def run_validate(client, model, ds, limit, threshold):
    path = f"got_datasets/{ds}.csv"
    rows = list(csv.DictReader(open(path)))
    sample = balanced_sample(rows, limit) if limit else rows
    print(f"[validate] judging {len(sample)} gold rows from {path} with {model} "
          f"(gate >= {threshold:.2f})...", flush=True)

    pairs, out = [], []
    for i, r in enumerate(sample):
        stem, completion, gold = row_to_judge_input(r)
        user = f'Stem: "{stem}"\nCompletion: "{completion}"'
        j = extract_json(ask(client, model, STEER_SYS, user, max_tokens=80)) or {}
        verdict = str(j.get("verdict", "INCOHERENT")).upper()
        if verdict not in ("TRUE", "FALSE", "INCOHERENT"):
            verdict = "INCOHERENT"
        pairs.append((gold, verdict))
        out.append({"statement": r.get("statement", ""), "stem": stem, "completion": completion,
                    "gold_label": gold, "verdict": verdict,
                    "agree": int((gold == 1 and verdict == "TRUE")
                                 or (gold == 0 and verdict == "FALSE"))})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    res = score_agreement(pairs)
    outpath = f"judge_validation_{ds}.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["statement", "stem", "completion", "gold_label",
                                          "verdict", "agree"])
        w.writeheader()
        w.writerows(out)
    print(f"[validate] wrote {outpath}")

    _print_report(res, threshold)
    return res


def _print_report(res, threshold):
    c = res["confusion"]
    print("\n=== confusion (gold \\ verdict) ===")
    print(f"{'':>10} {'TRUE':>6} {'FALSE':>6} {'INCOH':>6}")
    for gold, name in [(1, "true"), (0, "false")]:
        print(f"{name:>10} {c.get((gold,'TRUE'),0):>6} {c.get((gold,'FALSE'),0):>6} "
              f"{c.get((gold,'INCOHERENT'),0):>6}")
    ok = passes_gate(res["accuracy"], threshold)
    print(f"\n[validate] accuracy {res['accuracy']:.3f} on {res['n']} rows "
          f"({res['agree']} agree) -> {'PASS' if ok else 'FAIL'} (gate >= {threshold:.2f})")
    if not ok:
        print("[validate] judge is MISCALIBRATED for this concept — do NOT trust its steering "
              "curves; fall back to another backend or note the caveat.")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cities", help="gold dataset stem in got_datasets/")
    p.add_argument("--backend", choices=["anthropic", "olmo"], default="olmo",
                   help="chat judge to validate (same one you run in judge_results)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="anthropic model id")
    p.add_argument("--olmo-model", default="allenai/Olmo-3-7B-Instruct")
    p.add_argument("--device", default="cuda", help="device for the olmo judge (mps on Mac)")
    p.add_argument("--limit", type=int, default=100, help="balanced sample size (0 = all rows)")
    p.add_argument("--threshold", type=float, default=0.85, help="min accuracy to PASS the gate")
    return p


def main():
    args = build_parser().parse_args()
    if args.backend == "olmo":
        from judges.olmo_judge import OlmoJudge
        client, model = OlmoJudge(args.olmo_model, args.device), args.olmo_model
    else:
        client, model = get_client(), args.model
    res = run_validate(client, model, args.dataset, args.limit, args.threshold)
    sys.exit(0 if passes_gate(res["accuracy"], args.threshold) else 1)


if __name__ == "__main__":
    main()
