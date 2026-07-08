"""The money figure: DCT recovery (y) vs. behavioral causal salience (x), one point per concept.

Reads recovery_<concept>.csv and judge_steer_<concept>.csv for each concept, joins them, and
plots the spectrum. `present_map` says which judged verdict marks the concept-present dimension
per concept (truth: FALSE = lie induced; refusal: FALSE = still refusing; toxicity: FALSE = toxic).

    .venv/bin/python viz_spectrum.py --concepts cities refusal toxicity sycophancy
"""
import argparse
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_utils import concept_salience

DEFAULT_PRESENT = {"cities": "FALSE", "common_claim_true_false": "FALSE",
                   "refusal": "FALSE", "toxicity": "FALSE", "sycophancy": "FALSE"}


def build_points(concepts, present_map=None):
    present_map = present_map or DEFAULT_PRESENT
    out = []
    for c in concepts:
        rec = next(csv.DictReader(open(f"recovery_{c}.csv")))
        y = float(rec["ratio_vs_random"])
        rows = list(csv.DictReader(open(f"judge_steer_{c}.csv")))
        sal = concept_salience(rows, present_verdict=present_map.get(c, "FALSE"))
        out.append({"concept": c, "x_salience": sal["x_salience"], "y_recovery": y})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="+", required=True)
    args = p.parse_args()
    df = build_points(args.concepts)
    df.to_csv("spectrum_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["x_salience"], df["y_recovery"], s=80, color="#4477aa", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["concept"], (r["x_salience"], r["y_recovery"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="DCT = random (no recovery)")
    ax.set_xlabel("behavioral causal salience  (max judged flip vs. unsteered)")
    ax.set_ylabel("DCT recovery  (best |cos| ÷ random baseline)")
    ax.set_title("Does DCT recover a concept in proportion to its causal salience?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_findings_spectrum.png", dpi=150)
    print("saved plot_findings_spectrum.png and spectrum_points.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
