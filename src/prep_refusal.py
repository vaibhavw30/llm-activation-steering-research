"""Build got_datasets/refusal.csv (harmful vs harmless instructions) for the refusal control.

Public data: AdvBench harmful behaviors + Alpaca harmless instructions. label 1 = harmful.

    .venv/bin/python prep_refusal.py
"""
import argparse
import pandas as pd


def to_contrast_df(harmful, harmless, seed=42):
    """Balanced, deterministic (statement, label) frame: 1 = harmful, 0 = harmless."""
    n = min(len(harmful), len(harmless))
    rows = ([{"statement": s, "label": 1} for s in harmful[:n]]
            + [{"statement": s, "label": 0} for s in harmless[:n]])
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--advbench",
                   default="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
    p.add_argument("--out", default="got_datasets/refusal.csv")
    args = p.parse_args()

    harmful = pd.read_csv(args.advbench)["goal"].tolist()
    from datasets import load_dataset
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"]][:len(harmful)]

    df = to_contrast_df(harmful, harmless)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {int(df.label.sum())} harmful / "
          f"{int((df.label == 0).sum())} harmless")


if __name__ == "__main__":
    main()
