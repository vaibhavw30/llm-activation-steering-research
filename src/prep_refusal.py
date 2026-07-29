"""Build got_datasets/refusal.csv (+ refusal_holdout.csv) for the refusal control.

Public data: AdvBench harmful behaviors + Alpaca harmless instructions.

POLARITY (important): label 1 = HARMLESS, label 0 = harmful. reach_analyze computes
eps* for label-1 statements crossing into the label-0 halfspace, so this convention
makes the certificate describe INDUCING refusal on harmless instructions — steering a
harmless prompt into the "harmful" halfspace should make the model treat it as harmful
and refuse. That has full headroom (baseline refusal on harmless prompts is ~0). The
opposite direction (ablating refusal) needs the model to refuse unprompted, which a
base model may never do; reach_steer sweeps +/- scales, so we observe it for free.

HOLDOUT: the held-out prompts never enter direction fitting. They supply the mean-arm
generation prompts and the A1 screening prompts, so behavioral evaluation is
leakage-free.

CHAT TEMPLATE: --chat-template <model-id> renders every instruction through that
model's chat template ONCE, here, before anything downstream sees it. Doing it here
rather than at generation time is deliberate: extract.py, reach_margins.py and
reach_steer.py must all tokenize the identical string, or the certificate's
linearization point stops being the generation prompt's last token — which is exactly
the context-shift confound this control is designed to avoid. Leave it off for base
models, which have no template.

    .venv/bin/python src/prep_refusal.py
    .venv/bin/python src/prep_refusal.py --chat-template google/gemma-2-2b-it
"""
import argparse

import numpy as np
import pandas as pd

N_HOLDOUT = 32
ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")


def to_contrast_df(harmful, harmless, seed=42, label1="harmless"):
    """Balanced, deterministic (statement, label) frame. label1 selects which class
    gets label 1; default "harmless" — see the module docstring on polarity."""
    if label1 not in ("harmless", "harmful"):
        raise ValueError(f"label1 must be 'harmless' or 'harmful', got {label1!r}")
    n = min(len(harmful), len(harmless))
    pos, neg = (harmless, harmful) if label1 == "harmless" else (harmful, harmless)
    rows = ([{"statement": s, "label": 1} for s in list(pos)[:n]]
            + [{"statement": s, "label": 0} for s in list(neg)[:n]])
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_holdout(harmful, harmless, n_hold=N_HOLDOUT, seed=42):
    """Deterministically peel n_hold of each class off for behavioral evaluation.
    Returns (harmful_fit, harmless_fit, holdout_df[statement, kind])."""
    rng = np.random.default_rng(seed)
    out, keep = [], {}
    for kind, items in (("harmful", list(harmful)), ("harmless", list(harmless))):
        k = min(n_hold, len(items))
        idx = rng.permutation(len(items))
        hold, fit = sorted(idx[:k]), sorted(idx[k:])
        out += [{"statement": items[i], "kind": kind} for i in hold]
        keep[kind] = [items[i] for i in fit]
    return keep["harmful"], keep["harmless"], pd.DataFrame(out)


def apply_template(statements, model_name):
    """Render each instruction as a user turn in model_name's chat template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return [tok.apply_chat_template([{"role": "user", "content": str(s)}],
                                    tokenize=False, add_generation_prompt=True)
            for s in statements]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--advbench", default=ADVBENCH_URL)
    p.add_argument("--out", default="got_datasets/refusal.csv")
    p.add_argument("--holdout-out", default="got_datasets/refusal_holdout.csv")
    p.add_argument("--n-holdout", type=int, default=N_HOLDOUT)
    p.add_argument("--label1", default="harmless", choices=["harmless", "harmful"])
    p.add_argument("--chat-template", default=None,
                   help="model id whose chat template to apply to every instruction")
    args = p.parse_args()

    harmful = pd.read_csv(args.advbench)["goal"].astype(str).tolist()
    from datasets import load_dataset
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"]][:len(harmful)]
    if args.chat_template:
        harmful = apply_template(harmful, args.chat_template)
        harmless = apply_template(harmless, args.chat_template)
        print(f"applied {args.chat_template} chat template to all instructions")

    hf, hs, hold = split_holdout(harmful, harmless, args.n_holdout)
    df = to_contrast_df(hf, hs, label1=args.label1)
    df.to_csv(args.out, index=False)
    hold.to_csv(args.holdout_out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, label1={args.label1}, "
          f"{int(df.label.sum())} label-1 / {int((df.label == 0).sum())} label-0")
    print(f"wrote {args.holdout_out}: {len(hold)} held-out prompts "
          f"({int((hold.kind == 'harmful').sum())} harmful / "
          f"{int((hold.kind == 'harmless').sum())} harmless)")


if __name__ == "__main__":
    main()
