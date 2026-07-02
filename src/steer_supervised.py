"""
steer_supervised.py — Funnel Step 2 (PI priority #2): does the SUPERVISED truth direction
actually steer behavior?

Injects the mean-diff / gradient truth direction (precomputed by export_truth_dir.py on the
laptop, since it needs the big activations + sklearn) at the truth-peak layer, sweeping
magnitude, and prints unsteered-vs-steered completions on factual prompts. If the supervised
direction reads truth at ~99% but does NOT change truthfulness behavior when injected, that is
the cleanest possible demonstration of "decodable != causally salient" — shown directly, not
inferred from a cosine null.

Needs truth_dir_<ds>.npz on the cluster (rsync it over). Runs on the GH200:
    python steer_supervised.py --dataset cities --device cuda
Output: steer_supervised_<dataset>.md  + steer_supervised_<dataset>.csv (for the magnitude plot)

NOTE: gemma-2-2b is a BASE model, so "truthfulness behavior" is read from completions
(e.g. does steering flip "The capital of Japan is ___" away from Tokyo?). Read qualitatively.
"""

import argparse
import csv
import numpy as np
import torch

import dct_steer_utils as su

# Factual prompts with a known correct continuation — easiest place to SEE a truthfulness shift.
FACTUAL_PROMPTS = [
    "The capital of Japan is",
    "The city of Paris is in the country of",
    "Two plus two equals",
    "The chemical symbol for water is",
    "The Earth orbits the",
    "The first president of the United States was",
    "The opposite of hot is",
    "The sun rises in the",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--direction", choices=["mean_diff", "grad", "both"], default="both")
    p.add_argument("--scales", default="0,20,40,80",
                   help="comma-separated steering magnitudes to sweep (0 = unsteered)")
    p.add_argument("--max-new-tokens", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    z = np.load(f"truth_dir_{ds}.npz")
    layer = int(z["layer"])
    dirs = {}
    if args.direction in ("mean_diff", "both"):
        dirs["mean_diff"] = torch.tensor(z["mean_diff"], dtype=torch.float32)
    if args.direction in ("grad", "both"):
        dirs["grad"] = torch.tensor(z["grad"], dtype=torch.float32)
    scales = [float(x) for x in args.scales.split(",")]
    print(f"Dataset={ds} | layer={layer} | directions={list(dirs)} | scales={scales}", flush=True)

    tok, model, dev = su.load_model(args.device)
    prompts = FACTUAL_PROMPTS

    lines = [f"# Steering the SUPERVISED truth direction — {ds}",
             f"\nlayer={layer}. Each direction is unit-norm; we sweep magnitude. Read whether "
             f"steered completions become FALSE/incoherent (a behavioral effect) or stay put "
             f"(no effect = decodable-but-not-causal).\n"]
    rows = [("direction", "scale", "prompt", "completion")]

    with su.Steerer(model, layer) as st:
        for name, vec in dirs.items():
            lines.append(f"\n---\n\n## Direction: {name}\n")
            for s in scales:
                st.set(None if s == 0 else s * vec)
                tag = "unsteered" if s == 0 else f"scale={s:g}"
                lines.append(f"\n**{tag}**\n")
                lines.append("| prompt | completion |")
                lines.append("|---|---|")
                for p in prompts:
                    c = su.generate(model, tok, p, args.max_new_tokens)
                    lines.append(f"| {p} | {c} |")
                    rows.append((name, s, p, c))
                print(f"  {name} scale={s:g} done", flush=True)

    with open(f"steer_supervised_{ds}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"steer_supervised_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved steer_supervised_{ds}.md (read it) and .csv (for the magnitude plot).")
    print("Key question: does increasing scale flip the factual completions to false/garbage?")
    print("  - YES -> the supervised direction IS causal (truth is steerable; revisit the null).")
    print("  - NO  -> decodable-but-not-causal, shown directly (the strongest thesis result).")


if __name__ == "__main__":
    main()
