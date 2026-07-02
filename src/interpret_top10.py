"""
interpret_top10.py — Funnel Step 1 (PI priority #1): what ARE DCT's top causal directions?

Ranks the DCT vectors by downstream-effect magnitude ||U_i||, then for the top K steers the
model with each (input_scale * V[:,i]) at the source layer and generates on a set of probe
prompts. Emits an unsteered-vs-steered table you READ and LABEL (manual-first judge), with a
judge() hook to automate later. The PI's gate: read the top 10 — if none manipulate
truthfulness, the cosine null is legitimate; if one clearly does, the comparison was the issue.

Runs on the GH200 (generation). From the repo root on the cluster:
    python interpret_top10.py --dataset cities --top-k 10 --device cuda
Output: interpret_top10_<dataset>.md
"""

import argparse
import json
import os
import numpy as np
import torch

import dct_steer_utils as su


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--scale", type=float, default=None,
                   help="steering magnitude; default = input_scale from dct_meta_<ds>.json")
    return p.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    V = torch.load(f"dct_V_{ds}.pt", map_location="cpu").float()
    V = V / (V.norm(dim=0, keepdim=True) + 1e-8)          # unit columns
    U = torch.load(f"dct_U_{ds}.pt", map_location="cpu").float()
    meta = json.load(open(f"dct_meta_{ds}.json")) if os.path.exists(f"dct_meta_{ds}.json") else {}
    source_layer = int(meta.get("source_layer", 11))
    scale = args.scale if args.scale is not None else float(meta.get("input_scale", 20.0))

    potency = U.norm(dim=0)                                # downstream-effect magnitude per vector
    top = torch.argsort(potency, descending=True)[:args.top_k].tolist()
    print(f"Dataset={ds} | source_layer={source_layer} | scale={scale:.2f} | "
          f"top-{args.top_k} by ||U||: {top}", flush=True)

    tok, model, dev = su.load_model(args.device)
    prompts = su.PROBE_PROMPTS

    lines = [f"# DCT top-{args.top_k} interpretation — {ds}",
             f"\nsource_layer={source_layer}, steering scale={scale:.2f}, ranked by ||U|| "
             f"(downstream-effect magnitude).\n",
             "For each vector: read steered vs unsteered, write what it does on the **Label** "
             "line, and note if it affects **truthfulness**.\n"]

    with su.Steerer(model, source_layer) as st:
        # unsteered baselines once
        st.set(None)
        base = {p: su.generate(model, tok, p, args.max_new_tokens) for p in prompts}

        for rank, vi in enumerate(top):
            vec = scale * V[:, vi]
            lines.append(f"\n---\n\n## Rank {rank+1}: vector #{vi}  (‖U‖={potency[vi]:.2f})")
            lines.append("\n**Label (fill in):** ______    **Affects truthfulness? (y/n):** __\n")
            lines.append("| prompt | unsteered | steered |")
            lines.append("|---|---|---|")
            st.set(vec)
            for p in prompts:
                steered = su.generate(model, tok, p, args.max_new_tokens)
                j = su.judge(base[p], steered)
                cell = f"{steered}" + (f" _(judge: {j})_" if j else "")
                lines.append(f"| {p} | {base[p]} | {cell} |")
            print(f"  done rank {rank+1} (vec #{vi})", flush=True)

    out = f"interpret_top10_{ds}.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out} — read it and label each vector. "
          f"Key question: does ANY top vector manipulate truthfulness?")


if __name__ == "__main__":
    main()
