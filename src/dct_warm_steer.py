"""Inject each warm-DCT candidate direction at the source layer with the shared DCT input_scale,
sweep tau, generate short factual completions for the OLMo judge.

    python dct_warm_steer.py --dataset cities --device cuda
    python dct_warm_steer.py --dataset cities --device cpu --limit 3   # smoke

Writes dct_warm_steer_<ds>.csv (direction,scale,prompt,completion — scale=tau), the schema
judge_results.py --mode steer reads unchanged."""
import argparse
import csv
import json
import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from steer_supervised import FACTUAL_PROMPTS

TAUS = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]   # two-sided: -tau -> FALSE (lying), +tau -> TRUE
MAX_NEW_TOKENS = 8


def injected(tau, unit_dir, input_scale):
    return float(tau) * float(input_scale) * unit(np.asarray(unit_dir, np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke)")
    a = ap.parse_args()
    ds = a.dataset
    meta = json.load(open(f"dct_meta_{ds}.json"))
    layer = int(meta["source_layer"]); scale = float(meta["input_scale"])
    dirs = np.load(f"dct_warm_dirs_{ds}.npz")
    prompts = FACTUAL_PROMPTS[:a.limit] if a.limit else FACTUAL_PROMPTS
    print(f"[warm/steer] {ds}: layer={layer} input_scale={scale:.3f} "
          f"{len(dirs.files)} dirs x {len(TAUS)} taus x {len(prompts)} prompts", flush=True)

    tok, model, dev = su.load_model(a.device)
    rows = [("direction", "scale", "prompt", "completion")]
    for name in dirs.files:
        uvec = unit(np.asarray(dirs[name], np.float64))
        with su.Steerer(model, layer) as st:
            for tau in TAUS:
                vec = None if tau == 0.0 else torch.tensor(
                    injected(tau, uvec, scale), dtype=torch.float32)
                st.set(vec)
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((name, tau, p, c))
                print(f"  {name} tau={tau:+.1f} done", flush=True)
    with open(f"dct_warm_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[warm/steer] wrote dct_warm_steer_{ds}.csv")


if __name__ == "__main__":
    main()
