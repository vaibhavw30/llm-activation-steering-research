"""Assemble the candidate directions for the warm-DCT behavioral test and record their geometry.

    python dct_warm_directions.py --dataset cities
Writes dct_warm_dirs_<ds>.npz (name->unit vec at source layer) and dct_warm_geometry_<ds>.csv."""
import argparse
import csv
import os
import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

SEEDS = ["mean_diff", "grad"]
LAMS = [0.0, 0.3, 1.0, 3.0]


def aligned(vec, ref):
    v = unit(np.asarray(vec, np.float64))
    return v if float(v @ unit(np.asarray(ref, np.float64))) >= 0 else -v


def assemble_directions(ds, seeds=SEEDS, lams=LAMS):
    td = np.load(f"truth_dir_{ds}.npz")
    md = unit(np.asarray(td["mean_diff"], np.float64))
    grad = unit(np.asarray(td["grad"], np.float64))
    dirs = {"raw_mean_diff": md, "raw_grad": grad}
    seed_ref = {"mean_diff": md, "grad": grad}
    for s in seeds:
        for lam in lams:
            path = f"dct_warm_V_{ds}_{s}_{lam_tag(lam)}.pt"
            if not os.path.exists(path):
                print(f"[dirs] WARN: missing {path}; skipping (warm fit likely failed)")
                continue
            V = torch.load(path, map_location="cpu").float().numpy()
            dirs[f"warm_{s}_{lam_tag(lam)}"] = aligned(V[:, 0], seed_ref[s])
    V, U, _ = fu.load_dct(ds)                     # cold run — independent of warm fits
    top = fu.top_k_by_potency(V, U, 1)[0]         # ‖U‖-ranked top factor (model-free)
    dirs["cold_top"] = aligned(V[:, top], md)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    ds = a.dataset
    dirs = assemble_directions(ds)
    md = dirs["raw_mean_diff"]
    seed_ref = {"mean_diff": dirs["raw_mean_diff"], "grad": dirs["raw_grad"]}
    np.savez(f"dct_warm_dirs_{ds}.npz", **{k: v.astype(np.float32) for k, v in dirs.items()})
    rows = [("direction", "drift", "cos_to_mean_diff")]
    for name, v in dirs.items():
        # drift = 1 - cos to the seed axis: 0 for raw_mean_diff (its own ref); raw_grad and
        # cold_top are measured against mean_diff, so their drift is generally nonzero
        if name.startswith("warm_"):
            ref = seed_ref["mean_diff" if "mean_diff" in name else "grad"]
        else:
            ref = md
        drift = 1.0 - float(v @ ref)
        rows.append((name, round(drift, 4), round(float(v @ md), 4)))
    with open(f"dct_warm_geometry_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[dirs] wrote dct_warm_dirs_{ds}.npz ({len(dirs)} dirs) and dct_warm_geometry_{ds}.csv")


if __name__ == "__main__":
    main()
