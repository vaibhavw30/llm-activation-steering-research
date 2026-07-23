"""Supervised truth seeds at the DCT *target* layer, for U-space anchored DCT.
Standalone so export_concept_dir.py and truth_dir_<ds>.npz stay untouched.

    .venv/bin/python src/export_target_dir.py --dataset cities
"""
import argparse
import json
import numpy as np

import funnel_utils as fu
from export_concept_dir import concept_directions


def export(ds):
    tgt = int(json.load(open(f"dct_meta_{ds}.json"))["target_layer"])
    X, y = fu.load_acts(ds, tgt)
    mean_diff, grad = concept_directions(X, y)
    out = f"truth_dir_tgt_{ds}.npz"
    np.savez(out, mean_diff=mean_diff.astype(np.float32),
             grad=grad.astype(np.float32), layer=np.array(tgt))
    print(f"Saved {out}: mean_diff & grad at TARGET layer {tgt} (d={X.shape[1]}); "
          f"cos={float(mean_diff @ grad):+.3f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    export(p.parse_args().dataset)


if __name__ == "__main__":
    main()
