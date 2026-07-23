"""Assemble U-anchored warm-DCT candidate directions + their U/V geometry.

    python dct_uwarm_directions.py --dataset cities
Writes dct_uwarm_dirs_<ds>.npz (name -> unit vec at source layer) and
dct_uwarm_geometry_<ds>.csv (direction, cos_U0_mdtgt, cos_V0_mdsrc).

Sign convention: each uwarm V0 is flipped so its factor-0 EFFECT points toward TRUE
(U0 . mean_diff@target >= 0); the geometry CSV records the raw signed cosines pre-flip."""
import argparse
import csv
import os
import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

LAMS = (0.3, 1.0, 3.0)      # lam=0 == the cold run, already covered by cold_top


def assemble(ds, lams=LAMS):
    td = np.load(f"truth_dir_{ds}.npz")
    md_src = unit(np.asarray(td["mean_diff"], np.float64))
    tt = np.load(f"truth_dir_tgt_{ds}.npz")
    md_tgt = unit(np.asarray(tt["mean_diff"], np.float64))
    dirs = {"raw_mean_diff": md_src}
    rows = [("raw_mean_diff", float("nan"), 1.0)]
    Vc, Uc, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(Vc, Uc, 1)[0]
    cold = unit(Vc[:, top].astype(np.float64))
    cold = cold if float(cold @ md_src) >= 0 else -cold
    dirs["cold_top"] = cold
    rows.append(("cold_top", float("nan"), float(cold @ md_src)))
    for lam in lams:
        tag = lam_tag(lam)
        vp = f"dct_uwarm_V_{ds}_mean_diff_{tag}.pt"
        up = f"dct_uwarm_U_{ds}_mean_diff_{tag}.pt"
        if not (os.path.exists(vp) and os.path.exists(up)):
            print(f"[udirs] WARN: missing {vp} or {up}; skipping (uwarm fit likely failed)")
            continue
        v0 = unit(torch.load(vp, map_location="cpu").float().numpy()[:, 0].astype(np.float64))
        u0 = unit(torch.load(up, map_location="cpu").float().numpy()[:, 0].astype(np.float64))
        cos_u = float(u0 @ md_tgt)
        cos_v = float(v0 @ md_src)
        name = f"uwarm_mean_diff_{tag}"
        dirs[name] = v0 if cos_u >= 0 else -v0     # effect points toward TRUE
        rows.append((name, cos_u, cos_v))
    return dirs, rows


def write_outputs(ds, lams=LAMS):
    dirs, rows = assemble(ds, lams)
    np.savez(f"dct_uwarm_dirs_{ds}.npz",
             **{k: v.astype(np.float32) for k, v in dirs.items()})
    with open(f"dct_uwarm_geometry_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "cos_U0_mdtgt", "cos_V0_mdsrc"))
        w.writerows(rows)
    print(f"[udirs] wrote dct_uwarm_dirs_{ds}.npz ({len(dirs)} dirs) and "
          f"dct_uwarm_geometry_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    write_outputs(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
