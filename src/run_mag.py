# src/run_mag.py
"""run_mag.py — drive the MAG probes from cached mag_acts_<ds>.npz.

    python src/run_mag.py --dataset cities --probe directions
    python src/run_mag.py --dataset cities --probe all
    python src/run_mag.py --probe transfer            # across all 4 datasets
"""
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import funnel_utils as fu
from mag.config import DATASETS
from mag.directions import build_directions
from mag import probes
from mag.operators import operator_features


def _load(ds):
    return np.load(f"mag_acts_{ds}.npz", allow_pickle=True)


def _write_csv(path, rows):
    if not rows:
        print(f"[run_mag] no rows for {path}"); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[run_mag] wrote {path}")


def do_directions(ds, layer):
    c = _load(ds)
    y_gold = c["labels"].astype(int); y_yM = c["ymL"].astype(int)
    mean_diff = grad = dctV = None
    if os.path.exists(f"activations/acts_{ds}.npz"):
        X, y = fu.load_acts(ds, layer)
        mean_diff = fu.mean_diff_dir(X, y); grad = fu.grad_dir(X, y)
    if os.path.exists(f"dct_V_{ds}.pt"):
        V, U, _ = fu.load_dct(ds)
        dctV = V[:, fu.top_k_by_potency(V, U, 1)[0]]
    out = build_directions(c, layer, y_gold, y_yM, mean_diff, grad, dctV)
    np.savez(f"mag_dir_{ds}.npz", **out)
    print(f"[run_mag] wrote mag_dir_{ds}.npz  layer={layer}  "
          f"cos(u_gold,mean_diff)={float(out['cos_uGold_mean_diff']):+.3f}")


def do_e1(ds, layer):
    c = _load(ds)
    dct = fu.load_dct(ds) if os.path.exists(f"dct_V_{ds}.pt") else None
    rows = probes.E1_readability(c, layer, c["labels"].astype(int), c["ymL"].astype(int), dct)
    _write_csv(f"mag_readability_{ds}.csv", rows)


def do_e2(ds, layer):
    c = _load(ds)
    feat_mag = operator_features("InputDelta", c, layer)
    feat_raw = operator_features("Direct", c, layer)
    rows = probes.E2_disagreement(feat_mag, feat_raw, c["labels"].astype(int), c["ymL"].astype(int))
    _write_csv(f"mag_disagreement_{ds}.csv", rows)


def do_e3(ds, layer):
    c = _load(ds)
    extra = {}
    if os.path.exists(f"truth_dir_{ds}.npz"):
        td = np.load(f"truth_dir_{ds}.npz"); extra["mean_diff"] = td["mean_diff"]
    if os.path.exists(f"dct_V_{ds}.pt"):
        V, U, _ = fu.load_dct(ds); extra["dct"] = V[:, fu.top_k_by_potency(V, U, 1)[0]]
    rows = probes.E3_linearity(c, layer, extra)
    _write_csv(f"mag_linearity_{ds}.csv", rows)


def do_transfer():
    feats, labels = {}, {}
    for ds in DATASETS:
        if not os.path.exists(f"mag_acts_{ds}.npz"):
            continue
        c = _load(ds); layer = fu.BEST_LAYER.get(ds, 11)
        feats[ds] = operator_features("InputDelta", c, layer)
        labels[ds] = c["labels"].astype(int)
    if len(feats) < 2:
        print("[run_mag] transfer needs >=2 datasets extracted"); return
    res = probes.transfer_rank(feats, labels)
    _write_csv("mag_transfer.csv", res["rows"])
    print(f"[run_mag] transfer Top-1={res['top1']:.2f}  Spearman={res['spearman']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--probe", required=True,
                    choices=["directions", "e1", "e2", "e3", "transfer", "all"])
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    if a.probe == "transfer":
        do_transfer(); return
    ds = a.dataset
    if ds is None:
        raise SystemExit("--dataset required for this probe")
    # Lead at the truth-peak layer (matches DCT/probe results); --layer overrides (e.g. 26 = final).
    layer = a.layer if a.layer is not None else fu.BEST_LAYER.get(ds, 11)
    if a.probe in ("directions", "all"):
        do_directions(ds, layer)
    if a.probe in ("e1", "all"):
        do_e1(ds, layer)
    if a.probe in ("e2", "all"):
        do_e2(ds, layer)
    if a.probe in ("e3", "all"):
        do_e3(ds, layer)
    if a.probe == "all":
        do_transfer()


if __name__ == "__main__":
    main()
