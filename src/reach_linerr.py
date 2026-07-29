"""reach_linerr.py — Phase 5: linearization validity radius (the reviewer armor).

Relative error  ||F(eps*Delta) - F(0) - eps*J*Delta|| / ||eps*J*Delta||  for
Delta in {per-statement unit J^T w (w = mean_diff_tgt), mean_diff@src, top DCT V,
random}, 64 statements/dataset, eps log-swept over geomspace(0.05, 1.5, 8) x
input_scale. Validity radius per direction = eps at 20% median relative error;
overlaid as the trust region on the Phase-1 curves (viz_reach.fig_curves).

    PYTHONPATH=src python src/reach_linerr.py --dataset cities --device cuda
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset cities --summarize
"""
import argparse
import csv
import json

import numpy as np
import torch

from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, vjp_rows, jvp_cols, load_landmarks)

N_STMT, SEED = 64, 42
EPS_FRACS = np.geomspace(0.05, 1.5, 8)
ERR_THRESH = 0.2


def rel_error_curve(f, F0, delta_unit, eps_list):
    """f is a single-statement hop (B=1). Returns [rel_err(eps) for eps in eps_list].
    J*Delta comes from one jvp; each F(eps*Delta) is one forward."""
    d = delta_unit.shape[-1]
    delta = delta_unit.reshape(1, d).to(F0.dtype)
    _, JD = jvp_cols(f, torch.zeros_like(delta), delta)
    errs = []
    for eps in eps_list:
        Fe = f(float(eps) * delta)
        lin = float(eps) * JD
        denom = float(lin.norm())
        errs.append(float((Fe - F0 - lin).norm()) / max(denom, 1e-12))
    return errs


def validity_radius(eps_list, med_errs, thresh=ERR_THRESH):
    """Largest swept eps with median error <= thresh, linearly interpolated at the
    crossing; 0.0 if even the smallest eps exceeds thresh; max(eps) if none does."""
    eps = np.asarray(eps_list, np.float64)
    err = np.asarray(med_errs, np.float64)
    if err[0] > thresh:
        return 0.0
    over = np.where(err > thresh)[0]
    if len(over) == 0:
        return float(eps[-1])
    j = over[0]
    e0, e1, r0, r1 = eps[j - 1], eps[j], err[j - 1], err[j]
    return float(e0 + (thresh - r0) * (e1 - e0) / max(r1 - r0, 1e-12))


def compute(ds, device, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    names = [str(x) for x in dirs["names"]]
    w_md = torch.tensor(np.asarray(dirs["W"][names.index("mean_diff_tgt")],
                                   np.float32))
    lm = load_landmarks(ds)
    input_scale = float(json.load(open(f"dct_meta_{ds}.json"))["input_scale"])
    eps_list = (EPS_FRACS * input_scale).tolist()
    rng = np.random.default_rng(SEED)
    pick = [rng.choice(np.where(labels == lab)[0], size=N_STMT // 2, replace=False)
            for lab in (0, 1)]
    pick = np.sort(np.concatenate(pick))
    if limit:
        pick = pick[:limit]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    fixed = {"mean_diff_src": torch.tensor(lm["md_src"], dtype=torch.float32),
             "random": torch.tensor(unit(rng.standard_normal(len(lm["md_src"]))),
                                    dtype=torch.float32)}
    if "dct_v" in lm:
        fixed["dct_v_top"] = torch.tensor(lm["dct_v"], dtype=torch.float32)
    rows = [("stmt_index", "label", "direction", "eps", "eps_frac", "rel_err")]
    for i in pick:
        fb = forward_source_batch(model, tok, [stmts[i]], meta["src"], meta["tgt"], dev)
        f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
        d = fb["h_src_seq"].shape[-1]
        F0 = f(torch.zeros(1, d, device=dev))
        _, G = vjp_rows(f, torch.zeros(1, d, device=dev), w_md[None, :].to(dev))
        jtw = G[0, 0] / G[0, 0].norm().clamp_min(1e-12)
        deltas = {"jtw_mean_diff_tgt": jtw, **{k: v.to(dev) for k, v in fixed.items()}}
        for dname, delta in deltas.items():
            errs = rel_error_curve(f, F0, delta, eps_list)
            for frac, eps, e in zip(EPS_FRACS, eps_list, errs):
                rows.append((int(i), int(labels[i]), dname, f"{eps:.6g}",
                             f"{frac:.4g}", f"{e:.6g}"))
        print(f"[linerr] {ds} stmt {int(i)} done", flush=True)
    with open(f"reach_linerr_{ds}.csv", "w", newline="") as f2:
        csv.writer(f2).writerows(rows)
    print(f"[linerr] wrote reach_linerr_{ds}.csv")


def summarize(ds):
    per = {}
    with open(f"reach_linerr_{ds}.csv") as f:
        for r in csv.DictReader(f):
            per.setdefault(r["direction"], {}).setdefault(
                float(r["eps"]), []).append(float(r["rel_err"]))
    with open(f"reach_linerr_summary_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "eps20"))
        for dname, by_eps in per.items():
            eps = np.array(sorted(by_eps))
            med = np.array([np.median(by_eps[e]) for e in eps])
            r = validity_radius(eps, med)
            w.writerow((dname, f"{r:.6g}"))
            print(f"[linerr] {dname}: validity radius (20% err) = {r:.4g}")
    print(f"[linerr] wrote reach_linerr_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.dataset)
    else:
        compute(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
