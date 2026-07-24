"""reach_svd.py — Phase 2: full Jacobian + SVD for a 32-statement subsample.

Full J (d x d) per statement via batched jvp (TANGENT_CHUNK basis tangents per
forward-mode call), then SVD. One output file per statement so partial completion
is usable. Analysis turns "margin is small" into "truth lies in the singular tail":
energy of truth readouts in the top-k LEFT subspace, of mean_diff@src in the top-k
RIGHT subspace, and overlap of top right-singulars with DCT's V (consistency check —
a mismatch here is a stop-and-diagnose event, spec §4 P2).

    PYTHONPATH=src python src/reach_svd.py --dataset cities --device cuda
    PYTHONPATH=src .venv/bin/python src/reach_svd.py --dataset cities --analyze
"""
import argparse
import csv
import os

import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, jvp_cols)

N_STMT, SEED, TANGENT_CHUNK, TOPK_SAVE = 32, 42, 64, 64


def pick_svd_indices(labels, n=N_STMT, seed=SEED):
    """n/2 true + n/2 false statement indices, deterministic."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    pick = [rng.choice(idx[labels == lab], size=n // 2, replace=False)
            for lab in (1, 0)]
    return np.sort(np.concatenate(pick))


def full_jacobian(sliced, h_row, attn_row, device, chunk=TANGENT_CHUNK):
    """h_row (1,T,d). Returns J (d,d) float32 with J[:, j] = dF/dDelta_j, built
    column-block by column-block: the batch dimension carries `chunk` tangents of
    the SAME statement."""
    d = h_row.shape[-1]
    cols = []
    for j0 in range(0, d, chunk):
        k = min(chunk, d - j0)
        h_rep = h_row.expand(k, -1, -1)
        am_rep = attn_row.expand(k, -1)
        f = make_hop(sliced, h_rep, am_rep)
        delta0 = torch.zeros(k, d, device=device)
        T_rows = torch.zeros(k, d, device=device)
        T_rows[torch.arange(k), j0 + torch.arange(k)] = 1.0
        _, JT = jvp_cols(f, delta0, T_rows)          # (k, d): rows are J e_j
        cols.append(JT.T.float().cpu())              # (d, k) columns of J
    return torch.cat(cols, dim=1)                    # (d, d)


def compute(ds, device, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    picks = pick_svd_indices(labels)
    if limit:
        picks = picks[:limit]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    outdir = f"reach_svd_{ds}"
    os.makedirs(outdir, exist_ok=True)
    for i in picks:
        opath = os.path.join(outdir, f"stmt_{int(i):05d}.npz")
        if os.path.exists(opath):
            print(f"[svd] {opath} exists — skipping", flush=True)
            continue
        fb = forward_source_batch(model, tok, [stmts[i]], meta["src"], meta["tgt"], dev)
        J = full_jacobian(sliced, fb["h_src_seq"], fb["attn"], dev)
        U, S, Vh = torch.linalg.svd(J.to(dev), full_matrices=False)
        np.savez(opath, s=S.cpu().numpy().astype(np.float32),
                 U64=U[:, :TOPK_SAVE].cpu().numpy().astype(np.float16),
                 V64=Vh[:TOPK_SAVE].T.cpu().numpy().astype(np.float16),
                 label=int(labels[i]), stmt_index=int(i))
        print(f"[svd] {ds} stmt {int(i)} done  s1={float(S[0]):.4g}  "
              f"eff_rank={(S.sum() ** 2 / (S ** 2).sum()).item():.1f}", flush=True)


def analyze(ds):
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    W = np.asarray(dirs["W"], np.float64)
    w_md = W[names.index("mean_diff_tgt")]
    w_pg = W[names.index("probe_grad_tgt")]
    md_src = unit(np.asarray(np.load(f"truth_dir_{ds}.npz")["mean_diff"], np.float64))
    rng = np.random.default_rng(7)
    w_rand = unit(rng.standard_normal(W.shape[1]))
    V, U, _ = fu.load_dct(ds)
    dctV16 = np.linalg.qr(V[:, fu.top_k_by_potency(V, U, 16)].astype(np.float64))[0]
    files = sorted(fn for fn in os.listdir(f"reach_svd_{ds}") if fn.endswith(".npz"))
    ks = np.arange(1, TOPK_SAVE + 1)
    energy = {"w_mean_diff_tgt_in_U": [], "w_probe_grad_tgt_in_U": [],
              "md_src_in_V": [], "rand_in_U": []}
    summ_rows = []
    for fn in files:
        z = np.load(os.path.join(f"reach_svd_{ds}", fn))
        s = np.asarray(z["s"], np.float64)
        U64 = np.asarray(z["U64"], np.float64)
        V64 = np.asarray(z["V64"], np.float64)
        energy["w_mean_diff_tgt_in_U"].append(np.cumsum((U64.T @ w_md) ** 2))
        energy["w_probe_grad_tgt_in_U"].append(np.cumsum((U64.T @ w_pg) ** 2))
        energy["md_src_in_V"].append(np.cumsum((V64.T @ md_src) ** 2))
        energy["rand_in_U"].append(np.cumsum((U64.T @ w_rand) ** 2))
        ov = float(np.mean(np.sum((dctV16.T @ V64[:, :16]) ** 2, axis=0)))
        summ_rows.append((int(z["stmt_index"]), int(z["label"]),
                          float(s.sum() ** 2 / (s ** 2).sum()), float(s[0]),
                          float(s[TOPK_SAVE - 1] / s[0]), ov))
    with open(f"reach_svd_energy_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("k", "quantity", "mean_energy"))
        for q, rows in energy.items():
            mean = np.mean(np.stack(rows), axis=0)
            for k, v in zip(ks, mean):
                w.writerow((int(k), q, f"{v:.6g}"))
    with open(f"reach_svd_summary_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stmt_index", "label", "eff_rank", "s1", "s64_over_s1",
                    "dctV_overlap_top16"))
        w.writerows(summ_rows)
    mo = np.mean([r[5] for r in summ_rows])
    print(f"[svd] {ds}: {len(files)} statements, mean eff_rank="
          f"{np.mean([r[2] for r in summ_rows]):.1f}, DCT-V top-16 overlap={mo:.3f}")
    if mo < 0.3:
        print("[svd] WARNING: top right-singulars disagree with DCT's V — "
              "stop-and-diagnose before trusting the linearized picture (spec §4 P2)")
    print(f"[svd] wrote reach_svd_energy_{ds}.csv and reach_svd_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.dataset)
    else:
        compute(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
