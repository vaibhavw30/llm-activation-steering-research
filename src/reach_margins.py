"""reach_margins.py — Phase 1 extraction for the backward-reachability audit.

Three stages (spec §4 P1), run in order (--stage all runs them back-to-back):
  --stage acts : batched no-grad forwards; saves last-token h_src/h_tgt/h_final.
  --stage dirs : fits the direction battery + P(true)=0.2 thresholds from stage-1
                 activations (needs scikit-learn; seconds).
  --stage vjp  : per statement x direction, one vjp -> margins m_i(w) = ||J_i^T w||,
                 unit J^T w rows (stored for truth/truth_sub/dct_u groups), cosines
                 to source landmarks. Checkpoints every CHUNK statements; --resume
                 (default) skips completed chunks.

    PYTHONPATH=src python src/reach_margins.py --dataset cities --stage all --device cuda
    PYTHONPATH=src python src/reach_margins.py --dataset cities --stage vjp --limit 8 --device cpu   # smoke
"""
import argparse
import os

import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, vjp_rows, load_landmarks)

LOGIT_02 = float(np.log(0.2 / 0.8))          # P(true) <= 0.2 confidence margin
N_BOOT, BOOT_FRAC = 6, 0.5                   # bootstrap probes for truth_sub_k
K_DCT_U, N_RAND = 4, 64
MIN_ACC_1D = 0.6                             # threshold valid only above this 1-D acc
CHUNK, VJP_BATCH, ACTS_BATCH = 100, 16, 16
# landmark key -> stored array name. The STORED names must not change: viz_reach and
# every existing truth reach_margins_<ds>.npz depend on them.
COS_KEY = {"md_src": "cos_md_src", "v_q": "cos_vq", "dct_v": "cos_dctv"}
DATASET_SAMPLE = {"common_claim_true_false": 2000}   # stratified cap; cities runs full


# ---------------------------------------------------------------- pure helpers

def load_statements(ds, seed=42):
    """Statements + labels from got_datasets/<ds>.csv; stratified seed-42 subsample
    where DATASET_SAMPLE caps the dataset. Returns (statements, labels, row_index)."""
    import pandas as pd
    df = pd.read_csv(os.path.join("got_datasets", f"{ds}.csv"))
    labels = df["label"].values.astype(int)
    idx = np.arange(len(df))
    cap = DATASET_SAMPLE.get(ds)
    if cap and cap < len(df):
        rng = np.random.default_rng(seed)
        keep = [rng.choice(idx[labels == lab], size=cap // 2, replace=False)
                for lab in (0, 1)]
        idx = np.sort(np.concatenate(keep))
    return df["statement"].values[idx], labels[idx], idx


def gram_schmidt(vecs, tol=1e-6):
    """Orthonormal basis (rows) from a list of vectors; near-dependent ones dropped."""
    basis = []
    for v in vecs:
        w = np.asarray(v, np.float64).copy()
        for b in basis:
            w -= (w @ b) * b
        n = np.linalg.norm(w)
        if n > tol:
            basis.append(w / n)
    return np.stack(basis)


def fit_probe_dir(X, y, seed=None, frac=1.0):
    """Unit logistic-probe gradient direction (funnel_utils.grad_dir recipe), with an
    optional seeded row-subsample for the bootstrap battery members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(y), size=int(frac * len(y)), replace=False)
        X, y = X[keep], y[keep]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return unit(lr.coef_[0] / sc.scale_)


def fit_threshold(scores, y):
    """1-D logistic calibration of s = w.h. Returns (acc, slope_sign, t02) where t02
    is the score at calibrated P(true)=0.2 — only meaningful when slope > 0 (caller
    flips w and refits when slope_sign < 0)."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, np.float64).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000).fit(s, y)
    acc = float(lr.score(s, y))
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    sign = 1.0 if a > 0 else -1.0
    t02 = (LOGIT_02 - b) / a if a != 0 else float("nan")
    return acc, sign, t02


def build_battery(h_tgt, y, ds):
    """Direction battery at the target layer (spec §3). h_tgt (n,d) float, y (n,).
    Reads truth_dir_tgt_<ds>.npz and dct_U_<ds>.pt/dct_V_<ds>.pt from cwd."""
    h = np.asarray(h_tgt, np.float64)
    tt = np.load(f"truth_dir_tgt_{ds}.npz")
    md = unit(np.asarray(tt["mean_diff"], np.float64))
    pg = fit_probe_dir(h, y)
    boots = [fit_probe_dir(h, y, seed=s, frac=BOOT_FRAC) for s in range(N_BOOT)]
    sub = gram_schmidt([md, pg] + boots)
    rng = np.random.default_rng(123)
    raw = [("mean_diff_tgt", "truth", md), ("probe_grad_tgt", "truth", pg)]
    raw += [(f"truth_sub_{j}", "truth_sub", sub[j]) for j in range(sub.shape[0])]
    from reach_hop import optional_artifacts
    if optional_artifacts(ds)["dct"]:
        V, U, _ = fu.load_dct(ds)
        tops = fu.top_k_by_potency(V, U, K_DCT_U)
        raw += [(f"dct_u_{r}", "dct_u", U[:, t].astype(np.float64))
                for r, t in enumerate(tops)]
    raw += [(f"rand_{r}", "rand", rng.standard_normal(h.shape[1]))
            for r in range(N_RAND)]
    W, names, groups, acc1d, thresh02 = [], [], [], [], []
    for name, group, v in raw:
        v = unit(np.asarray(v, np.float64))
        acc, sign, t02 = fit_threshold(h @ v, y)
        if sign < 0:
            v = -v
            acc, sign, t02 = fit_threshold(h @ v, y)
        W.append(v); names.append(name); groups.append(group); acc1d.append(acc)
        thresh02.append(t02 if acc >= MIN_ACC_1D else float("nan"))
    store = [g in ("truth", "truth_sub", "dct_u") for g in groups]
    return {"W": np.stack(W).astype(np.float32),
            "names": np.array(names, dtype=object),
            "groups": np.array(groups, dtype=object),
            "acc1d": np.array(acc1d, np.float32),
            "thresh02": np.array(thresh02, np.float32),
            "store_jtw": np.array(store)}


# ---------------------------------------------------------------------- stages

def atomic_savez(path, **arrays):
    """Crash-safe np.savez: write to a temp sibling, then atomically rename into
    place so a partial write never leaves a truncated file at `path` that resume
    would treat as a completed chunk.

    Note: np.savez(str_path, ...) auto-appends ".npz" to any path that doesn't
    already end in ".npz" — since `path + ".tmp"` doesn't, passing that string
    straight to np.savez would silently write to `path + ".tmp.npz"` instead,
    and the following os.replace(tmp, path) would then raise FileNotFoundError.
    Writing through an open file handle avoids that auto-appending, so the file
    lands at exactly `tmp` as intended."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def stage_acts(ds, device, limit=0):
    validate_inputs(ds)
    stmts, labels, row_index = load_statements(ds)
    if limit:
        stmts, labels, row_index = stmts[:limit], labels[:limit], row_index[:limit]
    tok, model, _, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    hs_src, hs_tgt, hs_fin = [], [], []
    for b0 in range(0, len(stmts), ACTS_BATCH):
        fb = forward_source_batch(model, tok, stmts[b0:b0 + ACTS_BATCH],
                                  meta["src"], meta["tgt"], dev)
        hs_src.append(fb["h_src"].cpu().numpy())
        hs_tgt.append(fb["h_tgt"].cpu().numpy())
        hs_fin.append(fb["h_final"].cpu().numpy())
        print(f"[acts] {ds} {b0 + len(fb['h_src'])}/{len(stmts)}", flush=True)
    np.savez(f"reach_acts_{ds}.npz",
             h_src=np.concatenate(hs_src).astype(np.float32),
             h_tgt=np.concatenate(hs_tgt).astype(np.float32),
             h_final=np.concatenate(hs_fin).astype(np.float32),
             labels=labels, statements=np.array(stmts, dtype=object),
             row_index=row_index, src_layer=meta["src"], tgt_layer=meta["tgt"])
    print(f"[acts] wrote reach_acts_{ds}.npz  n={len(stmts)}")


def stage_dirs(ds):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    bat = build_battery(acts["h_tgt"], acts["labels"].astype(int), ds)
    # sanity: stored mean_diff_tgt must agree with this run's own class-mean diff
    own_md = unit(np.asarray(acts["h_tgt"], np.float64)[acts["labels"] == 1].mean(0)
                  - np.asarray(acts["h_tgt"], np.float64)[acts["labels"] == 0].mean(0))
    cos = float(abs(own_md @ np.asarray(bat["W"][0], np.float64)))
    print(f"[dirs] cos(own mean_diff, truth_dir_tgt mean_diff) = {cos:.3f}")
    if cos < 0.9:
        print("[dirs] WARNING: stored target-layer mean_diff disagrees with this run's "
              "activations — check sampling/layer conventions before trusting margins")
    np.savez(f"reach_dirs_{ds}.npz", src_layer=acts["src_layer"],
             tgt_layer=acts["tgt_layer"], **bat)
    n_valid = int(np.isfinite(bat["thresh02"]).sum())
    print(f"[dirs] wrote reach_dirs_{ds}.npz  K={len(bat['names'])} "
          f"(valid thresholds: {n_valid})")


def stage_vjp(ds, device, resume=True, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    W = torch.tensor(dirs["W"], dtype=torch.float32)
    store = np.asarray(dirs["store_jtw"])
    lm = load_landmarks(ds)
    lm_names = [k for k in ("md_src", "v_q", "dct_v") if k in lm]
    stmts = acts["statements"]
    n = min(limit, len(stmts)) if limit else len(stmts)
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    W_dev = W.to(dev)
    store_t = torch.tensor(np.asarray(store, bool), dtype=torch.bool, device=dev)
    lm_t = {k: torch.tensor(lm[k], dtype=torch.float32, device=dev)
            for k in lm_names}
    cdir = f"reach_chunks_{ds}"
    os.makedirs(cdir, exist_ok=True)
    for c0 in range(0, n, CHUNK):
        cpath = os.path.join(cdir, f"chunk_{c0:05d}.npz")
        if resume and os.path.exists(cpath):
            print(f"[vjp] {cpath} exists — skipping", flush=True)
            continue
        c1 = min(c0 + CHUNK, n)
        m_l, jtw_l = [], []
        cos_l = {k: [] for k in lm_names}
        for b0 in range(c0, c1, VJP_BATCH):
            batch = stmts[b0:min(b0 + VJP_BATCH, c1)]
            fb = forward_source_batch(model, tok, batch, meta["src"], meta["tgt"], dev)
            f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
            delta0 = torch.zeros(len(batch), fb["h_src_seq"].shape[-1], device=dev)
            _, G = vjp_rows(f, delta0, W_dev)              # (K, B, d)
            m = G.norm(dim=-1)                             # (K, B)
            Gu = G / m.clamp_min(1e-12)[..., None]
            m_l.append(m.T.cpu().numpy())
            for k in lm_names:
                cos_l[k].append((Gu @ lm_t[k]).T.cpu().numpy())
            jtw_l.append(Gu[store_t].permute(1, 0, 2).cpu().numpy().astype(np.float16))
            print(f"[vjp] {ds} statements {b0}-{b0 + len(batch)} done", flush=True)
        atomic_savez(cpath, margins=np.concatenate(m_l).astype(np.float32),
                     jtw=np.concatenate(jtw_l),
                     **{COS_KEY[k]: np.concatenate(v).astype(np.float32)
                        for k, v in cos_l.items()})
        print(f"[vjp] checkpointed {cpath}", flush=True)
    merge_chunks(ds, n, dirs)


def merge_chunks(ds, n, dirs):
    cdir = f"reach_chunks_{ds}"
    keys = list(np.load(os.path.join(cdir, f"chunk_{0:05d}.npz")).files)
    parts = {k: [] for k in keys}
    for c0 in range(0, n, CHUNK):
        z = np.load(os.path.join(cdir, f"chunk_{c0:05d}.npz"))
        for k in keys:
            parts[k].append(z[k])
    store = np.asarray(dirs["store_jtw"])
    np.savez(f"reach_margins_{ds}.npz",
             **{k: np.concatenate(v) for k, v in parts.items()},
             names=dirs["names"], groups=dirs["groups"],
             store_names=np.asarray(dirs["names"])[store])
    print(f"[vjp] merged {n} statements -> reach_margins_{ds}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", required=True, choices=["acts", "dirs", "vjp", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap statements (smoke)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    if a.stage in ("acts", "all"):
        stage_acts(a.dataset, a.device, a.limit)
    if a.stage in ("dirs", "all"):
        stage_dirs(a.dataset)
    if a.stage in ("vjp", "all"):
        stage_vjp(a.dataset, a.device, resume=not a.no_resume, limit=a.limit)


if __name__ == "__main__":
    main()
