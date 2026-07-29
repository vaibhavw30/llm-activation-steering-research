"""reach_stemjac.py — Horizon-0 items 0.4 + 0.5: stem-context J^T w geometry.

D1 said Jacobian gain collapses 8-35x across a one-word context shift. This
measures the shift itself: recompute J^T w (w = mean_diff_tgt) at the STEM
context for the same 200 statements whose full-context rows are already cached
in reach_margins_<ds>.npz, then compare —
  per statement : cos(full, stem), gain ratio m_stem/m_full,
                  first-order robust budget eps_robust = g / (m_full - ||Delta J^T w||)
  population    : principal angles between the full-row and stem-row subspaces,
                  the top stacked-SVD common direction and how well it aligns
                  with each family ("does ANY input direction survive the shift?")

  --compute (GPU): one vjp per stem batch -> reach_stemjac_<ds>.npz
  --analyze (CPU): reach_stemjac_summary_<ds>.csv + printed aggregates

    PYTHONPATH=src python src/reach_stemjac.py --dataset cities --compute --device cuda
    PYTHONPATH=src python src/reach_stemjac.py --dataset cities --analyze
"""
import argparse
import csv

import numpy as np

from reach_steer import stem_of, N_PER_STMT, SEED

VJP_BATCH = 16
Q_SUBSPACE = 8


# ---------------------------------------------------------------- pure helpers

def pair_metrics(jtw_full, m_full, jtw_stem, m_stem, g):
    """All inputs per-statement arrays; jtw_* are UNIT rows (n,d), m_* norms,
    g the full-point budget numerator. Returns dict of per-statement arrays."""
    jf = np.asarray(jtw_full, np.float64)
    js = np.asarray(jtw_stem, np.float64)
    m_f = np.asarray(m_full, np.float64)
    m_s = np.asarray(m_stem, np.float64)
    cos = np.sum(jf * js, axis=1)
    ratio = m_s / np.maximum(m_f, 1e-12)
    delta = np.linalg.norm(m_f[:, None] * jf - m_s[:, None] * js, axis=1)
    worst = m_f - delta
    g = np.asarray(g, np.float64)
    eps_star = np.where(g > 0, g / np.maximum(m_f, 1e-12), 0.0)
    eps_robust = np.where((g > 0) & (worst > 0), g / np.maximum(worst, 1e-12),
                          np.inf)
    return {"cos": cos, "ratio": ratio, "delta_norm": delta,
            "eps_star": eps_star, "eps_robust": eps_robust}


def principal_angles(A, B, q=Q_SUBSPACE):
    """Angles (degrees) between span(rows of A) and span(rows of B), each
    truncated to its top-q right-singular subspace."""
    Qa = np.linalg.svd(np.asarray(A, np.float64), full_matrices=False)[2][:q]
    Qb = np.linalg.svd(np.asarray(B, np.float64), full_matrices=False)[2][:q]
    s = np.clip(np.linalg.svd(Qa @ Qb.T, compute_uv=False), 0.0, 1.0)
    return np.degrees(np.arccos(s))


def common_direction(A, B):
    """Top right-singular vector of the stacked rows, plus |cos| of each row
    family against it: the best single input direction shared by both contexts."""
    stacked = np.concatenate([np.asarray(A, np.float64),
                              np.asarray(B, np.float64)])
    v1 = np.linalg.svd(stacked, full_matrices=False)[2][0]
    return v1, np.abs(np.asarray(A, np.float64) @ v1), \
        np.abs(np.asarray(B, np.float64) @ v1)


# --------------------------------------------------------------------- stages

def compute(ds, device, limit=0):
    import torch
    from reach_hop import (load_model_and_slice, forward_source_batch, make_hop,
                           vjp_rows)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    stmts = acts["statements"]
    y = np.asarray(acts["labels"]).astype(int)
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    picks = rng.permutation(idx1)[:N_PER_STMT]
    if limit:
        picks = picks[:limit]
    keep = [(int(i), stem_of(stmts[i])) for i in picks
            if stem_of(stmts[i]) is not None]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    W = torch.tensor(np.asarray(dirs["W"][names.index("mean_diff_tgt")],
                                np.float32))[None, :].to(dev)
    jtw_l, m_l = [], []
    for b0 in range(0, len(keep), VJP_BATCH):
        batch = keep[b0:b0 + VJP_BATCH]
        fb = forward_source_batch(model, tok, [s for _, s in batch],
                                  meta["src"], meta["tgt"], dev)
        f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
        delta0 = torch.zeros(len(batch), fb["h_src_seq"].shape[-1], device=dev)
        _, G = vjp_rows(f, delta0, W)                  # (1, B, d)
        m = G[0].norm(dim=-1)                          # (B,)
        jtw_l.append((G[0] / m.clamp_min(1e-12)[:, None]).cpu().numpy()
                     .astype(np.float16))
        m_l.append(m.cpu().numpy())
        print(f"[stemjac] {ds} {b0 + len(batch)}/{len(keep)}", flush=True)
    np.savez(f"reach_stemjac_{ds}.npz",
             jtw_stem=np.concatenate(jtw_l),
             m_stem=np.concatenate(m_l).astype(np.float32),
             stmt_index=np.array([i for i, _ in keep]))
    print(f"[stemjac] wrote reach_stemjac_{ds}.npz  n={len(keep)}")


def analyze(ds):
    sj = np.load(f"reach_stemjac_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    store_names = [str(x) for x in mz["store_names"]]
    k, ks = names.index("mean_diff_tgt"), store_names.index("mean_diff_tgt")
    idx = np.asarray(sj["stmt_index"], int)
    jtw_raw = mz["jtw"]        # ONE decompression — per-row access re-reads the zip
    sel = jtw_raw[idx][:, ks, :].astype(np.float64)
    jtw_full = sel / np.linalg.norm(sel, axis=1, keepdims=True)
    m_full = np.asarray(mz["margins"], np.float64)[idx, k]
    jtw_stem = np.asarray(sj["jtw_stem"], np.float64)
    jtw_stem /= np.linalg.norm(jtw_stem, axis=1, keepdims=True)   # f16 renorm
    m_stem = np.asarray(sj["m_stem"], np.float64)
    w = np.asarray(dirs["W"][k], np.float64)
    t02 = float(dirs["thresh02"][k])
    g = np.asarray(acts["h_tgt"], np.float64)[idx] @ w - t02
    pm = pair_metrics(jtw_full, m_full, jtw_stem, m_stem, g)
    ang = principal_angles(jtw_full, jtw_stem)
    v1, cA, cB = common_direction(jtw_full, jtw_stem)
    finite = np.isfinite(pm["eps_robust"])
    print(f"[stemjac] {ds}: n={len(idx)}")
    print(f"  gain ratio m_stem/m_full   median {np.median(pm['ratio']):.3f}  "
          f"[IQR {np.percentile(pm['ratio'], 25):.3f}, "
          f"{np.percentile(pm['ratio'], 75):.3f}]")
    print(f"  cos(jtw_full, jtw_stem)    median {np.median(pm['cos']):.3f}")
    print(f"  principal angles (q={Q_SUBSPACE}, deg): "
          + " ".join(f"{a:.0f}" for a in ang))
    print(f"  common direction v1: median |cos| full {np.median(cA):.3f}, "
          f"stem {np.median(cB):.3f}")
    print(f"  robust certificate finite for {int(finite.sum())}/{len(idx)} "
          f"({finite.mean():.0%}); median eps_robust/eps_star "
          f"{np.median((pm['eps_robust'] / np.maximum(pm['eps_star'], 1e-12))[finite]):.2f}"
          if finite.any() else
          f"  robust certificate finite for 0/{len(idx)} — no statement survives "
          f"the measured context shift at first order")
    with open(f"reach_stemjac_summary_{ds}.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(("stmt_index", "m_full", "m_stem", "ratio",
                       "cos_full_stem", "delta_norm", "eps_star", "eps_robust"))
        for j, i in enumerate(idx):
            wcsv.writerow((int(i), f"{m_full[j]:.6g}", f"{m_stem[j]:.6g}",
                           f"{pm['ratio'][j]:.6g}", f"{pm['cos'][j]:.6g}",
                           f"{pm['delta_norm'][j]:.6g}",
                           f"{pm['eps_star'][j]:.6g}",
                           f"{pm['eps_robust'][j]:.6g}"))
    print(f"[stemjac] wrote reach_stemjac_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap stems (smoke)")
    a = ap.parse_args()
    if a.compute:
        compute(a.dataset, a.device, a.limit)
    if a.analyze:
        analyze(a.dataset)
    if not (a.compute or a.analyze):
        raise SystemExit("pass --compute and/or --analyze")


if __name__ == "__main__":
    main()
