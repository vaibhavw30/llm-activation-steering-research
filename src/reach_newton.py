"""reach_newton.py — Horizon-0 item 0.6: iterated re-steering (negative control).

2-3 scalar-Newton steps toward the reachability boundary g = 0, recomputing
J^T w at the CURRENT steered state each step (vjp_rows accepts nonzero delta0).
Same-point context (full statement, teacher-forced) — the exact linearization
point of eps*. Prediction from the audit: converges in ~1 step here, because
the failure is context transfer, not curvature. Either outcome is decisive:
fast convergence confirms the D1 diagnosis; failure would mean the vjp margins
are locally miscalibrated after all (cross-check against reach_samepoint).

    PYTHONPATH=src python src/reach_newton.py --dataset cities --device cuda
    PYTHONPATH=src python src/reach_newton.py --dataset cities --device cpu --limit 2  # smoke
"""
import argparse
import csv

import numpy as np

from reach_steer import stem_of, N_PER_STMT, SEED

N_NEWTON = 32
K_STEPS = 3
CAP_FRAC = 1.5          # ||delta|| cap = CAP_FRAC * input_scale (scale_grid's cap)
CONV_FRAC = 0.05        # converged when |g| < CONV_FRAC * |g_0|


def newton_update(delta, g, jtw, cap):
    """One scalar-Newton step toward g = 0 along J^T w:
    delta <- delta - g * jtw / ||jtw||^2, then norm-cap at `cap`.
    Returns (delta_new, capped). Zero-gradient rows are left unchanged."""
    delta = np.asarray(delta, np.float64)
    jtw = np.asarray(jtw, np.float64)
    m2 = float(jtw @ jtw)
    if m2 < 1e-24:
        return delta.copy(), False
    new = delta - float(g) * jtw / m2
    n = float(np.linalg.norm(new))
    if n > cap:
        return new * (cap / n), True
    return new, False


def run(ds, device, limit=0):
    import torch
    from reach_hop import (load_meta, load_model_and_slice, forward_source_batch,
                           make_hop, vjp_rows)
    _, _, input_scale, _ = load_meta(ds)
    cap = CAP_FRAC * float(input_scale)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    k = names.index("mean_diff_tgt")
    w_np = np.asarray(dirs["W"][k], np.float64)
    t02 = float(dirs["thresh02"][k])
    stmts = acts["statements"]
    y = np.asarray(acts["labels"]).astype(int)
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    picks = [int(i) for i in rng.permutation(idx1)[:N_PER_STMT]
             if stem_of(stmts[i]) is not None][:N_NEWTON]
    if limit:
        picks = picks[:limit]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    W = torch.tensor(w_np, dtype=torch.float32)[None, :].to(dev)
    rows = [("stmt_index", "step", "g", "m", "delta_norm", "cos_jtw0", "capped")]
    g_first, g_last = [], []
    for i in picks:
        fb = forward_source_batch(model, tok, [stmts[i]], meta["src"],
                                  meta["tgt"], dev)
        f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
        delta = np.zeros(fb["h_src_seq"].shape[-1])
        jtw0, capped = None, False
        for step in range(K_STEPS + 1):
            d_t = torch.tensor(delta, dtype=torch.float32, device=dev)[None, :]
            F0, G = vjp_rows(f, d_t, W)
            g = float(np.asarray(F0[0].cpu(), np.float64) @ w_np - t02)
            jtw = np.asarray(G[0, 0].cpu(), np.float64)
            m = float(np.linalg.norm(jtw))
            if jtw0 is None:
                jtw0 = jtw / max(m, 1e-12)
            cos0 = float(jtw / max(m, 1e-12) @ jtw0)
            rows.append((i, step, f"{g:.6g}", f"{m:.6g}",
                         f"{np.linalg.norm(delta):.6g}", f"{cos0:.6g}",
                         int(capped)))
            if step == 0:
                g_first.append(g)
            if step == K_STEPS:
                g_last.append(g)
                break
            delta, capped = newton_update(delta, g, jtw, cap)
        print(f"  stmt {i} done  g0={g_first[-1]:+.2f} -> gK={g_last[-1]:+.2f}",
              flush=True)
    with open(f"reach_newton_{ds}.csv", "w", newline="") as fcsv:
        csv.writer(fcsv).writerows(rows)
    g0 = np.abs(np.array(g_first))
    gK = np.abs(np.array(g_last))
    conv = gK < CONV_FRAC * np.maximum(g0, 1e-12)
    print(f"[newton] {ds}: n={len(picks)}  median |g|: {np.median(g0):.2f} -> "
          f"{np.median(gK):.2f}  converged (<{CONV_FRAC:.0%} of |g0|): "
          f"{int(conv.sum())}/{len(conv)}")
    print(f"[newton] wrote reach_newton_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap statements (smoke)")
    a = ap.parse_args()
    run(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
