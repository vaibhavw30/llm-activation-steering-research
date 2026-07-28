"""reach_samepoint.py — same-point actuation control (deep-audit test #1).

The missing rung between P1 and P3. P1's eps* was computed at the FULL
STATEMENT's last token; P3's stmt arm steered and read one word earlier (the
stem) during generation, and realized slopes came out 8-35x below the predicted
margin m = ||J^T w||. This script repeats the stmt arm EXACTLY — same picks,
same per-statement unit J^T w direction, same scale grids, same Steerer hook —
except the prompt is the full statement and g is read teacher-forced at the
statement's last token: the exact linearization point of eps*. No generation.

Decomposition the summary prints, per dataset (medians over statements):
    m_pred  ->  slope_samepoint  ->  slope_stem
  calibration factor = slope_samepoint / m_pred   (vjp margin vs real hook-forward
                                                   slope at the SAME point)
  context factor     = slope_stem / slope_samepoint  (one-word context shift)

    PYTHONPATH=src python src/reach_samepoint.py --dataset cities --device cuda
    PYTHONPATH=src python src/reach_samepoint.py --dataset cities --device cpu --limit 3  # smoke
    PYTHONPATH=src python src/reach_samepoint.py --dataset cities --summarize
"""
import argparse
import csv
import os

import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from reach_steer import (_load_common, read_g, scale_grid, stem_of,
                         STMT_FRACS, N_PER_STMT, SEED)

MIN_SCALES_FIT = 3


def run(ds, device, limit=0):
    src, tgt, input_scale, summ, dirs, mz, acts, names, store_names = _load_common(ds)
    stmts = acts["statements"]
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    k = names.index("mean_diff_tgt")
    ks = store_names.index("mean_diff_tgt")
    t02 = float(dirs["thresh02"][k])
    w_vec = torch.tensor(np.asarray(dirs["W"][k], np.float32))
    h_tgt = np.asarray(acts["h_tgt"], np.float64)[:len(y)]
    g_all = h_tgt @ np.asarray(dirs["W"][k], np.float64) - t02
    m_all = np.asarray(mz["margins"], np.float64)[:, k]
    # identical selection to reach_steer.arm_per_stmt
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    picks = rng.permutation(idx1)[:N_PER_STMT]
    if limit:
        picks = picks[:limit]
    tok, model, dev = su.load_model(device)
    rows = [("stmt_index", "label", "eps_star", "m_pred", "g_full", "scale", "g_read")]
    with su.Steerer(model, src) as st:
        for i in picks:
            if stem_of(stmts[i]) is None:      # keep population identical to the stem arm
                continue
            stmt = str(stmts[i]).strip()
            eps_i = g_all[i] / max(m_all[i], 1e-12) if g_all[i] > 0 else 0.0
            jtw_i = unit(np.asarray(mz["jtw"], np.float64)[i, ks, :])
            for s in scale_grid(eps_i, input_scale, STMT_FRACS):
                st.set(None if s == 0.0 else torch.tensor(
                    s * jtw_i, dtype=torch.float32))
                g = read_g(model, tok, stmt, tgt, w_vec.to(dev), t02, dev)
                rows.append((int(i), int(y[i]), f"{eps_i:.6g}", f"{m_all[i]:.6g}",
                             f"{g_all[i]:.6g}", s, f"{g:.6g}"))
            print(f"  stmt {int(i)} done", flush=True)
    with open(f"reach_samepoint_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[samepoint] wrote reach_samepoint_{ds}.csv")


def per_stmt_slopes(recs):
    """recs: iterable of (stmt_index, scale, g) -> {stmt_index: (slope, g_at_0)}.
    Least-squares line per statement; statements with < MIN_SCALES_FIT distinct
    scales are skipped. g_at_0 is the recorded g at scale 0 (None if absent)."""
    by = {}
    for i, s, g in recs:
        by.setdefault(int(i), []).append((float(s), float(g)))
    out = {}
    for i, pts in by.items():
        xs = np.array([p[0] for p in pts]); gs = np.array([p[1] for p in pts])
        if len(np.unique(xs)) < MIN_SCALES_FIT:
            continue
        slope = float(np.polyfit(xs, gs, 1)[0])
        at0 = xs == 0.0
        out[i] = (slope, float(gs[at0][0]) if at0.any() else None)
    return out


def _read_recs(path, icol, scol, gcol):
    with open(path, newline="") as f:
        return [(r[icol], r[scol], r[gcol]) for r in csv.DictReader(f)]


def summarize(ds):
    sp = per_stmt_slopes(_read_recs(f"reach_samepoint_{ds}.csv",
                                    "stmt_index", "scale", "g_read"))
    m_pred, g_full = {}, {}
    with open(f"reach_samepoint_{ds}.csv", newline="") as f:
        for r in csv.DictReader(f):
            m_pred[int(r["stmt_index"])] = float(r["m_pred"])
            g_full[int(r["stmt_index"])] = float(r["g_full"])
    calib = np.array([sp[i][0] / m_pred[i] for i in sp if m_pred[i] > 0])
    offs = np.array([sp[i][1] - g_full[i] for i in sp if sp[i][1] is not None])
    print(f"[samepoint] {ds}: n={len(sp)}")
    print(f"  predicted margin m_pred        median {np.median([m_pred[i] for i in sp]):.3f}")
    print(f"  realized same-point slope      median {np.median([sp[i][0] for i in sp]):.3f}")
    print(f"  CALIBRATION factor slope/m     median {np.median(calib):.3f}  "
          f"[IQR {np.percentile(calib, 25):.3f}, {np.percentile(calib, 75):.3f}]")
    print(f"  baseline offset g(0)-g_full    median {np.median(offs):+.3f}  "
          f"(hook-forward vs stored-acts consistency; ~0 expected)")
    out_rows = [("stmt_index", "m_pred", "slope_samepoint", "calibration",
                 "slope_stem", "context_factor")]
    stem_path = f"reach_steer_stmt_meta_{ds}.csv"
    stem = (per_stmt_slopes(_read_recs(stem_path, "stmt_index", "scale", "g_read"))
            if os.path.exists(stem_path) else {})
    ctx = []
    for i in sorted(sp):
        s_sp = sp[i][0]
        s_st = stem[i][0] if i in stem else None
        if s_st is not None and s_sp > 0:
            ctx.append(s_st / s_sp)
        out_rows.append((i, f"{m_pred[i]:.6g}", f"{s_sp:.6g}",
                         f"{s_sp / m_pred[i]:.6g}" if m_pred[i] > 0 else "",
                         f"{s_st:.6g}" if s_st is not None else "",
                         f"{s_st / s_sp:.6g}" if s_st is not None and s_sp > 0 else ""))
    if ctx:
        print(f"  CONTEXT factor stem/samepoint  median {np.median(ctx):.3f}  "
              f"[IQR {np.percentile(ctx, 25):.3f}, {np.percentile(ctx, 75):.3f}]")
        print("  ladder: m_pred --(calibration)--> same-point slope --(context)--> stem slope")
    else:
        print(f"  (no {stem_path} — context factor skipped)")
    with open(f"reach_samepoint_summary_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print(f"[samepoint] wrote reach_samepoint_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap statements (smoke)")
    ap.add_argument("--summarize", action="store_true",
                    help="analyze existing CSV; no model load")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.dataset)
    else:
        run(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
