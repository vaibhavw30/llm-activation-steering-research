"""reach_analyze.py — Phase 1 analysis: reachability verdicts, curves, summary.

Linear closed-form test (spec §2), per statement i and readout w:
    reachable within eps  <=>  g_i(w) <= eps * m_i(w)
    g_i(w) = w.h_tgt_i - t02(w)      (signed distance to the FALSE threshold)
    m_i(w) = ||J_i^T w||             (controllability margin, from reach_margins)

Curves are computed over label==1 statements (pushing TRUE statements into the
FALSE halfspace — the lie direction). Runs locally in seconds.

    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
"""
import argparse
import csv
import json

import numpy as np

UNREACHABLE = 1e12          # sentinel eps* when the margin is (numerically) zero
EPS_POINTS = 61


def required_eps(g, m):
    """eps*_i = g/m where g>0; 0 where already in the target set; UNREACHABLE where
    the margin vanishes."""
    g = np.asarray(g, np.float64)
    m = np.asarray(m, np.float64)
    out = np.where(g <= 0, 0.0,
                   np.where(m > 1e-12, g / np.maximum(m, 1e-12), UNREACHABLE))
    return out


def frac_reachable_curve(eps_star, grid):
    es = np.asarray(eps_star, np.float64)
    return np.array([(es <= e).mean() for e in np.asarray(grid, np.float64)])


def subspace_best_margin(G_raw):
    """G_raw (K,d): rows J_i^T b_k for an orthonormal basis {b_k}. Best-case margin
    max_{w in span, ||w||=1} ||J_i^T w|| = sigma_max(G_raw)."""
    return float(np.linalg.svd(np.asarray(G_raw, np.float64), compute_uv=False)[0])


def eps_grid(input_scale, n=EPS_POINTS):
    return np.linspace(0.0, 1.5 * float(input_scale), n)


def analyze(ds):
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    input_scale = float(json.load(open(f"dct_meta_{ds}.json"))["input_scale"])
    W = np.asarray(dirs["W"], np.float64)                    # (K, d)
    names = [str(x) for x in dirs["names"]]
    groups = [str(x) for x in dirs["groups"]]
    thresh = np.asarray(dirs["thresh02"], np.float64)        # (K,)
    h_tgt = np.asarray(acts["h_tgt"], np.float64)            # (n, d)
    y = np.asarray(acts["labels"]).astype(int)
    margins = np.asarray(mz["margins"], np.float64)          # (n, K)
    n = margins.shape[0]
    h_tgt, y = h_tgt[:n], y[:n]                              # tolerate --limit runs
    lab1 = y == 1
    grid = eps_grid(input_scale)

    g_all = h_tgt @ W.T - thresh[None, :]                    # (n, K); NaN thresh -> NaN
    eps_star = {}
    for k, name in enumerate(names):
        if not np.isfinite(thresh[k]):
            continue
        eps_star[name] = required_eps(g_all[lab1, k], margins[lab1, k])

    # --- best-case over the truth subspace: per-statement min eps* over members
    sub_names = [nm for nm, gr in zip(names, groups) if gr in ("truth", "truth_sub")
                 and np.isfinite(thresh[names.index(nm)])]
    if sub_names:
        stack = np.stack([eps_star[nm] for nm in sub_names])          # (S, n1)
        eps_star["truth_sub_best"] = stack.min(axis=0)
        best_sub_name = sub_names[int(np.argmin([np.median(eps_star[nm])
                                                 for nm in sub_names]))]
    else:
        best_sub_name = ""

    # --- per-statement best-case MARGIN over the subspace (sigma_max of stacked J^T b)
    store_names = [str(x) for x in mz["store_names"]]
    sub_rows = [store_names.index(nm) for nm, gr in zip(names, groups)
                if gr in ("truth", "truth_sub")]
    jtw = np.asarray(mz["jtw"], np.float64)                  # (n, Ks, d) unit rows
    col = [names.index(store_names[r]) for r in sub_rows]
    best_margin = np.array([subspace_best_margin(
        jtw[i, sub_rows, :] * margins[i, col][:, None]) for i in range(n)])

    # --- write curves
    with open(f"reach_curve_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "eps", "frac_reachable"))
        for name, es in eps_star.items():
            for e, fr in zip(grid, frac_reachable_curve(es, grid)):
                w.writerow((name, f"{e:.6g}", f"{fr:.6g}"))

    # --- summary
    med_marg = {nm: float(np.median(margins[:, k]))
                for k, nm in enumerate(names)}
    rand_meds = np.array([med_marg[nm] for nm, gr in zip(names, groups)
                          if gr == "rand"])
    def rand_pct(v):
        return float((rand_meds < v).mean() * 100.0)
    summary = {
        "input_scale": input_scale, "n_label1": int(lab1.sum()),
        "rand_null": {"median": float(np.median(rand_meds)),
                      "q05": float(np.quantile(rand_meds, 0.05)),
                      "q95": float(np.quantile(rand_meds, 0.95))},
        "directions": {}, "best_sub_name": best_sub_name,
    }
    for k, nm in enumerate(names):
        d = {"group": groups[k], "median_margin": med_marg[nm],
             "rand_pct": rand_pct(med_marg[nm])}
        if nm in eps_star:
            d["median_eps_star"] = float(np.median(eps_star[nm]))
            d["frac_at_input_scale"] = float((eps_star[nm] <= input_scale).mean())
        summary["directions"][nm] = d
    bm_med = float(np.median(best_margin))
    summary["truth_sub_best"] = {
        "median_margin": bm_med,
        "median_eps_star": float(np.median(eps_star["truth_sub_best"]))
        if "truth_sub_best" in eps_star else None,
        "frac_at_input_scale": float((eps_star["truth_sub_best"] <= input_scale).mean())
        if "truth_sub_best" in eps_star else None,
    }
    summary["verdict"] = ("unreachable-tail"
                          if bm_med < summary["rand_null"]["median"]
                          else "reachable-candidate")
    with open(f"reach_summary_{ds}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] {ds}: verdict={summary['verdict']}  "
          f"best-case median margin={bm_med:.4g} vs rand median="
          f"{summary['rand_null']['median']:.4g}")
    for nm in ("mean_diff_tgt", "probe_grad_tgt", "truth_sub_best"):
        if nm in eps_star:
            print(f"  {nm}: median eps*={np.median(eps_star[nm]):.4g} "
                  f"(input_scale={input_scale:.4g}), frac reachable at budget="
                  f"{(eps_star[nm] <= input_scale).mean():.3f}")
    print(f"[analyze] wrote reach_curve_{ds}.csv and reach_summary_{ds}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    analyze(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
