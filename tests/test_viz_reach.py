import sys, os, json, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_reach as vr


def _make_synthetic(ds, tmp_path, n=20, d=8):
    rng = np.random.default_rng(0)
    names = ["mean_diff_tgt", "probe_grad_tgt", "truth_sub_0", "dct_u_0",
             "rand_0", "rand_1"]
    groups = ["truth", "truth", "truth_sub", "dct_u", "rand", "rand"]
    K, Ks = len(names), 4
    np.savez(tmp_path / f"reach_dirs_{ds}.npz",
             W=rng.standard_normal((K, d)).astype(np.float32),
             names=np.array(names, object), groups=np.array(groups, object),
             acc1d=np.full(K, 0.9, np.float32),
             thresh02=np.array([0, 0, 0, np.nan, np.nan, np.nan], np.float32),
             store_jtw=np.array([True, True, True, True, False, False]),
             src_layer=3, tgt_layer=5)
    np.savez(tmp_path / f"reach_margins_{ds}.npz",
             margins=rng.exponential(1.0, (n, K)).astype(np.float32),
             cos_md_src=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             cos_vq=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             cos_dctv=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             jtw=rng.standard_normal((n, Ks, d)).astype(np.float16),
             names=np.array(names, object), groups=np.array(groups, object),
             store_names=np.array(names[:Ks], object))
    grid = np.linspace(0, 60, 61)
    with open(tmp_path / f"reach_curve_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("direction", "eps", "frac_reachable"))
        for nm in ("mean_diff_tgt", "truth_sub_best"):
            for e in grid:
                w.writerow((nm, e, min(1.0, e / 60)))
    with open(tmp_path / f"reach_summary_{ds}.json", "w") as f:
        json.dump({"input_scale": 40.0, "best_sub_name": "truth_sub_0",
                   "rand_null": {"median": 1.0, "q05": 0.5, "q95": 2.0},
                   "verdict": "unreachable-tail",
                   "truth_sub_best": {"median_margin": 0.5}}, f)


def test_phase1_figures_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_synthetic("toy", tmp_path)
    vr.fig_margins("toy"); vr.fig_curves("toy"); vr.fig_geometry("toy")
    for p in ("plot_reach_margins_toy.png", "plot_reach_curves_toy.png",
              "plot_reach_geometry_toy.png"):
        assert os.path.exists(p), p


def test_fig_svd_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with open("reach_svd_energy_toy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("k", "quantity", "mean_energy"))
        for k in range(1, 65):
            w.writerow((k, "w_mean_diff_tgt_in_U", min(1.0, k / 64)))
            w.writerow((k, "rand_in_U", min(1.0, k / 128)))
    vr.fig_svd("toy")
    assert os.path.exists("plot_reach_svd_toy.png")


def test_fig_svd_skips_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    vr.fig_svd("toy")                      # must not raise
    assert "skip" in capsys.readouterr().out
