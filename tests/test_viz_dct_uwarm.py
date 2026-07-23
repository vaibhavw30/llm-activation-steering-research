import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import viz_dct_uwarm as vu


def _toy_files(tmp_path):
    rows = [("direction", "scale", "prompt", "completion", "verdict", "reason")]
    for d, _ in vu.PANELS:
        for tau in (-1.0, 0.0, 1.0):
            rows.append((d, tau, "p", "c", "TRUE", ""))
            rows.append((d, tau, "p2", "c2", "INCOHERENT", ""))
    with open(tmp_path / "judge_dct_uwarm_steer_toy.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(tmp_path / "dct_uwarm_geometry_toy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "cos_U0_mdtgt", "cos_V0_mdsrc"))
        w.writerow(("raw_mean_diff", "nan", "1.0"))
        w.writerow(("cold_top", "nan", "0.05"))
        w.writerow(("uwarm_mean_diff_lam0p3", "0.2", "0.1"))
        w.writerow(("uwarm_mean_diff_lam1", "0.6", "0.15"))
        w.writerow(("uwarm_mean_diff_lam3", "0.9", "0.2"))


def test_panels_cover_uwarm_lams():
    names = [d for d, _ in vu.PANELS]
    assert names == ["raw_mean_diff", "cold_top", "uwarm_mean_diff_lam0p3",
                     "uwarm_mean_diff_lam1", "uwarm_mean_diff_lam3"]


def test_plots_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_files(tmp_path)
    vu.plot_curves("toy")
    vu.plot_geometry("toy")
    assert os.path.exists("plot_dct_uwarm_curves_toy.png")
    assert os.path.exists("plot_dct_uwarm_geometry_toy.png")
