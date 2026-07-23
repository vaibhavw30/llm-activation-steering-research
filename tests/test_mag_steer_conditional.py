import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.steer_conditional import composed_vec, direction_name, TAUS7, GATES


def test_grid():
    assert TAUS7 == [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    assert GATES == (0, 1)


def test_composed_vec_baseline_is_none():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    assert composed_vec(0, 0.0, vq, md, 2.0) is None      # untouched shared baseline


def test_composed_vec_gate_only():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(1, 0.0, vq, md, 2.0)
    assert np.allclose(v, [2.0, 0.0])                     # apn * v_Q alone


def test_composed_vec_gate_plus_content():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(1, -1.0, vq, md, 2.0)
    assert np.allclose(v, [2.0, -2.0])                    # apn*v_Q + tau*apn*md


def test_composed_vec_content_only_matches_e4_calibration():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(0, 0.6, vq, md, 2.0)
    assert np.allclose(v, [0.0, 1.2])


def test_direction_name():
    assert direction_name(0) == "meandiff_gate0"
    assert direction_name(1) == "meandiff_gate1"


def test_viz_conditional(tmp_path, monkeypatch):
    import csv as _csv
    monkeypatch.chdir(tmp_path)
    rows = [("direction", "scale", "prompt", "completion", "verdict", "reason")]
    for d in ("meandiff_gate0", "meandiff_gate1"):
        for tau in (-1.0, 0.0, 1.0):
            rows.append((d, tau, "p", "c", "TRUE", ""))
            rows.append((d, tau, "p2", "c2", "FALSE" if d.endswith("1") else "TRUE", ""))
    with open(tmp_path / "judge_mag_conditional_toy.csv", "w", newline="") as f:
        _csv.writer(f).writerows(rows)
    from mag.viz_conditional import plot_conditional
    plot_conditional("toy")
    assert os.path.exists("plot_mag_conditional_toy.png")
