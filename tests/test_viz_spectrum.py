import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz_spectrum import build_points


def _write(tmp, name, header, row):
    path = os.path.join(tmp, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerow(row)
    return path


def test_build_points_joins_recovery_and_salience(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "recovery_demo.csv",
           ["concept", "best_abs_cos", "random_max_abs_cos", "ratio_vs_random",
            "subspace_frac", "subspace_chance"],
           ["demo", "0.30", "0.06", "5.0", "0.40", "0.22"])
    with open(tmp_path / "judge_steer_demo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "scale", "prompt", "completion", "verdict", "reason"])
        w.writerow(["grad", "0", "p", "c", "TRUE", ""])
        w.writerow(["grad", "-120", "p", "c", "FALSE", ""])

    df = build_points(["demo"], present_map={"demo": "FALSE"})
    assert list(df["concept"]) == ["demo"]
    assert abs(float(df["y_recovery"].iloc[0]) - 5.0) < 1e-9   # ratio_vs_random
    assert abs(float(df["x_salience"].iloc[0]) - 1.0) < 1e-9   # 0 -> 1 FALSE swing
