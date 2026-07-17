import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge_results import steer_paths, run_steer_local


def test_steer_paths_default_is_supervised():
    i, o, p = steer_paths("cities")
    assert i == "steer_supervised_cities.csv"
    assert o == "judge_steer_cities.csv"
    assert p == "plot_judge_steering_cities.png"


def test_steer_paths_mag_flag_selects_mag_artifacts():
    i, o, p = steer_paths("cities", mag=True)
    assert i == "mag_steer_cities.csv"
    assert o == "judge_mag_steer_cities.csv"
    assert p == "plot_judge_mag_steering_cities.png"


def test_steer_paths_explicit_overrides_win():
    i, o, p = steer_paths("cities", in_path="x.csv", out_path="y.csv", plot_path="z.png", mag=True)
    assert (i, o, p) == ("x.csv", "y.csv", "z.png")


class _FakeJudge:
    def score(self, prompt, completion):
        # "true" in the completion → truthful, else not; drives a deterministic verdict
        return {"truthful": "true" in completion.lower(), "informative": True}


def test_run_steer_local_reads_and_writes_custom_paths(tmp_path):
    in_path = tmp_path / "mag_steer_x.csv"
    out_path = tmp_path / "judge_mag_steer_x.csv"
    plot_path = tmp_path / "plot_judge_mag_steering_x.png"
    with open(in_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "scale", "prompt", "completion"])
        w.writerow(["mag_u_Prefixed", "-1.0", "The capital of France is", "true, it is Paris"])
        w.writerow(["mag_u_Prefixed", "1.0", "The capital of France is", "false gibberish xyz"])

    run_steer_local(_FakeJudge(), "x", 0, str(in_path), str(out_path), str(plot_path))

    assert out_path.exists() and plot_path.exists()
    rows = list(csv.DictReader(open(out_path)))
    assert [r["verdict"] for r in rows] == ["TRUE", "FALSE"]
    assert set(rows[0]) == {"direction", "scale", "prompt", "completion", "verdict", "reason"}
