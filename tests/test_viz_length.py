# tests/test_viz_length.py
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_length as vz


def test_mean_trajectory_averages_and_truncates():
    rows = [
        {"direction": "mean_diff", "tau": "0.0", "entropy": json.dumps([1.0, 2.0, 3.0])},
        {"direction": "mean_diff", "tau": "0.0", "entropy": json.dumps([3.0, 2.0])},   # shorter
    ]
    traj = vz.mean_trajectory(rows, "entropy")
    # common length is 2; averaged: [(1+3)/2, (2+2)/2] = [2.0, 2.0]
    assert traj[("mean_diff", "0.0")] == [2.0, 2.0]
