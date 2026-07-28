import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_samepoint as rsp


def test_per_stmt_slopes_recovers_known_line():
    """Perfect linear data at 5 scales -> exact slope and g(0) per statement."""
    recs = []
    for i, (slope, g0) in enumerate([(0.5, -52.0), (7.0, 80.0)]):
        for s in (-2.0, -1.0, 0.0, 1.0, 2.0):
            recs.append((i, s, g0 + slope * s))
    out = rsp.per_stmt_slopes(recs)
    assert set(out) == {0, 1}
    assert abs(out[0][0] - 0.5) < 1e-9 and abs(out[0][1] - (-52.0)) < 1e-9
    assert abs(out[1][0] - 7.0) < 1e-9 and abs(out[1][1] - 80.0) < 1e-9


def test_per_stmt_slopes_skips_underdetermined_and_missing_zero():
    """< MIN_SCALES_FIT distinct scales -> statement dropped; no scale-0 row -> g0 None."""
    recs = [(0, 0.0, 1.0), (0, 1.0, 2.0),                       # only 2 scales -> dropped
            (1, -1.0, 0.0), (1, 1.0, 4.0), (1, 3.0, 8.0)]       # 3 scales, no 0
    out = rsp.per_stmt_slopes(recs)
    assert 0 not in out
    slope, g0 = out[1]
    assert abs(slope - 2.0) < 1e-9 and g0 is None


def test_summarize_ladder_math(tmp_path, monkeypatch, capsys):
    """End-to-end summarize on synthetic CSVs: calibration and context factors."""
    monkeypatch.chdir(tmp_path)
    ds = "toy"
    # same-point: m_pred=8, realized slope=4 -> calibration 0.5; g_full matches g(0)
    with open(f"reach_samepoint_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stmt_index", "label", "eps_star", "m_pred", "g_full", "scale", "g_read"])
        for s in (-2.0, -1.0, 0.0, 1.0, 2.0):
            w.writerow([7, 1, "1.25", "8.0", "10.0", s, 10.0 + 4.0 * s])
    # stem arm: slope=1 -> context factor 1/4
    with open(f"reach_steer_stmt_meta_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stmt_index", "label", "eps_star", "scale", "g_read"])
        for s in (-2.0, -1.0, 0.0, 1.0, 2.0):
            w.writerow([7, 1, "1.25", s, -50.0 + 1.0 * s])
    rsp.summarize(ds)
    text = capsys.readouterr().out
    assert "CALIBRATION factor slope/m     median 0.500" in text
    assert "CONTEXT factor stem/samepoint  median 0.250" in text
    assert "offset g(0)-g_full    median +0.000" in text
    with open(f"reach_samepoint_summary_{ds}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["stmt_index"] == "7"
    assert float(rows[0]["calibration"]) == 0.5
    assert float(rows[0]["context_factor"]) == 0.25
