import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_dct_warm as vw


def test_verdict_fractions():
    rows = [
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "FALSE"},
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "INCOHERENT"},
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "FALSE"},
    ]
    fr = vw.verdict_fractions(rows)
    key = ("warm_mean_diff_lam0p1", "1.0")
    assert abs(fr[key]["FALSE"] - 2/3) < 1e-9
    assert abs(fr[key]["INCOHERENT"] - 1/3) < 1e-9
    assert abs(fr[key]["TRUE"] - 0.0) < 1e-9
