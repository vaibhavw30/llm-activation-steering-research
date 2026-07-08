import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum_utils import concept_salience


def _rows(scale_to_verdicts):
    out = []
    for scale, verdicts in scale_to_verdicts.items():
        for v in verdicts:
            out.append({"direction": "grad", "scale": str(scale), "verdict": v})
    return out


def test_salience_is_max_swing_in_present_rate():
    rows = _rows({
        0.0: ["TRUE", "TRUE", "TRUE", "TRUE"],          # present(FALSE)=0.0
        -120.0: ["FALSE", "FALSE", "TRUE", "TRUE"],     # present=0.5
    })
    res = concept_salience(rows, present_verdict="FALSE")
    assert abs(res["x_salience"] - 0.5) < 1e-9
    assert res["best_scale"] == -120.0


def test_scales_over_incoherence_budget_are_excluded():
    rows = _rows({
        0.0: ["TRUE", "TRUE"],
        -120.0: ["INCOHERENT", "INCOHERENT", "INCOHERENT", "FALSE"],  # incoherent=0.75 > 0.5 -> excluded
    })
    res = concept_salience(rows, present_verdict="FALSE", max_incoherent=0.5)
    assert res["x_salience"] == 0.0   # only scale 0 qualifies
