import math
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.probes import (
    reconstruction_error, wilson_ci, E1_readability, E2_disagreement,
    E3_linearity, transfer_rank,
)


def _cache(L=3, n=40, d=6, seed=1):
    rng = np.random.default_rng(seed)
    # make InputDelta linearly separable by gold so E1 acc > 0.5
    y = np.array([0, 1] * (n // 2))
    A_p = rng.standard_normal((L, n, d))
    A_Qp = A_p.copy()
    A_Qp[:, :, 0] += (y * 4.0 - 2.0)          # class signal in dim 0 of the shift
    per = lambda: rng.standard_normal((L, n, d))
    const = lambda: rng.standard_normal((L, d))
    c = {"A_p": A_p, "A_Qp": A_Qp, "A_Qpv": per(), "A_verdict": per(),
         "A_EQp": per(), "A_Q": const(), "A_empty": const()}
    return c, y


def test_reconstruction_error_zero_when_perfect():
    shifts = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert reconstruction_error(shifts, np.array([1.0, 0.0])) < 1e-12


def test_reconstruction_error_one_when_v_zero():
    shifts = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert abs(reconstruction_error(shifts, np.zeros(2)) - 1.0) < 1e-12


def test_wilson_ci_brackets_point():
    lo, hi = wilson_ci(7, 10)
    assert 0.0 <= lo < 0.7 < hi <= 1.0


def test_E1_returns_expected_keys_and_beats_half_on_signal():
    c, y = _cache()
    rows = E1_readability(c, layer=1, y_gold=y, y_yM=y, dct=None)
    assert rows and set(rows[0]) == {"operator", "target", "acc", "roc", "n"}
    idel = [r for r in rows if r["operator"] == "InputDelta" and r["target"] == "gold"]
    assert idel and idel[0]["acc"] > 0.6      # planted signal is decodable


def test_E1_near_degenerate_yM_returns_nan_not_crash():
    # y^M on a base model can be all-"yes" but for a single statement (e.g. 39/40
    # class 1, one class 0). Two classes exist, so a naive len(unique)<2 guard
    # passes, but 5-fold stratified CV then leaves a training split single-class.
    # The probe must degrade to nan, not raise.
    c, y = _cache()
    y_yM = np.ones_like(y); y_yM[0] = 0        # 39 ones, 1 zero
    rows = E1_readability(c, layer=1, y_gold=y, y_yM=y_yM, dct=None)
    ym_rows = [r for r in rows if r["target"] == "yM"]
    assert ym_rows and all(math.isnan(r["acc"]) and math.isnan(r["roc"]) for r in ym_rows)


def test_E2_match_rate_in_unit_interval():
    c, y = _cache()
    from mag.operators import operator_features
    feat = operator_features("InputDelta", c, 1)
    y_yM = y.copy(); y_yM[:5] = 1 - y_yM[:5]   # a few disagreements
    rows = E2_disagreement(feat, operator_features("Direct", c, 1), y, y_yM)
    for r in rows:
        assert 0.0 <= r["match_yM_rate"] <= 1.0


def test_E3_linearity_returns_vq_row_finite():
    c, y = _cache()
    d = c["A_p"].shape[2]
    probe = np.zeros(d); probe[0] = 1.0
    rows = E3_linearity(c, layer=1, extra_dirs={"probe": probe})
    vq = [r for r in rows if r["direction"] == "v_Q"]
    assert vq
    assert math.isfinite(vq[0]["eps_Q"]) and vq[0]["eps_Q"] >= 0.0
    assert -1.0 <= vq[0]["cos"] <= 1.0
    probe_rows = [r for r in rows if r["direction"] == "probe"]
    assert probe_rows


def test_transfer_rank_keys_and_ranges():
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * 6)
    feats_by_ds, labels_by_ds = {}, {}
    for name in ("ds_a", "ds_b", "ds_c"):
        X = rng.standard_normal((12, 4))
        X[:, 0] += (labels * 4.0 - 2.0)   # class-separating signal in dim 0
        feats_by_ds[name] = X
        labels_by_ds[name] = labels.copy()
    result = transfer_rank(feats_by_ds, labels_by_ds)
    assert set(result) == {"top1", "spearman", "rows"}
    assert 0.0 <= result["top1"] <= 1.0
    assert result["rows"]
    for row in result["rows"]:
        assert {"target", "candidate", "realized_delta", "geom_score"} <= set(row)
