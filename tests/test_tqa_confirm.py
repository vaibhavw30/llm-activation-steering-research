"""Tests for src/tqa_confirm.py (J-C, D3) and tqa_discovery.steer_block."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tqa_confirm as tc  # noqa: E402
import tqa_discovery as td  # noqa: E402


def _selection():
    return {"rules": {
        "S-none": [{"factor": 3, "sign": "both"}, {"factor": 5, "sign": "both"}],
        "S-geo": [{"factor": 7, "sign": -1}],
        "S-beh": [{"factor": 7, "sign": -1}]}}          # same pick as S-geo


def test_confirm_directions_signs_dedupes_and_orders():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((12, 10))
    q2 = rng.standard_normal(12)
    dirs = tc.confirm_directions(_selection(), V, q2, mag=None, n_rand=3)
    names = [n for n, _ in dirs]
    assert names == ["dct_S-none_3", "dct_S-none_5", "dct_S-geo_7", "q2_mean_diff",
                     "rand_0", "rand_1", "rand_2"]
    vec = dict(dirs)
    assert np.allclose(vec["dct_S-geo_7"], -td.unit(V[:, 7]))
    assert np.allclose(vec["dct_S-none_3"], td.unit(V[:, 3]))
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for v in vec.values())


def test_confirm_directions_mag_uyM_only_when_alive():
    rng = np.random.default_rng(1)
    V = rng.standard_normal((6, 4))
    mag = {"u_Q_gold": rng.standard_normal(6), "u_Q_yM": rng.standard_normal(6),
           "g0_alive": np.bool_(False)}
    names = [n for n, _ in tc.confirm_directions({"rules": {}}, V, np.ones(6), mag, n_rand=0)]
    assert names == ["q2_mean_diff", "mag_uQ"]
    mag["g0_alive"] = np.bool_(True)
    dirs = dict(tc.confirm_directions({"rules": {}}, V, np.ones(6), mag, n_rand=0))
    assert "mag_uyM" in dirs
    assert np.allclose(dirs["mag_uQ"], td.unit(mag["u_Q_gold"]))     # u_Q points at true


def test_fracs_signed_are_symmetric():
    f = tc.fracs_signed()
    assert sorted(f) == sorted(-x for x in f) and 0.0 not in f


def test_perm_p():
    assert tc.perm_p(0.5, [0.1, 0.2, 0.6]) == pytest.approx(2 / 4)
    assert tc.perm_p(0.9, [0.1] * 8) == pytest.approx(1 / 9)
    assert tc.perm_p(0.0, []) == 1.0


def _rows():
    rows = []
    for q in range(10):
        rows.append({"direction": "baseline", "frac": "0.0", "question": f"q{q}", "t": 0})
        rows.append({"direction": "good", "frac": "2.0", "question": f"q{q}",
                     "t": int(q < 8)})
        for j in range(4):
            rows.append({"direction": f"rand_{j}", "frac": "2.0", "question": f"q{q}",
                         "t": int(q == j)})
    return rows


def test_summarize_mcnemar_wilson_and_perm():
    out = {r["direction"]: r for r in tc.summarize(_rows(), "t")}
    g = out["good"]
    assert (g["gained"], g["lost"], g["n"]) == (8, 0, 10)
    assert g["rate"] == pytest.approx(0.8) and g["lo"] < 0.8 < g["hi"]
    assert g["mcnemar_p"] == pytest.approx(2 / 2 ** 8)
    assert g["perm_p"] == pytest.approx(1 / 5)
    assert out["rand_0"]["perm_p"] == ""


def test_main_refuses_limit_without_prefix():
    with pytest.raises(SystemExit):
        tc.main(["--stage", "steer", "--limit", "2"])


def test_steer_block_sets_the_vector_and_appends(tmp_path, monkeypatch):
    import dct_steer_utils as su
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(su, "generate_raw", lambda m, t, p, n: " An answer.\nQ: next?")

    class St:
        vec = "unset"

        def set(self, v):
            self.vec = v

    st = St()
    out = "o.csv"
    td.steer_block(None, None, st, "d", np.array([1.0, 0.0]), 2.0,
                   [{"question": "Why?", "half": "A"}], out, tag=-1.0)
    assert st.vec.tolist() == [2.0, 0.0]
    td.steer_block(None, None, st, "baseline", None, 0.0, [{"question": "Why?", "half": "A"}],
                   out, tag=0.0)
    assert st.vec is None
    rows = td.read_csv(out)
    assert [r["direction"] for r in rows] == ["d", "baseline"]
    r = rows[0]
    assert r["prompt"] == "Q: Why?\nA:" and r["answer"] == "An answer."
    assert r["completion"] == "An answer. Q: next?" and r["frac"] == "-1.0"
    assert r["half"] == "A" and float(r["scale"]) == 2.0
