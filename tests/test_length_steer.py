# tests/test_length_steer.py
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_steer as ls


def test_injected_vector_scales_and_normalizes():
    d = np.array([3.0, 4.0, 0.0])          # norm 5
    v = ls.injected_vector(2.0, d, 10.0)   # 2 * 10 * unit(d)
    assert np.allclose(np.linalg.norm(v), 20.0)
    assert np.allclose(v, np.array([12.0, 16.0, 0.0]))


def test_injected_vector_tau_zero_is_zero():
    d = np.array([1.0, 2.0, 2.0])
    assert np.allclose(ls.injected_vector(0.0, d, 99.0), 0.0)


def test_injected_vector_negative_tau_flips_direction():
    d = np.array([3.0, 4.0, 0.0])
    assert np.allclose(ls.injected_vector(-1.0, d, 10.0),
                       -ls.injected_vector(1.0, d, 10.0))


def test_taus_are_two_sided_and_symmetric():
    # -tau pushes mean_diff toward FALSE, +tau toward TRUE; 0.0 must be present (no-injection control)
    assert 0.0 in ls.TAUS
    assert sorted(ls.TAUS) == sorted(-t for t in ls.TAUS)   # symmetric about 0


class _FakeTok:
    # decode returns "t<id> " per token so cutoffs are countable
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def test_prefixes_for_cutoffs_truncates():
    ids = list(range(10))
    out = ls.prefixes_for_cutoffs(ids, [4, 8, 16], _FakeTok())
    assert set(out) == {4, 8}          # 16 > 10 generated tokens -> skipped
    assert out[4] == "t0 t1 t2 t3"
    assert out[8] == "t0 t1 t2 t3 t4 t5 t6 t7"
