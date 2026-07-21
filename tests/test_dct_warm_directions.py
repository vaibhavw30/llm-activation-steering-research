import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm_directions as dd


def test_aligned_flips_sign_toward_ref():
    ref = np.array([1.0, 0.0])
    v = np.array([-2.0, 0.0])
    out = dd.aligned(v, ref)
    assert np.allclose(out, [1.0, 0.0])          # unit + sign-flipped toward ref


def test_aligned_is_unit():
    out = dd.aligned(np.array([3.0, 4.0]), np.array([1.0, 1.0]))
    assert np.allclose(np.linalg.norm(out), 1.0)
