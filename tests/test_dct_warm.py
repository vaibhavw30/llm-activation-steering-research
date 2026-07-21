import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm as dw


def test_lam_tag():
    assert dw.lam_tag(0.0) == "lam0"
    assert dw.lam_tag(0.1) == "lam0p1"
    assert dw.lam_tag(0.3) == "lam0p3"


def test_load_seed_asserts_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savez("truth_dir_toy.npz",
             mean_diff=np.array([3.0, 4.0], np.float32),
             grad=np.array([1.0, 0.0], np.float32), layer=np.array(11))
    v = dw.load_seed("toy", "mean_diff", 11)
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.8])
    try:
        dw.load_seed("toy", "mean_diff", 7)      # wrong layer
        assert False, "expected AssertionError"
    except AssertionError:
        pass
