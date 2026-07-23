import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm_steer as ws


def test_injected_scales_by_tau_and_input_scale():
    d = np.array([0.0, 3.0, 4.0])                 # norm 5
    v = ws.injected(0.5, d, 10.0)                 # 0.5 * 10 * unit(d)
    assert np.allclose(np.linalg.norm(v), 5.0)
    assert np.allclose(v, [0.0, 3.0, 4.0])


def test_injected_tau0_zero():
    assert np.allclose(ws.injected(0.0, np.array([1.0, 1.0]), 7.0), 0.0)


def test_injected_negative_tau_flips_direction():
    d = np.array([0.0, 3.0, 4.0])
    assert np.allclose(ws.injected(-0.5, d, 10.0), -ws.injected(0.5, d, 10.0))


def test_taus_two_sided_symmetric():
    # -tau -> FALSE (lying), +tau -> TRUE; 0.0 present (no-injection control)
    assert 0.0 in ws.TAUS
    assert sorted(ws.TAUS) == sorted(-t for t in ws.TAUS)


def test_resolve_paths_defaults_and_overrides():
    import dct_warm_steer as dws
    assert dws.resolve_paths("cities") == ("dct_warm_dirs_cities.npz",
                                           "dct_warm_steer_cities.csv")
    assert dws.resolve_paths("cities", "dct_uwarm_dirs_cities.npz",
                             "dct_uwarm_steer_cities.csv") == (
        "dct_uwarm_dirs_cities.npz", "dct_uwarm_steer_cities.csv")
