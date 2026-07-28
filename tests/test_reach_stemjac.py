import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_stemjac as rsj


def test_pair_metrics_identical_rows():
    """Same direction, same gain -> ratio 1, cos 1, no shift, robust == star."""
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    pm = rsj.pair_metrics(u, [2.0, 4.0], u, [2.0, 4.0], g=[1.0, 8.0])
    assert np.allclose(pm["cos"], 1.0) and np.allclose(pm["ratio"], 1.0)
    assert np.allclose(pm["delta_norm"], 0.0)
    assert np.allclose(pm["eps_star"], [0.5, 2.0])
    assert np.allclose(pm["eps_robust"], pm["eps_star"])


def test_pair_metrics_orthogonal_rotation_kills_robust_margin():
    """Stem row orthogonal with equal gain: ||Delta|| = m*sqrt(2) > m -> inf."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    pm = rsj.pair_metrics(a, [3.0], b, [3.0], g=[6.0])
    assert abs(pm["cos"][0]) < 1e-12
    assert abs(pm["delta_norm"][0] - 3.0 * np.sqrt(2)) < 1e-9
    assert np.isinf(pm["eps_robust"][0])
    assert abs(pm["eps_star"][0] - 2.0) < 1e-12


def test_pair_metrics_pure_attenuation_inflates_budget():
    """Same direction, gain halved: ||Delta|| = m/2, worst = m/2 -> robust = 2x star."""
    u = np.array([[0.6, 0.8]])
    pm = rsj.pair_metrics(u, [4.0], u, [2.0], g=[4.0])
    assert abs(pm["eps_star"][0] - 1.0) < 1e-12
    assert abs(pm["eps_robust"][0] - 2.0) < 1e-12


def test_principal_angles_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 12))
    assert np.allclose(rsj.principal_angles(A, A, q=4), 0.0, atol=1e-3)
    B = np.zeros((4, 12)); B[np.arange(4), 8 + np.arange(4)] = 1.0
    C = np.zeros((4, 12)); C[np.arange(4), np.arange(4)] = 1.0
    assert np.allclose(rsj.principal_angles(B, C, q=4), 90.0, atol=1e-6)


def test_common_direction_recovers_planted_shared_vector():
    rng = np.random.default_rng(1)
    v = np.zeros(16); v[3] = 1.0
    noise = 0.05 * rng.standard_normal((30, 16))
    A = v + noise[:15]
    B = v + noise[15:]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    v1, cA, cB = rsj.common_direction(A, B)
    assert abs(abs(v1 @ v) - 1.0) < 0.01
    assert np.median(cA) > 0.97 and np.median(cB) > 0.97
