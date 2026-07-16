import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.steer import alpha, injected_vector


def test_alpha_zero_gives_zero():
    assert alpha(0.0, 12.3) == 0.0


def test_alpha_linear_in_tau():
    assert abs(alpha(0.3, 10.0) - 3.0) < 1e-12
    assert abs(alpha(-1.0, 10.0) + 10.0) < 1e-12


def test_injected_vector_zero_when_tau_zero():
    u = np.array([0.6, 0.8])            # unit
    assert np.allclose(injected_vector(0.0, u, 10.0), [0.0, 0.0])


def test_injected_vector_norm_is_calibrated():
    u = np.array([0.6, 0.8])            # ||u|| = 1
    v = injected_vector(0.3, u, 10.0)
    assert abs(np.linalg.norm(v) - 3.0) < 1e-9      # |tau| * A_prefix_norm


def test_injected_vector_sign_flips_with_tau():
    u = np.array([1.0, 0.0])
    assert np.allclose(injected_vector(-0.3, u, 10.0), -injected_vector(0.3, u, 10.0))
