import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_newton as rn


def test_newton_converges_in_one_step_on_linear_map():
    """g(delta) = g0 + a.delta with constant gradient a: one step lands g = 0."""
    a = np.array([3.0, 4.0])          # ||a|| = 5
    g0 = 10.0
    delta, capped = rn.newton_update(np.zeros(2), g0, a, cap=1e9)
    assert not capped
    assert abs(g0 + a @ delta) < 1e-12          # exactly on the boundary
    assert abs(np.linalg.norm(delta) - g0 / 5.0) < 1e-12   # minimum-norm step


def test_newton_cap_triggers_and_rescales():
    a = np.array([1.0, 0.0])
    delta, capped = rn.newton_update(np.zeros(2), 10.0, a, cap=2.0)
    assert capped
    assert abs(np.linalg.norm(delta) - 2.0) < 1e-12
    assert delta[0] < 0                          # still pointed downhill


def test_newton_zero_gradient_guard():
    start = np.array([1.0, -1.0])
    delta, capped = rn.newton_update(start, 5.0, np.zeros(2), cap=1.0)
    assert not capped
    assert np.array_equal(delta, start)
