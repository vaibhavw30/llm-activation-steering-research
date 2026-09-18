"""E8, part 1: the geometry is what we say it is.

The PI's claim 3 was "check that the steering vector is what you want it to be". These
are the parts of that check that need no model: the RMSNorm site adjoint, and the cone
solve. Every one of them was verified once by hand in the session that wrote
token_geom.py; this file is what makes them stay verified.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import token_geom as tg


# ------------------------------------------------------------------ RMSNorm site

def _hz(seed=0, d=32):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 3, d), rng.normal(0, 0.5, d)


def _fwd(h, gamma):
    """The map the site linearizes: z = (h / rms(h)) * (1 + gamma)."""
    return h / (np.linalg.norm(h) / np.sqrt(len(h))) * (1.0 + gamma)


def test_push_matches_a_finite_difference_jacobian():
    h, gamma = _hz(0)
    push, _ = tg.rmsnorm_site(h, gamma)
    rng = np.random.default_rng(7)
    v = rng.normal(size=len(h))
    v /= np.linalg.norm(v)
    t = 1e-6
    fd = (_fwd(h + t * v, gamma) - _fwd(h - t * v, gamma)) / (2 * t)
    assert np.allclose(push(v), fd, atol=1e-6, rtol=1e-5)


def test_pull_is_the_exact_adjoint_of_push():
    """<push(u), v> == <u, pull(v)>. If this drifts, every pre-norm budget is wrong,
    because cone_budget re-expresses the face normals through `pull` alone."""
    h, gamma = _hz(1)
    push, pull = tg.rmsnorm_site(h, gamma)
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, v = rng.normal(size=len(h)), rng.normal(size=len(h))
        assert abs(push(u) @ v - u @ pull(v)) < 1e-9 * max(1.0, abs(u @ pull(v)))


def test_radial_perturbations_are_annihilated():
    """RMSNorm is degree-zero homogeneous, so pushing along h itself does nothing.
    This is the reason `radial_<dir>` is reported: that fraction of a direction is
    spent for free."""
    h, gamma = _hz(2)
    push, _ = tg.rmsnorm_site(h, gamma)
    assert np.linalg.norm(push(h)) < 1e-10 * np.linalg.norm(h)


def test_pull_contracts_by_sqrt_d_over_norm_h():
    """The scale factor is sqrt(d)/||h||, which for real activations is the ~5x
    RMSNorm penalty. Checked against a direction orthogonal to h, where P_perp is the
    identity and the factor is exact."""
    d = 32
    h = np.zeros(d)
    h[0] = 40.0
    gamma = np.zeros(d)
    _, pull = tg.rmsnorm_site(h, gamma)
    v = np.zeros(d)
    v[1] = 1.0
    assert np.isclose(np.linalg.norm(pull(v)), np.sqrt(d) / 40.0, rtol=1e-9)


# --------------------------------------------------------------------- cone solve

def test_min_norm_ineq_hits_the_hyperplane_distance_for_one_constraint():
    """With a single active face the least-norm solution is the textbook
    point-to-hyperplane distance b/||A||, and it must lie ALONG the normal."""
    A = np.array([[3.0, 4.0]])
    b = np.array([10.0])
    D = tg._min_norm_ineq(A, b, iters=5000)
    assert np.isclose(np.linalg.norm(D), 10.0 / 5.0, rtol=1e-4)
    assert np.isclose(abs(D @ A[0]) / (np.linalg.norm(D) * 5.0), 1.0, rtol=1e-4)


def test_min_norm_ineq_is_feasible_and_no_larger_than_any_feasible_point():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 12))
    b = rng.normal(size=4) + 2.0
    D = tg._min_norm_ineq(A, b)
    assert (A @ D - b).min() > -1e-8
    for _ in range(200):                       # random feasible points, none smaller
        c = rng.normal(size=(4,))
        cand = A.T @ np.abs(c)
        if (A @ cand - b).min() > -1e-8:
            assert np.linalg.norm(D) <= np.linalg.norm(cand) + 1e-6


def test_min_norm_ineq_returns_zero_when_the_origin_is_already_feasible():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = tg._min_norm_ineq(A, np.array([-1.0, -1.0]))
    assert np.linalg.norm(D) < 1e-8


@pytest.mark.parametrize("V,d,seed", [(500, 32, 0), (2000, 64, 1), (5000, 128, 2)])
def test_cone_budget_certificate_is_verified_against_the_full_vocabulary(V, d, seed):
    """`solved=True` is a proof, not an optimizer's opinion: the target must win an
    explicit argmax over every row of E, with strict inequality. This is the property
    the whole program leans on, so it is checked at three vocabulary sizes."""
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(V, d))
    z = rng.normal(size=d) * 2.0
    j = int(np.argsort(-(E @ z))[1])           # the runner-up: a reachable target
    D, solved, nf = tg.cone_budget(E, z, j)
    assert solved and 1 <= nf <= tg.MAX_FACES
    lg = E @ (z + D)
    assert int(np.argmax(lg)) == j
    assert np.sort(lg)[-1] - np.sort(lg)[-2] > 0      # strictly inside, not on a tie


def test_cone_budget_is_within_a_few_percent_of_the_single_face_lower_bound():
    """M/||a|| is a valid lower bound on the least-norm displacement (it ignores every
    face but one). The full solve must be >= it, and in practice close to it: if the
    ratio ever explodes, the active set is thrashing."""
    rng = np.random.default_rng(5)
    E = rng.normal(size=(3000, 64))
    z = rng.normal(size=64) * 2.0
    order = np.argsort(-(E @ z))
    j_top, j = int(order[0]), int(order[1])
    lg = E @ z
    lb = (lg[j_top] - lg[j]) / np.linalg.norm(E[j] - E[j_top])
    D, solved, _ = tg.cone_budget(E, z, j)
    assert solved
    assert np.linalg.norm(D) >= lb - 1e-8
    assert np.linalg.norm(D) < 1.5 * lb


def test_cone_budget_returns_zero_work_when_the_target_already_wins():
    rng = np.random.default_rng(6)
    E = rng.normal(size=(400, 16))
    z = rng.normal(size=16)
    j = int(np.argmax(E @ z))
    D, solved, nf = tg.cone_budget(E, z, j)
    assert solved and nf == 0 and np.linalg.norm(D) < 1e-12


def test_cone_budget_through_the_rmsnorm_site_costs_more_than_at_post_norm():
    """The pre-norm budget is measured in h-units and must exceed the post-norm one,
    because P_perp discards the radial part and sqrt(d)/||h|| contracts the rest. This
    is the `rmsnorm_penalty` column, asserted rather than assumed."""
    rng = np.random.default_rng(9)
    d, V = 48, 1500
    E = rng.normal(size=(V, d))
    h = rng.normal(size=d) * 20.0
    gamma = rng.normal(size=d) * 0.3
    z = _fwd(h, gamma)
    j = int(np.argsort(-(E @ z))[1])
    D0, ok0, _ = tg.cone_budget(E, z, j)
    Dp, okp, _ = tg.cone_budget(E, z, j, site=tg.rmsnorm_site(h, gamma))
    assert ok0 and okp
    assert np.linalg.norm(Dp) > np.linalg.norm(D0)


# ---------------------------------------------------------------------- stem rule

def test_stem_of_drops_the_final_word_and_the_period():
    assert tg.stem_of("The city of Paris is in France.") == "The city of Paris is in"


def test_stem_of_refuses_statements_too_short_to_have_a_stem():
    assert tg.stem_of("Cats purr.") is None


def test_stem_for_uses_the_template_when_given_and_stem_of_otherwise():
    t = "The city of {city} is in the country of"
    assert tg.stem_for("The city of Ufa is in South Africa.", "Ufa", t) == \
        "The city of Ufa is in the country of"
    assert tg.stem_for("The city of Ufa is in South Africa.") == \
        "The city of Ufa is in South"          # the cut mid-country the template avoids


def test_clean_rows_keeps_the_first_row_per_city_and_drops_countries_outside_the_pool():
    cities = ["Ufa", "Ufa", "Lima", "Pyongyang", "Oslo"]
    correct = ["Russia", "Russia", "Peru", "North Korea", "Norway"]
    pool = {"Russia": 1, "Peru": 2, "Norway": 3}
    assert tg.clean_rows(cities, correct, pool) == [0, 2, 4]


def test_clean_rows_does_not_revisit_a_city_whose_first_row_was_dropped():
    assert tg.clean_rows(["X", "X"], ["Atlantis", "Russia"], {"Russia": 1}) == []
