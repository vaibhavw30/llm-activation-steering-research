import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sae_decompose import omp, explained, jaccard


def _dict(F=40, d=12, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((F, d))
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def test_omp_recovers_a_planted_two_atom_vector():
    D = _dict()
    v = 2.0 * D[3] - 1.5 * D[17]
    sup, coefs, resid = omp(v, D, k=2)
    assert set(sup.tolist()) == {3, 17}
    assert resid < 1e-8
    by_atom = {int(s): c for s, c in zip(sup, coefs)}
    assert abs(by_atom[3] - 2.0) < 1e-6 and abs(by_atom[17] + 1.5) < 1e-6


def test_omp_residual_is_monotone_nonincreasing_in_k():
    D = _dict(seed=1)
    v = np.random.default_rng(2).standard_normal(12)
    r = [omp(v, D, k=k)[2] for k in (1, 3, 6, 10)]
    assert all(r[i + 1] <= r[i] + 1e-12 for i in range(len(r) - 1))


def test_omp_never_repeats_an_atom():
    sup, _, _ = omp(np.random.default_rng(4).standard_normal(12), _dict(seed=3), k=8)
    assert len(set(sup.tolist())) == 8


def test_omp_rejects_k_larger_than_the_dictionary():
    with pytest.raises(ValueError):
        omp(np.ones(12), _dict(F=5), k=6)


def test_explained_is_one_for_an_exact_fit():
    D = _dict()
    v = D[0] * 3.0
    sup, coefs, _ = omp(v, D, k=1)
    assert abs(explained(v, D, sup, coefs) - 1.0) < 1e-9


def test_jaccard_edges():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([], []) == 0.0
    assert abs(jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-12
