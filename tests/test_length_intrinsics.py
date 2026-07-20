import sys, os, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_intrinsics as li


def test_max_prob_peaked_vs_flat():
    peaked = np.array([100.0, 0.0, 0.0, 0.0])   # near one-hot after softmax
    flat = np.zeros(4)                            # uniform
    assert li.token_max_prob(peaked) > 0.99
    assert abs(li.token_max_prob(flat) - 0.25) < 1e-6


def test_entropy_flat_is_log_vocab():
    flat = np.zeros(8)
    assert abs(li.token_entropy(flat) - math.log(8)) < 1e-6
    peaked = np.array([100.0, 0.0, 0.0])
    assert li.token_entropy(peaked) < 1e-3


def test_rep3_flags_detects_repeated_trigram():
    # tokens: a b c a b c  -> the second "c" (idx 5) completes a repeated trigram (a,b,c)
    ids = [1, 2, 3, 1, 2, 3]
    flags = li.rep3_flags(ids)
    assert flags == [0, 0, 0, 0, 0, 1]


def test_intrinsic_signals_lengths_and_keys():
    scores = [np.zeros(5) for _ in range(4)]
    ids = [0, 1, 2, 2]
    out = li.intrinsic_signals(scores, ids)
    assert set(out) == {"max_prob", "entropy", "rep3"}
    assert all(len(out[k]) == 4 for k in out)
