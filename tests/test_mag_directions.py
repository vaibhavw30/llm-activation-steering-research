import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.directions import v_Q, class_mean_diff, a_prefix_norm


def test_vq_is_mean_of_shifts():
    A_p = np.array([[0.0, 0.0], [1.0, 1.0]])
    A_Qp = np.array([[1.0, 0.0], [3.0, 1.0]])
    # shifts: [1,0] and [2,0] -> mean [1.5, 0]
    assert np.allclose(v_Q(A_Qp, A_p), [1.5, 0.0])


def test_class_mean_diff_is_neg_minus_pos():
    feats = np.array([[2.0, 0.0],    # label 0
                      [4.0, 0.0],    # label 0
                      [0.0, 0.0]])   # label 1
    labels = np.array([0, 0, 1])
    # mean(neg)=[3,0], mean(pos)=[0,0] -> [3,0]
    assert np.allclose(class_mean_diff(feats, labels), [3.0, 0.0])


def test_a_prefix_norm_positive_and_is_mean_row_norm():
    A_Qp = np.array([[3.0, 4.0], [0.0, 0.0]])   # norms 5 and 0 -> mean 2.5
    assert abs(a_prefix_norm(A_Qp) - 2.5) < 1e-9


def test_class_mean_diff_handles_missing_class():
    feats = np.array([[1.0, 1.0], [2.0, 2.0]])
    labels = np.array([0, 0])                    # no positives
    out = class_mean_diff(feats, labels)
    assert out.shape == (2,) and np.all(np.isfinite(out))
