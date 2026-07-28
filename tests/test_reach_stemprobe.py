import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_stemprobe as rsp


def test_stem_labels_scale0_true_false_only(tmp_path):
    path = tmp_path / "judge.csv"
    rows = [("jtw_stmt", "0.0", "stem A", "c", "TRUE", ""),
            ("jtw_stmt", "0.0", "stem B", "c", "FALSE", ""),
            ("jtw_stmt", "0.0", "stem C", "c", "INCOHERENT", ""),
            ("jtw_stmt", "2.5", "stem D", "c", "FALSE", "")]   # nonzero scale ignored
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "scale", "prompt", "completion", "verdict", "reason"))
        w.writerows(rows)
    assert rsp.stem_labels(path) == {"stem A": 1, "stem B": 0}


def test_transfer_metrics_hand_computed():
    g = np.array([2.0, -1.0, 3.0, -4.0])   # pred: 1 0 1 0
    y = np.array([1, 0, 0, 0])             # correct: 1 1 0 1 -> acc .75
    acc, bal = rsp.transfer_metrics(g, y)
    assert abs(acc - 0.75) < 1e-12
    assert abs(bal - (2 / 3 + 1.0) / 2) < 1e-12   # class0 2/3, class1 1/1


def test_cv_acc_separable_and_degenerate():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-3, 0.1, (20, 2)), rng.normal(3, 0.1, (20, 2))])
    y = np.array([0] * 20 + [1] * 20)
    acc = rsp._cv_acc(lambda: LogisticRegression(), X, y)
    assert acc > 0.95
    assert np.isnan(rsp._cv_acc(lambda: LogisticRegression(),
                                X[:21], np.array([0] * 20 + [1])))
