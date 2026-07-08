import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export_concept_dir import concept_directions


def test_directions_are_unit_and_point_toward_positive_class():
    rng = np.random.default_rng(0)
    # class 1 shifted +1 on dim 0; class 0 shifted -1 on dim 0
    X = rng.standard_normal((200, 5))
    y = (np.arange(200) % 2)
    X[y == 1, 0] += 3.0
    X[y == 0, 0] -= 3.0
    mean_diff, grad = concept_directions(X, y)
    assert abs(np.linalg.norm(mean_diff) - 1.0) < 1e-6
    assert abs(np.linalg.norm(grad) - 1.0) < 1e-6
    assert mean_diff[0] > 0.9   # dominated by dim 0, pointing toward class 1
