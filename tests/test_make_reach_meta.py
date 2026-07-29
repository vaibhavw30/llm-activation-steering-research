# tests/test_make_reach_meta.py
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from make_reach_meta import (pick_source_layer, meta_dict, layer_sweep,
                             HOP_DEPTH, MIN_SOURCE_LAYER)


def test_picks_best_linear_layer_that_leaves_room_for_the_hop():
    rows = [{"layer": 5, "linear_acc": 0.80}, {"layer": 11, "linear_acc": 0.95},
            {"layer": 24, "linear_acc": 0.99}]
    # layer 24 + 9 = 33 > max_layer 26, so it is ineligible
    assert pick_source_layer(rows, max_layer=26) == 11


def test_ties_break_to_the_earlier_layer():
    rows = [{"layer": 7, "linear_acc": 0.9}, {"layer": 13, "linear_acc": 0.9}]
    assert pick_source_layer(rows, max_layer=26) == 7


def test_saturated_accuracy_is_floored_at_min_source_layer():
    # a lexically separable concept can hit 1.0 at the embedding layer
    rows = [{"layer": L, "linear_acc": 1.0} for L in range(0, 18)]
    assert pick_source_layer(rows, max_layer=26) == MIN_SOURCE_LAYER


def test_near_ties_within_tolerance_pick_the_earlier_layer():
    rows = [{"layer": 11, "linear_acc": 0.950}, {"layer": 12, "linear_acc": 0.953}]
    assert pick_source_layer(rows, max_layer=26) == 11


def test_raises_when_no_layer_is_eligible():
    with pytest.raises(SystemExit):
        pick_source_layer([{"layer": 2, "linear_acc": 1.0}], max_layer=26)


def test_meta_dict_has_the_keys_validate_inputs_and_reach_analyze_read():
    m = meta_dict("refusal", "google/gemma-2-2b-it", 12)
    assert m["source_layer"] == 12
    assert m["target_layer"] == 12 + HOP_DEPTH
    assert m["model"] == "google/gemma-2-2b-it"
    assert m["input_scale"] is None          # filled by calibrate_scale.py
    assert json.dumps(m)                     # serializable


def test_layer_sweep_returns_one_row_per_layer_and_finds_the_separable_one():
    import numpy as np
    rng = np.random.default_rng(0)
    L, n, d = 4, 80, 6
    y = np.array([0, 1] * (n // 2))
    acts = rng.standard_normal((L, n, d))
    acts[2] += y[:, None] * 6.0              # layer 2 is linearly separable
    rows = layer_sweep(acts, y)
    assert [r["layer"] for r in rows] == [0, 1, 2, 3]
    assert max(rows, key=lambda r: r["linear_acc"])["layer"] == 2
