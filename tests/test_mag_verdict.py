import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.verdict import verdict_from_logits


def test_yes_wins():
    v = 8
    logits = np.full(v, -10.0); logits[2] = 5.0   # yes id
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 1 and r["margin"] > 0 and 0 < r["conf"] <= 1


def test_no_wins():
    v = 8
    logits = np.full(v, -10.0); logits[3] = 5.0   # no id
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 0 and r["margin"] < 0


def test_tie_defaults_to_zero():
    v = 8
    logits = np.zeros(v)                            # p_yes == p_no
    r = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r["ymL"] == 0                            # strict > means tie -> 0
    assert abs(r["margin"]) < 1e-9


def test_multi_variant_ids_are_summed():
    v = 8
    logits = np.full(v, -10.0); logits[2] = 2.0; logits[4] = 2.0   # two yes variants
    r_two = verdict_from_logits(logits, yes_ids=[2, 4], no_ids=[3])
    r_one = verdict_from_logits(logits, yes_ids=[2], no_ids=[3])
    assert r_two["p_yes"] > r_one["p_yes"]         # summing variants raises p_yes
