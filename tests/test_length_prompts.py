import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import length_prompts as lp


def test_cities_stem_strips_answer_and_period():
    stems = lp.cities_stems(n=50, seed=0)
    assert 0 < len(stems) <= 50
    for stem, ans in stems:
        # stem ends right before the country, no trailing period, answer is non-empty
        assert stem.endswith("the country of")
        assert not stem.endswith(".")
        assert ans and ans[0].isupper()
        # the answer must NOT appear in the stem (no leakage)
        assert ans not in stem


def test_cities_stems_deterministic():
    a = lp.cities_stems(n=20, seed=7)
    b = lp.cities_stems(n=20, seed=7)
    assert a == b
