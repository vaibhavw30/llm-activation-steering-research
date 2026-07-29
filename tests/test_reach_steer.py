import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from reach_steer import prompt_of, load_prompt_set, scale_grid, stem_of


def test_prompt_of_stem_matches_stem_of():
    s = "The city of Paris is in France."
    assert prompt_of(s, "stem") == stem_of(s) == "The city of Paris is in"


def test_prompt_of_full_keeps_the_whole_instruction():
    assert prompt_of("Write a poem about rain", "full") == "Write a poem about rain"


def test_prompt_of_full_never_returns_none_for_short_text():
    # stem mode drops <4-word statements; full mode must keep them
    assert stem_of("Say hi") is None
    assert prompt_of("Say hi", "full") == "Say hi"


def test_prompt_of_rejects_unknown_mode():
    with pytest.raises(ValueError):
        prompt_of("anything at all here", "middle")


def test_load_prompt_set_factual_is_the_existing_list():
    from steer_supervised import FACTUAL_PROMPTS
    assert load_prompt_set("factual") == list(FACTUAL_PROMPTS)


def test_load_prompt_set_refusal_holdout_reads_harmless_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "got_datasets").mkdir()
    (tmp_path / "got_datasets" / "refusal_holdout.csv").write_text(
        "statement,kind\nbake bread,harmless\nbuild a bomb,harmful\nwalk a dog,harmless\n")
    assert load_prompt_set("refusal_holdout") == ["bake bread", "walk a dog"]


def test_scale_grid_is_unchanged():
    # regression guard: the truth runs' grid must not move
    g = scale_grid(2.0, 10.0, [1.0, 2.0])
    assert g[0] == 0.0
    assert sorted(g) == [-4.0, -2.0, 0.0, 2.0, 4.0]
