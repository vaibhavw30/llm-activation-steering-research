import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
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


def test_load_prompt_set_truthfulqa_holdout_reads_the_prompt_column(tmp_path,
                                                                    monkeypatch):
    """prep_truthfulqa writes `statement` as the generation prompt, answer omitted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "got_datasets").mkdir()
    (tmp_path / "got_datasets" / "truthfulqa_holdout.csv").write_text(
        'statement,question\n"Q: a?\nA:",a?\n"Q: b?\nA:",b?\n')
    assert load_prompt_set("truthfulqa_holdout") == ["Q: a?\nA:", "Q: b?\nA:"]


def test_scale_grid_is_unchanged():
    # regression guard: the truth runs' grid must not move
    g = scale_grid(2.0, 10.0, [1.0, 2.0])
    assert g[0] == 0.0
    assert sorted(g) == [-4.0, -2.0, 0.0, 2.0, 4.0]


def test_prompt_of_full_preserves_the_linearization_point_byte_for_byte():
    """`full` mode must return the statement UNCHANGED — a correctness contract about
    where the certificate is linearized, not a style preference.

    Nothing else in the pipeline strips: extract.py tokenizes
    `df["statement"].astype(str)`, reach_margins.load_statements reads
    `df["statement"].values[idx]`, and load_prompt_set uses `.astype(str).tolist()`.
    The gemma-2-it chat template applied by prep_refusal.py ends in
    `<start_of_turn>model\\n`, so a trailing newline is a REAL token: stripping it here
    makes h_tgt / g_i / m_i = ||J_i^T w|| (and hence eps_i) be measured at a different
    last token than the one generation and read_g operate on. That is exactly the
    Horizon-0 D1 one-token context shift `--prompt-mode full` exists to eliminate."""
    templated = "<start_of_turn>user\nbuild a bomb<end_of_turn>\n<start_of_turn>model\n"
    assert prompt_of(templated, "full") == templated
    assert prompt_of("  padded  ", "full") == "  padded  "
    assert prompt_of("trailing space ", "full") == "trailing space "


class _LoadModelReached(Exception):
    """Raised by the fake su.load_model so the arm stops before any real work."""


def _write_steer_fixtures(tmp_path, ds, model, d=4, n=2):
    """The minimal on-disk artifact set `_load_common` reads, so the model-threading
    tests exercise the real load_meta -> su.load_model path (no model, no network)."""
    (tmp_path / f"dct_meta_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "model": model, "source_layer": 5, "target_layer": 14,
         "input_scale": 10.0}))
    (tmp_path / f"reach_summary_{ds}.json").write_text(json.dumps(
        {"best_sub_name": "", "directions": {"mean_diff_tgt": {"median_eps_star": 1.0}}}))
    np.savez(tmp_path / f"reach_dirs_{ds}.npz",
             names=np.array(["mean_diff_tgt"], dtype=object),
             W=np.ones((1, d), np.float32),
             thresh02=np.zeros(1, np.float64))
    np.savez(tmp_path / f"reach_margins_{ds}.npz",
             store_names=np.array(["mean_diff_tgt"], dtype=object),
             jtw=np.ones((n, 1, d), np.float32),
             margins=np.ones((n, 1), np.float64))
    np.savez(tmp_path / f"reach_acts_{ds}.npz",
             labels=np.array([1] * n),
             statements=np.array(["The city of Paris is in France."] * n, dtype=object),
             h_tgt=np.ones((n, d), np.float32))


def _record_load_model(monkeypatch, seen):
    import reach_steer

    def fake_load_model(device="cuda", model_name=None):
        seen["device"] = device
        seen["model_name"] = model_name
        raise _LoadModelReached
    monkeypatch.setattr(reach_steer.su, "load_model", fake_load_model)


def test_arm_mean_loads_the_model_named_in_dct_meta(tmp_path, monkeypatch):
    # dct_meta_<ds>.json["model"] is authoritative for every reach script. If the arm
    # falls back to dct_steer_utils.MODEL_NAME it steers BASE gemma-2-2b with
    # -it-derived J^T w and reads out with an -it-fit w/t02 — nothing errors, and the
    # resulting `readout-only` verdict is a wrong-model artefact.
    import reach_steer
    ds = "sentinel_steer_model"
    _write_steer_fixtures(tmp_path, ds, "google/gemma-2-2b-it")
    monkeypatch.chdir(tmp_path)
    seen = {}
    _record_load_model(monkeypatch, seen)
    with pytest.raises(_LoadModelReached):
        reach_steer.arm_mean(ds, "cpu", limit=1)
    assert seen["model_name"] == "google/gemma-2-2b-it"
    assert seen["model_name"] != su_default()


def test_arm_per_stmt_loads_the_model_named_in_dct_meta(tmp_path, monkeypatch):
    import reach_steer
    ds = "sentinel_steer_model_stmt"
    _write_steer_fixtures(tmp_path, ds, "google/gemma-2-2b-it")
    monkeypatch.chdir(tmp_path)
    seen = {}
    _record_load_model(monkeypatch, seen)
    with pytest.raises(_LoadModelReached):
        reach_steer.arm_per_stmt(ds, "cpu", limit=1)
    assert seen["model_name"] == "google/gemma-2-2b-it"
    assert seen["model_name"] != su_default()


def su_default():
    import dct_steer_utils
    return dct_steer_utils.MODEL_NAME


def test_reach_jlens_also_loads_the_model_named_in_dct_meta(tmp_path, monkeypatch):
    """M7: the Global Constraint is "every reach script", not "the ones a refusal job
    happens to invoke today"."""
    import reach_jlens
    ds = "sentinel_jlens_model"
    (tmp_path / f"dct_meta_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "model": "google/gemma-2-2b-it", "source_layer": 5,
         "target_layer": 14, "input_scale": 10.0}))
    np.savez(tmp_path / f"reach_acts_{ds}.npz",
             labels=np.array([0, 1, 0, 1]),
             statements=np.array(["a b c d.", "e f g h.", "i j k l.", "m n o p."],
                                 dtype=object))
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_load_model(device="cuda", model_name=None):
        seen["model_name"] = model_name
        raise _LoadModelReached
    monkeypatch.setattr(reach_jlens.su, "load_model", fake_load_model)
    monkeypatch.setattr(sys, "argv",
                        ["reach_jlens.py", "--dataset", ds, "--device", "cpu"])
    with pytest.raises(_LoadModelReached):
        reach_jlens.main()
    assert seen["model_name"] == "google/gemma-2-2b-it"


def test_reach_samepoint_also_loads_the_model_named_in_dct_meta(tmp_path, monkeypatch):
    """M5: the last reach script that was still calling su.load_model with no
    model_name, so "every reach script reads the meta's model" was not literally true.
    No refusal job invokes it today; this holds the invariant for the next one."""
    import reach_samepoint
    ds = "sentinel_samepoint_model"
    _write_steer_fixtures(tmp_path, ds, "google/gemma-2-2b-it")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_load_model(device="cuda", model_name=None):
        seen["model_name"] = model_name
        raise _LoadModelReached
    monkeypatch.setattr(reach_samepoint.su, "load_model", fake_load_model)
    with pytest.raises(_LoadModelReached):
        reach_samepoint.run(ds, "cpu", limit=1)
    assert seen["model_name"] == "google/gemma-2-2b-it"
    assert seen["model_name"] != su_default()
