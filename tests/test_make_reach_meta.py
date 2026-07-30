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


def test_layer_too_close_to_max_layer_is_ineligible_even_within_hop():
    # dct.SlicedModel.forward does self.model.model.layers = self.L[start:end+2]
    # (src/dct.py:238), so end_layer+2 must stay within len(self.L) == max_layer.
    # With hop=9, max_layer=26: layers 16 and 17 satisfy L+hop<=26 but not
    # L+hop+2<=26, so neither is eligible and no fallback layer exists here.
    rows = [{"layer": 16, "linear_acc": 0.99}, {"layer": 17, "linear_acc": 0.995}]
    with pytest.raises(SystemExit):
        pick_source_layer(rows, max_layer=26)


def test_run_refuses_to_overwrite_an_existing_meta_file_without_force(tmp_path, monkeypatch):
    import numpy as np
    import make_reach_meta as m
    monkeypatch.chdir(tmp_path)
    ds = "sentinel_ds_a4_fix"
    # A valid acts file, so that absent the overwrite guard, run() would succeed
    # end-to-end and actually clobber the existing meta file (a weak test would
    # only prove *some* SystemExit fired, e.g. from a missing acts file).
    rng = np.random.default_rng(1)
    acts = rng.standard_normal((20, 40, 6))
    labels = np.array([0, 1] * 20)
    np.savez(tmp_path / f"acts_{ds}.npz", activations=acts, labels=labels)
    meta_path = tmp_path / f"dct_meta_{ds}.json"
    original = {"sentinel": True, "input_scale": 47.716029511013176}
    meta_path.write_text(json.dumps(original))
    with pytest.raises(SystemExit):
        m.run(ds, "some-model")
    assert json.loads(meta_path.read_text()) == original


def test_calibrate_scale_refuses_to_recalibrate_an_already_calibrated_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from calibrate_scale import check_not_calibrated
    calibrated = {"input_scale": 47.716029511013176}
    # already-calibrated + no --force -> refuse (this is what protects
    # dct_meta_cities.json's real input_scale from a cluster job overwriting it)
    with pytest.raises(SystemExit):
        check_not_calibrated(calibrated, "dct_meta_sentinel.json", force=False)
    # --force explicitly permits recalibrating an already-calibrated meta
    check_not_calibrated(calibrated, "dct_meta_sentinel.json", force=True)
    # input_scale: null (make_reach_meta.py's normal output) is the ordinary
    # not-yet-calibrated refusal path and must proceed unchanged, force or not
    check_not_calibrated({"input_scale": None}, "dct_meta_sentinel.json", force=False)


def _write_acts(tmp_path, ds, model=None):
    import numpy as np
    rng = np.random.default_rng(3)
    kw = {"activations": rng.standard_normal((20, 40, 6)),
          "labels": np.array([0, 1] * 20)}
    if model is not None:
        kw["model"] = model
    np.savez(tmp_path / f"acts_{ds}.npz", **kw)


def test_run_refuses_when_the_acts_model_disagrees_with_the_meta_model(tmp_path,
                                                                      monkeypatch):
    # extract.py:152 records the checkpoint the activations came from; `extract --model`
    # and `make_reach_meta --model` are two independently operator-typed literals. A
    # manual re-run of one and not the other writes an AUTHORITATIVE meta describing a
    # different model than the activations the layer choice was derived from, and every
    # downstream stage trusts the meta.
    import make_reach_meta as m
    monkeypatch.chdir(tmp_path)
    ds = "sentinel_ds_model_xcheck"
    _write_acts(tmp_path, ds, model="google/gemma-2-2b")
    with pytest.raises(SystemExit) as e:
        m.run(ds, "google/gemma-2-2b-it")
    msg = str(e.value)
    # both values quoted (repr), so "google/gemma-2-2b" being a prefix of
    # "google/gemma-2-2b-it" cannot make a weaker assertion pass vacuously
    assert "'google/gemma-2-2b'" in msg and "'google/gemma-2-2b-it'" in msg
    assert "acts_" in msg and "model mismatch" in msg
    assert not os.path.exists(f"dct_meta_{ds}.json")   # nothing written


def test_run_accepts_a_matching_acts_model(tmp_path, monkeypatch):
    import make_reach_meta as m
    monkeypatch.chdir(tmp_path)
    ds = "sentinel_ds_model_match"
    _write_acts(tmp_path, ds, model="google/gemma-2-2b-it")
    m.run(ds, "google/gemma-2-2b-it")
    assert json.loads(open(f"dct_meta_{ds}.json").read())["model"] == \
        "google/gemma-2-2b-it"


def test_run_only_warns_when_the_acts_file_predates_the_model_key(tmp_path,
                                                                 monkeypatch, capsys):
    # Older acts_<ds>.npz files have no "model" member; that is not a mismatch and
    # must not block the refusal prep.
    import make_reach_meta as m
    monkeypatch.chdir(tmp_path)
    ds = "sentinel_ds_model_absent"
    _write_acts(tmp_path, ds, model=None)
    m.run(ds, "google/gemma-2-2b-it")
    assert "no 'model'" in capsys.readouterr().out
    assert os.path.exists(f"dct_meta_{ds}.json")
