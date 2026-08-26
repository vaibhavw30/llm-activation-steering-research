"""D1 control flow, with the model stubbed out.

The real sweep is 8,400 GPU generations per dataset, so the only cheap way to know the
harness is right is to replace generation with a function whose output encodes the injected
norm, then assert on that. Covers: the injected magnitude equals rel * median ||h_src||, the
baseline block is genuinely unsteered, --resume refills exactly the missing cells, a finished
run cannot be clobbered, and the dead-hook assert fires.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("transformers")
import dose_response as D  # noqa: E402

DS = "cities"
GRID = [0.01, 0.30, 1.00]
LIMIT = 3
N_DIRS, N_DOSES = 6, 7           # 3 magnitudes x 2 signs, plus zero
EXPECTED_ROWS = LIMIT + N_DIRS * (N_DOSES - 1) * LIMIT

_needs_artifacts = pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, f"mag_dir_{DS}.npz")),
    reason="needs the committed mag_dir / truth_dir / reach_* artifacts")


class _State:
    vec = None


class _FakeSteerer:
    def __init__(self, model, layer):
        self.layer = layer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _State.vec = None

    def set(self, v):
        _State.vec = v


def _injected_norm():
    return 0.0 if _State.vec is None else float(torch.linalg.norm(_State.vec))


def _fake_generate(model, tok, prompt, n):
    """The completion encodes the injected norm, so assertions can recover it."""
    return f"gen[{prompt[:6]}|{_injected_norm():.4f}]"


def _dead_generate(model, tok, prompt, n):
    """A hook that never fires: the text is the same whatever is injected."""
    return "identical"


def _fake_readout(model, tok, statement, yes_ids, no_ids, dev):
    return {"margin": 0.5 - _injected_norm() / 1000.0, "p_yes": 0.7, "p_no": 0.2}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(D.su, "load_model", lambda device, model_name=None: (None, None, "cpu"))
    monkeypatch.setattr(D.su, "generate", _fake_generate)
    monkeypatch.setattr(D.su, "Steerer", _FakeSteerer)
    monkeypatch.setattr(D, "yesno_readout", _fake_readout)
    monkeypatch.setattr(D, "first_token_ids", lambda tok, v: [1, 2])
    _State.vec = None
    return str(tmp_path / "stub")


def _run(prefix, **kw):
    return D.run(DS, "cpu", GRID, limit=LIMIT, prefix=prefix, **kw)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@_needs_artifacts
def test_dose_equals_relative_magnitude_times_median_activation_norm(stubbed):
    _run(stubbed)
    meta = json.load(open(f"{stubbed}_meta_{DS}.json"))
    rows = _read(f"{stubbed}_{DS}.csv")
    for rel in (0.01, 0.30, 1.00):
        got = [float(r["completion"].split("|")[1].rstrip("]")) for r in rows
               if r["direction"] == "sup_grad" and abs(float(r["scale"]) - rel) < 1e-9]
        assert got, f"no rows at rel={rel}"
        assert np.allclose(got, rel * meta["h_src_median"], rtol=1e-4)


@_needs_artifacts
def test_baseline_rows_are_unsteered_and_written_once(stubbed):
    _run(stubbed)
    rows = _read(f"{stubbed}_{DS}.csv")
    base = [r for r in rows if r["direction"] == "baseline"]
    assert len(base) == LIMIT
    assert all(r["completion"].endswith("|0.0000]") for r in base)
    yn = [r for r in _read(f"{stubbed}_yesno_{DS}.csv") if r["direction"] == "baseline"]
    assert all(float(r["d_margin"]) == 0.0 for r in yn)


@_needs_artifacts
def test_every_direction_and_dose_is_present_exactly_once(stubbed):
    _run(stubbed)
    rows = _read(f"{stubbed}_{DS}.csv")
    assert len(rows) == EXPECTED_ROWS
    keys = [(r["direction"], r["scale"], r["prompt"]) for r in rows]
    assert len(set(keys)) == len(keys)
    meta = json.load(open(f"{stubbed}_meta_{DS}.json"))
    assert set(r["direction"] for r in rows) == {"baseline", *meta["directions"]}
    assert len(meta["direction_cosines"]) == 15      # 6 choose 2


@_needs_artifacts
def test_resume_refills_only_the_missing_cells_even_after_a_reformat(stubbed):
    """The scale column is matched as a float, not as the string this script formatted. A
    partial file that has been through pandas writes 1.0 where we wrote +1.0000, and a string
    comparison would regenerate the entire run on top of itself."""
    _run(stubbed)
    for suffix in ("", "_yesno"):
        path = f"{stubbed}{suffix}_{DS}.csv"
        rows = _read(path)
        keep = [r for r in rows
                if not (r["direction"] == "rand_ctrl" and abs(float(r["scale"])) == 1.0)]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in keep:
                r["scale"] = str(float(r["scale"]))       # the reformat
                w.writerow(r)

    _run(stubbed, resume=True)
    rows = _read(f"{stubbed}_{DS}.csv")
    assert len(rows) == EXPECTED_ROWS
    keys = [(r["direction"], round(float(r["scale"]), 6), r["prompt"]) for r in rows]
    assert len(set(keys)) == len(keys)


@_needs_artifacts
def test_refuses_to_clobber_a_finished_run(stubbed):
    _run(stubbed)
    with pytest.raises(SystemExit, match="already exists"):
        _run(stubbed)


@_needs_artifacts
def test_dead_hook_aborts_before_generating_anything(stubbed, monkeypatch):
    monkeypatch.setattr(D.su, "generate", _dead_generate)
    with pytest.raises(SystemExit, match="HOOK IS DEAD"):
        _run(stubbed)


@_needs_artifacts
def test_layer_disagreement_refuses_to_run(stubbed, monkeypatch):
    """A shared magnitude axis is meaningless if the direction families were fit at different
    layers, so the script must stop rather than produce a plottable wrong answer."""
    orig_load = json.load
    monkeypatch.setattr(D.json, "load",
                        lambda f: ({**orig_load(f), "source_layer": 99}
                                   if f.name.startswith("dct_meta") else orig_load(f)))
    with pytest.raises(SystemExit, match="injection layer disagrees"):
        D.load_geometry(DS)


@_needs_artifacts
def test_a_cell_written_to_only_one_file_is_regenerated_in_both(stubbed):
    """A kill between the factual writes and the yes/no writes of one cell leaves that cell in
    one file only. Trusting either file alone loses data: skip on the factual file and the
    cell's yes/no rows are missing forever; do not skip and its factual rows are duplicated."""
    _run(stubbed)
    fac_path, yn_path = f"{stubbed}_{DS}.csv", f"{stubbed}_yesno_{DS}.csv"
    victim = ("mag_u_gold", 0.30)

    yn_rows = _read(yn_path)
    kept = [r for r in yn_rows
            if not (r["direction"] == victim[0] and float(r["scale"]) == victim[1])]
    assert len(kept) < len(yn_rows)
    with open(yn_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(yn_rows[0].keys()))
        w.writeheader()
        w.writerows(kept)

    _run(stubbed, resume=True)
    fac, yn = _read(fac_path), _read(yn_path)
    assert len(fac) == EXPECTED_ROWS
    assert len(yn) == EXPECTED_ROWS
    for rows in (fac, yn):
        key = "prompt" if "prompt" in rows[0] else "statement"
        keys = [(r["direction"], round(float(r["scale"]), 6), r[key]) for r in rows]
        assert len(set(keys)) == len(keys), "half-written cell was duplicated on resume"
    assert sum(1 for r in yn if r["direction"] == victim[0]
               and float(r["scale"]) == victim[1]) == LIMIT
