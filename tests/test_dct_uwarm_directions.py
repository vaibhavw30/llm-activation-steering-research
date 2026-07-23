import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dct_uwarm_directions as dud


def _toy_setup(tmp_path, d=4):
    np.savez(tmp_path / "truth_dir_toy.npz",
             mean_diff=np.array([1.0, 0, 0, 0], np.float32),
             grad=np.array([0, 1.0, 0, 0], np.float32), layer=np.array(1))
    np.savez(tmp_path / "truth_dir_tgt_toy.npz",
             mean_diff=np.array([0, 0, 1.0, 0], np.float32),
             grad=np.array([0, 0, 0, 1.0], np.float32), layer=np.array(3))
    # cold artifacts for cold_top
    V = torch.eye(d)[:, :2]; U = torch.zeros(d, 2); U[:, 1] = 5.0
    torch.save(V, tmp_path / "dct_V_toy.pt"); torch.save(U, tmp_path / "dct_U_toy.pt")
    json.dump({}, open(tmp_path / "dct_meta_toy.json", "w"))
    # one uwarm fit at lam=1: V0 along e1, U0 pointing AWAY from md_tgt (tests sign flip)
    Vw = torch.zeros(d, 2); Vw[1, 0] = 1.0; Vw[0, 1] = 1.0
    Uw = torch.zeros(d, 2); Uw[2, 0] = -2.0; Uw[3, 1] = 1.0
    torch.save(Vw, tmp_path / "dct_uwarm_V_toy_mean_diff_lam1.pt")
    torch.save(Uw, tmp_path / "dct_uwarm_U_toy_mean_diff_lam1.pt")


def test_assemble_flips_v0_to_true_pointing_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_setup(tmp_path)
    dirs, rows = dud.assemble("toy", lams=(1.0,))
    assert set(dirs) == {"raw_mean_diff", "cold_top", "uwarm_mean_diff_lam1"}
    # U0 . md_tgt = -1 -> V0 must be sign-flipped: stored dir = -e1
    assert np.allclose(dirs["uwarm_mean_diff_lam1"], [0, -1.0, 0, 0])
    geo = {r[0]: r for r in rows}
    assert np.isclose(geo["uwarm_mean_diff_lam1"][1], -1.0)   # raw signed cos_U0_mdtgt
    assert np.isclose(geo["uwarm_mean_diff_lam1"][2], 0.0)    # cos_V0_mdsrc


def test_assemble_skips_missing_lams(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _toy_setup(tmp_path)
    dirs, _ = dud.assemble("toy", lams=(1.0, 3.0))            # lam3 file absent
    assert "uwarm_mean_diff_lam3" not in dirs
    assert "uwarm_mean_diff_lam1" in dirs
    assert "WARN" in capsys.readouterr().out


def test_main_writes_npz_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_setup(tmp_path)
    dud.write_outputs("toy", lams=(1.0,))
    assert os.path.exists("dct_uwarm_dirs_toy.npz")
    assert os.path.exists("dct_uwarm_geometry_toy.csv")
    z = np.load("dct_uwarm_dirs_toy.npz")
    for k in z.files:
        assert np.isclose(np.linalg.norm(z[k]), 1.0, atol=1e-5)
