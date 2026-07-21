import sys, os
import json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm_directions as dd


def test_aligned_flips_sign_toward_ref():
    ref = np.array([1.0, 0.0])
    v = np.array([-2.0, 0.0])
    out = dd.aligned(v, ref)
    assert np.allclose(out, [1.0, 0.0])          # unit + sign-flipped toward ref


def test_aligned_is_unit():
    out = dd.aligned(np.array([3.0, 4.0]), np.array([1.0, 1.0]))
    assert np.allclose(np.linalg.norm(out), 1.0)


def test_assemble_directions_skips_missing_warm_fit(tmp_path, monkeypatch):
    """A warm .pt file missing (e.g. the slurm job's training loop hit a failed fit and
    continued past it) must not abort assembly — it should be skipped with a warning, and
    the raw/cold directions (which don't depend on any warm fit) must still come through."""
    monkeypatch.chdir(tmp_path)

    mean_diff = np.array([1.0, 0.0, 0.0, 0.0])  # tiny toy hidden dim (d=4)
    grad = np.array([0.0, 1.0, 0.0, 0.0])
    np.savez(tmp_path / "truth_dir_toy.npz", mean_diff=mean_diff, grad=grad, layer=5)

    # cold DCT factors: 2 factors, factor 0 potent and roughly mean_diff-aligned.
    V_cold = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    U_cold = torch.tensor([[3.0, 0.1], [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]])
    torch.save(V_cold, tmp_path / "dct_V_toy.pt")
    torch.save(U_cold, tmp_path / "dct_U_toy.pt")
    json.dump({"source_layer": 5}, open(tmp_path / "dct_meta_toy.json", "w"))

    # Only the lam0 warm fit for mean_diff succeeded; lam0p3 is deliberately absent
    # (as would happen after a `!!!! ... FAILED — continuing` line in the slurm log).
    V_warm_lam0 = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    torch.save(V_warm_lam0, tmp_path / "dct_warm_V_toy_mean_diff_lam0.pt")

    dirs = dd.assemble_directions("toy", seeds=["mean_diff"], lams=[0.0, 0.3])

    assert "raw_mean_diff" in dirs
    assert "raw_grad" in dirs
    assert "cold_top" in dirs
    assert "warm_mean_diff_lam0" in dirs
    assert "warm_mean_diff_lam0p3" not in dirs
