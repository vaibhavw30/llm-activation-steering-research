"""E6 harness tests: the sensitivity spectrum reads its inputs correctly.

Nothing here touches a model. What it pins down is that the direction families are
unit-normalized (a gain of ||dz||/eps is only comparable across directions if every
direction has the same length), that per-statement families stay aligned with their
statements, and that the two summary statistics come out of the frame the way the
runbook says they do.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
import token_sens as tsn


@pytest.fixture()
def ds(tmp_path, monkeypatch):
    """A miniature token_geom/token_acts pair on disk, in the layout the real run uses."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    n, d = 6, 16
    np.savez(f"token_geom_{'toy'}.npz",
             delta_opt=rng.normal(size=(n, d)).astype(np.float32),
             a_unit=(lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(
                 rng.normal(size=(n, d))).astype(np.float32),
             delta_cone=np.abs(rng.normal(2, 0.3, n)),
             margin=np.abs(rng.normal(5, 1, n)),
             row_index=np.arange(n))
    np.savez("token_acts_toy.npz",
             z_full=rng.normal(size=(n, d)), h_full=rng.normal(size=(n, d)),
             labels=np.array([1, 0, 1, 0, 1, 0]),
             stems=np.array([f"stem {i}" for i in range(n)], object),
             row_index=np.arange(n))
    return "toy", n, d


def test_every_family_direction_is_a_unit_vector(ds):
    name, n, d = ds
    g = np.load(f"token_geom_{name}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{name}.npz", allow_pickle=True)
    fam = tsn.families(name, g, a, "postnorm", None, 4, np.random.default_rng(1))
    assert len(fam) >= 4 + 3                       # randoms + a_unit + oracle + md_full
    for nm, U in fam:
        norms = (np.linalg.norm(U, axis=1) if np.ndim(U) == 2
                 else [np.linalg.norm(U)])
        assert np.allclose(norms, 1.0, atol=1e-9), nm


def test_per_statement_families_have_one_row_per_statement(ds):
    name, n, d = ds
    g = np.load(f"token_geom_{name}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{name}.npz", allow_pickle=True)
    fam = dict(tsn.families(name, g, a, "postnorm", None, 2,
                            np.random.default_rng(1)))
    for nm in ("a_unit", "oracle"):
        assert fam[nm].shape == (n, d)             # misalignment here silently pairs
                                                   # each statement with another's target


def test_oracle_family_is_the_normalized_least_norm_displacement(ds):
    name, _, _ = ds
    g = np.load(f"token_geom_{name}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{name}.npz", allow_pickle=True)
    fam = dict(tsn.families(name, g, a, "postnorm", None, 1,
                            np.random.default_rng(1)))
    raw = np.asarray(g["delta_opt"], np.float64)
    cos = np.einsum("ij,ij->i", fam["oracle"], raw / np.linalg.norm(
        raw, axis=1, keepdims=True))
    assert np.allclose(cos, 1.0, atol=1e-9)


def test_random_directions_are_reproducible_under_the_same_seed(ds):
    name, _, _ = ds
    g = np.load(f"token_geom_{name}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{name}.npz", allow_pickle=True)
    f1 = dict(tsn.families(name, g, a, "postnorm", None, 3,
                           np.random.default_rng(tsn.SEED)))
    f2 = dict(tsn.families(name, g, a, "postnorm", None, 3,
                           np.random.default_rng(tsn.SEED)))
    assert np.allclose(f1["random0"], f2["random0"])
    assert not np.allclose(f1["random0"], f1["random1"])


def test_md_full_is_read_from_the_site_matched_position(ds):
    """Post-norm coordinates come from z_full, pre-norm ones from h_full. Reading the
    wrong one gives a direction in the wrong space, which is silent and wrong."""
    name, _, _ = ds
    g = np.load(f"token_geom_{name}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{name}.npz", allow_pickle=True)
    post = dict(tsn.families(name, g, a, "postnorm", None, 0,
                             np.random.default_rng(0)))["md_full"]
    pre = dict(tsn.families(name, g, a, "prenorm", None, 0,
                            np.random.default_rng(0)))["md_full"]
    assert not np.allclose(post, pre)
    lab = np.asarray(a["labels"], int)
    Z = np.asarray(a["z_full"], np.float64)
    want = Z[lab == 1].mean(0) - Z[lab == 0].mean(0)
    assert np.allclose(post, want / np.linalg.norm(want))


def test_summarize_reports_both_statistics_without_crashing(capsys):
    """The heterogeneity spread and the scale-dependence ratio are different readings
    of the same table; a run that prints only one of them is not answering claim 4."""
    rows = []
    for stmt in range(3):
        for nm, gain in (("random0", 1.0), ("jtw", 40.0)):
            for f in (0.01, 1.0):
                rows.append(dict(stmt=stmt, direction=nm, frac=f, eps=f,
                                 gain=gain, dM=gain * 0.5, slope=0.5,
                                 cos_first=0.5))
    tsn.summarize(pd.DataFrame(rows), "layer:13")
    out = capsys.readouterr().out
    assert "spread p90/p10" in out
    assert "scale-dependence" in out
    assert "40" in out                              # the anisotropy is visible, not averaged away
