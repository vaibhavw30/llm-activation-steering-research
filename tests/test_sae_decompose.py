import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sae_decompose import omp, explained, jaccard


def _dict(F=40, d=12, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((F, d))
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def test_omp_recovers_a_planted_two_atom_vector():
    D = _dict()
    v = 2.0 * D[3] - 1.5 * D[17]
    sup, coefs, resid = omp(v, D, k=2)
    assert set(sup.tolist()) == {3, 17}
    assert resid < 1e-8
    by_atom = {int(s): c for s, c in zip(sup, coefs)}
    assert abs(by_atom[3] - 2.0) < 1e-6 and abs(by_atom[17] + 1.5) < 1e-6


def test_omp_residual_is_monotone_nonincreasing_in_k():
    D = _dict(seed=1)
    v = np.random.default_rng(2).standard_normal(12)
    r = [omp(v, D, k=k)[2] for k in (1, 3, 6, 10)]
    assert all(r[i + 1] <= r[i] + 1e-12 for i in range(len(r) - 1))


def test_omp_never_repeats_an_atom():
    sup, _, _ = omp(np.random.default_rng(4).standard_normal(12), _dict(seed=3), k=8)
    assert len(set(sup.tolist())) == 8


def test_omp_rejects_k_larger_than_the_dictionary():
    with pytest.raises(ValueError):
        omp(np.ones(12), _dict(F=5), k=6)


def test_explained_is_one_for_an_exact_fit():
    D = _dict()
    v = D[0] * 3.0
    sup, coefs, _ = omp(v, D, k=1)
    assert abs(explained(v, D, sup, coefs) - 1.0) < 1e-9


def test_jaccard_edges():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([], []) == 0.0
    assert abs(jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-12


def _reach_fixture(tmp_path, d=6, n=4, with_stem=False):
    """Minimal reach artifacts: two target-layer readouts, one stored J^T w row set."""
    import json
    np.random.seed(0)
    json.dump({"source_layer": 1, "target_layer": 10, "input_scale": 1.0,
               "model": "m"}, open(tmp_path / "dct_meta_ds.json", "w"))
    np.savez(tmp_path / "reach_acts_ds.npz",
             labels=np.array([0, 1, 1, 1][:n]),
             statements=np.array(["a", "b", "c", "d"][:n], dtype=object))
    np.savez(tmp_path / "reach_dirs_ds.npz",
             W=np.eye(2, d).astype(np.float32),
             names=np.array(["mean_diff_tgt", "probe_grad_tgt"], dtype=object))
    jtw = np.tile(np.eye(1, d, 2).astype(np.float16), (n, 1, 1))   # (n, 1, d)
    np.savez(tmp_path / "reach_margins_ds.npz", jtw=jtw,
             store_names=np.array(["mean_diff_tgt"], dtype=object))
    if with_stem:
        np.savez(tmp_path / "reach_stemjac_ds.npz",
                 jtw_stem=np.tile(np.eye(1, d, 3).astype(np.float16), (2, 1)),
                 m_stem=np.ones(2, np.float32),
                 stmt_index=np.array([1, 3]))


def test_collect_vectors_returns_named_vectors_with_their_layer_space(tmp_path,
                                                                      monkeypatch):
    import sae_decompose as sd
    _reach_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    got = sd.collect_vectors("ds")
    assert got["w_mean_diff_tgt"][1] == "tgt"
    assert got["jtw_mean"][1] == "src"
    assert got["w_mean_diff_tgt"][0].shape == (6,)
    # jtw rows are unit; the mean of identical unit rows is that row
    assert abs(np.linalg.norm(got["jtw_mean"][0]) - 1.0) < 1e-6
    # no stemjac and no reach_svd dir -> those entries are absent, not empty
    assert "jtw_stem_mean" not in got
    assert not any(k.startswith("V64_") for k in got)


def test_collect_vectors_aligns_stem_rows_by_stmt_index(tmp_path, monkeypatch):
    import sae_decompose as sd
    _reach_fixture(tmp_path, with_stem=True)
    monkeypatch.chdir(tmp_path)
    got = sd.collect_vectors("ds")
    assert sd.D2_PAIR == "jtw_full_matched_mean|jtw_stem_mean"
    assert got["jtw_full_matched_mean"][0].shape == got["jtw_stem_mean"][0].shape
    assert "common_v1" in got
    # the fixture's full rows are e_2 and its stem rows are e_3, so the matched mean
    # stays on e_2 — proving it read the full matrix, not the stem one
    assert abs(got["jtw_full_matched_mean"][0][2] - 1.0) < 1e-6


def test_run_refuses_a_meta_whose_model_is_not_gemma_2_2b(tmp_path, monkeypatch):
    # Track B is gemma-2-2b-only: sae_load.REPO is hardcoded to
    # google/gemma-scope-2b-pt-res, so pointing this at a `refusal` run done on
    # google/gemma-2-2b-it would silently decompose -it directions in base-model SAE
    # atoms. Must fail before any SAE download.
    import json
    import sae_decompose
    monkeypatch.chdir(tmp_path)
    ds = "sentinel_sae_model"
    (tmp_path / f"dct_meta_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "model": "google/gemma-2-2b-it", "source_layer": 11,
         "target_layer": 20, "input_scale": 10.0}))
    with pytest.raises(SystemExit, match="gemma-scope-2b-pt-res"):
        sae_decompose.run(ds)
