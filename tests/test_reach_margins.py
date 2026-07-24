import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_margins as rm


def test_gram_schmidt_orthonormal_and_drops_dependent():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    v3 = 2.0 * v1 + 3.0 * v2          # dependent — must be dropped
    B = rm.gram_schmidt([v1, v2, v3])
    assert B.shape == (2, 3)
    assert np.allclose(B @ B.T, np.eye(2), atol=1e-8)


def test_fit_threshold_separable():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(-2, 0.1, 200), rng.normal(2, 0.1, 200)])
    y = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    acc, sign, t02 = rm.fit_threshold(s, y)
    assert acc > 0.95 and sign > 0
    assert -2 < t02 < 2                # FALSE-side threshold sits between the classes
    # P(true)=0.2 threshold must sit BELOW the midpoint (on the FALSE side of 0)
    assert t02 < 0


def test_fit_threshold_flipped_scores_get_negative_slope():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(2, 0.1, 200), rng.normal(-2, 0.1, 200)])
    y = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    _, sign, _ = rm.fit_threshold(s, y)
    assert sign < 0                    # caller flips w and refits


def test_fit_probe_dir_recovers_separating_axis():
    rng = np.random.default_rng(1)
    n, d = 400, 8
    y = (rng.random(n) > 0.5).astype(int)
    X = rng.normal(0, 1, (n, d))
    X[:, 3] += 4.0 * y                 # axis 3 separates the classes
    w = rm.fit_probe_dir(X, y)
    assert abs(w[3]) > 0.8
    assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-6)


def test_build_battery_shapes_and_orientation(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n, d = 300, 16
    y = (rng.random(n) > 0.5).astype(int)
    h = rng.normal(0, 1, (n, d))
    h[:, 0] += 3.0 * y                 # truth axis = e0
    # fake input artifacts in a temp cwd
    monkeypatch.chdir(tmp_path)
    md = np.zeros(d, np.float32); md[0] = -1.0     # deliberately anti-oriented
    np.savez("truth_dir_tgt_toy.npz", mean_diff=md, grad=md, layer=np.array(5))
    import torch
    V = torch.randn(d, 6); U = torch.randn(d, 6)
    torch.save(V, "dct_V_toy.pt"); torch.save(U, "dct_U_toy.pt")
    bat = rm.build_battery(h, y, "toy")
    K = len(bat["names"])
    assert bat["W"].shape == (K, d)
    assert np.allclose(np.linalg.norm(bat["W"], axis=1), 1.0, atol=1e-5)
    # groups present with pinned sizes: 2 truth + <=8 sub + 4 dct_u + 64 rand
    g = list(bat["groups"])
    assert g.count("truth") == 2 and g.count("dct_u") == rm.K_DCT_U
    assert g.count("rand") == rm.N_RAND and 1 <= g.count("truth_sub") <= 8
    # every truth-group direction ends up TRUE-positive-oriented with a valid threshold
    for i, grp in enumerate(bat["groups"]):
        if grp == "truth":
            s = h @ bat["W"][i]
            assert s[y == 1].mean() > s[y == 0].mean()
            assert np.isfinite(bat["thresh02"][i])
    # jtw storage flag: truth, truth_sub, dct_u stored; rand not
    for i, grp in enumerate(bat["groups"]):
        assert bat["store_jtw"][i] == (grp in ("truth", "truth_sub", "dct_u"))


def test_load_statements_stratified_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("got_datasets")
    import csv
    with open("got_datasets/toy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["statement", "label"])
        for i in range(50):
            w.writerow([f"s{i}", i % 2])
    monkeypatch.setitem(rm.DATASET_SAMPLE, "toy", 20)
    stmts, labels, idx = rm.load_statements("toy")
    assert len(stmts) == 20 and labels.sum() == 10
    assert np.all(np.diff(idx) > 0)    # sorted, deterministic
    stmts2, labels2, idx2 = rm.load_statements("toy")
    assert np.array_equal(idx, idx2)   # seed-stable


def test_atomic_savez_writes_completely_and_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "chunk_00000.npz")
    rm.atomic_savez(p, a=np.arange(5), b=np.ones((2, 3)))
    assert os.path.exists(p)
    assert not os.path.exists(p + ".tmp")           # temp cleaned up by rename
    z = np.load(p)
    assert np.array_equal(z["a"], np.arange(5))
    assert z["b"].shape == (2, 3)


def test_atomic_savez_overwrites_stale_tmp(tmp_path):
    p = str(tmp_path / "chunk_00000.npz")
    with open(p + ".tmp", "wb") as fh:
        fh.write(b"garbage")                          # leftover from a prior crash
    rm.atomic_savez(p, a=np.arange(3))
    assert not os.path.exists(p + ".tmp")
    assert np.array_equal(np.load(p)["a"], np.arange(3))
