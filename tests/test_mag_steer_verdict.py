import csv
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.steer_verdict import sample_statements, build_directions, TAUS7


def _write_toy_dataset(tmp_path, n=40):
    os.makedirs(tmp_path / "got_datasets", exist_ok=True)
    with open(tmp_path / "got_datasets" / "toy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statement", "label"])
        for i in range(n):
            w.writerow([f"Statement number {i}.", i % 2])


def test_taus7_grid():
    assert TAUS7 == [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]


def test_sample_statements_balanced_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_dataset(tmp_path)
    pairs = sample_statements("toy", n_per_class=3, seed=42)
    labels = [lab for _, lab in pairs]
    assert labels.count(1) == 3 and labels.count(0) == 3
    assert pairs == sample_statements("toy", n_per_class=3, seed=42)   # deterministic


def test_sample_statements_capped_by_test_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_dataset(tmp_path, n=40)          # test split = 8 rows, 4 per class
    pairs = sample_statements("toy", n_per_class=64, seed=42)
    labels = [lab for _, lab in pairs]
    assert labels.count(1) == 4 and labels.count(0) == 4


def test_sample_statements_avoids_train_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_dataset(tmp_path)
    from sklearn.model_selection import train_test_split
    rows = list(csv.DictReader(open("got_datasets/toy.csv")))
    labels = [int(r["label"]) for r in rows]
    idx = np.arange(len(rows))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    test_statements = {rows[i]["statement"] for i in test_idx}
    pairs = sample_statements("toy", n_per_class=64, seed=42)
    assert all(s in test_statements for s, _ in pairs)


def _write_toy_directions(tmp_path, d=6):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "truth_dir_toy.npz",
             mean_diff=rng.standard_normal(d).astype(np.float32),
             grad=rng.standard_normal(d).astype(np.float32), layer=np.array(3))
    np.savez(tmp_path / "mag_dir_toy.npz", layer=np.array(3),
             A_prefix_norm=np.array(2.5),
             resid_pc1_unit=rng.standard_normal(d).astype(np.float32))
    V = torch.randn(d, 4); U = torch.zeros(d, 4); U[:, 2] = 9.0   # factor 2 most potent
    torch.save(V, tmp_path / "dct_V_toy.pt")
    torch.save(U, tmp_path / "dct_U_toy.pt")
    json.dump({"input_scale": 1.0}, open(tmp_path / "dct_meta_toy.json", "w"))
    return V


def test_build_directions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    V = _write_toy_directions(tmp_path)
    dirs, apn = build_directions("toy")
    assert apn == 2.5
    names = [n for n, _, _ in dirs]
    assert names == ["sup_mean_diff", "sup_grad", "mag_resid_pc1", "cold_top", "random_unit"]
    for _, vec, layer in dirs:
        assert layer == 3
        assert np.isclose(np.linalg.norm(vec), 1.0)
    cold = dict((n, v) for n, v, _ in dirs)["cold_top"]
    v2 = V[:, 2].numpy().astype(np.float64)
    assert np.isclose(abs(cold @ (v2 / np.linalg.norm(v2))), 1.0)   # potency-top factor
    dirs2, _ = build_directions("toy")
    rand = dict((n, v) for n, v, _ in dirs)["random_unit"]
    rand2 = dict((n, v) for n, v, _ in dirs2)["random_unit"]
    assert np.allclose(rand, rand2)                                  # seeded, reproducible
