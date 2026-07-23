import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import export_target_dir as etd


def _toy_setup(tmp_path, tgt=2):
    os.makedirs(tmp_path / "activations", exist_ok=True)
    rng = np.random.default_rng(0)
    n, d, layers = 40, 5, 4
    acts = rng.standard_normal((layers, n, d)).astype(np.float32)
    labels = np.array([i % 2 for i in range(n)])
    acts[tgt, labels == 1] += 3.0          # plant a separation at the target layer
    np.savez(tmp_path / "activations" / "acts_toy.npz",
             activations=acts, labels=labels)
    json.dump({"target_layer": tgt}, open(tmp_path / "dct_meta_toy.json", "w"))


def test_export_writes_target_layer_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_setup(tmp_path)
    out = etd.export("toy")
    assert out == "truth_dir_tgt_toy.npz"
    td = np.load(out)
    assert int(td["layer"]) == 2
    assert np.isclose(np.linalg.norm(td["mean_diff"]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(td["grad"]), 1.0, atol=1e-5)
    # the planted separation is along +ones: mean_diff should point that way
    md = np.asarray(td["mean_diff"], np.float64)
    assert float(md @ (np.ones(5) / np.sqrt(5))) > 0.9


def test_export_never_touches_source_layer_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_setup(tmp_path)
    np.savez("truth_dir_toy.npz", mean_diff=np.ones(5, np.float32),
             grad=np.ones(5, np.float32), layer=np.array(1))
    before = np.load("truth_dir_toy.npz")["layer"].item()
    etd.export("toy")
    assert np.load("truth_dir_toy.npz")["layer"].item() == before
