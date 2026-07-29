import sys, os
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_hop as rh


class _ToySlice(nn.Module):
    """Stand-in for dct.SlicedModel: nonlinear per-position map + causal cross-position
    mixing, (B,T,d) -> (B,T,d). Frozen weights, like the real model."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64), requires_grad=False)
        self.W2 = nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64), requires_grad=False)

    def forward(self, h):
        z = torch.tanh(h @ self.W1)
        return z.cumsum(dim=1) @ self.W2      # position t sees positions <= t


def _setup(B=2, T=4, d=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    sl = _ToySlice(d=d)
    h = torch.randn(B, T, d, generator=g, dtype=torch.float64)
    attn = torch.ones(B, T, dtype=torch.long)
    attn[1, -1] = 0                            # row 1 has one right-pad
    return sl, h, attn, B, T, d


def test_last_nonpad_index():
    _, _, attn, _, T, _ = _setup()
    last = rh.last_nonpad_index(attn)
    assert last.tolist() == [T - 1, T - 2]


def test_f0_equals_clean_forward():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    out = f(torch.zeros(B, d))
    ref = sl(h)
    last = rh.last_nonpad_index(attn)
    for b in range(B):
        assert torch.allclose(out[b], ref[b, last[b]], atol=1e-6)


def test_broadcast_convention():
    """f(delta) must equal slicing h with delta added at EVERY position of that row."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    delta = torch.randn(B, d)
    out = f(delta)
    ref = sl(h + delta[:, None, :])
    last = rh.last_nonpad_index(attn)
    for b in range(B):
        assert torch.allclose(out[b], ref[b, last[b]], atol=1e-6)


def _fd_jacobian(f, B, d, h=1e-4):
    """Full J_b per row by central finite differences. Returns (B, d, d)."""
    J = torch.zeros(B, d, d, dtype=torch.float64)
    for j in range(d):
        e = torch.zeros(B, d, dtype=torch.float64); e[:, j] = h
        J[:, :, j] = (f(e) - f(-e)) / (2 * h)
    return J


def test_vjp_matches_finite_differences():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    J = _fd_jacobian(f, B, d)
    W = torch.randn(3, d, dtype=torch.float64)
    _, G = rh.vjp_rows(f, torch.zeros(B, d, dtype=torch.float64), W)     # (K,B,d)
    # Toy path runs in float64, so this finite-difference check holds tightly
    # (no float32 roundoff floor to clear).
    for k in range(3):
        for b in range(B):
            ref = J[b].T @ W[k]
            assert torch.allclose(G[k, b], ref, atol=1e-5), (k, b)


def test_transposition_identity():
    """w·(Jv) via jvp == (Jᵀw)·v via vjp — guards Phases 1 and 2 simultaneously."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    torch.manual_seed(7)
    w = torch.randn(d, dtype=torch.float64); v = torch.randn(d, dtype=torch.float64)
    _, G = rh.vjp_rows(f, torch.zeros(B, d, dtype=torch.float64), w[None, :])
    _, JT = rh.jvp_cols(f, torch.zeros(B, d, dtype=torch.float64), v.expand(B, d).contiguous())
    for b in range(B):
        lhs = float(JT[b] @ w)
        rhs = float(G[0, b] @ v)
        assert abs(lhs - rhs) < 1e-5


def test_vjp_rows_are_per_row_independent():
    """Row b's gradient must not mix in row b'. Compare batch-of-2 vs single-row runs."""
    sl, h, attn, B, T, d = _setup()
    w = torch.randn(1, d)
    f2 = rh.make_hop(sl, h, attn)
    _, G2 = rh.vjp_rows(f2, torch.zeros(B, d), w)
    for b in range(B):
        f1 = rh.make_hop(sl, h[b:b + 1], attn[b:b + 1])
        _, G1 = rh.vjp_rows(f1, torch.zeros(1, d), w)
        assert torch.allclose(G2[0, b], G1[0, 0], atol=1e-5)


class _NormalizerlessSlice:
    """Mimics dct.SlicedModel under transformers >= 5: it pre-divides gemma-2
    inputs by sqrt(d) but the HF model no longer re-multiplies inputs_embeds,
    so the net map is true_map(h / sqrt(d))."""
    def __init__(self, true_map, d):
        self.true_map, self.d = true_map, d

    def __call__(self, h):
        return self.true_map(h / self.d ** 0.5)


def test_calibrate_slice_identity_passthrough():
    """A faithful slice is returned unwrapped with factor 1."""
    sl, h, attn, B, T, d = _setup()
    last = rh.last_nonpad_index(attn)
    target = sl(h)[torch.arange(B), last]
    cand, cos, factor = rh.calibrate_slice(sl, h, target, last)
    assert cand is sl and factor == 1.0 and cos > 0.999


def test_calibrate_slice_compensates_missing_normalizer():
    """The sqrt(d) compensation restores fidelity and the wrapped map matches
    the true map everywhere (so its Jacobian is the true Jacobian)."""
    sl, h, attn, B, T, d = _setup()
    last = rh.last_nonpad_index(attn)
    target = sl(h)[torch.arange(B), last]
    broken = _NormalizerlessSlice(sl, d)
    cand, cos, factor = rh.calibrate_slice(broken, h, target, last)
    assert isinstance(cand, rh.CompensatedSlice) and factor == d ** 0.5
    assert cos > 0.999
    assert torch.allclose(cand(h), sl(h), atol=1e-10)


def test_calibrate_slice_raises_when_uncompensatable():
    import pytest
    sl, h, attn, B, T, d = _setup()
    last = rh.last_nonpad_index(attn)
    target = sl(h)[torch.arange(B), last]
    with pytest.raises(SystemExit, match="unfaithful"):
        rh.calibrate_slice(lambda x: torch.zeros_like(x) + 1.0, h, target, last)


def test_optional_artifacts_reports_absence(tmp_path, monkeypatch):
    import json
    import numpy as np
    import reach_hop
    monkeypatch.chdir(tmp_path)
    (tmp_path / "got_datasets").mkdir()
    (tmp_path / "got_datasets" / "ds.csv").write_text("statement,label\na,1\n")
    json.dump({"model": "m", "source_layer": 1, "target_layer": 10,
               "input_scale": 2.0}, open("dct_meta_ds.json", "w"))
    for nm, layer in (("truth_dir_ds.npz", 1), ("truth_dir_tgt_ds.npz", 10)):
        np.savez(nm, mean_diff=np.ones(4, np.float32), grad=np.ones(4, np.float32),
                 layer=np.array(layer))
    assert reach_hop.optional_artifacts("ds") == {"dct": False, "mag": False}
    reach_hop.validate_inputs("ds")          # must NOT raise: both are optional
    assert set(reach_hop.load_landmarks("ds")) == {"md_src"}


def test_load_meta_returns_model_from_json(tmp_path, monkeypatch):
    import json
    import reach_hop
    monkeypatch.chdir(tmp_path)
    json.dump({"source_layer": 3, "target_layer": 12, "input_scale": 7.5,
               "model": "google/gemma-2-2b-it"}, open("dct_meta_ds.json", "w"))
    assert reach_hop.load_meta("ds") == (3, 12, 7.5, "google/gemma-2-2b-it")


def test_load_meta_defaults_model_when_absent(tmp_path, monkeypatch):
    import json
    import reach_hop
    monkeypatch.chdir(tmp_path)
    json.dump({"source_layer": 3, "target_layer": 12, "input_scale": 7.5},
              open("dct_meta_ds.json", "w"))
    assert reach_hop.load_meta("ds")[3] == "google/gemma-2-2b"


def test_load_meta_rejects_an_uncalibrated_meta(tmp_path, monkeypatch):
    import json
    import pytest
    import reach_hop
    monkeypatch.chdir(tmp_path)
    json.dump({"source_layer": 3, "target_layer": 12, "input_scale": None},
              open("dct_meta_ds.json", "w"))
    with pytest.raises(SystemExit, match="calibrate_scale"):
        reach_hop.load_meta("ds")


def test_validate_inputs_still_fails_on_missing_required(tmp_path, monkeypatch):
    import json
    import pytest
    import reach_hop
    monkeypatch.chdir(tmp_path)
    json.dump({"source_layer": 1, "target_layer": 10, "input_scale": 1.0},
              open("dct_meta_ds.json", "w"))
    with pytest.raises(SystemExit) as e:
        reach_hop.validate_inputs("ds")
    assert "truth_dir_ds.npz" in str(e.value)
