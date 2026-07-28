# Backward-Reachability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five-phase backward-reachability audit (spec: `docs/superpowers/specs/2026-07-24-reachability-audit-design.md`): per-statement controllability margins ‖Jᵀw‖ and reachability curves for the FALSE-probe halfspace, full-Jacobian SVD mechanism subsample, Jᵀw steering with behavioral judging, per-layer J-lens margins, and linearization-error trust regions — for gemma-2-2b hops cities 11→20 and common_claim 13→22.

**Architecture:** One shared autograd library (`reach_hop.py`) defines the hop map F(Δ) = target-layer last-token activation under a Δ broadcast to every position at the source layer, built on the repo's proven `dct.SlicedModel` (which `dct.py` already drives with `torch.func.vjp`/`jvp` on the real model). Everything else is thin: staged extraction with checkpoints (`reach_margins.py`), pure-numpy analysis (`reach_analyze.py`), and per-phase scripts that reuse the existing steering/judge harness (`dct_steer_utils.Steerer`, `judge_results.py --mode steer`).

**Tech Stack:** PyTorch (`torch.func.vjp/jvp`), transformers (eager attention, 4.51.3 pin on cluster), numpy, scikit-learn (probe fits), matplotlib (Agg), pytest with stand-in `nn.Module` stubs (no HF model in tests), SLURM on DeltaAI GH200.

## Global Constraints

- Work on branch `feat/reach-audit` (created in Task 1 from current HEAD of `feat/mag-e4-steering`).
- **Never `git add -A` or `git add .`** — add only the files named in the task's commit step. NEVER stage these uncommitted files from other tracks: `src/spectrum_utils.py`, `src/viz_spectrum.py`, `tests/test_spectrum_utils.py`, `tests/test_viz_spectrum.py`, `docs/RESULTS_SINCE_LAST_MEETING_PART2.md`, `docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md`, `src/viz_mag_linearity.py`, `plot_mag_linearity_v2.png`.
- Run tests as: `PYTHONPATH=src .venv/bin/python -m pytest tests/<file> -q` (tests also do their own `sys.path.insert`, matching existing tests).
- **STRICT TASK GATE:** a task is complete only when (a) every step's checkbox is done, (b) its named test command passes, (c) the FULL suite `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` is green, and (d) the task reviewer approves. Do not start the next task before the gate passes.
- All new scripts live in `src/`, run with `PYTHONPATH=src`, read/write artifacts in the repo root (repo convention). No GPU/model access in unit tests — model-touching paths are exercised only via `--limit` smoke runs on the cluster or local `--device cpu|mps`.
- Layer convention (verified): `hidden_states[ℓ]` = residual stream entering block ℓ; `dct.SlicedModel(model, start_layer=src, end_layer=tgt, layers_name="model.layers")` maps `hidden_states[src] → hidden_states[tgt]`; `Steerer` adds its vector at exactly `hidden_states[src]`, every position.
- Sign convention: every truth readout w is oriented so larger w·h ⇒ TRUE; the FALSE target set is {h : w·h ≤ t02(w)} with t02 at calibrated P(true)=0.2 (`LOGIT_02 = log(0.2/0.8) ≈ −1.3863`). +τ→TRUE / −τ→FALSE for steering.
- Fixed numbers (from spec, pinned): common_claim stratified sample 2000 (seed 42); rand_64 seed 123; bootstrap probes seeds 0–5 at 50% subsample; dct_u_top k=4; ε grid `linspace(0, 1.5·input_scale, 61)`; vjp batch 16; checkpoint chunk 100; SVD: 32 statements (16T/16F, seed 42), tangent chunk 64, save top-64 U/V as float16; steer: 200 label-1 statements (seed 42), strengths {0.5,1,1.5,2}×ε*, both signs, cap 1.5·input_scale; jlens sample 256 (seed 42); linerr: 64 statements (seed 42), ε ∈ `geomspace(0.05, 1.5, 8)·input_scale`, validity radius at 20% relative error; 1-D threshold valid only if 1-D probe accuracy ≥ 0.6.

---

### Task 1: `reach_hop.py` — the shared hop/autograd library

**Files:**
- Create: `src/reach_hop.py`
- Test: `tests/test_reach_hop.py`

**Interfaces:**
- Consumes: `dct.SlicedModel`, `dct_steer_utils.load_model`, `funnel_utils.unit/load_dct/top_k_by_potency` (existing).
- Produces (used by Tasks 2, 5, 6, 8):
  - `last_nonpad_index(attn_mask) -> LongTensor (B,)`
  - `make_hop(sliced, h_src_seq, attn_mask) -> f` where `f(delta_rows: (B,d)) -> (B,d)`
  - `vjp_rows(f, delta0, W) -> (F0 (B,d), G (K,B,d))` with `G[k,b] = J_bᵀ W[k]`
  - `jvp_cols(f, delta0, T_rows) -> (F0 (B,d), JT (B,d))` with `JT[b] = J_b T_rows[b]`
  - `load_meta(ds) -> (src, tgt, input_scale)`
  - `load_model_and_slice(ds, device) -> (tok, model, sliced, meta_dict)` where `meta_dict = {"src","tgt","input_scale","device"}`
  - `forward_source_batch(model, tok, statements, src, tgt, device, max_length=64) -> dict` with keys `h_src_seq (B,T,d)`, `attn (B,T)`, `h_src`, `h_tgt`, `h_final` (each `(B,d)`, detached)
  - `load_landmarks(ds) -> {"md_src","v_q","dct_v"}` (unit float64 numpy)
  - `validate_inputs(ds)` — fail-fast startup validation (spec §6)

- [ ] **Step 0: Create the branch**

```bash
git checkout -b feat/reach-audit
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reach_hop.py`:

```python
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
        self.W1 = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)
        self.W2 = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        z = torch.tanh(h @ self.W1)
        return z.cumsum(dim=1) @ self.W2      # position t sees positions <= t


def _setup(B=2, T=4, d=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    sl = _ToySlice(d=d)
    h = torch.randn(B, T, d, generator=g)
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
    J = torch.zeros(B, d, d)
    for j in range(d):
        e = torch.zeros(B, d); e[:, j] = h
        J[:, :, j] = (f(e) - f(-e)) / (2 * h)
    return J


def test_vjp_matches_finite_differences():
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    J = _fd_jacobian(f, B, d)
    W = torch.randn(3, d)
    _, G = rh.vjp_rows(f, torch.zeros(B, d), W)     # (K,B,d)
    for k in range(3):
        for b in range(B):
            ref = J[b].T @ W[k]
            assert torch.allclose(G[k, b], ref, atol=1e-3), (k, b)


def test_transposition_identity():
    """w·(Jv) via jvp == (Jᵀw)·v via vjp — guards Phases 1 and 2 simultaneously."""
    sl, h, attn, B, T, d = _setup()
    f = rh.make_hop(sl, h, attn)
    torch.manual_seed(7)
    w = torch.randn(d); v = torch.randn(d)
    _, G = rh.vjp_rows(f, torch.zeros(B, d), w[None, :])
    _, JT = rh.jvp_cols(f, torch.zeros(B, d), v.expand(B, d).contiguous())
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_hop.py -q`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'reach_hop'`.

- [ ] **Step 3: Write the implementation**

Create `src/reach_hop.py`:

```python
"""reach_hop.py — shared machinery for the backward-reachability audit (Phases 1-5).

Hop map, per statement i:  F_i : Delta in R^d  ->  h_tgt(last non-pad token) in R^d,
where Delta is added to the residual stream entering block `source_layer` at EVERY
token position — the exact convention of dct_steer_utils.Steerer (forward_pre_hook on
model.model.layers[src]) — and blocks src..tgt run via dct.SlicedModel, so that
F_i(0) == hidden_states[tgt] (run_dct_minimal's SlicedModel sanity check).

J_i = dF_i/dDelta at 0 is a d x d per-statement matrix, never materialized here:
    vjp_rows(f, delta0, W)   -> per-row J_i^T w      (one backward per direction w)
    jvp_cols(f, delta0, T)   -> per-row J_i v        (one forward-mode call per chunk)
dct.py already drives torch.func.vjp/jvp through SlicedModel on the real model
(dct.py lines ~419/~729), so these transforms are proven on this architecture.
"""
import json
import os

import numpy as np
import torch
from torch.func import vjp, jvp

import dct
import dct_steer_utils as su
import funnel_utils as fu
from funnel_utils import unit


def last_nonpad_index(attn_mask):
    """(B, T) 0/1 right-padded mask -> (B,) index of the last real token."""
    am = attn_mask
    return am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)


def make_hop(sliced, h_src_seq, attn_mask):
    """Return f(delta_rows (B,d)) -> (B,d): target-layer activation at the last
    non-pad token, with row b's delta broadcast over ALL of row b's positions.
    h_src_seq (B,T,d) is treated as a constant (detached)."""
    last = last_nonpad_index(attn_mask)
    idx = torch.arange(h_src_seq.shape[0], device=h_src_seq.device)
    h0 = h_src_seq.detach()

    def f(delta_rows):
        out = sliced(h0 + delta_rows[:, None, :])     # (B,T,d) = hidden_states[tgt]
        return out[idx, last, :]
    return f


def vjp_rows(f, delta0, W):
    """W (K,d) target-space directions. Returns (F0 (B,d), G (K,B,d)) with
    G[k,b] = J_b^T W[k]. One saved forward, K backward passes. Rows are independent
    because output row b depends only on delta row b."""
    B = delta0.shape[0]
    F0, pull = vjp(f, delta0)
    grads = []
    for w in W:
        cot = w.to(F0.dtype).to(F0.device).expand(B, -1).contiguous()
        grads.append(pull(cot)[0])
    return F0, torch.stack(grads)


def jvp_cols(f, delta0, T_rows):
    """T_rows (B,d) per-row tangents. Returns (F0, JT (B,d)) with JT[b] = J_b T_rows[b]."""
    return jvp(f, (delta0,), (T_rows,))


def load_meta(ds):
    m = json.load(open(f"dct_meta_{ds}.json"))
    return int(m["source_layer"]), int(m["target_layer"]), float(m["input_scale"])


def load_model_and_slice(ds, device):
    src, tgt, scale = load_meta(ds)
    tok, model, dev = su.load_model(device)
    sliced = dct.SlicedModel(model, start_layer=src, end_layer=tgt,
                             layers_name="model.layers")
    return tok, model, sliced, {"src": src, "tgt": tgt, "input_scale": scale,
                                "device": dev}


def forward_source_batch(model, tok, statements, src, tgt, device, max_length=64):
    """One no-grad full forward. Returns h_src_seq (B,T,d) + attn (B,T) for the hop,
    and last-token rows h_src / h_tgt / h_final (each (B,d), detached)."""
    tok.padding_side = "right"
    enc = tok(list(statements), return_tensors="pt", padding=True,
              truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    am = enc["attention_mask"]
    last = last_nonpad_index(am)
    idx = torch.arange(am.shape[0], device=am.device)
    return {"h_src_seq": hs[src].detach(), "attn": am,
            "h_src": hs[src][idx, last].detach(),
            "h_tgt": hs[tgt][idx, last].detach(),
            "h_final": hs[-1][idx, last].detach()}


def load_landmarks(ds):
    """Source-layer landmark unit vectors for cosine bookkeeping (spec §3)."""
    td = np.load(f"truth_dir_{ds}.npz")
    md_src = unit(np.asarray(td["mean_diff"], np.float64))
    mg = np.load(f"mag_dir_{ds}.npz")
    v_q = unit(np.asarray(mg["v_Q_unit"], np.float64))
    V, U, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(V, U, 1)[0]
    dct_v = unit(V[:, top].astype(np.float64))
    return {"md_src": md_src, "v_q": v_q, "dct_v": dct_v}


def validate_inputs(ds):
    """Fail-fast startup validation (spec §6): die in seconds on a config error,
    not after an hour of forwards."""
    problems = []
    checks = [
        (f"dct_meta_{ds}.json", None),
        (f"truth_dir_{ds}.npz", ("mean_diff", "grad", "layer")),
        (f"truth_dir_tgt_{ds}.npz", ("mean_diff", "grad", "layer")),
        (f"mag_dir_{ds}.npz", ("v_Q_unit",)),
        (f"dct_V_{ds}.pt", None),
        (f"dct_U_{ds}.pt", None),
        (os.path.join("got_datasets", f"{ds}.csv"), None),
    ]
    for path, keys in checks:
        if not os.path.exists(path):
            problems.append(f"missing {path}")
        elif keys is not None:
            z = np.load(path)
            missing = [k for k in keys if k not in z.files]
            if missing:
                problems.append(f"{path} lacks keys {missing}")
    if problems:
        raise SystemExit("[reach] input validation FAILED:\n  " + "\n  ".join(problems))
    src, tgt, _ = load_meta(ds)
    src_l = int(np.load(f"truth_dir_{ds}.npz")["layer"])
    tgt_l = int(np.load(f"truth_dir_tgt_{ds}.npz")["layer"])
    if src_l != src or tgt_l != tgt:
        raise SystemExit(f"[reach] layer mismatch: truth_dir layer {src_l} vs dct src {src}; "
                         f"truth_dir_tgt layer {tgt_l} vs dct tgt {tgt}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_hop.py -q`
Expected: 6 passed.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: everything green (pre-existing suite unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/reach_hop.py tests/test_reach_hop.py
git commit -m "feat(reach): hop-map library — broadcast-injection F(Delta), vjp/jvp rows"
```

---

### Task 2: `reach_margins.py` — Phase 1 extraction (stages acts / dirs / vjp)

**Files:**
- Create: `src/reach_margins.py`
- Test: `tests/test_reach_margins.py`

**Interfaces:**
- Consumes (Task 1): `reach_hop.validate_inputs`, `load_model_and_slice`, `forward_source_batch`, `make_hop`, `vjp_rows`, `load_landmarks`.
- Produces artifacts:
  - `reach_acts_<ds>.npz`: `h_src`, `h_tgt`, `h_final` (n,d f32), `labels` (n,), `statements` (n, object), `row_index` (n,), `src_layer`, `tgt_layer`.
  - `reach_dirs_<ds>.npz`: `W` (K,d f32 unit rows), `names` (K, object), `groups` (K, object ∈ {truth, truth_sub, dct_u, rand}), `acc1d` (K,), `thresh02` (K, NaN where 1-D acc < 0.6), `store_jtw` (K, bool), `src_layer`, `tgt_layer`.
  - `reach_margins_<ds>.npz`: `margins`, `cos_md_src`, `cos_vq`, `cos_dctv` (n,K f32), `jtw` (n,Ks,d f16 — unit Jᵀw rows for stored dirs only), `names`, `groups`, `store_names` (Ks, object).
- Produces functions (imported by tests and Task 3/6): `gram_schmidt(vecs, tol=1e-6)`, `fit_probe_dir(X, y, seed=None, frac=1.0)`, `fit_threshold(scores, y) -> (acc, slope_sign, t02)`, `build_battery(h_tgt, y, ds) -> dict`, `load_statements(ds, seed=42) -> (statements, labels, row_index)`, constants `LOGIT_02, N_BOOT=6, BOOT_FRAC=0.5, K_DCT_U=4, N_RAND=64, MIN_ACC_1D=0.6, CHUNK=100, VJP_BATCH=16, DATASET_SAMPLE={"common_claim_true_false": 2000}`.
- CLI: `PYTHONPATH=src python src/reach_margins.py --dataset cities --stage acts|dirs|vjp|all --device cuda [--limit N] [--no-resume]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reach_margins.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_margins.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'reach_margins'`.

- [ ] **Step 3: Write the implementation**

Create `src/reach_margins.py`:

```python
"""reach_margins.py — Phase 1 extraction for the backward-reachability audit.

Three stages (spec §4 P1), run in order (--stage all runs them back-to-back):
  --stage acts : batched no-grad forwards; saves last-token h_src/h_tgt/h_final.
  --stage dirs : fits the direction battery + P(true)=0.2 thresholds from stage-1
                 activations (needs scikit-learn; seconds).
  --stage vjp  : per statement x direction, one vjp -> margins m_i(w) = ||J_i^T w||,
                 unit J^T w rows (stored for truth/truth_sub/dct_u groups), cosines
                 to source landmarks. Checkpoints every CHUNK statements; --resume
                 (default) skips completed chunks.

    PYTHONPATH=src python src/reach_margins.py --dataset cities --stage all --device cuda
    PYTHONPATH=src python src/reach_margins.py --dataset cities --stage vjp --limit 8 --device cpu   # smoke
"""
import argparse
import os

import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, vjp_rows, load_landmarks)

LOGIT_02 = float(np.log(0.2 / 0.8))          # P(true) <= 0.2 confidence margin
N_BOOT, BOOT_FRAC = 6, 0.5                   # bootstrap probes for truth_sub_k
K_DCT_U, N_RAND = 4, 64
MIN_ACC_1D = 0.6                             # threshold valid only above this 1-D acc
CHUNK, VJP_BATCH, ACTS_BATCH = 100, 16, 16
DATASET_SAMPLE = {"common_claim_true_false": 2000}   # stratified cap; cities runs full


# ---------------------------------------------------------------- pure helpers

def load_statements(ds, seed=42):
    """Statements + labels from got_datasets/<ds>.csv; stratified seed-42 subsample
    where DATASET_SAMPLE caps the dataset. Returns (statements, labels, row_index)."""
    import pandas as pd
    df = pd.read_csv(os.path.join("got_datasets", f"{ds}.csv"))
    labels = df["label"].values.astype(int)
    idx = np.arange(len(df))
    cap = DATASET_SAMPLE.get(ds)
    if cap and cap < len(df):
        rng = np.random.default_rng(seed)
        keep = [rng.choice(idx[labels == lab], size=cap // 2, replace=False)
                for lab in (0, 1)]
        idx = np.sort(np.concatenate(keep))
    return df["statement"].values[idx], labels[idx], idx


def gram_schmidt(vecs, tol=1e-6):
    """Orthonormal basis (rows) from a list of vectors; near-dependent ones dropped."""
    basis = []
    for v in vecs:
        w = np.asarray(v, np.float64).copy()
        for b in basis:
            w -= (w @ b) * b
        n = np.linalg.norm(w)
        if n > tol:
            basis.append(w / n)
    return np.stack(basis)


def fit_probe_dir(X, y, seed=None, frac=1.0):
    """Unit logistic-probe gradient direction (funnel_utils.grad_dir recipe), with an
    optional seeded row-subsample for the bootstrap battery members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(y), size=int(frac * len(y)), replace=False)
        X, y = X[keep], y[keep]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return unit(lr.coef_[0] / sc.scale_)


def fit_threshold(scores, y):
    """1-D logistic calibration of s = w.h. Returns (acc, slope_sign, t02) where t02
    is the score at calibrated P(true)=0.2 — only meaningful when slope > 0 (caller
    flips w and refits when slope_sign < 0)."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, np.float64).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000).fit(s, y)
    acc = float(lr.score(s, y))
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    sign = 1.0 if a > 0 else -1.0
    t02 = (LOGIT_02 - b) / a if a != 0 else float("nan")
    return acc, sign, t02


def build_battery(h_tgt, y, ds):
    """Direction battery at the target layer (spec §3). h_tgt (n,d) float, y (n,).
    Reads truth_dir_tgt_<ds>.npz and dct_U_<ds>.pt/dct_V_<ds>.pt from cwd."""
    h = np.asarray(h_tgt, np.float64)
    tt = np.load(f"truth_dir_tgt_{ds}.npz")
    md = unit(np.asarray(tt["mean_diff"], np.float64))
    pg = fit_probe_dir(h, y)
    boots = [fit_probe_dir(h, y, seed=s, frac=BOOT_FRAC) for s in range(N_BOOT)]
    sub = gram_schmidt([md, pg] + boots)
    V, U, _ = fu.load_dct(ds)
    tops = fu.top_k_by_potency(V, U, K_DCT_U)
    rng = np.random.default_rng(123)
    raw = [("mean_diff_tgt", "truth", md), ("probe_grad_tgt", "truth", pg)]
    raw += [(f"truth_sub_{j}", "truth_sub", sub[j]) for j in range(sub.shape[0])]
    raw += [(f"dct_u_{r}", "dct_u", U[:, t].astype(np.float64))
            for r, t in enumerate(tops)]
    raw += [(f"rand_{r}", "rand", rng.standard_normal(h.shape[1]))
            for r in range(N_RAND)]
    W, names, groups, acc1d, thresh02 = [], [], [], [], []
    for name, group, v in raw:
        v = unit(np.asarray(v, np.float64))
        acc, sign, t02 = fit_threshold(h @ v, y)
        if sign < 0:
            v = -v
            acc, sign, t02 = fit_threshold(h @ v, y)
        W.append(v); names.append(name); groups.append(group); acc1d.append(acc)
        thresh02.append(t02 if acc >= MIN_ACC_1D else float("nan"))
    store = [g in ("truth", "truth_sub", "dct_u") for g in groups]
    return {"W": np.stack(W).astype(np.float32),
            "names": np.array(names, dtype=object),
            "groups": np.array(groups, dtype=object),
            "acc1d": np.array(acc1d, np.float32),
            "thresh02": np.array(thresh02, np.float32),
            "store_jtw": np.array(store)}


# ---------------------------------------------------------------------- stages

def stage_acts(ds, device, limit=0):
    validate_inputs(ds)
    stmts, labels, row_index = load_statements(ds)
    if limit:
        stmts, labels, row_index = stmts[:limit], labels[:limit], row_index[:limit]
    tok, model, _, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    hs_src, hs_tgt, hs_fin = [], [], []
    for b0 in range(0, len(stmts), ACTS_BATCH):
        fb = forward_source_batch(model, tok, stmts[b0:b0 + ACTS_BATCH],
                                  meta["src"], meta["tgt"], dev)
        hs_src.append(fb["h_src"].cpu().numpy())
        hs_tgt.append(fb["h_tgt"].cpu().numpy())
        hs_fin.append(fb["h_final"].cpu().numpy())
        print(f"[acts] {ds} {b0 + len(fb['h_src'])}/{len(stmts)}", flush=True)
    np.savez(f"reach_acts_{ds}.npz",
             h_src=np.concatenate(hs_src).astype(np.float32),
             h_tgt=np.concatenate(hs_tgt).astype(np.float32),
             h_final=np.concatenate(hs_fin).astype(np.float32),
             labels=labels, statements=np.array(stmts, dtype=object),
             row_index=row_index, src_layer=meta["src"], tgt_layer=meta["tgt"])
    print(f"[acts] wrote reach_acts_{ds}.npz  n={len(stmts)}")


def stage_dirs(ds):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    bat = build_battery(acts["h_tgt"], acts["labels"].astype(int), ds)
    # sanity: stored mean_diff_tgt must agree with this run's own class-mean diff
    own_md = unit(np.asarray(acts["h_tgt"], np.float64)[acts["labels"] == 1].mean(0)
                  - np.asarray(acts["h_tgt"], np.float64)[acts["labels"] == 0].mean(0))
    cos = float(abs(own_md @ np.asarray(bat["W"][0], np.float64)))
    print(f"[dirs] cos(own mean_diff, truth_dir_tgt mean_diff) = {cos:.3f}")
    if cos < 0.9:
        print("[dirs] WARNING: stored target-layer mean_diff disagrees with this run's "
              "activations — check sampling/layer conventions before trusting margins")
    np.savez(f"reach_dirs_{ds}.npz", src_layer=acts["src_layer"],
             tgt_layer=acts["tgt_layer"], **bat)
    n_valid = int(np.isfinite(bat["thresh02"]).sum())
    print(f"[dirs] wrote reach_dirs_{ds}.npz  K={len(bat['names'])} "
          f"(valid thresholds: {n_valid})")


def stage_vjp(ds, device, resume=True, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    W = torch.tensor(dirs["W"], dtype=torch.float32)
    store = np.asarray(dirs["store_jtw"])
    lm = load_landmarks(ds)
    stmts = acts["statements"]
    n = min(limit, len(stmts)) if limit else len(stmts)
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    W_dev = W.to(dev)
    store_t = torch.tensor(np.asarray(store, bool), dtype=torch.bool, device=dev)
    lm_t = {k: torch.tensor(v, dtype=torch.float32, device=dev)
            for k, v in lm.items()}
    cdir = f"reach_chunks_{ds}"
    os.makedirs(cdir, exist_ok=True)
    for c0 in range(0, n, CHUNK):
        cpath = os.path.join(cdir, f"chunk_{c0:05d}.npz")
        if resume and os.path.exists(cpath):
            print(f"[vjp] {cpath} exists — skipping", flush=True)
            continue
        c1 = min(c0 + CHUNK, n)
        m_l, cmd_l, cvq_l, cdv_l, jtw_l = [], [], [], [], []
        for b0 in range(c0, c1, VJP_BATCH):
            batch = stmts[b0:min(b0 + VJP_BATCH, c1)]
            fb = forward_source_batch(model, tok, batch, meta["src"], meta["tgt"], dev)
            f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
            delta0 = torch.zeros(len(batch), fb["h_src_seq"].shape[-1], device=dev)
            _, G = vjp_rows(f, delta0, W_dev)              # (K, B, d)
            m = G.norm(dim=-1)                             # (K, B)
            Gu = G / m.clamp_min(1e-12)[..., None]
            m_l.append(m.T.cpu().numpy())
            cmd_l.append((Gu @ lm_t["md_src"]).T.cpu().numpy())
            cvq_l.append((Gu @ lm_t["v_q"]).T.cpu().numpy())
            cdv_l.append((Gu @ lm_t["dct_v"]).T.cpu().numpy())
            jtw_l.append(Gu[store_t].permute(1, 0, 2).cpu().numpy().astype(np.float16))
            print(f"[vjp] {ds} statements {b0}-{b0 + len(batch)} done", flush=True)
        np.savez(cpath, margins=np.concatenate(m_l).astype(np.float32),
                 cos_md_src=np.concatenate(cmd_l).astype(np.float32),
                 cos_vq=np.concatenate(cvq_l).astype(np.float32),
                 cos_dctv=np.concatenate(cdv_l).astype(np.float32),
                 jtw=np.concatenate(jtw_l))
        print(f"[vjp] checkpointed {cpath}", flush=True)
    merge_chunks(ds, n, dirs)


def merge_chunks(ds, n, dirs):
    cdir = f"reach_chunks_{ds}"
    parts = {k: [] for k in ("margins", "cos_md_src", "cos_vq", "cos_dctv", "jtw")}
    for c0 in range(0, n, CHUNK):
        z = np.load(os.path.join(cdir, f"chunk_{c0:05d}.npz"))
        for k in parts:
            parts[k].append(z[k])
    store = np.asarray(dirs["store_jtw"])
    np.savez(f"reach_margins_{ds}.npz",
             **{k: np.concatenate(v) for k, v in parts.items()},
             names=dirs["names"], groups=dirs["groups"],
             store_names=np.asarray(dirs["names"])[store])
    print(f"[vjp] merged {n} statements -> reach_margins_{ds}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", required=True, choices=["acts", "dirs", "vjp", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap statements (smoke)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    if a.stage in ("acts", "all"):
        stage_acts(a.dataset, a.device, a.limit)
    if a.stage in ("dirs", "all"):
        stage_dirs(a.dataset)
    if a.stage in ("vjp", "all"):
        stage_vjp(a.dataset, a.device, resume=not a.no_resume, limit=a.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_margins.py -q`
Expected: 6 passed.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reach_margins.py tests/test_reach_margins.py
git commit -m "feat(reach): staged Phase-1 extraction — acts, direction battery, checkpointed vjp margins"
```

---

### Task 3: `reach_analyze.py` — Phase 1 analysis

**Files:**
- Create: `src/reach_analyze.py`
- Test: `tests/test_reach_analyze.py`

**Interfaces:**
- Consumes: `reach_acts_<ds>.npz`, `reach_dirs_<ds>.npz`, `reach_margins_<ds>.npz`, `dct_meta_<ds>.json` (input_scale).
- Produces artifacts:
  - `reach_curve_<ds>.csv`: columns `direction,eps,frac_reachable` — one row per (direction with a valid threshold, ε on the 61-point grid), plus the synthetic direction `truth_sub_best` (per-statement min ε* over truth ∪ truth_sub members). Curves computed over **label==1** statements (pushing TRUE statements into the FALSE halfspace = making the model lie).
  - `reach_summary_<ds>.json`: `{"input_scale": float, "n_label1": int, "directions": {name: {"group","median_margin","rand_pct","median_eps_star","frac_at_input_scale"}}, "rand_null": {"median","q05","q95"}, "best_sub_name": str, "truth_sub_best": {"median_margin","median_eps_star","frac_at_input_scale"}, "verdict": str}` — `rand_pct` = percentile of the direction's median margin within the 64 random-direction median margins; `verdict` = the spec's decision criterion string (`"unreachable-tail"` if truth-subspace best-case median margin < random-null median, else `"reachable-candidate"`).
- Produces functions (imported by tests and Task 6): `required_eps(g, m)`, `frac_reachable_curve(eps_star, grid)`, `subspace_best_margin(G_raw)`, `eps_grid(input_scale, n=61)`.
- CLI: `PYTHONPATH=src python src/reach_analyze.py --dataset cities`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reach_analyze.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_analyze as ra


def test_required_eps_hand_case():
    # 2-D hand case: J = diag(2, 1), w = e0  =>  J^T w = (2,0), m = 2.
    # h_tgt.w = 1.0, threshold t02 = 0  =>  g = 1.0, eps* = 0.5.
    g = np.array([1.0]); m = np.array([2.0])
    assert np.allclose(ra.required_eps(g, m), [0.5])


def test_required_eps_already_in_target_and_zero_margin():
    g = np.array([-0.3, 1.0]); m = np.array([1.0, 0.0])
    out = ra.required_eps(g, m)
    assert out[0] == 0.0               # already past the threshold: reachable at eps=0
    assert out[1] > 1e10               # zero margin: unreachable at any finite eps


def test_frac_reachable_monotone_in_eps():
    rng = np.random.default_rng(0)
    eps_star = rng.exponential(1.0, 500)
    grid = np.linspace(0, 5, 21)
    curve = ra.frac_reachable_curve(eps_star, grid)
    assert np.all(np.diff(curve) >= 0)
    assert curve[0] == (eps_star <= 0).mean()
    assert curve[-1] <= 1.0


def test_frac_reachable_exact_boundary():
    curve = ra.frac_reachable_curve(np.array([0.5, 1.0, 2.0]), np.array([0.4, 0.5, 1.0]))
    assert np.allclose(curve, [0.0, 1 / 3, 2 / 3])   # reachable iff eps* <= eps


def test_subspace_best_margin_is_top_singular_value():
    # rows J^T b_k for orthonormal {b_k}: best-case ||J^T w|| over unit w in span
    G = np.array([[3.0, 0.0, 0.0],
                  [0.0, 4.0, 0.0]])
    assert np.isclose(ra.subspace_best_margin(G), 4.0)
    # rotation of the basis must not change the answer
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.isclose(ra.subspace_best_margin(R @ G), 4.0)


def test_eps_grid_spans_1p5_input_scale():
    g = ra.eps_grid(40.0)
    assert len(g) == 61 and g[0] == 0.0 and np.isclose(g[-1], 60.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_analyze.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'reach_analyze'`.

- [ ] **Step 3: Write the implementation**

Create `src/reach_analyze.py`:

```python
"""reach_analyze.py — Phase 1 analysis: reachability verdicts, curves, summary.

Linear closed-form test (spec §2), per statement i and readout w:
    reachable within eps  <=>  g_i(w) <= eps * m_i(w)
    g_i(w) = w.h_tgt_i - t02(w)      (signed distance to the FALSE threshold)
    m_i(w) = ||J_i^T w||             (controllability margin, from reach_margins)

Curves are computed over label==1 statements (pushing TRUE statements into the
FALSE halfspace — the lie direction). Runs locally in seconds.

    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
"""
import argparse
import csv
import json

import numpy as np

UNREACHABLE = 1e12          # sentinel eps* when the margin is (numerically) zero
EPS_POINTS = 61


def required_eps(g, m):
    """eps*_i = g/m where g>0; 0 where already in the target set; UNREACHABLE where
    the margin vanishes."""
    g = np.asarray(g, np.float64)
    m = np.asarray(m, np.float64)
    out = np.where(g <= 0, 0.0,
                   np.where(m > 1e-12, g / np.maximum(m, 1e-12), UNREACHABLE))
    return out


def frac_reachable_curve(eps_star, grid):
    es = np.asarray(eps_star, np.float64)
    return np.array([(es <= e).mean() for e in np.asarray(grid, np.float64)])


def subspace_best_margin(G_raw):
    """G_raw (K,d): rows J_i^T b_k for an orthonormal basis {b_k}. Best-case margin
    max_{w in span, ||w||=1} ||J_i^T w|| = sigma_max(G_raw)."""
    return float(np.linalg.svd(np.asarray(G_raw, np.float64), compute_uv=False)[0])


def eps_grid(input_scale, n=EPS_POINTS):
    return np.linspace(0.0, 1.5 * float(input_scale), n)


def analyze(ds):
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    input_scale = float(json.load(open(f"dct_meta_{ds}.json"))["input_scale"])
    W = np.asarray(dirs["W"], np.float64)                    # (K, d)
    names = [str(x) for x in dirs["names"]]
    groups = [str(x) for x in dirs["groups"]]
    thresh = np.asarray(dirs["thresh02"], np.float64)        # (K,)
    h_tgt = np.asarray(acts["h_tgt"], np.float64)            # (n, d)
    y = np.asarray(acts["labels"]).astype(int)
    margins = np.asarray(mz["margins"], np.float64)          # (n, K)
    n = margins.shape[0]
    h_tgt, y = h_tgt[:n], y[:n]                              # tolerate --limit runs
    lab1 = y == 1
    grid = eps_grid(input_scale)

    g_all = h_tgt @ W.T - thresh[None, :]                    # (n, K); NaN thresh -> NaN
    eps_star = {}
    for k, name in enumerate(names):
        if not np.isfinite(thresh[k]):
            continue
        eps_star[name] = required_eps(g_all[lab1, k], margins[lab1, k])

    # --- best-case over the truth subspace: per-statement min eps* over members
    sub_names = [nm for nm, gr in zip(names, groups) if gr in ("truth", "truth_sub")
                 and np.isfinite(thresh[names.index(nm)])]
    if sub_names:
        stack = np.stack([eps_star[nm] for nm in sub_names])          # (S, n1)
        eps_star["truth_sub_best"] = stack.min(axis=0)
        best_sub_name = sub_names[int(np.argmin([np.median(eps_star[nm])
                                                 for nm in sub_names]))]
    else:
        best_sub_name = ""

    # --- per-statement best-case MARGIN over the subspace (sigma_max of stacked J^T b)
    store_names = [str(x) for x in mz["store_names"]]
    sub_rows = [store_names.index(nm) for nm, gr in zip(names, groups)
                if gr in ("truth", "truth_sub")]
    jtw = np.asarray(mz["jtw"], np.float64)                  # (n, Ks, d) unit rows
    col = [names.index(store_names[r]) for r in sub_rows]
    best_margin = np.array([subspace_best_margin(
        jtw[i, sub_rows, :] * margins[i, col][:, None]) for i in range(n)])

    # --- write curves
    with open(f"reach_curve_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "eps", "frac_reachable"))
        for name, es in eps_star.items():
            for e, fr in zip(grid, frac_reachable_curve(es, grid)):
                w.writerow((name, f"{e:.6g}", f"{fr:.6g}"))

    # --- summary
    med_marg = {nm: float(np.median(margins[:, k]))
                for k, nm in enumerate(names)}
    rand_meds = np.array([med_marg[nm] for nm, gr in zip(names, groups)
                          if gr == "rand"])
    def rand_pct(v):
        return float((rand_meds < v).mean() * 100.0)
    summary = {
        "input_scale": input_scale, "n_label1": int(lab1.sum()),
        "rand_null": {"median": float(np.median(rand_meds)),
                      "q05": float(np.quantile(rand_meds, 0.05)),
                      "q95": float(np.quantile(rand_meds, 0.95))},
        "directions": {}, "best_sub_name": best_sub_name,
    }
    for k, nm in enumerate(names):
        d = {"group": groups[k], "median_margin": med_marg[nm],
             "rand_pct": rand_pct(med_marg[nm])}
        if nm in eps_star:
            d["median_eps_star"] = float(np.median(eps_star[nm]))
            d["frac_at_input_scale"] = float((eps_star[nm] <= input_scale).mean())
        summary["directions"][nm] = d
    bm_med = float(np.median(best_margin))
    summary["truth_sub_best"] = {
        "median_margin": bm_med,
        "median_eps_star": float(np.median(eps_star["truth_sub_best"]))
        if "truth_sub_best" in eps_star else None,
        "frac_at_input_scale": float((eps_star["truth_sub_best"] <= input_scale).mean())
        if "truth_sub_best" in eps_star else None,
    }
    summary["verdict"] = ("unreachable-tail"
                          if bm_med < summary["rand_null"]["median"]
                          else "reachable-candidate")
    with open(f"reach_summary_{ds}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] {ds}: verdict={summary['verdict']}  "
          f"best-case median margin={bm_med:.4g} vs rand median="
          f"{summary['rand_null']['median']:.4g}")
    for nm in ("mean_diff_tgt", "probe_grad_tgt", "truth_sub_best"):
        if nm in eps_star:
            print(f"  {nm}: median eps*={np.median(eps_star[nm]):.4g} "
                  f"(input_scale={input_scale:.4g}), frac reachable at budget="
                  f"{(eps_star[nm] <= input_scale).mean():.3f}")
    print(f"[analyze] wrote reach_curve_{ds}.csv and reach_summary_{ds}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    analyze(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_analyze.py -q`
Expected: 6 passed.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reach_analyze.py tests/test_reach_analyze.py
git commit -m "feat(reach): Phase-1 analysis — closed-form reachability, curves, subspace best-case"
```

---

### Task 4: `viz_reach.py` — Phase 1 figures

**Files:**
- Create: `src/viz_reach.py`
- Test: `tests/test_viz_reach.py`

**Interfaces:**
- Consumes: `reach_margins_<ds>.npz`, `reach_dirs_<ds>.npz`, `reach_curve_<ds>.csv`, `reach_summary_<ds>.json`; optionally `reach_linerr_summary_<ds>.csv` (Task 8 — overlay drawn only if the file exists).
- Produces: `plot_reach_margins_<ds>.png`, `plot_reach_curves_<ds>.png`, `plot_reach_geometry_<ds>.png`.
- Functions: `fig_margins(ds)`, `fig_curves(ds)`, `fig_geometry(ds)`, `main()` CLI `--dataset`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_viz_reach.py`:

```python
import sys, os, json, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_reach as vr


def _make_synthetic(ds, tmp_path, n=20, d=8):
    rng = np.random.default_rng(0)
    names = ["mean_diff_tgt", "probe_grad_tgt", "truth_sub_0", "dct_u_0",
             "rand_0", "rand_1"]
    groups = ["truth", "truth", "truth_sub", "dct_u", "rand", "rand"]
    K, Ks = len(names), 4
    np.savez(tmp_path / f"reach_dirs_{ds}.npz",
             W=rng.standard_normal((K, d)).astype(np.float32),
             names=np.array(names, object), groups=np.array(groups, object),
             acc1d=np.full(K, 0.9, np.float32),
             thresh02=np.array([0, 0, 0, np.nan, np.nan, np.nan], np.float32),
             store_jtw=np.array([True, True, True, True, False, False]),
             src_layer=3, tgt_layer=5)
    np.savez(tmp_path / f"reach_margins_{ds}.npz",
             margins=rng.exponential(1.0, (n, K)).astype(np.float32),
             cos_md_src=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             cos_vq=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             cos_dctv=rng.uniform(-1, 1, (n, K)).astype(np.float32),
             jtw=rng.standard_normal((n, Ks, d)).astype(np.float16),
             names=np.array(names, object), groups=np.array(groups, object),
             store_names=np.array(names[:Ks], object))
    grid = np.linspace(0, 60, 61)
    with open(tmp_path / f"reach_curve_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("direction", "eps", "frac_reachable"))
        for nm in ("mean_diff_tgt", "truth_sub_best"):
            for e in grid:
                w.writerow((nm, e, min(1.0, e / 60)))
    json.dump({"input_scale": 40.0, "best_sub_name": "truth_sub_0",
               "rand_null": {"median": 1.0, "q05": 0.5, "q95": 2.0},
               "verdict": "unreachable-tail",
               "truth_sub_best": {"median_margin": 0.5}},
              open(tmp_path / f"reach_summary_{ds}.json", "w"))


def test_phase1_figures_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_synthetic("toy", tmp_path)
    vr.fig_margins("toy"); vr.fig_curves("toy"); vr.fig_geometry("toy")
    for p in ("plot_reach_margins_toy.png", "plot_reach_curves_toy.png",
              "plot_reach_geometry_toy.png"):
        assert os.path.exists(p), p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_reach.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'viz_reach'`.

- [ ] **Step 3: Write the implementation**

Create `src/viz_reach.py`:

```python
"""viz_reach.py — figures for the backward-reachability audit.

Phase 1: margin distributions, reachability curves (with Phase-5 trust region
overlaid when reach_linerr_summary_<ds>.csv exists), and J^T w source geometry.
Later tasks extend this module with SVD (Phase 2) and J-lens (Phase 4) figures.

    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GRAY, GREEN, RED, ORANGE = "#4477aa", "#999999", "#228833", "#cc3311", "#ee7733"


def _load(ds):
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    dz = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    summ = json.load(open(f"reach_summary_{ds}.json"))
    names = [str(x) for x in mz["names"]]
    groups = [str(x) for x in mz["groups"]]
    return mz, dz, summ, names, groups


def fig_margins(ds):
    mz, dz, summ, names, groups = _load(ds)
    m = np.asarray(mz["margins"], np.float64)
    sel = {"truth readouts": [k for k, g in enumerate(groups) if g == "truth"],
           "truth subspace": [k for k, g in enumerate(groups) if g == "truth_sub"],
           "DCT U top (control)": [k for k, g in enumerate(groups) if g == "dct_u"],
           "random null (64)": [k for k, g in enumerate(groups) if g == "rand"]}
    colors = [BLUE, GREEN, RED, GRAY]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    data = [np.log10(np.clip(m[:, ks].ravel(), 1e-12, None)) for ks in sel.values()]
    parts = ax.violinplot(data, showmedians=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(sel) + 1)); ax.set_xticklabels(sel.keys(), fontsize=9)
    ax.set_ylabel("log10 controllability margin  ||J^T w||")
    ax.axhline(np.log10(max(summ["rand_null"]["median"], 1e-12)), color=GRAY,
               ls="--", lw=1, label="random-null median")
    ax.set_title(f"{ds}: controllability margins per readout group "
                 f"(verdict: {summ['verdict']})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"plot_reach_margins_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_margins_{ds}.png")


def _read_curves(ds):
    curves = {}
    with open(f"reach_curve_{ds}.csv") as f:
        for r in csv.DictReader(f):
            curves.setdefault(r["direction"], []).append(
                (float(r["eps"]), float(r["frac_reachable"])))
    return {k: np.array(v) for k, v in curves.items()}


def fig_curves(ds):
    summ = json.load(open(f"reach_summary_{ds}.json"))
    curves = _read_curves(ds)
    scale = summ["input_scale"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    palette = [BLUE, GREEN, ORANGE, RED, GRAY, "#aa3377", "#66ccee"]
    show = ["mean_diff_tgt", "probe_grad_tgt", "truth_sub_best"]
    show += [k for k in curves if k.startswith("dct_u")]
    show += [k for k in curves if k not in show][:2]
    for i, nm in enumerate([s for s in show if s in curves]):
        c = curves[nm]
        lw = 2.4 if nm == "truth_sub_best" else 1.4
        ax.plot(c[:, 0], c[:, 1], label=nm, color=palette[i % len(palette)], lw=lw)
    ax.axvline(scale, color="k", ls="--", lw=1)
    ax.text(scale, 1.02, "input_scale", ha="center", fontsize=8)
    lin = f"reach_linerr_summary_{ds}.csv"
    if os.path.exists(lin):
        with open(lin) as f:
            rows = {r["direction"]: float(r["eps20"]) for r in csv.DictReader(f)}
        e20 = rows.get("jtw_mean_diff_tgt")
        if e20 is not None:
            ax.axvspan(e20, ax.get_xlim()[1], color=GRAY, alpha=0.15)
            ax.text(e20, 0.5, " beyond linear validity (rel err > 20%)",
                    fontsize=8, color="#555555", rotation=90, va="center")
    ax.set_xlabel("perturbation budget eps (activation norm units)")
    ax.set_ylabel("fraction of TRUE statements reachable into FALSE halfspace")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(f"{ds}: first-order reachability of the FALSE-probe halfspace",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"plot_reach_curves_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_curves_{ds}.png")


def fig_geometry(ds):
    mz, dz, summ, names, groups = _load(ds)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    k_md = names.index("mean_diff_tgt")
    picks = [("mean_diff_tgt", k_md, BLUE)]
    if "dct_u_0" in names:
        picks.append(("dct_u_0", names.index("dct_u_0"), RED))
    for ax, key, lab in ((axes[0], "cos_md_src", "cos(J^T w, mean_diff@src)"),
                         (axes[1], "cos_vq", "cos(J^T w, v_Q)")):
        arr = np.asarray(mz[key], np.float64)
        for nm, k, c in picks:
            ax.hist(arr[:, k], bins=40, alpha=0.55, color=c, label=f"w={nm}")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab); ax.set_xlim(-1, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylabel("statements")
    fig.suptitle(f"{ds}: where J^T w points at the source layer", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_reach_geometry_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_geometry_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    fig_margins(ds); fig_curves(ds); fig_geometry(ds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_reach.py -q`
Expected: 1 passed.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/viz_reach.py tests/test_viz_reach.py
git commit -m "feat(reach): Phase-1 figures — margins, reachability curves, source geometry"
```

---

### Task 5: `reach_svd.py` — Phase 2 full-Jacobian subsample

**Files:**
- Create: `src/reach_svd.py`
- Modify: `src/viz_reach.py` (add `fig_svd`), `tests/test_viz_reach.py` (extend smoke)

**Interfaces:**
- Consumes (Task 1): `load_model_and_slice`, `forward_source_batch`, `make_hop`, `jvp_cols`, `validate_inputs`; (Task 2 artifacts): `reach_acts_<ds>.npz`, `reach_dirs_<ds>.npz`; `funnel_utils.load_dct/top_k_by_potency`; `truth_dir_<ds>.npz`.
- Produces: `reach_svd_<ds>/stmt_<i>.npz` (`s` (d,) f32 singular values, `U64` (d,64) f16, `V64` (d,64) f16, `label`, `stmt_index`); `--analyze` writes `reach_svd_energy_<ds>.csv` (`k,quantity,mean_energy` for `quantity ∈ {w_mean_diff_tgt_in_U, w_probe_grad_tgt_in_U, md_src_in_V, rand_in_U}`) and `reach_svd_summary_<ds>.csv` (`stmt_index,label,eff_rank,s1,s64_over_s1,dctV_overlap_top16`).
- CLI: `PYTHONPATH=src python src/reach_svd.py --dataset cities --device cuda [--limit N]` (compute) / `--analyze` (local).

- [ ] **Step 1: Write the implementation** (no new unit-test file — the jvp path is covered by `test_reach_hop.py::test_transposition_identity`; the analysis math gets a viz smoke in Step 2)

Create `src/reach_svd.py`:

```python
"""reach_svd.py — Phase 2: full Jacobian + SVD for a 32-statement subsample.

Full J (d x d) per statement via batched jvp (TANGENT_CHUNK basis tangents per
forward-mode call), then SVD. One output file per statement so partial completion
is usable. Analysis turns "margin is small" into "truth lies in the singular tail":
energy of truth readouts in the top-k LEFT subspace, of mean_diff@src in the top-k
RIGHT subspace, and overlap of top right-singulars with DCT's V (consistency check —
a mismatch here is a stop-and-diagnose event, spec §4 P2).

    PYTHONPATH=src python src/reach_svd.py --dataset cities --device cuda
    PYTHONPATH=src .venv/bin/python src/reach_svd.py --dataset cities --analyze
"""
import argparse
import csv
import os

import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, jvp_cols)

N_STMT, SEED, TANGENT_CHUNK, TOPK_SAVE = 32, 42, 64, 64


def pick_svd_indices(labels, n=N_STMT, seed=SEED):
    """n/2 true + n/2 false statement indices, deterministic."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    pick = [rng.choice(idx[labels == lab], size=n // 2, replace=False)
            for lab in (1, 0)]
    return np.sort(np.concatenate(pick))


def full_jacobian(sliced, h_row, attn_row, device, chunk=TANGENT_CHUNK):
    """h_row (1,T,d). Returns J (d,d) float32 with J[:, j] = dF/dDelta_j, built
    column-block by column-block: the batch dimension carries `chunk` tangents of
    the SAME statement."""
    d = h_row.shape[-1]
    cols = []
    for j0 in range(0, d, chunk):
        k = min(chunk, d - j0)
        h_rep = h_row.expand(k, -1, -1)
        am_rep = attn_row.expand(k, -1)
        f = make_hop(sliced, h_rep, am_rep)
        delta0 = torch.zeros(k, d, device=device)
        T_rows = torch.zeros(k, d, device=device)
        T_rows[torch.arange(k), j0 + torch.arange(k)] = 1.0
        _, JT = jvp_cols(f, delta0, T_rows)          # (k, d): rows are J e_j
        cols.append(JT.T.float().cpu())              # (d, k) columns of J
    return torch.cat(cols, dim=1)                    # (d, d)


def compute(ds, device, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    picks = pick_svd_indices(labels)
    if limit:
        picks = picks[:limit]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    outdir = f"reach_svd_{ds}"
    os.makedirs(outdir, exist_ok=True)
    for i in picks:
        opath = os.path.join(outdir, f"stmt_{int(i):05d}.npz")
        if os.path.exists(opath):
            print(f"[svd] {opath} exists — skipping", flush=True)
            continue
        fb = forward_source_batch(model, tok, [stmts[i]], meta["src"], meta["tgt"], dev)
        J = full_jacobian(sliced, fb["h_src_seq"], fb["attn"], dev)
        U, S, Vh = torch.linalg.svd(J.to(dev), full_matrices=False)
        np.savez(opath, s=S.cpu().numpy().astype(np.float32),
                 U64=U[:, :TOPK_SAVE].cpu().numpy().astype(np.float16),
                 V64=Vh[:TOPK_SAVE].T.cpu().numpy().astype(np.float16),
                 label=int(labels[i]), stmt_index=int(i))
        print(f"[svd] {ds} stmt {int(i)} done  s1={float(S[0]):.4g}  "
              f"eff_rank={(S.sum() ** 2 / (S ** 2).sum()).item():.1f}", flush=True)


def analyze(ds):
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    W = np.asarray(dirs["W"], np.float64)
    w_md = W[names.index("mean_diff_tgt")]
    w_pg = W[names.index("probe_grad_tgt")]
    md_src = unit(np.asarray(np.load(f"truth_dir_{ds}.npz")["mean_diff"], np.float64))
    rng = np.random.default_rng(7)
    w_rand = unit(rng.standard_normal(W.shape[1]))
    V, U, _ = fu.load_dct(ds)
    dctV16 = np.linalg.qr(V[:, fu.top_k_by_potency(V, U, 16)].astype(np.float64))[0]
    files = sorted(fn for fn in os.listdir(f"reach_svd_{ds}") if fn.endswith(".npz"))
    ks = np.arange(1, TOPK_SAVE + 1)
    energy = {"w_mean_diff_tgt_in_U": [], "w_probe_grad_tgt_in_U": [],
              "md_src_in_V": [], "rand_in_U": []}
    summ_rows = []
    for fn in files:
        z = np.load(os.path.join(f"reach_svd_{ds}", fn))
        s = np.asarray(z["s"], np.float64)
        U64 = np.asarray(z["U64"], np.float64)
        V64 = np.asarray(z["V64"], np.float64)
        energy["w_mean_diff_tgt_in_U"].append(np.cumsum((U64.T @ w_md) ** 2))
        energy["w_probe_grad_tgt_in_U"].append(np.cumsum((U64.T @ w_pg) ** 2))
        energy["md_src_in_V"].append(np.cumsum((V64.T @ md_src) ** 2))
        energy["rand_in_U"].append(np.cumsum((U64.T @ w_rand) ** 2))
        ov = float(np.mean(np.sum((dctV16.T @ V64[:, :16]) ** 2, axis=0)))
        summ_rows.append((int(z["stmt_index"]), int(z["label"]),
                          float(s.sum() ** 2 / (s ** 2).sum()), float(s[0]),
                          float(s[TOPK_SAVE - 1] / s[0]), ov))
    with open(f"reach_svd_energy_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("k", "quantity", "mean_energy"))
        for q, rows in energy.items():
            mean = np.mean(np.stack(rows), axis=0)
            for k, v in zip(ks, mean):
                w.writerow((int(k), q, f"{v:.6g}"))
    with open(f"reach_svd_summary_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stmt_index", "label", "eff_rank", "s1", "s64_over_s1",
                    "dctV_overlap_top16"))
        w.writerows(summ_rows)
    mo = np.mean([r[5] for r in summ_rows])
    print(f"[svd] {ds}: {len(files)} statements, mean eff_rank="
          f"{np.mean([r[2] for r in summ_rows]):.1f}, DCT-V top-16 overlap={mo:.3f}")
    if mo < 0.3:
        print("[svd] WARNING: top right-singulars disagree with DCT's V — "
              "stop-and-diagnose before trusting the linearized picture (spec §4 P2)")
    print(f"[svd] wrote reach_svd_energy_{ds}.csv and reach_svd_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.dataset)
    else:
        compute(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `fig_svd` to `src/viz_reach.py` and extend the smoke test**

Append to `src/viz_reach.py` (before `main`), and add `fig_svd(ds)` to `main()`'s calls guarded by file existence:

```python
def fig_svd(ds):
    path = f"reach_svd_energy_{ds}.csv"
    if not os.path.exists(path):
        print(f"skip fig_svd: {path} missing")
        return
    series = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            series.setdefault(r["quantity"], []).append(
                (int(r["k"]), float(r["mean_energy"])))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    style = {"w_mean_diff_tgt_in_U": (BLUE, "truth mean_diff (left/U)"),
             "w_probe_grad_tgt_in_U": (GREEN, "truth probe grad (left/U)"),
             "md_src_in_V": (ORANGE, "mean_diff@src (right/V)"),
             "rand_in_U": (GRAY, "random readout (left/U)")}
    for q, pts in series.items():
        pts = np.array(sorted(pts))
        c, lab = style.get(q, (RED, q))
        ax.plot(pts[:, 0], pts[:, 1], color=c, label=lab)
    ax.set_xlabel("top-k singular subspace"); ax.set_ylabel("captured energy of w")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{ds}: does truth live in the singular tail of the hop?",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"plot_reach_svd_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_svd_{ds}.png")
```

In `main()` change the call line to:

```python
    fig_margins(ds); fig_curves(ds); fig_geometry(ds); fig_svd(ds)
```

Append to `tests/test_viz_reach.py`:

```python
def test_fig_svd_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with open("reach_svd_energy_toy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("k", "quantity", "mean_energy"))
        for k in range(1, 65):
            w.writerow((k, "w_mean_diff_tgt_in_U", min(1.0, k / 64)))
            w.writerow((k, "rand_in_U", min(1.0, k / 128)))
    vr.fig_svd("toy")
    assert os.path.exists("plot_reach_svd_toy.png")


def test_fig_svd_skips_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    vr.fig_svd("toy")                      # must not raise
    assert "skip" in capsys.readouterr().out
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_reach.py tests/test_reach_hop.py -q`
Expected: all pass (3 viz tests + 6 hop tests).

- [ ] **Step 4: Sanity-import the new module**

Run: `PYTHONPATH=src .venv/bin/python -c "import reach_svd; print(reach_svd.pick_svd_indices(__import__('numpy').array([0,1]*40)))"`
Expected: prints 32 sorted indices, 16 with label 1 and 16 with label 0.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reach_svd.py src/viz_reach.py tests/test_viz_reach.py
git commit -m "feat(reach): Phase-2 full-Jacobian SVD subsample + singular-energy figure"
```

---

### Task 6: `reach_steer.py` — Phase 3 steering + readout logging

**Files:**
- Create: `src/reach_steer.py`

**Interfaces:**
- Consumes: `dct_steer_utils` (`load_model`, `Steerer`, `generate`), `steer_supervised.FACTUAL_PROMPTS`, `reach_analyze.required_eps` (not needed — ε* read from summary), artifacts `reach_summary_<ds>.json`, `reach_margins_<ds>.npz`, `reach_dirs_<ds>.npz`, `reach_acts_<ds>.npz`, `dct_meta_<ds>.json`.
- Produces:
  - `reach_steer_<ds>.csv` — schema `(direction, scale, prompt, completion)`, exactly what `judge_results.py --mode steer --steer-input` reads (verified against `dct_warm_steer.py` and `deltaai/run_dct_uwarm_judge.slurm`).
  - `reach_steer_readout_<ds>.csv` — `(direction, scale, prompt, g_read)` where `g_read = w·h_tgt(last prompt token) − t02(w)` measured with steering ON (the readout half of the 2×2).
  - per-statement arm: `reach_steer_stmt_<ds>.csv` (same 4-column judge schema, prompt = statement stem) and `reach_steer_stmt_meta_<ds>.csv` (`stmt_index,label,eps_star,scale,g_read`).
- Steering convention: inject `scale · unit(dir)` at the source layer via `Steerer` (every position, matching Phase-1's J). Direction sign: `Jᵀw` points toward increasing w·h (toward TRUE, since w is TRUE-oriented) ⇒ negative scales push toward FALSE. Scales: `±{0.5,1,1.5,2}×ε*` (mean arm) / `±{1,2}×ε*_i` (per-statement arm), all clipped to `1.5·input_scale` in absolute value, plus scale 0 baseline.
- CLI: `PYTHONPATH=src python src/reach_steer.py --dataset cities --device cuda --arm mean|per_stmt [--limit N]`.

- [ ] **Step 1: Write the implementation** (glue over tested components — the vjp math is Task 1-tested, ε* is Task 3-tested; per spec §7 this gets a smoke run, not unit tests)

Create `src/reach_steer.py`:

```python
"""reach_steer.py — Phase 3: steer along J^T w and behaviorally judge (spec §4 P3).

Arm "mean":     dataset-mean unit J^T w for w in {mean_diff_tgt, best truth-subspace
                member}, swept at ±{0.5,1,1.5,2} x median eps*(w), capped at
                1.5 x input_scale, on the 32 FACTUAL_PROMPTS.
Arm "per_stmt": per-statement unit J^T w (w = mean_diff_tgt) on 200 label-1
                statements, prompts = the statement minus its final word, swept at
                ±{1,2} x that statement's own eps*.

Every steered generation also logs the target-layer probe readout
g_read = w.h_tgt(last prompt token) - t02(w), so the deliverable 2x2
(readout moved x behavior moved) is directly tabulable. "Readout flips, model
still won't lie" is a first-class outcome (LiSeCo activation->behavior gap).

    PYTHONPATH=src python src/reach_steer.py --dataset cities --device cuda --arm mean
    PYTHONPATH=src python src/reach_steer.py --dataset cities --device cpu --arm mean --limit 2   # smoke
"""
import argparse
import csv
import json

import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from steer_supervised import FACTUAL_PROMPTS
from reach_hop import load_meta, last_nonpad_index

MEAN_FRACS = [0.5, 1.0, 1.5, 2.0]
STMT_FRACS = [1.0, 2.0]
N_PER_STMT, SEED = 200, 42
MIN_STEM_WORDS = 4
MAX_NEW_TOKENS = 8


def scale_grid(eps_star, input_scale, fracs):
    """Signed steering scales bracketing the reachability boundary, capped at
    1.5 x input_scale; includes the 0 baseline."""
    cap = 1.5 * float(input_scale)
    mags = sorted({min(f * float(eps_star), cap) for f in fracs if eps_star > 0})
    return [0.0] + [s * m for m in mags for s in (+1.0, -1.0)]


def stem_of(statement):
    words = str(statement).rstrip(" .").split()
    if len(words) < MIN_STEM_WORDS:
        return None
    return " ".join(words[:-1])


def read_g(model, tok, prompt, tgt_layer, w_t, t02, dev):
    """Target-layer probe reading at the last prompt token (steering hook active)."""
    enc = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    last = last_nonpad_index(enc["attention_mask"])[0]
    h = hs[tgt_layer][0, last]
    return float(h @ w_t) - float(t02)


def _load_common(ds):
    src, tgt, input_scale = load_meta(ds)
    summ = json.load(open(f"reach_summary_{ds}.json"))
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    store_names = [str(x) for x in mz["store_names"]]
    return src, tgt, input_scale, summ, dirs, mz, acts, names, store_names


def arm_mean(ds, device, limit=0):
    src, tgt, input_scale, summ, dirs, mz, acts, names, store_names = _load_common(ds)
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    lab1 = y == 1
    w_names = ["mean_diff_tgt"]
    best = summ.get("best_sub_name", "")
    if best and best != "mean_diff_tgt":
        w_names.append(best)
    prompts = FACTUAL_PROMPTS[:limit] if limit else FACTUAL_PROMPTS
    tok, model, dev = su.load_model(device)
    rows = [("direction", "scale", "prompt", "completion")]
    readout = [("direction", "scale", "prompt", "g_read")]
    for wn in w_names:
        k = names.index(wn)
        ks = store_names.index(wn)
        jtw = np.asarray(mz["jtw"], np.float64)[lab1, ks, :]      # unit rows
        mean_dir = unit(jtw.mean(axis=0))
        w_vec = torch.tensor(np.asarray(dirs["W"][k], np.float32))
        t02 = float(dirs["thresh02"][k])
        eps_star = summ["directions"][wn]["median_eps_star"]
        vec64 = mean_dir
        dname = f"jtw_{wn}"
        with su.Steerer(model, src) as st:
            for s in scale_grid(eps_star, input_scale, MEAN_FRACS):
                st.set(None if s == 0.0 else torch.tensor(
                    s * vec64, dtype=torch.float32))
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((dname, s, p, c))
                    g = read_g(model, tok, p, tgt, w_vec.to(dev), t02, dev)
                    readout.append((dname, s, p, f"{g:.6g}"))
                print(f"  {dname} scale={s:+.3g} done", flush=True)
    with open(f"reach_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(f"reach_steer_readout_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(readout)
    print(f"[steer] wrote reach_steer_{ds}.csv and reach_steer_readout_{ds}.csv")


def arm_per_stmt(ds, device, limit=0):
    src, tgt, input_scale, summ, dirs, mz, acts, names, store_names = _load_common(ds)
    stmts = acts["statements"]
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    k = names.index("mean_diff_tgt")
    ks = store_names.index("mean_diff_tgt")
    t02 = float(dirs["thresh02"][k])
    w_vec = torch.tensor(np.asarray(dirs["W"][k], np.float32))
    h_tgt = np.asarray(acts["h_tgt"], np.float64)[:len(y)]
    g_all = h_tgt @ np.asarray(dirs["W"][k], np.float64) - t02
    m_all = np.asarray(mz["margins"], np.float64)[:, k]
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    picks = rng.permutation(idx1)[:N_PER_STMT]
    if limit:
        picks = picks[:limit]
    tok, model, dev = su.load_model(device)
    rows = [("direction", "scale", "prompt", "completion")]
    meta_rows = [("stmt_index", "label", "eps_star", "scale", "g_read")]
    with su.Steerer(model, src) as st:
        for i in picks:
            stem = stem_of(stmts[i])
            if stem is None:
                continue
            eps_i = g_all[i] / max(m_all[i], 1e-12) if g_all[i] > 0 else 0.0
            jtw_i = unit(np.asarray(mz["jtw"], np.float64)[i, ks, :])
            for s in scale_grid(eps_i, input_scale, STMT_FRACS):
                st.set(None if s == 0.0 else torch.tensor(
                    s * jtw_i, dtype=torch.float32))
                c = su.generate(model, tok, stem, MAX_NEW_TOKENS)
                g = read_g(model, tok, stem, tgt, w_vec.to(dev), t02, dev)
                rows.append(("jtw_stmt", s, stem, c))
                meta_rows.append((int(i), int(y[i]), f"{eps_i:.6g}", s, f"{g:.6g}"))
            print(f"  stmt {int(i)} done", flush=True)
    with open(f"reach_steer_stmt_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(f"reach_steer_stmt_meta_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(meta_rows)
    print(f"[steer] wrote reach_steer_stmt_{ds}.csv and reach_steer_stmt_meta_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", required=True, choices=["mean", "per_stmt"])
    ap.add_argument("--limit", type=int, default=0, help="cap prompts/statements (smoke)")
    a = ap.parse_args()
    if a.arm == "mean":
        arm_mean(a.dataset, a.device, a.limit)
    else:
        arm_per_stmt(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Unit-test the two pure helpers by extending `tests/test_reach_analyze.py`**

Append:

```python
def test_scale_grid_caps_and_brackets():
    import reach_steer as rs
    g = rs.scale_grid(eps_star=10.0, input_scale=40.0, fracs=[0.5, 1.0, 1.5, 2.0])
    assert 0.0 in g
    mags = sorted({abs(s) for s in g if s != 0})
    assert mags == [5.0, 10.0, 15.0, 20.0]
    g2 = rs.scale_grid(eps_star=100.0, input_scale=40.0, fracs=[1.0, 2.0])
    assert max(abs(s) for s in g2) == 60.0        # capped at 1.5 x input_scale
    assert rs.scale_grid(0.0, 40.0, [1.0]) == [0.0]


def test_stem_of():
    import reach_steer as rs
    assert rs.stem_of("The city of Paris is in France.") == "The city of Paris is in"
    assert rs.stem_of("Too short.") is None
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_analyze.py -q`
Expected: 8 passed (6 prior + 2 new). (`reach_steer` imports `dct_steer_utils`/`steer_supervised`, which import torch/transformers but load no model at import time.)

- [ ] **Step 4: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/reach_steer.py tests/test_reach_analyze.py
git commit -m "feat(reach): Phase-3 J^T-w steering with target-layer readout logging"
```

---

### Task 7: `reach_jlens.py` — Phase 4 per-layer margins

**Files:**
- Create: `src/reach_jlens.py`
- Modify: `src/viz_reach.py` (add `fig_jlens`), `tests/test_viz_reach.py` (extend smoke)

**Interfaces:**
- Consumes: `dct_steer_utils.load_model`, `reach_hop.last_nonpad_index`, `reach_margins.fit_probe_dir/fit_threshold` (probe at final layer), `mag.config.Q_TRUTH/Q_SUFFIX/YES_VARIANTS/NO_VARIANTS`, `mag.verdict.first_token_ids`, `reach_acts_<ds>.npz` (statement subsample source), `dct_meta_<ds>.json`.
- Produces: `reach_jlens_<ds>.csv` — columns `mode,layer,w_name,margin_mean,margin_median,margin_se,n` for `mode ∈ {decl, quest}`, `layer ∈ 0..25`, `w_name ∈ {verdict, truth_final, v_q_final}`.
- Core method (self-contained, no external dependency): per batch, one forward from `inputs_embeds` with grad enabled; for each final-layer readout w, one `torch.autograd.grad` call returns gradients at ALL hidden_states simultaneously; the broadcast-Δ margin at layer ℓ is `‖Σ_t ∂(w·h_final,last)/∂h_ℓ[t]‖` (sum over positions = the chain rule of the every-position injection). This IS the exact per-statement J-lens-style margin — `anthropics/jacobian-lens` (fitted, averaged lenses) is an optional cross-check behind `--use-jlens`, isolated so it can never block the phase (spec §6).

- [ ] **Step 1: Write the implementation**

Create `src/reach_jlens.py`:

```python
"""reach_jlens.py — Phase 4: per-source-layer controllability margins ||J_l^T w||
for final-layer readouts (the workspace-selectivity test, spec §4 P4).

Readouts w (final-residual basis):
  verdict     : unit( mean W_U[yes first-tokens] - mean W_U[no first-tokens] )
  truth_final : logistic-probe gradient fit on this run's final-layer declarative acts
  v_q_final   : unit mean( h_final(quest) - h_final(decl) )   (the question-mode shift)

Inputs run in two modes: decl = the bare statement; quest = Q_TRUTH+statement+Q_SUFFIX.
Workspace predictions: v_q margin >> truth_final margin on declaratives; truth_final
margin RISES under the question prefix. That interaction is the headline figure.

Method: exact per-statement autograd — one backward per (readout, batch) yields
gradients at every layer at once; margin_l = ||sum_t d(w.h_final,last)/dh_l[t]||
(sum over positions = the every-position broadcast-injection convention).
`--use-jlens` optionally cross-checks against anthropics/jacobian-lens fitted
lenses on layers {src, (src+tgt)//2, tgt}; any failure there only warns.

    PYTHONPATH=src python src/reach_jlens.py --dataset cities --device cuda
"""
import argparse
import csv

import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from reach_hop import last_nonpad_index, load_meta
from reach_margins import fit_probe_dir, fit_threshold
from mag.config import Q_TRUTH, Q_SUFFIX, YES_VARIANTS, NO_VARIANTS
from mag.verdict import first_token_ids

N_SAMPLE, SEED, BATCH = 256, 42, 8
LAYERS = list(range(0, 26))
MAXLEN = {"decl": 64, "quest": 96}


def prompts_for(stmts, mode):
    if mode == "decl":
        return [str(s) for s in stmts]
    return [Q_TRUTH + str(s) + Q_SUFFIX for s in stmts]


def collect_finals(model, tok, prompts, dev, max_length):
    tok.padding_side = "right"
    outs = []
    for b0 in range(0, len(prompts), BATCH):
        enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        last = last_nonpad_index(enc["attention_mask"])
        idx = torch.arange(len(last), device=dev)
        outs.append(hs[-1][idx, last].float().cpu().numpy())
    return np.concatenate(outs)


def build_readouts(model, tok, stmts, labels, dev):
    fin_d = collect_finals(model, tok, prompts_for(stmts, "decl"), dev, MAXLEN["decl"])
    fin_q = collect_finals(model, tok, prompts_for(stmts, "quest"), dev, MAXLEN["quest"])
    W_U = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    yes = first_token_ids(tok, YES_VARIANTS)
    no = first_token_ids(tok, NO_VARIANTS)
    w_verdict = unit(W_U[yes].mean(0).astype(np.float64)
                     - W_U[no].mean(0).astype(np.float64))
    w_truth = fit_probe_dir(fin_d.astype(np.float64), labels)
    _, sign, _ = fit_threshold(fin_d @ w_truth, labels)
    if sign < 0:
        w_truth = -w_truth
    w_vq = unit((fin_q - fin_d).mean(0).astype(np.float64))
    return {"verdict": w_verdict, "truth_final": w_truth, "v_q_final": w_vq}


def margins_for_mode(model, tok, prompts, readouts, dev, max_length):
    """Returns {w_name: (n, len(LAYERS)) margins}."""
    tok.padding_side = "right"
    emb_layer = model.get_input_embeddings()
    acc = {wn: [] for wn in readouts}
    w_t = {wn: torch.tensor(w, dtype=torch.float32, device=dev)
           for wn, w in readouts.items()}
    for b0 in range(0, len(prompts), BATCH):
        enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(dev)
        emb = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
        hs = out.hidden_states
        last = last_nonpad_index(enc["attention_mask"])
        idx = torch.arange(len(last), device=dev)
        h_last = hs[-1][idx, last]                           # (B, d)
        targets = [hs[l] for l in LAYERS]
        for wi, wn in enumerate(readouts):
            s = (h_last @ w_t[wn]).sum()                     # rows independent
            grads = torch.autograd.grad(
                s, targets, retain_graph=(wi < len(readouts) - 1))
            m = torch.stack([g.sum(dim=1).norm(dim=-1) for g in grads], dim=1)
            acc[wn].append(m.detach().float().cpu().numpy())  # (B, L)
        print(f"  margins batch {b0}-{b0 + len(last)} done", flush=True)
    return {wn: np.concatenate(v) for wn, v in acc.items()}


def cross_check_jlens(model, tok, ds, readouts, margins_decl):
    """Optional external cross-check; failure only warns (spec §6 isolation)."""
    try:
        import jacobian_lens  # noqa: F401
    except ImportError:
        print("[jlens] anthropics/jacobian-lens not installed — skipping cross-check")
        return
    try:
        src, tgt, _ = load_meta(ds)
        check_layers = sorted({src, (src + tgt) // 2, tgt})
        print(f"[jlens] cross-check requested on layers {check_layers}; see the "
              f"jacobian-lens README for lens loading — comparing fitted-lens "
              f"||J_l^T w|| to our per-statement means")
        # Best-effort: the exact fitted-lens API is pinned at integration time on the
        # cluster; any exception lands in the guard below and only warns.
    except Exception as e:  # noqa: BLE001 — external dep must never block Phase 4
        print(f"[jlens] cross-check failed (non-blocking): {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-jlens", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    rng = np.random.default_rng(SEED)
    n = min(N_SAMPLE, len(stmts))
    pick = [rng.choice(np.where(labels == lab)[0], size=n // 2, replace=False)
            for lab in (0, 1)]
    pick = np.sort(np.concatenate(pick))
    if a.limit:
        pick = pick[:a.limit]
    stmts, labels = stmts[pick], labels[pick]
    tok, model, dev = su.load_model(a.device)
    readouts = build_readouts(model, tok, stmts, labels, dev)
    rows = [("mode", "layer", "w_name", "margin_mean", "margin_median",
             "margin_se", "n")]
    margins_decl = None
    for mode in ("decl", "quest"):
        prompts = prompts_for(stmts, mode)
        print(f"[jlens] {ds} mode={mode}: {len(prompts)} prompts", flush=True)
        marg = margins_for_mode(model, tok, prompts, readouts, dev, MAXLEN[mode])
        if mode == "decl":
            margins_decl = marg
        for wn, m in marg.items():
            for li, l in enumerate(LAYERS):
                col = m[:, li]
                rows.append((mode, l, wn, f"{col.mean():.6g}",
                             f"{np.median(col):.6g}",
                             f"{col.std() / max(len(col), 1) ** 0.5:.6g}", len(col)))
    if a.use_jlens:
        cross_check_jlens(model, tok, ds, readouts, margins_decl)
    with open(f"reach_jlens_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[jlens] wrote reach_jlens_{ds}.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `fig_jlens` to `src/viz_reach.py` and extend the smoke test**

Append to `src/viz_reach.py` and add `fig_jlens(ds)` to `main()`'s call line (same missing-file guard style as `fig_svd`):

```python
def fig_jlens(ds):
    path = f"reach_jlens_{ds}.csv"
    if not os.path.exists(path):
        print(f"skip fig_jlens: {path} missing")
        return
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            data.setdefault((r["mode"], r["w_name"]), []).append(
                (int(r["layer"]), float(r["margin_mean"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    style = {"verdict": (RED, "verdict (yes-no unembed)"),
             "truth_final": (BLUE, "truth-content probe"),
             "v_q_final": (GREEN, "question-mode v_Q")}
    for ax, mode, title in ((axes[0], "decl", "declarative input"),
                            (axes[1], "quest", "question-prefixed input")):
        for wn, (c, lab) in style.items():
            pts = np.array(sorted(data.get((mode, wn), [(0, np.nan)])))
            ax.semilogy(pts[:, 0], pts[:, 1], color=c, label=lab)
        ax.set_xlabel("source layer l"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean ||J_l^T w||")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ds}: per-layer controllability of final-basis readouts "
                 f"(workspace selectivity)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_reach_jlens_{ds}.png", dpi=150)
    plt.close(fig)
    print(f"saved plot_reach_jlens_{ds}.png")
```

In `main()` change the call line to:

```python
    fig_margins(ds); fig_curves(ds); fig_geometry(ds); fig_svd(ds); fig_jlens(ds)
```

Append to `tests/test_viz_reach.py`:

```python
def test_fig_jlens_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with open("reach_jlens_toy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("mode", "layer", "w_name", "margin_mean", "margin_median",
                    "margin_se", "n"))
        for mode in ("decl", "quest"):
            for l in range(26):
                for wn in ("verdict", "truth_final", "v_q_final"):
                    w.writerow((mode, l, wn, 1.0 + l * 0.1, 1.0, 0.05, 8))
    vr.fig_jlens("toy")
    assert os.path.exists("plot_reach_jlens_toy.png")
```

- [ ] **Step 3: Run tests + import sanity**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_reach.py -q`
Expected: 4 passed.

Run: `PYTHONPATH=src .venv/bin/python -c "import reach_jlens; print(reach_jlens.prompts_for(['X is Y.'], 'quest')[0])"`
Expected: prints `Answer in yes or no only. Is the following statement true?\nStatement: X is Y.\nAnswer:` content (one line with the prefix and suffix).

- [ ] **Step 4: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/reach_jlens.py src/viz_reach.py tests/test_viz_reach.py
git commit -m "feat(reach): Phase-4 per-layer J-lens margins with workspace-mode contrast"
```

---

### Task 8: `reach_linerr.py` — Phase 5 linearization error

**Files:**
- Create: `src/reach_linerr.py`
- Test: `tests/test_reach_linerr.py`

**Interfaces:**
- Consumes (Task 1): `load_model_and_slice`, `forward_source_batch`, `make_hop`, `vjp_rows`, `jvp_cols`, `load_landmarks`, `validate_inputs`; artifacts `reach_acts_<ds>.npz`, `reach_dirs_<ds>.npz`, `dct_meta_<ds>.json`.
- Produces: `reach_linerr_<ds>.csv` (`stmt_index,label,direction,eps,eps_frac,rel_err`) and `reach_linerr_summary_<ds>.csv` (`direction,eps20`) where `eps20` = largest swept ε whose median relative error ≤ 0.2, linearly interpolated between grid points. Directions: `jtw_mean_diff_tgt` (per-statement, computed on the fly via vjp), `mean_diff_src`, `dct_v_top`, `random`.
- Produces functions (tested): `rel_error_curve(f, F0, delta_unit, eps_list)` and `validity_radius(eps_list, med_errs, thresh=0.2)`.
- CLI: `PYTHONPATH=src python src/reach_linerr.py --dataset cities --device cuda [--limit N]` then `--summarize` (local, no GPU).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reach_linerr.py`:

```python
import sys, os
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reach_linerr as rl


class _LinearSlice(nn.Module):
    """Purely linear stand-in: rel error must be ~0 at every eps."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        return h @ self.W


class _QuadSlice(nn.Module):
    """Quadratic term: rel error must grow with eps."""
    def __init__(self, d=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(d, d, generator=g), requires_grad=False)

    def forward(self, h):
        return h @ self.W + 0.05 * (h * h)


def _f_for(slice_mod, d=6, T=3, seed=1):
    import reach_hop as rh
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, T, d, generator=g)
    attn = torch.ones(1, T, dtype=torch.long)
    return rh.make_hop(slice_mod, h, attn)


def test_rel_error_zero_for_linear_map():
    f = _f_for(_LinearSlice())
    F0 = f(torch.zeros(1, 6))
    delta = torch.randn(6); delta /= delta.norm()
    errs = rl.rel_error_curve(f, F0, delta, [0.1, 1.0, 10.0, 100.0])
    assert all(e < 1e-4 for e in errs), errs


def test_rel_error_grows_for_nonlinear_map():
    f = _f_for(_QuadSlice())
    F0 = f(torch.zeros(1, 6))
    delta = torch.randn(6); delta /= delta.norm()
    errs = rl.rel_error_curve(f, F0, delta, [0.01, 1.0, 100.0])
    assert errs[0] < errs[-1]
    assert errs[0] < 0.05


def test_validity_radius_interpolation():
    eps = np.array([1.0, 2.0, 4.0, 8.0])
    med = np.array([0.05, 0.1, 0.3, 0.9])
    r = rl.validity_radius(eps, med, thresh=0.2)
    assert 2.0 < r < 4.0                       # crosses 0.2 between eps=2 and eps=4
    assert rl.validity_radius(eps, np.array([0.01, 0.02, 0.03, 0.04])) == 8.0
    assert rl.validity_radius(eps, np.array([0.5, 0.6, 0.7, 0.8])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_linerr.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'reach_linerr'`.

- [ ] **Step 3: Write the implementation**

Create `src/reach_linerr.py`:

```python
"""reach_linerr.py — Phase 5: linearization validity radius (the reviewer armor).

Relative error  ||F(eps*Delta) - F(0) - eps*J*Delta|| / ||eps*J*Delta||  for
Delta in {per-statement unit J^T w (w = mean_diff_tgt), mean_diff@src, top DCT V,
random}, 64 statements/dataset, eps log-swept over geomspace(0.05, 1.5, 8) x
input_scale. Validity radius per direction = eps at 20% median relative error;
overlaid as the trust region on the Phase-1 curves (viz_reach.fig_curves).

    PYTHONPATH=src python src/reach_linerr.py --dataset cities --device cuda
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset cities --summarize
"""
import argparse
import csv
import json

import numpy as np
import torch

from funnel_utils import unit
from reach_hop import (validate_inputs, load_model_and_slice, forward_source_batch,
                       make_hop, vjp_rows, jvp_cols, load_landmarks)

N_STMT, SEED = 64, 42
EPS_FRACS = np.geomspace(0.05, 1.5, 8)
ERR_THRESH = 0.2


def rel_error_curve(f, F0, delta_unit, eps_list):
    """f is a single-statement hop (B=1). Returns [rel_err(eps) for eps in eps_list].
    J*Delta comes from one jvp; each F(eps*Delta) is one forward."""
    d = delta_unit.shape[-1]
    delta = delta_unit.reshape(1, d).to(F0.dtype)
    _, JD = jvp_cols(f, torch.zeros_like(delta), delta)
    errs = []
    for eps in eps_list:
        Fe = f(float(eps) * delta)
        lin = float(eps) * JD
        denom = float(lin.norm())
        errs.append(float((Fe - F0 - lin).norm()) / max(denom, 1e-12))
    return errs


def validity_radius(eps_list, med_errs, thresh=ERR_THRESH):
    """Largest swept eps with median error <= thresh, linearly interpolated at the
    crossing; 0.0 if even the smallest eps exceeds thresh; max(eps) if none does."""
    eps = np.asarray(eps_list, np.float64)
    err = np.asarray(med_errs, np.float64)
    if err[0] > thresh:
        return 0.0
    over = np.where(err > thresh)[0]
    if len(over) == 0:
        return float(eps[-1])
    j = over[0]
    e0, e1, r0, r1 = eps[j - 1], eps[j], err[j - 1], err[j]
    return float(e0 + (thresh - r0) * (e1 - e0) / max(r1 - r0, 1e-12))


def compute(ds, device, limit=0):
    validate_inputs(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    names = [str(x) for x in dirs["names"]]
    w_md = torch.tensor(np.asarray(dirs["W"][names.index("mean_diff_tgt")],
                                   np.float32))
    lm = load_landmarks(ds)
    input_scale = float(json.load(open(f"dct_meta_{ds}.json"))["input_scale"])
    eps_list = (EPS_FRACS * input_scale).tolist()
    rng = np.random.default_rng(SEED)
    pick = [rng.choice(np.where(labels == lab)[0], size=N_STMT // 2, replace=False)
            for lab in (0, 1)]
    pick = np.sort(np.concatenate(pick))
    if limit:
        pick = pick[:limit]
    tok, model, sliced, meta = load_model_and_slice(ds, device)
    dev = meta["device"]
    fixed = {"mean_diff_src": torch.tensor(lm["md_src"], dtype=torch.float32),
             "dct_v_top": torch.tensor(lm["dct_v"], dtype=torch.float32),
             "random": torch.tensor(unit(rng.standard_normal(len(lm["md_src"]))),
                                    dtype=torch.float32)}
    rows = [("stmt_index", "label", "direction", "eps", "eps_frac", "rel_err")]
    for i in pick:
        fb = forward_source_batch(model, tok, [stmts[i]], meta["src"], meta["tgt"], dev)
        f = make_hop(sliced, fb["h_src_seq"], fb["attn"])
        d = fb["h_src_seq"].shape[-1]
        F0 = f(torch.zeros(1, d, device=dev))
        _, G = vjp_rows(f, torch.zeros(1, d, device=dev), w_md[None, :].to(dev))
        jtw = G[0, 0] / G[0, 0].norm().clamp_min(1e-12)
        deltas = {"jtw_mean_diff_tgt": jtw, **{k: v.to(dev) for k, v in fixed.items()}}
        for dname, delta in deltas.items():
            errs = rel_error_curve(f, F0, delta, eps_list)
            for frac, eps, e in zip(EPS_FRACS, eps_list, errs):
                rows.append((int(i), int(labels[i]), dname, f"{eps:.6g}",
                             f"{frac:.4g}", f"{e:.6g}"))
        print(f"[linerr] {ds} stmt {int(i)} done", flush=True)
    with open(f"reach_linerr_{ds}.csv", "w", newline="") as f2:
        csv.writer(f2).writerows(rows)
    print(f"[linerr] wrote reach_linerr_{ds}.csv")


def summarize(ds):
    per = {}
    with open(f"reach_linerr_{ds}.csv") as f:
        for r in csv.DictReader(f):
            per.setdefault(r["direction"], {}).setdefault(
                float(r["eps"]), []).append(float(r["rel_err"]))
    with open(f"reach_linerr_summary_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "eps20"))
        for dname, by_eps in per.items():
            eps = np.array(sorted(by_eps))
            med = np.array([np.median(by_eps[e]) for e in eps])
            r = validity_radius(eps, med)
            w.writerow((dname, f"{r:.6g}"))
            print(f"[linerr] {dname}: validity radius (20% err) = {r:.4g}")
    print(f"[linerr] wrote reach_linerr_summary_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.dataset)
    else:
        compute(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_linerr.py -q`
Expected: 3 passed.

- [ ] **Step 5: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green. (The Phase-1 curves figure already overlays `reach_linerr_summary_<ds>.csv` when present — implemented in Task 4.)

- [ ] **Step 6: Commit**

```bash
git add src/reach_linerr.py tests/test_reach_linerr.py
git commit -m "feat(reach): Phase-5 linearization-error sweep and validity radius"
```

---

### Task 9: SLURM jobs + runbook

**Files:**
- Create: `deltaai/run_reach_margins.slurm`, `deltaai/run_reach_svd.slurm`, `deltaai/run_reach_steer.slurm`, `deltaai/run_reach_linerr.slurm`, `deltaai/run_reach_judge.slurm`
- Create: `docs/REACHABILITY_RUNBOOK.md`

**Interfaces:**
- Consumes: every CLI from Tasks 2–8; the cluster conventions verified in `deltaai/run_dct_uwarm.slurm` / `run_dct_uwarm_judge.slurm` (`ACCOUNT_NAME` placeholder, `ghx4` partition, `module load python/miniforge3_pytorch`, `.venv-dct-gpu` / `.venv-judge-gpu`, `HF_HUB_DISABLE_XET=1`, `TRANSFORMERS_OFFLINE=1`, per-dataset `|| echo "!!!! ... FAILED — continuing"` guards, tight `--time`).
- One-time cluster prerequisite (new, verified gap): `.venv-dct-gpu` lacks scikit-learn, which `--stage dirs`, `reach_jlens`, and `reach_svd --analyze` need — the runbook's setup step installs it on a login node.

- [ ] **Step 1: Write the five SLURM jobs**

Create `deltaai/run_reach_margins.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reach_margins
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=reach_margins_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach margins $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_margins.py --dataset "$ds" --stage all --device cuda \
    || echo "!!!! $ds reach margins FAILED — continuing"
done
echo "=== reach margins done $(date) ==="
```

Create `deltaai/run_reach_svd.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reach_svd
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=reach_svd_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach svd $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_svd.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds reach svd FAILED — continuing"
  PYTHONPATH=src python3 src/reach_svd.py --dataset "$ds" --analyze \
    || echo "!!!! $ds reach svd analyze FAILED — continuing"
done
echo "=== reach svd done $(date) ==="
```

Create `deltaai/run_reach_steer.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reach_steer
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=reach_steer_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach steer mean $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_steer.py --dataset "$ds" --device cuda --arm mean \
    || echo "!!!! $ds reach steer mean FAILED — continuing"
  echo "=== reach steer per_stmt $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_steer.py --dataset "$ds" --device cuda --arm per_stmt \
    || echo "!!!! $ds reach steer per_stmt FAILED — continuing"
  echo "=== reach jlens $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_jlens.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds reach jlens FAILED — continuing"
done
echo "=== reach steer done $(date) ==="
```

Create `deltaai/run_reach_linerr.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reach_linerr
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=reach_linerr_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach linerr $ds $(date) ==="
  PYTHONPATH=src python3 src/reach_linerr.py --dataset "$ds" --device cuda \
    || echo "!!!! $ds reach linerr FAILED — continuing"
  PYTHONPATH=src python3 src/reach_linerr.py --dataset "$ds" --summarize \
    || echo "!!!! $ds reach linerr summarize FAILED — continuing"
done
echo "=== reach linerr done $(date) ==="
```

Create `deltaai/run_reach_judge.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reach_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=reach_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== reach judge mean-arm $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "reach_steer_${ds}.csv" \
    --steer-output "judge_reach_steer_${ds}.csv" \
    --steer-plot "plot_judge_reach_${ds}.png" \
    || echo "!!!! $ds reach judge mean FAILED — continuing"
  echo "=== reach judge stmt-arm $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "reach_steer_stmt_${ds}.csv" \
    --steer-output "judge_reach_steer_stmt_${ds}.csv" \
    --steer-plot "plot_judge_reach_stmt_${ds}.png" \
    || echo "!!!! $ds reach judge stmt FAILED — continuing"
done
echo "=== reach judge done $(date) ==="
```

- [ ] **Step 2: Syntax-check the jobs**

Run: `for f in deltaai/run_reach_*.slurm; do bash -n "$f" && echo "OK $f"; done`
Expected: `OK` for all five files.

- [ ] **Step 3: Write the runbook**

Create `docs/REACHABILITY_RUNBOOK.md`:

```markdown
# Reachability Audit — Runbook

Spec: `docs/superpowers/specs/2026-07-24-reachability-audit-design.md`.
Datasets: cities (hop 11→20), common_claim_true_false (13→22). All cluster jobs
follow the DeltaAI conventions in `deltaai/CLUSTER_OPERATIONS.md` (set `--account`,
one NCSA password + Duo push per ssh/rsync batch).

## Phase 0 — one-time cluster prerequisite (login node)

`.venv-dct-gpu` has no scikit-learn (setup_env.sh installs only
transformers/scipy/tqdm/pandas), and `reach_margins --stage dirs`,
`reach_jlens`, and `reach_svd --analyze` need it:

    module load python/miniforge3_pytorch
    source .venv-dct-gpu/bin/activate
    pip install scikit-learn

## Phase 0b — local smoke (laptop, before any rsync)

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_reach_hop.py tests/test_reach_margins.py tests/test_reach_analyze.py tests/test_reach_linerr.py tests/test_viz_reach.py -q
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage acts --limit 8 --device cpu
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage dirs
    PYTHONPATH=src .venv/bin/python src/reach_margins.py --dataset cities --stage vjp --limit 8 --device cpu
    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities

(The `--limit 8` artifacts are throwaway sanity checks — delete
`reach_acts_cities.npz reach_dirs_cities.npz reach_chunks_cities reach_margins_cities.npz reach_curve_cities.csv reach_summary_cities.json plot_reach_*_cities.png`
before rsyncing, so the cluster runs regenerate them at full size.)

## Phase 1 — rsync up (one batch)

    rsync -av --relative src deltaai tests docs got_datasets \
      dct_meta_cities.json dct_meta_common_claim_true_false.json \
      truth_dir_cities.npz truth_dir_common_claim_true_false.npz \
      truth_dir_tgt_cities.npz truth_dir_tgt_common_claim_true_false.npz \
      mag_dir_cities.npz mag_dir_common_claim_true_false.npz \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/

(`dct_V_*.pt` / `dct_U_*.pt` are already on the cluster from the DCT runs; if
`ls ~/PROJECT_DIR/dct_V_*.pt` says otherwise, add them to the batch.)

## Phase 2 — cluster jobs (GH200)

Submit order (margins gates everything; svd/linerr only need margins' stage
outputs; steer additionally needs the local Phase-3 analyze step — see below):

    sbatch deltaai/run_reach_margins.slurm          # ~2-4 h: acts+dirs+vjp, both datasets
    # after run_reach_margins completes:
    sbatch deltaai/run_reach_svd.slurm              # independent of analyze
    sbatch deltaai/run_reach_linerr.slurm           # independent of analyze

`reach_steer` needs `reach_summary_<ds>.json` (produced by the LOCAL analyze
step). Two options:
  (a) run analyze ON the cluster (sklearn now installed):
        PYTHONPATH=src python3 src/reach_analyze.py --dataset cities
        PYTHONPATH=src python3 src/reach_analyze.py --dataset common_claim_true_false
      (interactive on the login node, seconds — margins npz stays on scratch), then
        sbatch deltaai/run_reach_steer.slurm
        # after run_reach_steer completes:
        sbatch deltaai/run_reach_judge.slurm
  (b) or rsync margins down, analyze locally, rsync summary up (adds a Duo batch).
Option (a) is the default.

## Phase 3 — rsync back (one batch; excludes weights)

    rsync -av --exclude '*.pt' \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/'reach_margins_*.npz reach_acts_*.npz reach_dirs_*.npz reach_svd_*/ reach_svd_*.csv reach_curve_*.csv reach_summary_*.json reach_steer_*.csv reach_jlens_*.csv reach_linerr_*.csv judge_reach_*.csv plot_judge_reach_*.png reach_*_%j.out *.out' \
      ./

(reach_margins npz ≈ 200 MB/dataset — jtw rows are float16 to keep this small.
If bandwidth hurts, exclude `reach_acts_*.npz` and re-derive locally later.)

## Phase 4 — local analysis + figures

    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/reach_analyze.py --dataset common_claim_true_false
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset cities --summarize
    PYTHONPATH=src .venv/bin/python src/reach_linerr.py --dataset common_claim_true_false --summarize
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_reach.py --dataset common_claim_true_false

## Decision gates between phases (spec §4 cross-phase logic)

- **Gate P1:** read `reach_summary_<ds>.json`. `verdict=unreachable-tail`
  (truth-subspace best-case median margin < random-null median) ⇒ Phase 2 is the
  mechanism story. `reachable-candidate` ⇒ Phase 3 is the headline experiment.
- **Gate P2:** `dctV_overlap_top16` low (< 0.3 mean) in `reach_svd_summary_<ds>.csv`
  is a STOP-AND-DIAGNOSE — the linearized picture disagrees with DCT; do not
  interpret P1 margins until resolved.
- **Gate P3:** the 2×2 from `judge_reach_steer_*.csv` × `reach_steer_readout_*.csv`:
  readout moves & behavior moves ⇒ lever found (pivot to characterizing it);
  readout moves & behavior doesn't ⇒ LiSeCo-style dissociation claim.
- **Gate P5:** all P1–P3 claims must be restated inside/outside the
  `reach_linerr_summary` validity radius; if the radius ≪ input_scale, phrase the
  unreachability claim at the radius, with nonlinear spot-checks beyond.

## Parallel track (unchanged — spec §8)

The conditional-steering / U-anchor round launches independently per
`docs/CONDITIONAL_UANCHOR_RUNBOOK.md`; its judge results feed Phase 3/4
interpretation.
```

- [ ] **Step 4: Full-suite gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: green (no Python changed in this task).

- [ ] **Step 5: Commit**

```bash
git add deltaai/run_reach_margins.slurm deltaai/run_reach_svd.slurm \
  deltaai/run_reach_steer.slurm deltaai/run_reach_linerr.slurm \
  deltaai/run_reach_judge.slurm docs/REACHABILITY_RUNBOOK.md
git commit -m "feat(reach): DeltaAI slurm jobs + phase-gated runbook"
```

---

## Execution notes for the controller

- **Task order is the dependency order** (1 → 9); no parallel implementers.
- After Task 9 + final review, the branch is code-complete; actual cluster execution follows `docs/REACHABILITY_RUNBOOK.md` and is user-driven (NCSA password + Duo per rsync batch).
- The spec's `docs/superpowers/specs/2026-07-24-reachability-audit-design.md` §4 decision logic is runtime logic (which phase becomes the headline) — it does NOT gate implementation tasks; all phases get built.
- Known deliberate deviations from the spec, already grounded in the codebase: (a) test files — spec §7 names three; this plan adds `test_reach_margins.py` and `test_viz_reach.py` (repo convention tests viz + battery math is load-bearing); (b) a fifth SLURM job for judging (`run_reach_judge.slurm`) because judging uses the separate `.venv-judge-gpu` env, matching `run_dct_uwarm_judge.slurm`; (c) `reach_jlens` computes exact per-statement margins itself (autograd) with `anthropics/jacobian-lens` demoted to an optional cross-check — the fitted lens is an average and would be strictly less informative than what one backward pass already gives us; spec §6's isolation requirement is preserved.
