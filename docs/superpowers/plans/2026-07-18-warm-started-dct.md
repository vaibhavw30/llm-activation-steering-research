# Warm-Started / Semi-Supervised DCT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seed DCT's causal gradient search at the supervised truth axis with a soft anchor and test whether the refined direction becomes a causal truth-lever where raw `mean_diff` is inert; audit whether cold DCT's top factor is truth-aligned.

**Architecture:** A small, non-invasive addition to `src/dct.py` gives `ExponentialDCT.fit` a `init="warm"` path (seed factor 0 at a supervised axis) plus an anchor term in its power-iteration update. A training wrapper (`dct_warm.py`) clones the existing `run_dct_data.py` extraction/calibration and drives the warm fits (2 seeds × 3 λ per dataset), reusing the cold run's layers and `input_scale` from `dct_meta_<ds>.json`. A model-free assembly script gathers the candidate directions (raw supervised axes, warm factor-0s, cold-top-by-potency) and their geometry. A steering script injects each at the DCT source layer with a single shared norm and writes a judge-schema CSV, so the existing OLMo judge scores it unchanged. A plotting script produces the drift, verdict, and audit figures. Runs on the GH200.

**Tech Stack:** Python 3.13, PyTorch (fp32, eager attention), transformers, numpy, pandas, matplotlib (Agg). Model `google/gemma-2-2b` (base). OLMo judge via `judge_results.py --backend olmo`.

## Global Constraints

- The existing cold-start artifacts are authoritative for config: `dct_meta_<ds>.json` gives `source_layer` (11 cities / 13 common_claim), `target_layer` (20 / 22), `input_scale` (≈47.716 / ≈86.733), `num_iters` (30), `num_samples` (64), `token_idxs` (`-3:`). Warm runs MUST read these and match them (pass `--scale <input_scale>` to skip recalibration), changing only `init`, the seed, `anchor_lambda`, and `num_factors` (64 for warm).
- The supervised seed is `truth_dir_<ds>.npz["mean_diff"]` / `["grad"]`. These are **already at the DCT source layer** (`truth_dir.layer == dct_meta.source_layer`, verified: 11/13); the code MUST assert this equality and use them directly — no recompute from `mag_acts` needed (recompute is only the fallback if the assert fails).
- DCT `V`/`U` are saved **unit-normalized** (columns unit norm), matching `run_dct_data.py`'s save. Warm factor-0 is column index 0 of the saved `V`.
- Behavioral injection: `τ · input_scale · unit(dir)` at `source_layer` via `dct_steer_utils.Steerer`, one shared `input_scale` per dataset for every direction. Sign convention: `mean_diff` is stored as `unit(mean(true) − mean(false))` (points toward TRUE, matching `funnel_utils.mean_diff_dir` and how `src/mag/steer.py` loads it unnegated); all candidate directions are sign-aligned to `mean_diff` (→ TRUE), so **+τ pushes toward TRUE and −τ pushes toward FALSE (lying)**.
- Taus: `{-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0}` — two-sided (consistent with the length-steering project) so the run probes both toward-FALSE (−τ, where the causal truth→false flip is observable) and toward-TRUE (+τ); a one-sided +τ push toward TRUE on already-truthful factual prompts could never reveal the lever. τ=0 ⇒ no vector injected. Seeds: `mean_diff`, `grad` (separate warm runs — never seeded into one pool).
- Anchor is **scale-relative**: each iteration `V0 += λ·‖G_V[:,0]‖·seed`, so `λ` is a scale-free fraction of the update magnitude and transfers across datasets (an absolute `λ·seed` is negligible against the real gradient, whose column norm ‖G_V[:,0]‖≈5–6 at model scale — measured — so an absolute λ∈{0,0.1,0.3} gives ~full drift, cos≈0.05). Lambdas: `{0.0, 0.3, 1.0, 3.0}` — with the relative anchor these give factor-0-vs-seed cos ≈ `{0, 0.29, 0.71, 0.95}` (cos ≈ λ/√(1+λ²) when seed ⟂ G_V), spanning free-drift → strongly-anchored. `0.0` = init-only free-drift ablation.
- Datasets: `cities`, `common_claim_true_false`.
- The steer CSV MUST use header `direction,scale,prompt,completion` (with `scale = τ`) so `judge_results.py --mode steer` reads it unchanged.
- Tests run with `PYTHONPATH=src .venv/bin/python -m pytest`; the `dct.py` mechanism test uses a model-free linear stub (no gemma load).

---

### Task 1: `init="warm"` + soft anchor in `dct.py`

**Files:**
- Modify: `src/dct.py` — add two module-level helpers; extend `ExponentialDCT.fit` (dct.py:633), the init assert (dct.py:671), the init dispatch (dct.py:689-693), and the V-update (dct.py:790).
- Test: `tests/test_dct_warm_mech.py`

**Interfaces:**
- Produces:
  - `warm_init_column(V, seed) -> Tensor` — returns `V` with column 0 replaced by `F.normalize(seed, dim=0)` (other columns untouched).
  - `anchor_step(V_update, anchor, lam) -> Tensor` — returns `V_update` with column 0 incremented by `lam * ‖V_update[:,0]‖ * anchor` (scale-relative: the pull is `lam` times the column's own magnitude, so `lam` is a scale-free fraction; no-op when `lam == 0`).
  - `ExponentialDCT.fit(..., init="warm", warm_seed=<Tensor d_source>, anchor_lambda=<float>)` — seeds factor 0 and anchors it each iteration; returns `(U, V)` as before with `V[:,0]` aligned to the seed direction (up to the QR sign, which downstream extraction re-aligns).

- [ ] **Step 1: Write the failing mechanism tests**

```python
# tests/test_dct_warm_mech.py
import sys, os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct


def test_warm_init_column_sets_unit_seed():
    V = torch.randn(5, 3)
    seed = torch.tensor([0.0, 3.0, 0.0, 4.0, 0.0])   # norm 5
    out = dct.warm_init_column(V.clone(), seed)
    assert torch.allclose(out[:, 0].norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[:, 0], seed / 5.0, atol=1e-6)
    assert torch.allclose(out[:, 1:], V[:, 1:])          # other columns untouched


def test_anchor_step_scales_col0_by_its_norm():
    V_update = torch.zeros(4, 2)
    V_update[:, 0] = torch.tensor([3.0, 4.0, 0.0, 0.0])   # col0 norm 5
    anchor = torch.tensor([0.0, 0.0, 1.0, 0.0])           # unit, orthogonal to col0
    out = dct.anchor_step(V_update.clone(), anchor, 0.5)
    # col0 += 0.5 * ||col0||(=5) * anchor  =>  +2.5 in dim 2
    assert torch.allclose(out[:, 0], torch.tensor([3.0, 4.0, 2.5, 0.0]))
    assert torch.allclose(out[:, 1], torch.zeros(4))      # other columns untouched
    # lam=0 is a no-op
    assert torch.allclose(dct.anchor_step(V_update.clone(), anchor, 0.0), V_update)


class _LinearDelta(nn.Module):
    """Model-free DeltaActivations stub: delta(theta) = (x+theta)Wᵀ - y, averaged over seq."""
    def __init__(self, W):
        super().__init__()
        self.W = W
        self.device = W.device
        self.attention_mask = None

    def forward(self, theta, x, y, attention_mask=None):
        steered = (x + theta) @ self.W.T
        return (steered - y).mean(dim=1)


def test_fit_warm_keeps_factor0_near_seed_when_anchored():
    torch.manual_seed(0)
    d_src, d_tgt, n, seq = 6, 6, 4, 3
    W = torch.eye(d_tgt, d_src)
    da = _LinearDelta(W)
    X = torch.randn(n, seq, d_src)
    Y = torch.randn(n, seq, d_tgt)
    seed = torch.zeros(d_src); seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    U, V = m.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="warm",
                 warm_seed=seed, anchor_lambda=5.0, input_scale=1.0, max_iters=5,
                 orthogonalize=True)
    cos = abs(float(torch.dot(V[:, 0] / V[:, 0].norm(), seed)))
    assert cos > 0.9         # strong anchor pins factor 0 to the seed direction
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_mech.py -q`
Expected: FAIL — `AttributeError: module 'dct' has no attribute 'warm_init_column'`.

- [ ] **Step 3: Add the two helpers to `dct.py`**

Insert after `soft_ortho` (before `class SlicedModel`, ~dct.py:188):

```python
def warm_init_column(V, seed):
    """Replace column 0 of V with the unit-normalized seed (other columns untouched)."""
    V = V.clone()
    V[:, 0] = F.normalize(seed.to(V.device).to(V.dtype), dim=0)
    return V


def anchor_step(V_update, anchor, lam):
    """Pull column 0 of a V-update toward `anchor` by rate `lam`, scaled to the column's own
    magnitude so `lam` is a scale-free fraction of the update (no-op when lam == 0)."""
    if lam and lam > 0.0:
        V_update = V_update.clone()
        anchor = anchor.to(V_update.device).to(V_update.dtype)
        col_norm = torch.norm(V_update[:, 0])
        V_update[:, 0] = V_update[:, 0] + lam * col_norm * anchor
    return V_update
```

- [ ] **Step 4: Extend `ExponentialDCT.fit`**

At the assert (dct.py:671), allow `"warm"`:

```python
    assert(init in ["random", "rand_backward", "warm"])
```

Add `warm_seed=None, anchor_lambda=0.0` to the `fit` signature (dct.py:633-635 parameter list). After `self.beta = beta` (dct.py:679) add:

```python
        self.anchor_lambda = float(anchor_lambda)
        self.V_anchor = None
```

In the init dispatch (dct.py:689-693), add the warm branch:

```python
        elif init == "warm":
            assert warm_seed is not None, "init='warm' requires warm_seed"
            self._init_rand(delta_acts, X, Y, attention_mask)     # random V, U (free factors)
            self.V = warm_init_column(self.V, torch.as_tensor(warm_seed, device=self.device))
            self.V_anchor = self.V[:, 0].detach().clone()
```

In the update (dct.py:790), replace the single `self.V.data = F.normalize(...)` line with:

```python
                V_update = self.beta * G_V + (1 - self.beta) * self.V.data
                if self.anchor_lambda > 0.0 and self.V_anchor is not None:
                    V_update = anchor_step(V_update, self.V_anchor, self.anchor_lambda)
                self.V.data = F.normalize(V_update, dim=0)
```

(Leave the `deflation` branch below it unchanged.)

- [ ] **Step 5: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_mech.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Confirm no regression to existing init modes**

Run: `PYTHONPATH=src .venv/bin/python -c "import dct, torch, torch.nn as nn
W=torch.eye(6); 
class D(nn.Module):
 def __init__(s): super().__init__(); s.W=W; s.device=W.device; s.attention_mask=None
 def forward(s,t,x,y,m=None): return ((x+t)@s.W.T - y).mean(1)
X=torch.randn(4,3,6); Y=torch.randn(4,3,6)
U,V=dct.ExponentialDCT(num_factors=4).fit(D(),X,Y,init='random',max_iters=3)
print('random init OK', V.shape)"`
Expected: prints `random init OK torch.Size([6, 4])` — the default path still works with `anchor_lambda` defaulted to 0.

- [ ] **Step 7: Commit**

```bash
git add src/dct.py tests/test_dct_warm_mech.py
git commit -m "feat(dct): init='warm' seed + soft-anchor power-iteration update"
```

---

### Task 2: Warm training wrapper (`dct_warm.py`)

**Files:**
- Create: `src/dct_warm.py`
- Test: `tests/test_dct_warm.py` (pure helpers: seed loading, filename tag)

**Interfaces:**
- Consumes: `dct_meta_<ds>.json`, `truth_dir_<ds>.npz`, `got_datasets/<ds>.csv`, `dct.py` (warm fit), and the extraction path of `run_dct_data.py` (imported helpers `load_statements`, `parse_token_idxs`).
- Produces:
  - `lam_tag(lam) -> str` — filename-safe λ (`0.0→"lam0"`, `0.1→"lam0p1"`, `0.3→"lam0p3"`).
  - `load_seed(ds, seed_name, source_layer) -> np.ndarray` — loads `truth_dir_<ds>.npz[seed_name]`, asserts `int(truth_dir.layer) == source_layer`, returns a unit vector.
  - CLI writing `dct_warm_V_<ds>_<seed>_<lamtag>.pt` and `..._U.pt`.

- [ ] **Step 1: Write the failing helper tests**

```python
# tests/test_dct_warm.py
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm as dw


def test_lam_tag():
    assert dw.lam_tag(0.0) == "lam0"
    assert dw.lam_tag(0.1) == "lam0p1"
    assert dw.lam_tag(0.3) == "lam0p3"


def test_load_seed_asserts_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savez("truth_dir_toy.npz",
             mean_diff=np.array([3.0, 4.0], np.float32),
             grad=np.array([1.0, 0.0], np.float32), layer=np.array(11))
    v = dw.load_seed("toy", "mean_diff", 11)
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.8])
    try:
        dw.load_seed("toy", "mean_diff", 7)      # wrong layer
        assert False, "expected AssertionError"
    except AssertionError:
        pass
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dct_warm'`.

- [ ] **Step 3: Implement `dct_warm.py`**

```python
# src/dct_warm.py
"""Warm-started DCT: seed factor 0 at a supervised truth axis + soft anchor, matching the
existing cold run's config (layers, input_scale, num_iters) from dct_meta_<ds>.json.

Mirrors run_dct_data.py's extraction/calibration; changes only the DCT init.

    python dct_warm.py --dataset cities --seed-name mean_diff --anchor-lambda 0.1 --device cuda
    python dct_warm.py --dataset cities --seed-name grad --anchor-lambda 0.0 --num-factors 8 \
        --num-iters 3 --num-samples 8 --device cpu     # smoke

Writes: dct_warm_V_<ds>_<seed>_<lamtag>.pt  and  ..._U.pt
"""
import argparse
import json
import os
import numpy as np
import torch

import dct
from funnel_utils import unit
from run_dct_data import load_statements, parse_token_idxs, MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer


def lam_tag(lam):
    return "lam" + str(lam).replace(".", "p").rstrip("p0") if lam else "lam0"


def load_seed(ds, seed_name, source_layer):
    td = np.load(f"truth_dir_{ds}.npz")
    assert int(td["layer"]) == int(source_layer), (
        f"truth_dir_{ds}.layer={int(td['layer'])} != dct source_layer={source_layer}; "
        f"seed must live in source-layer space")
    return unit(np.asarray(td[seed_name], np.float64))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed-name", choices=["mean_diff", "grad"], required=True)
    p.add_argument("--anchor-lambda", type=float, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-factors", type=int, default=64)
    # overrides (default: read from dct_meta_<ds>.json)
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--factor-batch-size", type=int, default=64)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=325)
    return p.parse_args()


def main():
    a = parse_args()
    ds = a.dataset
    meta = json.load(open(f"dct_meta_{ds}.json"))
    src, tgt = int(meta["source_layer"]), int(meta["target_layer"])
    input_scale = float(meta["input_scale"])
    num_iters = a.num_iters or int(meta["num_iters"])
    num_samples = a.num_samples or int(meta["num_samples"])
    token_idxs = parse_token_idxs(meta.get("token_idxs", "-3:"))
    seed_vec = load_seed(ds, a.seed_name, src)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[warm] {ds} seed={a.seed_name} lam={a.anchor_lambda} src={src}->tgt={tgt} "
          f"scale={input_scale:.3f} factors={a.num_factors} iters={num_iters}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", truncation_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad = False
    d_model = model.config.hidden_size

    sliced = dct.SlicedModel(model, start_layer=src, end_layer=tgt, layers_name="model.layers")
    statements, _labels = load_statements(ds, num_samples, True, a.seed)
    n = len(statements)
    enc = tok(statements, return_tensors="pt", padding="longest", truncation=True,
              max_length=a.max_length)
    attn = enc["attention_mask"].to(torch.float)
    X = torch.zeros(n, enc["input_ids"].shape[1], d_model, device="cpu")
    Y = torch.zeros(n, enc["input_ids"].shape[1], d_model, device="cpu")
    for t in range(0, n, a.forward_batch_size):
        with torch.no_grad():
            ids = enc["input_ids"][t:t+a.forward_batch_size].to(device)
            msk = enc["attention_mask"][t:t+a.forward_batch_size].to(device)
            hs = model(ids, attention_mask=msk, output_hidden_states=True).hidden_states
            X[t:t+a.forward_batch_size] = hs[src].cpu()
            Y[t:t+a.forward_batch_size] = sliced(hs[src]).cpu()

    delta_acts_single = dct.DeltaActivations(sliced, target_position_indices=token_idxs)
    cpu_mask = attn.cpu()
    m = dct.ExponentialDCT(num_factors=a.num_factors)
    U, V = m.fit(
        delta_acts_single, X, Y, batch_size=1, factor_batch_size=a.factor_batch_size,
        init="warm", warm_seed=torch.as_tensor(seed_vec, dtype=torch.float32),
        anchor_lambda=a.anchor_lambda, input_scale=input_scale, max_iters=num_iters,
        beta=1.0, orthogonalize=True, deflation=False,
        attention_mask=cpu_mask.to(delta_acts_single.device), separate_u=False)

    tag = lam_tag(a.anchor_lambda)
    torch.save(V.detach().cpu(), f"dct_warm_V_{ds}_{a.seed_name}_{tag}.pt")
    torch.save(U.detach().cpu(), f"dct_warm_U_{ds}_{a.seed_name}_{tag}.pt")
    print(f"[warm] wrote dct_warm_V_{ds}_{a.seed_name}_{tag}.pt  V{tuple(V.shape)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the helper tests to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: CPU smoke run (tiny, real model)**

Run: `PYTHONPATH=src .venv/bin/python src/dct_warm.py --dataset cities --seed-name mean_diff --anchor-lambda 3.0 --num-factors 8 --num-iters 3 --num-samples 8 --device cpu`
Expected: loads gemma once, prints the `[warm]` config line, and writes `dct_warm_V_cities_mean_diff_lam3.pt` of shape `(2304, 8)`. Then verify factor 0 is near the seed:
`PYTHONPATH=src .venv/bin/python -c "import torch, numpy as np; from funnel_utils import unit; V=torch.load('dct_warm_V_cities_mean_diff_lam3.pt').numpy(); s=unit(np.load('truth_dir_cities.npz')['mean_diff'].astype(float)); print('cos(factor0, seed)=', abs(float(unit(V[:,0])@s)))"`
Expected: `cos(factor0, seed)=` a value > 0.85 (with the scale-relative anchor, λ=3 ≈ 0.95 in the seed-⟂ limit; strong anchor keeps factor 0 near `mean_diff`). Delete the smoke `.pt` after (`rm dct_warm_V_cities_mean_diff_lam3.pt dct_warm_U_cities_mean_diff_lam3.pt`).

- [ ] **Step 6: Commit**

```bash
git add src/dct_warm.py tests/test_dct_warm.py
git commit -m "feat(dct): warm-start training wrapper (matches cold config, drives seed+anchor fits)"
```

---

### Task 3: Direction assembly + geometry (`dct_warm_directions.py`)

**Files:**
- Create: `src/dct_warm_directions.py`
- Test: `tests/test_dct_warm_directions.py`

**Interfaces:**
- Consumes: `truth_dir_<ds>.npz` (`mean_diff`, `grad`), the 6 `dct_warm_V_<ds>_<seed>_<lamtag>.pt`, cold `dct_V_<ds>.pt` + `dct_U_<ds>.pt` (via `funnel_utils.load_dct` / `top_k_by_potency`).
- Produces:
  - `aligned(vec, ref) -> np.ndarray` — unit `vec`, sign-flipped so `dot(vec, ref) >= 0`.
  - `assemble_directions(ds, seeds, lams) -> dict[str, np.ndarray]` — `raw_mean_diff`, `raw_grad`, `warm_<seed>_<lamtag>` (factor 0, sign-aligned to its seed), `cold_top` (cold `V[:, argmax‖U‖]`, sign-aligned to `mean_diff`).
  - CLI writing `dct_warm_dirs_<ds>.npz` (name→vec) and `dct_warm_geometry_<ds>.csv` (columns: `direction, drift, cos_to_mean_diff`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dct_warm_directions.py
import sys, os
import numpy as np
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
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_directions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dct_warm_directions'`.

- [ ] **Step 3: Implement `dct_warm_directions.py`**

```python
# src/dct_warm_directions.py
"""Assemble the candidate directions for the warm-DCT behavioral test and record their geometry.

    python dct_warm_directions.py --dataset cities
Writes dct_warm_dirs_<ds>.npz (name->unit vec at source layer) and dct_warm_geometry_<ds>.csv."""
import argparse
import csv
import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

SEEDS = ["mean_diff", "grad"]
LAMS = [0.0, 0.3, 1.0, 3.0]


def aligned(vec, ref):
    v = unit(np.asarray(vec, np.float64))
    return v if float(v @ unit(np.asarray(ref, np.float64))) >= 0 else -v


def assemble_directions(ds, seeds=SEEDS, lams=LAMS):
    td = np.load(f"truth_dir_{ds}.npz")
    md = unit(np.asarray(td["mean_diff"], np.float64))
    grad = unit(np.asarray(td["grad"], np.float64))
    dirs = {"raw_mean_diff": md, "raw_grad": grad}
    seed_ref = {"mean_diff": md, "grad": grad}
    for s in seeds:
        for lam in lams:
            path = f"dct_warm_V_{ds}_{s}_{lam_tag(lam)}.pt"
            V = torch.load(path, map_location="cpu").float().numpy()
            dirs[f"warm_{s}_{lam_tag(lam)}"] = aligned(V[:, 0], seed_ref[s])
    V, U, _ = fu.load_dct(ds)                     # cold run
    top = fu.top_k_by_potency(V, U, 1)[0]         # ‖U‖-ranked top factor (model-free)
    dirs["cold_top"] = aligned(V[:, top], md)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    ds = a.dataset
    dirs = assemble_directions(ds)
    md = dirs["raw_mean_diff"]
    seed_ref = {"mean_diff": dirs["raw_mean_diff"], "grad": dirs["raw_grad"]}
    np.savez(f"dct_warm_dirs_{ds}.npz", **{k: v.astype(np.float32) for k, v in dirs.items()})
    rows = [("direction", "drift", "cos_to_mean_diff")]
    for name, v in dirs.items():
        # drift = 1 - cos to the seed axis (0 for raw axes / cold_top's own ref is mean_diff)
        if name.startswith("warm_"):
            ref = seed_ref["mean_diff" if "mean_diff" in name else "grad"]
        else:
            ref = md
        drift = 1.0 - float(v @ ref)
        rows.append((name, round(drift, 4), round(float(v @ md), 4)))
    with open(f"dct_warm_geometry_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[dirs] wrote dct_warm_dirs_{ds}.npz ({len(dirs)} dirs) and dct_warm_geometry_{ds}.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_directions.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dct_warm_directions.py tests/test_dct_warm_directions.py
git commit -m "feat(dct): assemble warm/cold/raw candidate directions + geometry CSV"
```

---

### Task 4: Behavioral steering (`dct_warm_steer.py`)

**Files:**
- Create: `src/dct_warm_steer.py`
- Test: `tests/test_dct_warm_steer.py` (pure injection helper)

**Interfaces:**
- Consumes: `dct_warm_dirs_<ds>.npz`, `dct_meta_<ds>.json` (`input_scale`, `source_layer`), `dct_steer_utils` (`load_model`, `Steerer`, `generate`), `steer_supervised.FACTUAL_PROMPTS`, `funnel_utils.unit`.
- Produces:
  - `injected(tau, unit_dir, input_scale) -> np.ndarray` — `tau * input_scale * unit(unit_dir)`.
  - CLI writing `dct_warm_steer_<ds>.csv` (schema `direction,scale,prompt,completion`, `scale = τ`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dct_warm_steer.py
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dct_warm_steer as ws


def test_injected_scales_by_tau_and_input_scale():
    d = np.array([0.0, 3.0, 4.0])                 # norm 5
    v = ws.injected(0.5, d, 10.0)                 # 0.5 * 10 * unit(d)
    assert np.allclose(np.linalg.norm(v), 5.0)
    assert np.allclose(v, [0.0, 3.0, 4.0])


def test_injected_tau0_zero():
    assert np.allclose(ws.injected(0.0, np.array([1.0, 1.0]), 7.0), 0.0)


def test_injected_negative_tau_flips_direction():
    d = np.array([0.0, 3.0, 4.0])
    assert np.allclose(ws.injected(-0.5, d, 10.0), -ws.injected(0.5, d, 10.0))


def test_taus_two_sided_symmetric():
    # -tau -> FALSE (lying), +tau -> TRUE; 0.0 present (no-injection control)
    assert 0.0 in ws.TAUS
    assert sorted(ws.TAUS) == sorted(-t for t in ws.TAUS)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_steer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dct_warm_steer'`.

- [ ] **Step 3: Implement `dct_warm_steer.py`**

```python
# src/dct_warm_steer.py
"""Inject each warm-DCT candidate direction at the source layer with the shared DCT input_scale,
sweep tau, generate short factual completions for the OLMo judge.

    python dct_warm_steer.py --dataset cities --device cuda
    python dct_warm_steer.py --dataset cities --device cpu --limit 3   # smoke

Writes dct_warm_steer_<ds>.csv (direction,scale,prompt,completion — scale=tau), the schema
judge_results.py --mode steer reads unchanged."""
import argparse
import csv
import json
import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from steer_supervised import FACTUAL_PROMPTS

TAUS = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]   # two-sided: -tau -> FALSE (lying), +tau -> TRUE
MAX_NEW_TOKENS = 8


def injected(tau, unit_dir, input_scale):
    return float(tau) * float(input_scale) * unit(np.asarray(unit_dir, np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke)")
    a = ap.parse_args()
    ds = a.dataset
    meta = json.load(open(f"dct_meta_{ds}.json"))
    layer = int(meta["source_layer"]); scale = float(meta["input_scale"])
    dirs = np.load(f"dct_warm_dirs_{ds}.npz")
    prompts = FACTUAL_PROMPTS[:a.limit] if a.limit else FACTUAL_PROMPTS
    print(f"[warm/steer] {ds}: layer={layer} input_scale={scale:.3f} "
          f"{len(dirs.files)} dirs x {len(TAUS)} taus x {len(prompts)} prompts", flush=True)

    tok, model, dev = su.load_model(a.device)
    rows = [("direction", "scale", "prompt", "completion")]
    for name in dirs.files:
        uvec = unit(np.asarray(dirs[name], np.float64))
        with su.Steerer(model, layer) as st:
            for tau in TAUS:
                vec = None if tau == 0.0 else torch.tensor(
                    injected(tau, uvec, scale), dtype=torch.float32)
                st.set(vec)
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((name, tau, p, c))
                print(f"  {name} tau={tau:+.1f} done", flush=True)
    with open(f"dct_warm_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[warm/steer] wrote dct_warm_steer_{ds}.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dct_warm_steer.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dct_warm_steer.py tests/test_dct_warm_steer.py
git commit -m "feat(dct): warm-DCT behavioral steering CSV (judge-schema, shared input_scale)"
```

---

### Task 5: Plots (`viz_dct_warm.py`)

**Files:**
- Create: `src/viz_dct_warm.py`
- Test: `tests/test_viz_dct_warm.py`

**Interfaces:**
- Consumes: `dct_warm_geometry_<ds>.csv`, `judge_dct_warm_steer_<ds>.csv` (judge output, schema `direction,scale,prompt,completion,verdict,reason`), `dct_U_<ds>.pt`/`dct_V_<ds>.pt` + `truth_dir` for the audit.
- Produces:
  - `verdict_fractions(rows) -> dict[(direction,tau), dict]` — TRUE/FALSE/INCOHERENT fractions per (direction, τ).
  - `plot_drift(ds)` → `plot_dct_warm_drift_<ds>.png` (drift vs λ per seed, from geometry CSV).
  - `plot_verdict(ds)` → `plot_dct_warm_verdict_<ds>.png` (FALSE vs INCOHERENT bars at strongest τ: raw_mean_diff vs warm_* vs cold_top).
  - `plot_audit(ds)` → `plot_dct_warm_audit_<ds>.png` (cold factors by ‖U‖ potency, colored by |cos| to `mean_diff`).
  - CLI: `python viz_dct_warm.py --dataset cities`.

- [ ] **Step 1: Write the failing test for `verdict_fractions`**

```python
# tests/test_viz_dct_warm.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viz_dct_warm as vw


def test_verdict_fractions():
    rows = [
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "FALSE"},
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "INCOHERENT"},
        {"direction": "warm_mean_diff_lam0p1", "scale": "1.0", "verdict": "FALSE"},
    ]
    fr = vw.verdict_fractions(rows)
    key = ("warm_mean_diff_lam0p1", "1.0")
    assert abs(fr[key]["FALSE"] - 2/3) < 1e-9
    assert abs(fr[key]["INCOHERENT"] - 1/3) < 1e-9
    assert abs(fr[key]["TRUE"] - 0.0) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_dct_warm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz_dct_warm'`.

- [ ] **Step 3: Implement `viz_dct_warm.py`**

```python
# src/viz_dct_warm.py
"""Warm-DCT figures: drift-vs-lambda, verdict head-to-head, and the note-#2 cold-factor audit.

    python viz_dct_warm.py --dataset cities"""
import argparse
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

LAMS = [0.0, 0.3, 1.0, 3.0]


def verdict_fractions(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], str(float(r["scale"]))), []).append(r["verdict"])
    out = {}
    for key, vs in groups.items():
        n = len(vs) or 1
        out[key] = {v: sum(x == v for x in vs) / n for v in ("TRUE", "FALSE", "INCOHERENT")}
    return out


def plot_drift(ds):
    rows = list(csv.DictReader(open(f"dct_warm_geometry_{ds}.csv")))
    drift = {r["direction"]: float(r["drift"]) for r in rows}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for seed, color in [("mean_diff", "#228833"), ("grad", "#4477aa")]:
        ys = [drift.get(f"warm_{seed}_{lam_tag(l)}", np.nan) for l in LAMS]
        ax.plot(LAMS, ys, "o-", color=color, label=seed)
    ax.set_xlabel("anchor λ"); ax.set_ylabel("drift (1 − cos to seed)")
    ax.set_title(f"{ds}: how far the causal search wanders vs anchor strength")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_drift_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_drift_{ds}.png")


def plot_verdict(ds):
    rows = list(csv.DictReader(open(f"judge_dct_warm_steer_{ds}.csv")))
    fr = verdict_fractions(rows)
    taus = sorted({k[1] for k in fr}, key=float)
    # directions point toward TRUE; the causal truth->false flip lives at the most-negative tau
    strongest = taus[0]
    names = sorted({k[0] for k in fr})
    false_v = [fr.get((n, strongest), {}).get("FALSE", 0.0) for n in names]
    incoh_v = [fr.get((n, strongest), {}).get("INCOHERENT", 0.0) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.6))
    ax.bar(x - 0.2, false_v, 0.4, color="#cc3311", label="FALSE (causal)")
    ax.bar(x + 0.2, incoh_v, 0.4, color="#999999", label="INCOHERENT (degrade)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1.0)
    ax.set_title(f"{ds}: causal lever vs degrader at τ={strongest}")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_verdict_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_verdict_{ds}.png")


def plot_audit(ds):
    V, U, _ = fu.load_dct(ds)
    md = unit(np.asarray(np.load(f"truth_dir_{ds}.npz")["mean_diff"], np.float64))
    potency = np.linalg.norm(U, axis=0)
    order = np.argsort(potency)[::-1][:30]
    cos = np.array([abs(float(unit(V[:, i].astype(float)) @ md)) for i in order])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(range(len(order)), potency[order], c=cos, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("cold DCT factor (ranked by ‖U‖ potency)")
    ax.set_ylabel("‖U‖ potency")
    ax.set_title(f"{ds}: is the top cold DCT factor truth-aligned? (color = |cos| to mean_diff)")
    fig.colorbar(sc, label="|cos| to mean_diff"); fig.tight_layout()
    fig.savefig(f"plot_dct_warm_audit_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_warm_audit_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_drift(a.dataset)
    plot_verdict(a.dataset)
    plot_audit(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_viz_dct_warm.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/viz_dct_warm.py tests/test_viz_dct_warm.py
git commit -m "feat(dct): warm-DCT drift, verdict, and cold-factor audit plots"
```

---

### Task 6: Cluster job scripts + runbook

**Files:**
- Create: `deltaai/run_dct_warm.slurm`
- Create: `deltaai/run_dct_warm_judge.slurm`
- Create: `deltaai/DCT_WARM_RUN.md`

**Interfaces:**
- Consumes: `.venv-dct-gpu` (warm train + steer), `.venv-judge-gpu` (judge). Reuses `judge_results.py --mode steer --backend olmo` with `--steer-input dct_warm_steer_<ds>.csv --steer-output judge_dct_warm_steer_<ds>.csv`.

- [ ] **Step 1: Write `run_dct_warm.slurm`** (train the 6 warm runs, assemble directions, then steer — all in the gen env)

```bash
#!/bin/bash
#SBATCH --job-name=dct_warm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=dct_warm_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  for seed in mean_diff grad; do
    for lam in 0.0 0.3 1.0 3.0; do
      echo "=== warm train $ds $seed lam=$lam $(date) ==="
      PYTHONPATH=src python3 src/dct_warm.py --dataset "$ds" --seed-name "$seed" \
        --anchor-lambda "$lam" --device cuda || echo "!!!! $ds $seed $lam FAILED — continuing"
    done
  done
  echo "=== assemble directions $ds ==="
  PYTHONPATH=src python3 src/dct_warm_directions.py --dataset "$ds"
  echo "=== warm steer $ds ==="
  PYTHONPATH=src python3 src/dct_warm_steer.py --dataset "$ds" --device cuda \
    || echo "!!!! steer $ds FAILED — continuing"
done
echo "=== dct warm done $(date) ==="
```

- [ ] **Step 2: Write `run_dct_warm_judge.slurm`** (clone of the length/MAG judge job, pointed at the warm CSVs)

```bash
#!/bin/bash
#SBATCH --job-name=dct_warm_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=dct_warm_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== warm judge $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "dct_warm_steer_${ds}.csv" \
    --steer-output "judge_dct_warm_steer_${ds}.csv" \
    --steer-plot "plot_judge_dct_warm_${ds}.png" \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== dct warm judge done $(date) ==="
```

- [ ] **Step 3: Write `DCT_WARM_RUN.md`**

Self-contained runbook modeled on `deltaai/MAG_E4_RUN.md` (real commands, no placeholders):
1. **Laptop:** confirm `dct_meta_<ds>.json`, `truth_dir_<ds>.npz`, cold `dct_V_<ds>.pt`/`dct_U_<ds>.pt` exist. rsync code up excluding `*.npz` and `*.pt`, then a **second** rsync of the needed artifacts: `rsync -av dct_meta_*.json truth_dir_*.npz dct_V_*.pt dct_U_*.pt vwudaru@...:~/llm-activation-steering-research/` (copy the `--exclude '*.npz'` / two-rsync warning verbatim from `MAG_E4_RUN.md`, and note `.pt` must also be excluded from the first rsync and sent explicitly).
2. **Cluster:** account into both scripts: `ACC=$(grep -o -- '--account=[^ ]*' deltaai/run_dct.slurm | head -1 | cut -d= -f2); sed -i "s/ACCOUNT_NAME/$ACC/" deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm`.
3. `sbatch deltaai/run_dct_warm.slurm` — done when `dct_warm_dirs_*.npz` and `dct_warm_steer_*.csv` (2 each) exist. (~2–4 h: 16 warm fits — 2 seeds × 4 λ × 2 datasets — at num_factors=64.)
4. After step 3's CSVs exist: `sbatch deltaai/run_dct_warm_judge.slurm` — done when `judge_dct_warm_steer_*.csv` (2) exist.
5. **Laptop:** rsync back `dct_warm_geometry_*.csv dct_warm_steer_*.csv judge_dct_warm_steer_*.csv dct_warm_dirs_*.npz`, then `PYTHONPATH=src .venv/bin/python src/viz_dct_warm.py --dataset cities` and `--dataset common_claim_true_false`.
Include an "If something goes wrong" table (adapt from `MAG_E4_RUN.md`): warm fit `AssertionError` on layer mismatch → the `truth_dir`/`dct_meta` layers disagree, re-export `truth_dir` at the source layer; missing cold `dct_V` on cluster → send it via the second rsync; judge input missing → confirm `dct_warm_steer_*.csv` exist first.

- [ ] **Step 4: Verify slurm scripts parse**

Run: `bash -n deltaai/run_dct_warm.slurm && bash -n deltaai/run_dct_warm_judge.slurm && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add deltaai/run_dct_warm.slurm deltaai/run_dct_warm_judge.slurm deltaai/DCT_WARM_RUN.md
git commit -m "feat(dct): GH200 warm-train+steer and judge slurm scripts + runbook"
```

---

## Self-Review

- **Spec coverage:** headline (causal lever near inert axis) → Tasks 1–4 (warm mechanism → warm runs → directions → behavioral FALSE/INCOHERENT read); note-#2 audit → Task 3 `cold_top` + Task 5 `plot_audit`; warm mechanism (init="warm" + anchor) → Task 1; seed at source layer (assert) → Global Constraints + Task 2 `load_seed`; mean_diff+grad × λ{0,0.3,1,3} (scale-relative anchor) separate runs → Task 2 + Task 6 slurm loop; reuse cold `dct_V` → Task 3 via `fu.load_dct`; shared `input_scale` calibration + two-sided τ{−1,−.6,−.3,0,.3,.6,1} → Task 4; artifacts (warm V files, dirs npz, geometry/steer/judge CSVs, 3 plots, runbook) → Tasks 2–6; success criteria (drift + verdict + audit plots) → Task 5.
- **Deviations from spec (intentional, simpler & repo-faithful, flagged for the reviewer):** (a) warm uses `init="random"` for the non-seeded factors instead of `rand_backward` — only factor 0 is the object of interest, and this avoids the expensive Jacobian; (b) `cold_top` is ranked by `‖U‖` potency via the existing `funnel_utils.top_k_by_potency` (model-free) rather than the model-in-the-loop `alphas`; (c) the seed is loaded directly from `truth_dir_<ds>.npz` (verified already at the source layer) rather than recomputed from `mag_acts`, with the layer-equality assert as the guard. All three are noted in Global Constraints / task interfaces.
- **Placeholder scan:** none — every code step is complete; the runbook step (Task 6 Step 3) enumerates exact commands to copy from `MAG_E4_RUN.md`.
- **Type consistency:** `lam_tag` defined once (Task 2) and imported by Tasks 3 & 5; `dct_warm_V_<ds>_<seed>_<lamtag>.pt` filename identical across Task 2 (writer), Task 3 (`assemble_directions` reader), Task 6 (slurm); the steer CSV schema `direction,scale,prompt,completion` (scale=τ) is consistent across Task 4 (writer), Task 6 (judge `--steer-input`), Task 5 (`verdict_fractions` reads the judged output); `injected`/`aligned`/`warm_init_column`/`anchor_step` signatures match their tests.
