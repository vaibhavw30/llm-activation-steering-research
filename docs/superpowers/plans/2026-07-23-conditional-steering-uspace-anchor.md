# Conditional Steering + U-Space Anchored DCT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-arm follow-up run — A1 verdict-mode steering with logit readout, A2 v_Q-gate + mean_diff composed steering, B U-space anchored DCT — plus the A0 judge backfill plumbing, per the approved spec `docs/superpowers/specs/2026-07-23-conditional-steering-uspace-anchor-design.md`.

**Architecture:** Arms A1/A2 are new scripts under `src/mag/` that reuse the existing `Steerer`/`generate`/verdict-readout machinery and the `mag_dir_<ds>.npz` calibration. Arm B extends the existing warm-DCT stack (`dct.py` anchor mechanics, `dct_warm.py` wrapper, directions-assembly, shared steer script) with a U-space anchor mode, new target-layer seeds, and its own directions/viz scripts. Cluster pieces mirror the existing `deltaai/*.slurm` + runbook pattern.

**Tech Stack:** Python 3.13 (`.venv`), torch (CPU/MPS locally, CUDA on GH200), numpy, scikit-learn, matplotlib (Agg), pytest. Model `google/gemma-2-2b` fp32, eager attention.

## Global Constraints

- Sign convention: directions point toward TRUE; +τ→TRUE, −τ→FALSE. τ grid everywhere: `[-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]`.
- MAG-layer calibration: injected vector = `τ · A_prefix_norm · unit(dir)`, `A_prefix_norm` and `layer` from `mag_dir_<ds>.npz` (cities layer 11, common_claim 13 — equal to the DCT source layers).
- Steer CSV schema consumed by the judge is exactly `direction,scale,prompt,completion` (`src/judge_results.py --mode steer`). Do not change it.
- Datasets for this run: `cities` and `common_claim_true_false` only.
- New DCT artifacts use the `dct_uwarm_` stem; never overwrite `dct_warm_*` or `truth_dir_<ds>.npz`.
- Do NOT touch `src/spectrum_utils.py`, `src/viz_spectrum.py`, `tests/test_spectrum_utils.py`, `tests/test_viz_spectrum.py` — they hold unrelated uncommitted work.
- Never push; commits are local only. Commit messages follow the repo's `feat(scope): ...` style.
- Tests: pytest, files under `tests/`, `sys.path.insert` header pattern used by every existing test.
- Run tests with `.venv/bin/python -m pytest tests/<file> -v` from the repo root.
- Slurm files keep the existing header pattern (`ACCOUNT_NAME` placeholder, `ghx4` partition, `HF_HUB_DISABLE_XET=1`, `TRANSFORMERS_OFFLINE=1`, per-step `|| echo "!!!! ... FAILED — continuing"`).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/mag/steer_verdict.py` | create | Arm A1: steer 5 directions over τ grid, read p_yes/p_no logit margins on test-split statements |
| `src/mag/viz_verdict.py` | create | Arm A1 figures: label-split margin curves + accuracy curves |
| `src/mag/steer_conditional.py` | create | Arm A2: composed v_Q-gate + mean_diff steering, free-form completions for the judge |
| `src/mag/viz_conditional.py` | create | Arm A2 figure: gate0 vs gate1 verdict-vs-τ panels |
| `src/export_target_dir.py` | create | Arm B seeds: mean_diff/grad at the DCT target layer → `truth_dir_tgt_<ds>.npz` |
| `src/dct.py` | modify | `fit(..., u_anchor=, u_anchor_lambda=)` — anchor the U update's column 0 |
| `src/dct_warm.py` | modify | `--anchor-space {v,u}`; u mode trains with U-anchor, writes `dct_uwarm_*` artifacts |
| `src/dct_uwarm_directions.py` | create | Assemble uwarm candidate directions + U/V geometry CSV |
| `src/dct_warm_steer.py` | modify | `--dirs-file` / `--out` args so the same script steers uwarm directions |
| `src/viz_dct_uwarm.py` | create | Arm B figures: verdict-vs-τ panels + geometry ladder |
| `deltaai/run_mag_cond_judge.slurm` | create | A0 + A2 judge job |
| `deltaai/run_dct_uwarm.slurm` | create | Arm B train + directions + steer job |
| `deltaai/run_dct_uwarm_judge.slurm` | create | Arm B judge job |
| `docs/CONDITIONAL_UANCHOR_RUNBOOK.md` | create | End-to-end run order, rsync lists, local run commands |

Tests: `tests/test_mag_steer_verdict.py`, `tests/test_mag_viz_verdict.py`, `tests/test_mag_steer_conditional.py`, `tests/test_export_target_dir.py`, additions to `tests/test_dct_warm_mech.py` and `tests/test_dct_warm.py`, `tests/test_dct_uwarm_directions.py`, additions to `tests/test_dct_warm_steer.py`, `tests/test_viz_dct_uwarm.py`.

---

### Task 1: `src/mag/steer_verdict.py` (Arm A1)

**Files:**
- Create: `src/mag/steer_verdict.py`
- Test: `tests/test_mag_steer_verdict.py`

**Interfaces:**
- Consumes: `mag.config.Q_TRUTH/Q_SUFFIX/YES_VARIANTS/NO_VARIANTS`, `mag.verdict.first_token_ids/verdict_from_logits`, `mag.steer.injected_vector`, `dct_steer_utils.load_model/Steerer`, `funnel_utils.unit/load_dct/top_k_by_potency`, files `got_datasets/<ds>.csv`, `mag_dir_<ds>.npz`, `truth_dir_<ds>.npz`, `dct_V_<ds>.pt`/`dct_U_<ds>.pt`/`dct_meta_<ds>.json`.
- Produces: `mag_verdict_logits_<ds>.csv` with header `direction,tau,statement,label,p_yes,p_no,margin`; functions `sample_statements(ds, n_per_class, seed)`, `build_directions(ds)` → `(list[(name, unit_vec, layer)], a_prefix_norm)`, `verdict_rows(model, tok, prompts, device, batch_size)` → `list[dict]`. Task 2 reads the CSV.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mag_steer_verdict.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mag_steer_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.steer_verdict'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/steer_verdict.py
"""Arm A1 — verdict-mode steering with logit readout. Steer each direction at its layer with
alpha(tau) = tau * A_prefix_norm, and read the model's first-token p_yes/p_no on
Q_TRUTH + statement + Q_SUFFIX over balanced test-split statements. No sampling, no judge.

    python -m mag.steer_verdict --dataset cities --device mps
    python -m mag.steer_verdict --dataset cities --device cpu --limit 2 --only sup_mean_diff
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dct_steer_utils as su
import funnel_utils as fu
from funnel_utils import unit
from mag.config import Q_TRUTH, Q_SUFFIX, YES_VARIANTS, NO_VARIANTS
from mag.steer import injected_vector
from mag.verdict import first_token_ids, verdict_from_logits

TAUS7 = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
N_PER_CLASS = 64
SPLIT_SEED = 42


def sample_statements(ds, n_per_class=N_PER_CLASS, seed=SPLIT_SEED):
    """Balanced (statement, label) pairs drawn only from the standard 80/20 test split,
    so none were used to fit mean_diff/grad. Capped by the smaller class pool."""
    rows = list(csv.DictReader(open(f"got_datasets/{ds}.csv")))
    labels = [int(r["label"]) for r in rows]
    idx = np.arange(len(rows))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    out = []
    for lab in (1, 0):
        pool = sorted(i for i in test_idx if labels[i] == lab)
        if len(pool) > n_per_class:
            pool = sorted(rng.choice(pool, n_per_class, replace=False))
        out += [(rows[i]["statement"], lab) for i in pool]
    return out


def build_directions(ds):
    """The 5 A1 directions as (name, unit_vec, layer); plus the shared A_prefix_norm."""
    md_npz = np.load(f"mag_dir_{ds}.npz")
    layer = int(md_npz["layer"]); apn = float(md_npz["A_prefix_norm"])
    td = np.load(f"truth_dir_{ds}.npz")
    dirs = [
        ("sup_mean_diff", unit(np.asarray(td["mean_diff"], np.float64)), int(td["layer"])),
        ("sup_grad", unit(np.asarray(td["grad"], np.float64)), int(td["layer"])),
        ("mag_resid_pc1", unit(np.asarray(md_npz["resid_pc1_unit"], np.float64)), layer),
    ]
    V, U, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(V, U, 1)[0]
    dirs.append(("cold_top", unit(V[:, top].astype(np.float64)), layer))
    rng = np.random.default_rng(SPLIT_SEED)
    dirs.append(("random_unit", unit(rng.standard_normal(V.shape[0])), layer))
    return dirs, apn


def verdict_rows(model, tok, prompts, device, batch_size=16):
    """Per-prompt first-token verdict dicts (p_yes/p_no/margin), batched with right padding.
    Mirrors mag.verdict.compute_verdicts but keeps every row's full readout."""
    yes_ids = first_token_ids(tok, YES_VARIANTS)
    no_ids = first_token_ids(tok, NO_VARIANTS)
    tok.padding_side = "right"
    out = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(device)
            logits = model(**enc).logits
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                row = logits[b, int(last[b].item()), :].float().cpu().numpy()
                out.append(verdict_from_logits(row, yes_ids, no_ids))
    return out


def run_a1(ds, device, n_per_class=N_PER_CLASS, only=None, taus=TAUS7):
    pairs = sample_statements(ds, n_per_class)
    dirs, apn = build_directions(ds)
    if only:
        subs = [s.strip() for s in only.split(",") if s.strip()]
        dirs = [d for d in dirs if any(s in d[0] for s in subs)]
        if not dirs:
            raise SystemExit(f"--only {only!r} matched no directions")
    prompts = [Q_TRUTH + s + Q_SUFFIX for s, _ in pairs]
    print(f"[a1] {ds}: {len(dirs)} dirs x {len(taus)} taus x {len(prompts)} statements "
          f"(apn={apn:.3f})", flush=True)
    tok, model, dev = su.load_model(device)
    rows = [("direction", "tau", "statement", "label", "p_yes", "p_no", "margin")]
    for name, uvec, layer in dirs:
        with su.Steerer(model, layer) as st:
            for tau in taus:
                st.set(None if tau == 0.0 else torch.tensor(
                    injected_vector(tau, uvec, apn), dtype=torch.float32))
                res = verdict_rows(model, tok, prompts, dev)
                for (s, lab), r in zip(pairs, res):
                    rows.append((name, tau, s, lab, r["p_yes"], r["p_no"], r["margin"]))
                print(f"  {name} tau={tau:+.1f} done", flush=True)
    out = f"mag_verdict_logits_{ds}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[a1] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap statements per class (smoke)")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep only matching direction names")
    a = ap.parse_args()
    run_a1(a.dataset, a.device, n_per_class=a.limit or N_PER_CLASS, only=a.only)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mag_steer_verdict.py -v`
Expected: 5 PASS

- [ ] **Step 5: Regression-check the imports it leans on**

Run: `.venv/bin/python -m pytest tests/test_mag_verdict.py tests/test_mag_steer.py -v`
Expected: all PASS (no changes to shared modules)

- [ ] **Step 6: Commit**

```bash
git add src/mag/steer_verdict.py tests/test_mag_steer_verdict.py
git commit -m "feat(mag): A1 verdict-mode steering with p_yes/p_no logit readout"
```

---

### Task 2: `src/mag/viz_verdict.py` (Arm A1 figures)

**Files:**
- Create: `src/mag/viz_verdict.py`
- Test: `tests/test_mag_viz_verdict.py`

**Interfaces:**
- Consumes: `mag_verdict_logits_<ds>.csv` (Task 1 schema `direction,tau,statement,label,p_yes,p_no,margin`).
- Produces: `plot_mag_verdict_margin_<ds>.png` (per-direction panels, mean margin vs τ split by gold label), `plot_mag_verdict_acc_<ds>.png` (verdict accuracy vs τ, one line per direction). Functions `load_rows(ds)`, `margin_stats(rows)`, `accuracy(rows)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mag_viz_verdict.py
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.viz_verdict import load_rows, margin_stats, accuracy, plot_margin, plot_accuracy


def _write_toy_csv(tmp_path):
    rows = [("direction", "tau", "statement", "label", "p_yes", "p_no", "margin")]
    for d in ("sup_mean_diff", "random_unit"):
        for tau in (-1.0, 0.0, 1.0):
            # true statements: margin follows tau; false statements: margin fixed positive
            rows.append((d, tau, "s_true", 1, 0.8, 0.1, 0.5 * tau))
            rows.append((d, tau, "s_false", 0, 0.7, 0.2, 0.4))
    with open(tmp_path / "mag_verdict_logits_toy.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_margin_stats_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_csv(tmp_path)
    rows = load_rows("toy")
    ms = margin_stats(rows)
    assert ms[("sup_mean_diff", -1.0, 1)][0] == -0.5      # mean margin, true class
    assert ms[("sup_mean_diff", -1.0, 0)][0] == 0.4
    acc = accuracy(rows)
    # tau=-1: true stmt margin<0 -> predicted 0 (wrong); false stmt margin>0 -> predicted 1 (wrong)
    assert acc[("sup_mean_diff", -1.0)] == 0.0
    # tau=+1: true stmt correct, false stmt wrong
    assert acc[("sup_mean_diff", 1.0)] == 0.5


def test_plots_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_csv(tmp_path)
    plot_margin("toy")
    plot_accuracy("toy")
    assert os.path.exists("plot_mag_verdict_margin_toy.png")
    assert os.path.exists("plot_mag_verdict_acc_toy.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mag_viz_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.viz_verdict'`

- [ ] **Step 3: Write the implementation**

```python
# src/mag/viz_verdict.py
"""Arm A1 figures: label-split margin-vs-tau curves and verdict accuracy-vs-tau.

    python -m mag.viz_verdict --dataset cities"""
import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_COLOR = {1: "#228833", 0: "#cc3311"}     # green = gold-true, red = gold-false


def load_rows(ds):
    out = []
    for r in csv.DictReader(open(f"mag_verdict_logits_{ds}.csv")):
        out.append({"direction": r["direction"], "tau": float(r["tau"]),
                    "label": int(r["label"]), "margin": float(r["margin"])})
    return out


def margin_stats(rows):
    """(direction, tau, label) -> (mean margin, sem)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["tau"], r["label"]), []).append(r["margin"])
    return {k: (float(np.mean(v)), float(np.std(v) / max(1, len(v)) ** 0.5))
            for k, v in groups.items()}


def accuracy(rows):
    """(direction, tau) -> fraction where (margin > 0) == gold label."""
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["tau"]), []).append(
            int((r["margin"] > 0) == bool(r["label"])))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def _directions_and_taus(rows):
    dirs = sorted({r["direction"] for r in rows})
    taus = sorted({r["tau"] for r in rows})
    return dirs, taus


def plot_margin(ds):
    rows = load_rows(ds)
    dirs, taus = _directions_and_taus(rows)
    ms = margin_stats(rows)
    ncols = 3
    nrows_ = int(np.ceil(len(dirs) / ncols))
    fig, axes = plt.subplots(nrows_, ncols, figsize=(4.6 * ncols, 3.6 * nrows_),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, d in zip(flat, dirs):
        for lab in (1, 0):
            ys = [ms.get((d, t, lab), (np.nan, 0.0)) for t in taus]
            mean = [y[0] for y in ys]; sem = [y[1] for y in ys]
            ax.errorbar(taus, mean, yerr=sem, fmt="o-", color=LABEL_COLOR[lab],
                        label=f"gold {'true' if lab else 'false'}", lw=2, ms=4)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_title(d, fontsize=10); ax.grid(alpha=0.25)
    for ax in flat[len(dirs):]:
        ax.axis("off")
    flat[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tau  (-=push FALSE, +=push TRUE)")
    for r in range(nrows_):
        axes[r][0].set_ylabel("mean p_yes - p_no")
    fig.suptitle(f"{ds}: verdict margin vs tau, split by gold label\n"
                 f"(differential movement = latent truth knowledge; parallel = content-blind)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"plot_mag_verdict_margin_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_verdict_margin_{ds}.png")


def plot_accuracy(ds):
    rows = load_rows(ds)
    dirs, taus = _directions_and_taus(rows)
    acc = accuracy(rows)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for d in dirs:
        ax.plot(taus, [acc.get((d, t), np.nan) for t in taus], "o-", label=d, lw=2, ms=4)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel("tau"); ax.set_ylabel("verdict accuracy vs gold label")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{ds}: does steering make the model's verdict more accurate?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(f"plot_mag_verdict_acc_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_verdict_acc_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_margin(a.dataset)
    plot_accuracy(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mag_viz_verdict.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mag/viz_verdict.py tests/test_mag_viz_verdict.py
git commit -m "feat(mag): A1 verdict-margin and accuracy figures"
```

---

### Task 3: `src/mag/steer_conditional.py` + `src/mag/viz_conditional.py` (Arm A2)

**Files:**
- Create: `src/mag/steer_conditional.py`, `src/mag/viz_conditional.py`
- Test: `tests/test_mag_steer_conditional.py`

**Interfaces:**
- Consumes: `mag_dir_<ds>.npz` (`v_Q_unit`, `A_prefix_norm`, `layer`), `truth_dir_<ds>.npz` (`mean_diff`, `layer`), `steer_supervised.FACTUAL_PROMPTS`, `dct_steer_utils.load_model/Steerer/generate`; viz consumes `judge_mag_conditional_<ds>.csv` and reuses `viz_dct_warm.verdict_fractions/_sorted_taus/_curve/VERDICT_COLOR`.
- Produces: `mag_conditional_<ds>.csv` (`direction,scale,prompt,completion`; direction names `meandiff_gate0`, `meandiff_gate1`), `plot_mag_conditional_<ds>.png`. Function `composed_vec(gate, tau, vq_unit, md_unit, apn)` → np array or `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mag_steer_conditional.py
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.steer_conditional import composed_vec, direction_name, TAUS7, GATES


def test_grid():
    assert TAUS7 == [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    assert GATES == (0, 1)


def test_composed_vec_baseline_is_none():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    assert composed_vec(0, 0.0, vq, md, 2.0) is None      # untouched shared baseline


def test_composed_vec_gate_only():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(1, 0.0, vq, md, 2.0)
    assert np.allclose(v, [2.0, 0.0])                     # apn * v_Q alone


def test_composed_vec_gate_plus_content():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(1, -1.0, vq, md, 2.0)
    assert np.allclose(v, [2.0, -2.0])                    # apn*v_Q + tau*apn*md


def test_composed_vec_content_only_matches_e4_calibration():
    vq = np.array([1.0, 0.0]); md = np.array([0.0, 1.0])
    v = composed_vec(0, 0.6, vq, md, 2.0)
    assert np.allclose(v, [0.0, 1.2])


def test_direction_name():
    assert direction_name(0) == "meandiff_gate0"
    assert direction_name(1) == "meandiff_gate1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mag_steer_conditional.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mag.steer_conditional'`

- [ ] **Step 3: Write the steering implementation**

```python
# src/mag/steer_conditional.py
"""Arm A2 — composed steering: inject the question-mode vector v_Q (gate) together with
tau * mean_diff (content) on free-form factual stems, for OLMo judging. Hypothesis: the truth
axis is causally potent only when the model is in 'being asked about truth' mode.

    python -m mag.steer_conditional --dataset cities --device mps
    python -m mag.steer_conditional --dataset cities --device cpu --limit 2   # smoke
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dct_steer_utils as su
from funnel_utils import unit
from steer_supervised import FACTUAL_PROMPTS

TAUS7 = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
GATES = (0, 1)
MAX_NEW_TOKENS = 8          # matches dct_warm_steer.py so judged runs are comparable


def direction_name(gate):
    return f"meandiff_gate{gate}"


def composed_vec(gate, tau, vq_unit, md_unit, apn):
    """gate*apn*v_Q + tau*apn*mean_diff; None for the untouched (gate=0, tau=0) baseline."""
    if gate == 0 and float(tau) == 0.0:
        return None
    vq = unit(np.asarray(vq_unit, np.float64))
    md = unit(np.asarray(md_unit, np.float64))
    return float(gate) * float(apn) * vq + float(tau) * float(apn) * md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke)")
    a = ap.parse_args()
    ds = a.dataset
    md_npz = np.load(f"mag_dir_{ds}.npz")
    layer = int(md_npz["layer"]); apn = float(md_npz["A_prefix_norm"])
    vq = md_npz["v_Q_unit"]
    td = np.load(f"truth_dir_{ds}.npz")
    assert int(td["layer"]) == layer, (
        f"truth_dir layer {int(td['layer'])} != mag layer {layer}; both vectors must live "
        f"in the same layer space to be co-injected")
    md = td["mean_diff"]
    prompts = FACTUAL_PROMPTS[:a.limit] if a.limit else FACTUAL_PROMPTS
    print(f"[a2] {ds}: layer={layer} apn={apn:.3f} gates={GATES} "
          f"{len(TAUS7)} taus x {len(prompts)} prompts", flush=True)
    tok, model, dev = su.load_model(a.device)
    rows = [("direction", "scale", "prompt", "completion")]
    with su.Steerer(model, layer) as st:
        for gate in GATES:
            for tau in TAUS7:
                vec = composed_vec(gate, tau, vq, md, apn)
                st.set(None if vec is None else torch.tensor(vec, dtype=torch.float32))
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((direction_name(gate), tau, p, c))
                print(f"  gate={gate} tau={tau:+.1f} done", flush=True)
    out = f"mag_conditional_{ds}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[a2] wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the viz implementation**

```python
# src/mag/viz_conditional.py
"""Arm A2 figure: gate0 vs gate1 verdict-vs-tau panels from the judged CSV.

    python -m mag.viz_conditional --dataset cities"""
import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_dct_warm import verdict_fractions, _sorted_taus, _curve, VERDICT_COLOR

PANELS = [("meandiff_gate0", "mean_diff alone (gate off) — known inert"),
          ("meandiff_gate1", "v_Q + mean_diff (gate on) — conditional test")]


def plot_conditional(ds):
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_mag_conditional_{ds}.csv"))))
    tk = _sorted_taus(fr)
    taus = [float(t) for t in tk]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, (d, title) in zip(axes, PANELS):
        for v in ("TRUE", "FALSE", "INCOHERENT"):
            ax.plot(taus, _curve(fr, d, tk, v), "o-", color=VERDICT_COLOR[v],
                    label=v.title(), lw=2, ms=4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
        ax.set_xlabel("tau  (-=push FALSE, +=push TRUE)")
    axes[0].set_ylabel("verdict fraction")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{ds}: does the question-mode gate unlock the truth axis?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"plot_mag_conditional_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_mag_conditional_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_conditional(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add a viz test to the same test file**

Append to `tests/test_mag_steer_conditional.py`:

```python
def test_viz_conditional(tmp_path, monkeypatch):
    import csv as _csv
    monkeypatch.chdir(tmp_path)
    rows = [("direction", "scale", "prompt", "completion", "verdict", "reason")]
    for d in ("meandiff_gate0", "meandiff_gate1"):
        for tau in (-1.0, 0.0, 1.0):
            rows.append((d, tau, "p", "c", "TRUE", ""))
            rows.append((d, tau, "p2", "c2", "FALSE" if d.endswith("1") else "TRUE", ""))
    with open(tmp_path / "judge_mag_conditional_toy.csv", "w", newline="") as f:
        _csv.writer(f).writerows(rows)
    from mag.viz_conditional import plot_conditional
    plot_conditional("toy")
    assert os.path.exists("plot_mag_conditional_toy.png")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mag_steer_conditional.py -v`
Expected: 7 PASS

- [ ] **Step 7: Commit**

```bash
git add src/mag/steer_conditional.py src/mag/viz_conditional.py tests/test_mag_steer_conditional.py
git commit -m "feat(mag): A2 composed v_Q-gate + mean_diff conditional steering and viz"
```

---

### Task 4: `src/export_target_dir.py` (Arm B seeds)

**Files:**
- Create: `src/export_target_dir.py`
- Test: `tests/test_export_target_dir.py`

**Interfaces:**
- Consumes: `dct_meta_<ds>.json` (`target_layer`), `activations/acts_<ds>.npz` via `funnel_utils.load_acts`, `export_concept_dir.concept_directions`.
- Produces: `truth_dir_tgt_<ds>.npz` with keys `mean_diff` (float32 unit), `grad` (float32 unit), `layer` (int == target_layer). Task 6 loads it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_export_target_dir.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_target_dir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_target_dir'`

- [ ] **Step 3: Write the implementation**

```python
# src/export_target_dir.py
"""Supervised truth seeds at the DCT *target* layer, for U-space anchored DCT.
Standalone so export_concept_dir.py and truth_dir_<ds>.npz stay untouched.

    .venv/bin/python src/export_target_dir.py --dataset cities
"""
import argparse
import json
import numpy as np

import funnel_utils as fu
from export_concept_dir import concept_directions


def export(ds):
    tgt = int(json.load(open(f"dct_meta_{ds}.json"))["target_layer"])
    X, y = fu.load_acts(ds, tgt)
    mean_diff, grad = concept_directions(X, y)
    out = f"truth_dir_tgt_{ds}.npz"
    np.savez(out, mean_diff=mean_diff.astype(np.float32),
             grad=grad.astype(np.float32), layer=np.array(tgt))
    print(f"Saved {out}: mean_diff & grad at TARGET layer {tgt} (d={X.shape[1]}); "
          f"cos={float(mean_diff @ grad):+.3f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    export(p.parse_args().dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_target_dir.py tests/test_export_concept_dir.py -v`
Expected: all PASS

- [ ] **Step 5: Run for real (local; activations are on this machine)**

```bash
.venv/bin/python src/export_target_dir.py --dataset cities
.venv/bin/python src/export_target_dir.py --dataset common_claim_true_false
```
Expected: `truth_dir_tgt_cities.npz` (layer 20) and `truth_dir_tgt_common_claim_true_false.npz` (layer 22) created; printed cos(mean_diff, grad) is finite.

- [ ] **Step 6: Commit**

```bash
git add src/export_target_dir.py tests/test_export_target_dir.py
git commit -m "feat(dct): export target-layer supervised seeds for U-space anchoring"
```

---

### Task 5: `src/dct.py` — U-anchor in `fit`

**Files:**
- Modify: `src/dct.py` (fit signature ~line 650, init block ~line 699, update block ~lines 812-818)
- Test: `tests/test_dct_warm_mech.py` (append)

**Interfaces:**
- Consumes: existing `anchor_step` (`dct.py:196`), existing update loop.
- Produces: `ExponentialDCT.fit(..., u_anchor=None, u_anchor_lambda=0.0)` — when set, each iteration's U update gets `anchor_step(U_update, U_anchor, u_anchor_lambda)` on column 0 before normalization. Requires `separate_u=False`. Task 6 calls it.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dct_warm_mech.py`; reuses its `_LinearDelta` stub)

```python
def test_fit_u_anchor_pins_u0_near_target_seed():
    torch.manual_seed(0)
    d_src, d_tgt, n, seq = 6, 6, 4, 3
    W = torch.eye(d_tgt, d_src)
    da = _LinearDelta(W)
    X = torch.randn(n, seq, d_src)
    Y = torch.randn(n, seq, d_tgt)
    u_seed = torch.zeros(d_tgt); u_seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    U, V = m.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="random",
                 u_anchor=u_seed, u_anchor_lambda=5.0, input_scale=1.0, max_iters=5,
                 orthogonalize=True)
    cos = abs(float(torch.dot(U[:, 0] / U[:, 0].norm(), u_seed)))
    assert cos > 0.9                       # strong U-anchor pins factor 0's effect direction
    assert torch.isfinite(V).all() and torch.isfinite(U).all()
    for j in range(4):                     # V columns stay unit (input side fully free)
        assert torch.allclose(V[:, j].norm(), torch.tensor(1.0), atol=1e-4)


def test_fit_u_anchor_zero_lambda_is_inert():
    torch.manual_seed(0)
    d, n, seq = 6, 4, 3
    da = _LinearDelta(torch.eye(d))
    X = torch.randn(n, seq, d); Y = torch.randn(n, seq, d)
    u_seed = torch.zeros(d); u_seed[0] = 1.0
    m = dct.ExponentialDCT(num_factors=4)
    torch.manual_seed(1)                      # reset RNG immediately before each fit so the
    U, V = m.fit(da, X, Y, batch_size=1,      # random inits are identical across the two runs
                 factor_batch_size=4, init="random",
                 u_anchor=u_seed, u_anchor_lambda=0.0, input_scale=1.0, max_iters=3,
                 orthogonalize=True)
    m2 = dct.ExponentialDCT(num_factors=4)
    torch.manual_seed(1)
    U2, V2 = m2.fit(da, X, Y, batch_size=1, factor_batch_size=4, init="random",
                    input_scale=1.0, max_iters=3, orthogonalize=True)
    assert torch.allclose(U, U2) and torch.allclose(V, V2)   # lam=0 == no anchor


def test_fit_u_anchor_rejects_separate_u():
    da = _LinearDelta(torch.eye(4))
    X = torch.randn(2, 3, 4); Y = torch.randn(2, 3, 4)
    m = dct.ExponentialDCT(num_factors=2)
    try:
        m.fit(da, X, Y, batch_size=1, factor_batch_size=2, init="random",
              u_anchor=torch.ones(4), u_anchor_lambda=1.0, input_scale=1.0,
              max_iters=1, separate_u=True)
        assert False, "expected AssertionError"
    except AssertionError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dct_warm_mech.py -v`
Expected: existing tests PASS; the 3 new tests FAIL with `TypeError: fit() got an unexpected keyword argument 'u_anchor'`

- [ ] **Step 3: Implement**

3a. Extend the `fit` signature (`dct.py:650-653`):

```python
    def fit(self, delta_acts_single, X, Y, batch_size=1, factor_batch_size=16, init="random",
            input_scale=1.0, max_iters=10, beta=1.0, orthogonalize=True, deflation=False,
            soft_ortho_temp=1.0, soft_ortho_iterations=10, attention_mask=None, separate_u=False,
            warm_seed=None, anchor_lambda=0.0, u_anchor=None, u_anchor_lambda=0.0):
```

3b. After the existing `self.V_anchor = None` line (~699), add:

```python
        self.u_anchor_lambda = float(u_anchor_lambda)
        self.U_anchor = None
        if u_anchor is not None:
            assert not separate_u, "u_anchor requires separate_u=False (single U matrix)"
            self.U_anchor = F.normalize(
                torch.as_tensor(u_anchor, dtype=torch.float32, device=self.device), dim=0)
```

3c. Replace the U-update line inside the update block (`dct.py:814`, currently `self.U.data = F.normalize(self.beta*G_U+(1-self.beta)*self.U.data, dim=0)`):

```python
                U_update = self.beta * G_U + (1 - self.beta) * self.U.data
                if self.u_anchor_lambda > 0.0 and self.U_anchor is not None:
                    U_update = anchor_step(U_update, self.U_anchor, self.u_anchor_lambda)
                self.U.data = F.normalize(U_update, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dct_warm_mech.py -v`
Expected: all PASS (old + 3 new)

- [ ] **Step 5: Regression-check the V-warm consumers**

Run: `.venv/bin/python -m pytest tests/test_dct_warm.py tests/test_dct_warm_directions.py tests/test_dct_warm_steer.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/dct.py tests/test_dct_warm_mech.py
git commit -m "feat(dct): u_anchor/u_anchor_lambda — anchor factor-0 effect direction in fit"
```

---

### Task 6: `src/dct_warm.py` — `--anchor-space u`

**Files:**
- Modify: `src/dct_warm.py`
- Test: `tests/test_dct_warm.py` (append)

**Interfaces:**
- Consumes: Task 5's `u_anchor` fit params; Task 4's `truth_dir_tgt_<ds>.npz`.
- Produces: CLI `--anchor-space {v,u}` (default `v`, existing behavior byte-identical); functions `load_seed_tgt(ds, seed_name, target_layer)` and `out_stem(anchor_space)` → `"dct_warm"`/`"dct_uwarm"`. In `u` mode writes `dct_uwarm_V_<ds>_<seed>_<lamtag>.pt` and `dct_uwarm_U_...pt`. Task 7 loads these.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dct_warm.py`)

```python
def test_out_stem():
    assert dw.out_stem("v") == "dct_warm"
    assert dw.out_stem("u") == "dct_uwarm"


def test_load_seed_tgt_asserts_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savez("truth_dir_tgt_toy.npz",
             mean_diff=np.array([3.0, 4.0], np.float32),
             grad=np.array([1.0, 0.0], np.float32), layer=np.array(20))
    v = dw.load_seed_tgt("toy", "mean_diff", 20)
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.8])
    try:
        dw.load_seed_tgt("toy", "mean_diff", 22)     # wrong layer
        assert False, "expected AssertionError"
    except AssertionError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dct_warm.py -v`
Expected: existing tests PASS; 2 new FAIL with `AttributeError: module 'dct_warm' has no attribute 'out_stem'`

- [ ] **Step 3: Implement**

3a. Add next to `load_seed` in `src/dct_warm.py`:

```python
def load_seed_tgt(ds, seed_name, target_layer):
    td = np.load(f"truth_dir_tgt_{ds}.npz")
    assert int(td["layer"]) == int(target_layer), (
        f"truth_dir_tgt_{ds}.layer={int(td['layer'])} != dct target_layer={target_layer}; "
        f"U-anchor seed must live in target-layer space")
    return unit(np.asarray(td[seed_name], np.float64))


def out_stem(anchor_space):
    return "dct_uwarm" if anchor_space == "u" else "dct_warm"
```

3b. Add to `parse_args`:

```python
    p.add_argument("--anchor-space", choices=["v", "u"], default="v",
                   help="v: anchor the input direction (original warm-DCT); "
                        "u: anchor the factor-0 EFFECT toward the target-layer truth axis")
```

3c. In `main`, replace the single `seed_vec = load_seed(...)` line (currently before model load) and the `m.fit(...)` call with an anchor-space branch. The seed line becomes:

```python
    if a.anchor_space == "v":
        seed_vec = load_seed(ds, a.seed_name, src)
    else:
        seed_vec = load_seed_tgt(ds, a.seed_name, tgt)
```

and the fit call becomes:

```python
    if a.anchor_space == "v":
        U, V = m.fit(
            delta_acts_single, X, Y, batch_size=1, factor_batch_size=a.factor_batch_size,
            init="warm", warm_seed=torch.as_tensor(seed_vec, dtype=torch.float32),
            anchor_lambda=a.anchor_lambda, input_scale=input_scale, max_iters=num_iters,
            beta=1.0, orthogonalize=True, deflation=False,
            attention_mask=cpu_mask.to(delta_acts_single.device), separate_u=False)
    else:
        U, V = m.fit(
            delta_acts_single, X, Y, batch_size=1, factor_batch_size=a.factor_batch_size,
            init="random", u_anchor=torch.as_tensor(seed_vec, dtype=torch.float32),
            u_anchor_lambda=a.anchor_lambda, input_scale=input_scale, max_iters=num_iters,
            beta=1.0, orthogonalize=True, deflation=False,
            attention_mask=cpu_mask.to(delta_acts_single.device), separate_u=False)
```

3d. Save with the stem (replace the two `torch.save` lines and print):

```python
    stem = out_stem(a.anchor_space)
    tag = lam_tag(a.anchor_lambda)
    torch.save(V.detach().cpu(), f"{stem}_V_{ds}_{a.seed_name}_{tag}.pt")
    torch.save(U.detach().cpu(), f"{stem}_U_{ds}_{a.seed_name}_{tag}.pt")
    print(f"[warm] wrote {stem}_V_{ds}_{a.seed_name}_{tag}.pt  V{tuple(V.shape)}")
```

Also update the `[warm]` startup print to include `space={a.anchor_space}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dct_warm.py tests/test_dct_warm_mech.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dct_warm.py tests/test_dct_warm.py
git commit -m "feat(dct): --anchor-space u — U-anchored warm fits with dct_uwarm_ artifacts"
```

---

### Task 7: `src/dct_uwarm_directions.py`

**Files:**
- Create: `src/dct_uwarm_directions.py`
- Test: `tests/test_dct_uwarm_directions.py`

**Interfaces:**
- Consumes: `dct_uwarm_V/U_<ds>_mean_diff_<lamtag>.pt` (Task 6), `truth_dir_<ds>.npz`, `truth_dir_tgt_<ds>.npz`, `funnel_utils.load_dct/top_k_by_potency/unit`, `dct_warm.lam_tag`.
- Produces: `dct_uwarm_dirs_<ds>.npz` (name → float32 unit vec at SOURCE layer; names `raw_mean_diff`, `cold_top`, `uwarm_mean_diff_lam0p3/lam1/lam3`) and `dct_uwarm_geometry_<ds>.csv` (`direction,cos_U0_mdtgt,cos_V0_mdsrc` — signed cosines BEFORE sign-alignment; NaN for non-uwarm rows' cos_U0). Sign convention: each uwarm V0 is flipped so its U0 effect points toward TRUE (`U0 · mean_diff@tgt ≥ 0`). Function `assemble(ds, lams)` → `(dirs_dict, geometry_rows)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dct_uwarm_directions.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dct_uwarm_directions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dct_uwarm_directions'`

- [ ] **Step 3: Write the implementation**

```python
# src/dct_uwarm_directions.py
"""Assemble U-anchored warm-DCT candidate directions + their U/V geometry.

    python dct_uwarm_directions.py --dataset cities
Writes dct_uwarm_dirs_<ds>.npz (name -> unit vec at source layer) and
dct_uwarm_geometry_<ds>.csv (direction, cos_U0_mdtgt, cos_V0_mdsrc).

Sign convention: each uwarm V0 is flipped so its factor-0 EFFECT points toward TRUE
(U0 . mean_diff@target >= 0); the geometry CSV records the raw signed cosines pre-flip."""
import argparse
import csv
import os
import numpy as np
import torch

import funnel_utils as fu
from funnel_utils import unit
from dct_warm import lam_tag

LAMS = (0.3, 1.0, 3.0)      # lam=0 == the cold run, already covered by cold_top


def assemble(ds, lams=LAMS):
    td = np.load(f"truth_dir_{ds}.npz")
    md_src = unit(np.asarray(td["mean_diff"], np.float64))
    tt = np.load(f"truth_dir_tgt_{ds}.npz")
    md_tgt = unit(np.asarray(tt["mean_diff"], np.float64))
    dirs = {"raw_mean_diff": md_src}
    rows = [("raw_mean_diff", float("nan"), 1.0)]
    Vc, Uc, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(Vc, Uc, 1)[0]
    cold = unit(Vc[:, top].astype(np.float64))
    cold = cold if float(cold @ md_src) >= 0 else -cold
    dirs["cold_top"] = cold
    rows.append(("cold_top", float("nan"), float(cold @ md_src)))
    for lam in lams:
        tag = lam_tag(lam)
        vp = f"dct_uwarm_V_{ds}_mean_diff_{tag}.pt"
        up = f"dct_uwarm_U_{ds}_mean_diff_{tag}.pt"
        if not (os.path.exists(vp) and os.path.exists(up)):
            print(f"[udirs] WARN: missing {vp} or {up}; skipping (uwarm fit likely failed)")
            continue
        v0 = unit(torch.load(vp, map_location="cpu").float().numpy()[:, 0].astype(np.float64))
        u0 = unit(torch.load(up, map_location="cpu").float().numpy()[:, 0].astype(np.float64))
        cos_u = float(u0 @ md_tgt)
        cos_v = float(v0 @ md_src)
        name = f"uwarm_mean_diff_{tag}"
        dirs[name] = v0 if cos_u >= 0 else -v0     # effect points toward TRUE
        rows.append((name, cos_u, cos_v))
    return dirs, rows


def write_outputs(ds, lams=LAMS):
    dirs, rows = assemble(ds, lams)
    np.savez(f"dct_uwarm_dirs_{ds}.npz",
             **{k: v.astype(np.float32) for k, v in dirs.items()})
    with open(f"dct_uwarm_geometry_{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "cos_U0_mdtgt", "cos_V0_mdsrc"))
        w.writerows(rows)
    print(f"[udirs] wrote dct_uwarm_dirs_{ds}.npz ({len(dirs)} dirs) and "
          f"dct_uwarm_geometry_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    write_outputs(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dct_uwarm_directions.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dct_uwarm_directions.py tests/test_dct_uwarm_directions.py
git commit -m "feat(dct): assemble U-anchored directions with effect-sign convention + geometry"
```

---

### Task 8: `src/dct_warm_steer.py` — `--dirs-file` / `--out`

**Files:**
- Modify: `src/dct_warm_steer.py`
- Test: `tests/test_dct_warm_steer.py` (append)

**Interfaces:**
- Consumes: existing script (reads a dirs npz, writes `direction,scale,prompt,completion`).
- Produces: `resolve_paths(ds, dirs_file=None, out=None)` → `(dirs_path, out_path)` with defaults `dct_warm_dirs_<ds>.npz` / `dct_warm_steer_<ds>.csv`; CLI `--dirs-file`, `--out`. Task 10's uwarm slurm passes `--dirs-file dct_uwarm_dirs_<ds>.npz --out dct_uwarm_steer_<ds>.csv`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_dct_warm_steer.py`)

```python
def test_resolve_paths_defaults_and_overrides():
    import dct_warm_steer as dws
    assert dws.resolve_paths("cities") == ("dct_warm_dirs_cities.npz",
                                           "dct_warm_steer_cities.csv")
    assert dws.resolve_paths("cities", "dct_uwarm_dirs_cities.npz",
                             "dct_uwarm_steer_cities.csv") == (
        "dct_uwarm_dirs_cities.npz", "dct_uwarm_steer_cities.csv")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dct_warm_steer.py -v`
Expected: existing PASS; new FAIL with `AttributeError: ... has no attribute 'resolve_paths'`

- [ ] **Step 3: Implement**

Add to `src/dct_warm_steer.py` (below `injected`):

```python
def resolve_paths(ds, dirs_file=None, out=None):
    return (dirs_file or f"dct_warm_dirs_{ds}.npz",
            out or f"dct_warm_steer_{ds}.csv")
```

In `main()`, add args and use them:

```python
    ap.add_argument("--dirs-file", default=None, help="directions npz (default warm dirs)")
    ap.add_argument("--out", default=None, help="output csv (default dct_warm_steer_<ds>.csv)")
```

Replace `dirs = np.load(f"dct_warm_dirs_{ds}.npz")` with:

```python
    dirs_path, out_path = resolve_paths(ds, a.dirs_file, a.out)
    dirs = np.load(dirs_path)
```

and the final write block's path with `out_path` (both the `open(...)` and the printed name).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dct_warm_steer.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dct_warm_steer.py tests/test_dct_warm_steer.py
git commit -m "feat(dct): parameterize warm steer script over dirs npz and output csv"
```

---

### Task 9: `src/viz_dct_uwarm.py`

**Files:**
- Create: `src/viz_dct_uwarm.py`
- Test: `tests/test_viz_dct_uwarm.py`

**Interfaces:**
- Consumes: `judge_dct_uwarm_steer_<ds>.csv` (judge schema + `verdict`), `dct_uwarm_geometry_<ds>.csv` (Task 7), reuses `viz_dct_warm.verdict_fractions/_sorted_taus/_curve/VERDICT_COLOR`.
- Produces: `plot_dct_uwarm_curves_<ds>.png` (5 panels: raw_mean_diff, cold_top, uwarm λ=0.3/1/3) and `plot_dct_uwarm_geometry_<ds>.png` (cos ladders vs λ).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viz_dct_uwarm.py
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import viz_dct_uwarm as vu


def _toy_files(tmp_path):
    rows = [("direction", "scale", "prompt", "completion", "verdict", "reason")]
    for d, _ in vu.PANELS:
        for tau in (-1.0, 0.0, 1.0):
            rows.append((d, tau, "p", "c", "TRUE", ""))
            rows.append((d, tau, "p2", "c2", "INCOHERENT", ""))
    with open(tmp_path / "judge_dct_uwarm_steer_toy.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(tmp_path / "dct_uwarm_geometry_toy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("direction", "cos_U0_mdtgt", "cos_V0_mdsrc"))
        w.writerow(("raw_mean_diff", "nan", "1.0"))
        w.writerow(("cold_top", "nan", "0.05"))
        w.writerow(("uwarm_mean_diff_lam0p3", "0.2", "0.1"))
        w.writerow(("uwarm_mean_diff_lam1", "0.6", "0.15"))
        w.writerow(("uwarm_mean_diff_lam3", "0.9", "0.2"))


def test_panels_cover_uwarm_lams():
    names = [d for d, _ in vu.PANELS]
    assert names == ["raw_mean_diff", "cold_top", "uwarm_mean_diff_lam0p3",
                     "uwarm_mean_diff_lam1", "uwarm_mean_diff_lam3"]


def test_plots_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toy_files(tmp_path)
    vu.plot_curves("toy")
    vu.plot_geometry("toy")
    assert os.path.exists("plot_dct_uwarm_curves_toy.png")
    assert os.path.exists("plot_dct_uwarm_geometry_toy.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_viz_dct_uwarm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'viz_dct_uwarm'`

- [ ] **Step 3: Write the implementation**

```python
# src/viz_dct_uwarm.py
"""U-anchored warm-DCT figures: verdict-vs-tau panels and the U/V geometry ladder.

    python viz_dct_uwarm.py --dataset cities"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_dct_warm import verdict_fractions, _sorted_taus, _curve, VERDICT_COLOR
from dct_warm import lam_tag

PANELS = [
    ("raw_mean_diff", "supervised axis (mean_diff)"),
    ("cold_top", "cold DCT top factor (free)"),
    ("uwarm_mean_diff_lam0p3", "U-anchored λ=0.3"),
    ("uwarm_mean_diff_lam1", "U-anchored λ=1"),
    ("uwarm_mean_diff_lam3", "U-anchored λ=3"),
]
UWARM_LAMS = [0.3, 1.0, 3.0]


def plot_curves(ds):
    fr = verdict_fractions(list(csv.DictReader(open(f"judge_dct_uwarm_steer_{ds}.csv"))))
    tk = _sorted_taus(fr)
    taus = [float(t) for t in tk]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (d, title) in zip(flat, PANELS):
        for v in ("TRUE", "FALSE", "INCOHERENT"):
            ax.plot(taus, _curve(fr, d, tk, v), "o-", color=VERDICT_COLOR[v],
                    label=v.title(), lw=2, ms=4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    for ax in flat[len(PANELS):]:
        ax.axis("off")
    flat[0].legend(fontsize=9, loc="center left")
    for ax in axes[1]:
        ax.set_xlabel("τ  (−=push FALSE, +=push TRUE)")
    for r in (0, 1):
        axes[r][0].set_ylabel("verdict fraction")
    fig.suptitle(f"{ds}: U-anchored DCT — verdict vs steering strength τ\n"
                 f"(input free, effect anchored toward the truth readout)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"plot_dct_uwarm_curves_{ds}.png", dpi=140)
    print(f"[viz] wrote plot_dct_uwarm_curves_{ds}.png")


def plot_geometry(ds):
    rows = list(csv.DictReader(open(f"dct_uwarm_geometry_{ds}.csv")))
    by = {r["direction"]: r for r in rows}
    cos_u, cos_v = [], []
    for lam in UWARM_LAMS:
        r = by.get(f"uwarm_mean_diff_{lam_tag(lam)}")
        cos_u.append(abs(float(r["cos_U0_mdtgt"])) if r else np.nan)
        cos_v.append(abs(float(r["cos_V0_mdsrc"])) if r else np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(UWARM_LAMS, cos_u, "o-", color="#aa3377",
            label="|cos(U₀, mean_diff@target)| — does the anchor knob work?")
    ax.plot(UWARM_LAMS, cos_v, "s-", color="#4477aa",
            label="|cos(V₀, mean_diff@source)| — what input gets chosen?")
    ax.set_xlabel("U-anchor λ"); ax.set_ylabel("|cos|"); ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{ds}: effect-space anchoring geometry")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"plot_dct_uwarm_geometry_{ds}.png", dpi=150)
    print(f"[viz] wrote plot_dct_uwarm_geometry_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    plot_curves(a.dataset)
    plot_geometry(a.dataset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_viz_dct_uwarm.py tests/test_viz_dct_warm.py -v`
Expected: all PASS (uwarm viz + no regression in the shared helpers' home module)

- [ ] **Step 5: Commit**

```bash
git add src/viz_dct_uwarm.py tests/test_viz_dct_uwarm.py
git commit -m "feat(dct): uwarm verdict-curve and geometry-ladder figures"
```

---

### Task 10: Slurm scripts + runbook

**Files:**
- Create: `deltaai/run_mag_cond_judge.slurm`, `deltaai/run_dct_uwarm.slurm`, `deltaai/run_dct_uwarm_judge.slurm`, `docs/CONDITIONAL_UANCHOR_RUNBOOK.md`

**Interfaces:**
- Consumes: everything above; existing `deltaai/run_dct_warm.slurm` / `run_dct_warm_judge.slurm` header conventions; `judge_results.py --mode steer --steer-input/--steer-output/--steer-plot` flags.
- Produces: three sbatch files (syntax-checked) and the runbook.

- [ ] **Step 1: Write `deltaai/run_dct_uwarm.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=dct_uwarm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=dct_uwarm_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-dct-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  for lam in 0.3 1.0 3.0; do
    echo "=== uwarm train $ds mean_diff lam=$lam $(date) ==="
    PYTHONPATH=src python3 src/dct_warm.py --dataset "$ds" --seed-name mean_diff \
      --anchor-space u --anchor-lambda "$lam" --device cuda \
      || echo "!!!! $ds uwarm $lam FAILED — continuing"
  done
  echo "=== assemble uwarm directions $ds ==="
  PYTHONPATH=src python3 src/dct_uwarm_directions.py --dataset "$ds"
  echo "=== uwarm steer $ds ==="
  PYTHONPATH=src python3 src/dct_warm_steer.py --dataset "$ds" --device cuda \
    --dirs-file "dct_uwarm_dirs_${ds}.npz" --out "dct_uwarm_steer_${ds}.csv" \
    || echo "!!!! uwarm steer $ds FAILED — continuing"
done
echo "=== dct uwarm done $(date) ==="
```

- [ ] **Step 2: Write `deltaai/run_dct_uwarm_judge.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=dct_uwarm_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=dct_uwarm_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== uwarm judge $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "dct_uwarm_steer_${ds}.csv" \
    --steer-output "judge_dct_uwarm_steer_${ds}.csv" \
    --steer-plot "plot_judge_dct_uwarm_${ds}.png" \
    || echo "!!!! $ds FAILED — continuing"
done
echo "=== dct uwarm judge done $(date) ==="
```

- [ ] **Step 3: Write `deltaai/run_mag_cond_judge.slurm`** (A0 backfill + A2)

```bash
#!/bin/bash
#SBATCH --job-name=mag_cond_judge
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=mag_cond_judge_%j.out
set -e
module load python/miniforge3_pytorch
source .venv-judge-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRANSFORMERS_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi
for ds in cities common_claim_true_false; do
  echo "=== A0 backfill: judge existing MAG E4 free-form $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "mag_steer_${ds}.csv" \
    --steer-output "judge_mag_steer_${ds}.csv" \
    --steer-plot "plot_judge_mag_steer_${ds}.png" \
    || echo "!!!! A0 $ds FAILED — continuing"
  echo "=== A2: judge conditional steering $ds $(date) ==="
  PYTHONPATH=src python3 src/judge_results.py --mode steer --backend olmo --device cuda \
    --dataset "$ds" \
    --steer-input "mag_conditional_${ds}.csv" \
    --steer-output "judge_mag_conditional_${ds}.csv" \
    --steer-plot "plot_judge_mag_conditional_${ds}.png" \
    || echo "!!!! A2 $ds FAILED — continuing"
done
echo "=== mag cond judge done $(date) ==="
```

- [ ] **Step 4: Syntax-check all three**

Run: `bash -n deltaai/run_dct_uwarm.slurm deltaai/run_dct_uwarm_judge.slurm deltaai/run_mag_cond_judge.slurm && echo OK`
Expected: `OK`

- [ ] **Step 5: Write `docs/CONDITIONAL_UANCHOR_RUNBOOK.md`**

```markdown
# Conditional Steering + U-Anchored DCT — Runbook

Spec: `docs/superpowers/specs/2026-07-23-conditional-steering-uspace-anchor-design.md`.
Datasets: cities, common_claim_true_false. Sign convention: +τ→TRUE, −τ→FALSE.

## Phase 1 — local (laptop, .venv, MPS)

1. Target-layer seeds (Arm B prerequisite, ~seconds):
   `.venv/bin/python src/export_target_dir.py --dataset cities`
   `.venv/bin/python src/export_target_dir.py --dataset common_claim_true_false`
2. Arm A1 smoke, then full (~1-2 h/dataset on MPS):
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset cities --device mps --limit 2 --only sup_mean_diff`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset cities --device mps`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_verdict --dataset common_claim_true_false --device mps`
3. A1 figures (immediately — A1 needs no judge):
   `PYTHONPATH=src .venv/bin/python -m mag.viz_verdict --dataset cities`
   `PYTHONPATH=src .venv/bin/python -m mag.viz_verdict --dataset common_claim_true_false`
4. Arm A2 generation (~1 h/dataset):
   `PYTHONPATH=src .venv/bin/python -m mag.steer_conditional --dataset cities --device mps`
   `PYTHONPATH=src .venv/bin/python -m mag.steer_conditional --dataset common_claim_true_false --device mps`

## Phase 2 — rsync up (one batch: one NCSA password + Duo push)

Code + inputs (from repo root; adjust remote path to the existing project dir):

    rsync -av --relative src deltaai tests docs \
      truth_dir_tgt_cities.npz truth_dir_tgt_common_claim_true_false.npz \
      mag_conditional_cities.csv mag_conditional_common_claim_true_false.csv \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/

(`mag_steer_*.csv` for A0 are already on the cluster from the MAG run; if not, add them.)

## Phase 3 — cluster jobs (GH200)

    sbatch deltaai/run_mag_cond_judge.slurm     # A0 + A2 judging (needs mag_conditional_*.csv)
    sbatch deltaai/run_dct_uwarm.slurm          # Arm B: 6 fits + directions + steer
    # after run_dct_uwarm completes:
    sbatch deltaai/run_dct_uwarm_judge.slurm

`run_mag_cond_judge` and `run_dct_uwarm` are independent — submit together.

## Phase 4 — rsync back (one batch; excludes .pt/.npz weights, matching the two-rsync pattern)

    rsync -av --exclude '*.pt' --exclude '*.npz' \
      USER@dt-login.delta.ncsa.illinois.edu:~/PROJECT_DIR/'judge_mag_steer_*.csv judge_mag_conditional_*.csv judge_dct_uwarm_steer_*.csv dct_uwarm_geometry_*.csv plot_judge_*.png *.out' \
      ./

## Phase 5 — figures + interpretation (local)

    PYTHONPATH=src .venv/bin/python -m mag.viz_conditional --dataset cities
    PYTHONPATH=src .venv/bin/python -m mag.viz_conditional --dataset common_claim_true_false
    PYTHONPATH=src .venv/bin/python src/viz_dct_uwarm.py --dataset cities
    PYTHONPATH=src .venv/bin/python src/viz_dct_uwarm.py --dataset common_claim_true_false

Read results against the interpretation matrix in the spec (§Interpretation matrix).

## Outputs checklist

| Arm | Data | Figures |
|---|---|---|
| A0 | judge_mag_steer_{ds}.csv | plot_judge_mag_steer_{ds}.png |
| A1 | mag_verdict_logits_{ds}.csv | plot_mag_verdict_margin/acc_{ds}.png |
| A2 | mag_conditional_{ds}.csv, judge_mag_conditional_{ds}.csv | plot_mag_conditional_{ds}.png |
| B | dct_uwarm_geometry_{ds}.csv, judge_dct_uwarm_steer_{ds}.csv | plot_dct_uwarm_curves/geometry_{ds}.png |
```

- [ ] **Step 6: Full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_spectrum_utils.py --ignore=tests/test_viz_spectrum.py`
Expected: all PASS (spectrum tests excluded — they belong to unrelated uncommitted work; run them separately unmodified if quick confirmation is wanted)

- [ ] **Step 7: Commit**

```bash
git add deltaai/run_mag_cond_judge.slurm deltaai/run_dct_uwarm.slurm deltaai/run_dct_uwarm_judge.slurm docs/CONDITIONAL_UANCHOR_RUNBOOK.md
git commit -m "feat(run): slurm jobs + runbook for conditional-steering and U-anchor round"
```

---

## Self-Review Notes

- **Spec coverage:** A0 → Task 10 (`run_mag_cond_judge.slurm`); A1 → Tasks 1-2; A2 → Task 3; B seeds → Task 4; B anchor mechanics → Tasks 5-6; B directions/steer/viz → Tasks 7-9; cluster + runbook → Task 10. The spec's interpretation matrix is referenced from the runbook.
- **Type consistency:** `build_directions` returns `(name, unit_vec, layer)` triples consumed only inside Task 1; `assemble` returns `(dirs_dict, rows)` consumed by `write_outputs` in the same file; `resolve_paths` tuple order `(dirs_path, out_path)` matches Task 10's flags.
- **Deliberate scope cuts (per spec):** no `grad` seeds in Arm B; `truth_dir_tgt` stores `grad` anyway (free, one line); λ_u=0 not trained (≡ cold run).
- The `verdict_rows` helper deliberately lives in `steer_verdict.py` rather than modifying shared `mag/verdict.py` — A1 needs per-row p_yes/p_no, and leaving `compute_verdicts` untouched keeps the committed MAG battery byte-identical.
