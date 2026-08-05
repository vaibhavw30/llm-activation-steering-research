# Token-Space Conclusions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the finished token-space cluster artifacts into a claim-by-claim verdict document for the PI, backed by a tested module holding every computation whose number appears in that document.

**Architecture:** A tested module `src/token_conclusions.py` carries the four computations that planning has already confirmed produce reportable numbers (shared-statement restriction, the legacy readout reconstruction, margin consumption, completion forensics). Scratchpad scripts carry the five analyses whose value is still unknown. Both feed `docs/TOKEN_SPACE_RAW_FINDINGS.md`, which is then read cold to write `docs/TOKEN_SPACE_FINDINGS.md`.

**Tech Stack:** Python 3.13 in `./.venv`, pandas, numpy, `difflib` from the standard library, pytest. No torch, no xgboost, no model loads, no cluster.

---

## Deviation from the spec, and why

The spec (`docs/superpowers/specs/2026-08-04-token-space-conclusions-design.md`) ordered extraction first and promotion second, on the reasoning that writing tests for nine analyses when perhaps five produce reportable numbers is waste. That reasoning is sound and is preserved for the five analyses whose value is still unknown.

Planning ran the extraction for A1, A2, A3, A5, A6, A7 and A8 in order to ground the spec's assumptions. Those numbers are now known, so for four computations the question the spec deferred is already answered, and they go straight into the tested module. The numbers appear in the tasks below as expected values, which makes each task verifiable rather than exploratory.

That same grounding pass falsified six statements in the spec. Each correction is carried into Global Constraints below.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Prose and git**

- No em dashes in any prose written by this plan.
- Never `git add -A` or `git add .`. Stage named files only.
- Never stage the other track's uncommitted files: `src/spectrum_utils.py`, `src/viz_spectrum.py`, `tests/test_spectrum_utils.py`, `tests/test_viz_spectrum.py`, `docs/RESULTS_SINCE_LAST_MEETING_PART2.md`, `docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md`, `plot_mag_linearity_v2.png`.
- `git push` is blocked by the permission classifier and is the researcher's to run.

**Data handling**

- Every script here READS only. Never re-run, regenerate or overwrite any existing artifact.
- Never import `xgboost` in the same process as `torch`.
- Never index into a loaded `.npz` inside a loop. Bind every member to a local array once.
- **Read every `token_steer_*.csv` through `token_conclusions.load_arm`, never a bare `pd.read_csv`.** It applies `keep_default_na=False, na_values=[]` and normalises the delta-column header. With pandas defaults the 740 empty completions in `token_steer_cities_postnorm_all_rp1.csv` silently become `NaN`, which destroys the distinction the analysis depends on. See correction 3. Numeric dtypes are unaffected: verified that `scale`, `frac`, `frac_margin` and `readout_delta` still parse as `float64` under this setting, so no casts are needed.
- Run everything with `./.venv/bin/python`. There is no bare `python` on this machine.

**Corrections to the spec, established by measurement during planning. Seven items; the last is a change that landed mid-plan rather than an error.**

1. **The layer arms do not contain `oracle` or `md_full`.** Spec §3 says the layer arms carry `oracle`, `md_full`, `jtw_token`. Measured: `token_steer_cities_layer16_*.csv` and `token_steer_common_claim_true_false_layer8_*.csv` carry exactly `jtw_legacy` and `jtw_token`. The post-norm and pre-norm arms carry `oracle`, `md_full`, `jtw_legacy`.

2. **`margins` in `reach_margins_<ds>.npz` is `m = ||J^T w||`, not `eps*`.** Spec A1 says "read `eps*_legacy` per statement from `margins`". There is no `eps*` on disk anywhere. It must be reconstructed as `eps* = g / m` with `g = w.h_tgt - t02`, exactly as `src/reach_analyze.py:24` `required_eps` defines it. `w` and `t02` come from `reach_dirs_<ds>.npz` (`W`, `thresh02`) at the index where `names == "mean_diff_tgt"`; `h_tgt` and the dataset row index come from `reach_acts_<ds>.npz`.

3. **There are no missing completions, only empty ones.** Spec test 3 says "Null completions are excluded, not counted as degenerate". Measured on `token_steer_cities_postnorm_all_rp1.csv`: 5,400 rows, 4,660 non-empty, 740 empty strings, and 4,660 + 740 = 5,400. Nothing was ever "not generated". All 740 belong to `md_full`, 739 of them at negative fracs where `|scale|` runs from 409 to 1,636, and their `argmax_tok` is `\n\n` (505), `\n` (219), `<eos>` (15) or a space (1). The model emitted a newline and stopped. **An empty completion is the loudest evidence of `md_full`'s failure mode and must be counted as degenerate, not dropped.**

4. **`broadcast_gain` at cities layer 16 is 1.355, not 1.55.** Measured from `token_jac_cities.csv`. Layer 8 is 1.720. The `m_all_median / m_last_median` ratio at layer 16 is 1.380. None of these is 1.55, so the figure carried in the project record is wrong and A5 must use 1.355.

5. **`jtw_legacy`'s swept scale is `frac * delta_cone`, identical at every site.** `src/token_steer.py:316` builds `alpha_of` only for directions with an `alpha_<name>` column in `token_geom_<ds>.csv`. Those columns are `alpha_md_full`, `alpha_mean_diff_tgt_asis`, `alpha_probe_grad_tgt_asis`. There is no `alpha_jtw_legacy`, so line 338's rescale never fires for it and `required_scale` returns `delta_cone[i]` unchanged. Consequence: the `crossed` column is byte-identical across all four cities arms, so **A1 produces one readout table per dataset, not one per arm.** Only `flipped` varies across arms, and it is 0.000 in every one.

6. **The `readout_delta` column has been renamed at the source but not in the files.** Task `task_427f63e2` landed during planning. `src/token_steer.py` now writes `tgt_minus_top_delta` and exposes `LEGACY_COLUMN_ALIASES` and `load_steer_csv`. The twelve CSVs on disk were correctly left alone and still carry the old header, so both spellings are live and `load_arm` must normalise. Note `token_steer.load_steer_csv` cannot be used directly here: it calls a bare `pd.read_csv` and would turn the 740 empty completions into NaN, and importing the module pulls in torch.

7. **The SAE artifacts decompose different vectors at different layers.** `sae_features_cities.csv` carries `w_mean_diff_tgt` at layer 20 and `jtw_mean`, `jtw_stem_mean`, `jtw_full_matched_mean`, `common_v1`, `V64_common_0..3` at layer 11. GemmaScope layer-11 and layer-20 dictionaries are unrelated, so **a feature-ID overlap between the truth readout and the steering vectors cannot be computed from these files.** The spec's "prior expectation of overlap near 0.05" is not testable here and A6 must say so instead of reporting a number.

**Measured values the tasks assert against**

| Quantity | cities | common_claim |
|---|---|---|
| `eps*_legacy` median, label 1 | 0.660 | 5.81 |
| median `scale / eps*_legacy` at frac 1.0, label 1 | 2.42 | 0.172 |
| `crossed` fraction at frac 1.0, label 1 | 0.881 (n=101) | 0.068 (n=44) |
| `jtw_legacy` hit rate, every arm, every frac | 0.000 | 0.000 |
| `jtw_legacy` median \|`frac_margin`\| at frac 1.0, postnorm | 0.0174 | 0.0174 |
| `md_full` median \|`frac_margin`\| at frac 1.0, postnorm | 1.088 | 1.001 |
| `oracle` hit rate at frac 1.0, postnorm | 1.000 | 1.000 |
| `broadcast_gain` at the swept layer | 1.355 (L16) | 1.720 (L8) |
| median `rmsnorm_penalty` | 4.380 | 5.177 |
| median `delta_cone` | 6.369 | 1.738 |
| median `delta_rel_z` | 0.0339 | 0.0106 |

---

## File Structure

| Path | Responsibility | Created by |
|---|---|---|
| `src/token_conclusions.py` | The four computations behind numbers in the verdict document | Tasks 1 to 4 |
| `tests/test_token_conclusions.py` | Synthetic-frame tests for those computations | Tasks 1 to 4 |
| `<scratchpad>/a4_alpha.py`, `a5_broadcast.py`, `a6_sae.py`, `a7_split.py`, `a8_prenorm.py` | The five analyses whose value is unknown | Task 5 |
| `<scratchpad>/extract_all.py` | Drives the module plus the five scripts, writes the raw dump | Task 5 |
| `docs/TOKEN_SPACE_RAW_FINDINGS.md` | Unpolished Pass 1 output | Task 5 |
| `docs/TOKEN_SPACE_FINDINGS.md` | Claim-by-claim verdict document | Task 6 |

`<scratchpad>` is `/private/tmp/claude-501/-Users-vaibhav-wudaru-llm-activation-steering-research/c8145135-7f79-4eff-81b2-f40f140cba80/scratchpad`.

---

### Task 1: Arm loading and shared-statement restriction

The foundation every later computation sits on. `jtw_legacy` covers 90 of 200 statements on common_claim because `reach_margins` was computed on a 2,000-row stratified subsample of a 4,450-row dataset. Any cross-direction statistic that does not restrict to the intersection first compares different statement populations.

**Files:**
- Create: `src/token_conclusions.py`
- Test: `tests/test_token_conclusions.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_arm(path) -> pd.DataFrame`, `shared_statements(df) -> list[int]`, `restrict_to_shared(df) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_token_conclusions.py`:

```python
"""Tests for src/token_conclusions.py.

Every test builds its own small DataFrame. None of them read a real artifact, so the
suite runs in well under a second and does not break if an artifact is ever
regenerated with different row counts.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import token_conclusions as tc                                      # noqa: E402


def _arm(rows):
    """rows: (direction, stmt, frac, scale, hit_target, frac_margin, completion)."""
    return pd.DataFrame(rows, columns=["direction", "stmt", "frac", "scale",
                                       "hit_target", "frac_margin", "completion"])


def test_the_shared_set_is_the_intersection_not_the_smallest_direction():
    """Two directions each short in a different place must yield the intersection.

    Taking the smaller direction's set would return {1, 2} here, which contains a
    statement direction B never ran.
    """
    df = _arm([("A", 1, 1.0, 1.0, 0, 0.1, "x"),
               ("A", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("A", 3, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 3, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 4, 1.0, 1.0, 0, 0.1, "x")])
    assert tc.shared_statements(df) == [2, 3]


def test_restriction_changes_the_rate_when_coverage_differs():
    """The defect this guards against is silent: an unrestricted mean is still a
    number, just a number about a different population."""
    df = _arm([("A", 1, 1.0, 1.0, 1, 0.1, "x"),
               ("A", 2, 1.0, 1.0, 0, 0.1, "x"),
               ("B", 1, 1.0, 1.0, 0, 0.1, "x")])
    assert df.groupby("direction").hit_target.mean()["A"] == pytest.approx(0.5)
    r = tc.restrict_to_shared(df)
    assert r.groupby("direction").hit_target.mean()["A"] == pytest.approx(1.0)


def test_load_arm_keeps_empty_completions_as_empty_strings(tmp_path):
    """pandas turns an empty CSV field into NaN by default. All 740 empty completions
    in the cities post-norm arm are md_full emitting a newline and stopping, which is
    the evidence, not missing data. Losing them to NaN would drop it."""
    p = tmp_path / "arm.csv"
    p.write_text("direction,stmt,frac,scale,hit_target,frac_margin,completion\n"
                 "md_full,1,-1.0,-500.0,0,-1.0,\n"
                 "oracle,1,1.0,6.0,1,1.0,in China\n")
    df = tc.load_arm(str(p))
    assert df.completion.iloc[0] == ""
    assert df.completion.isna().sum() == 0


def test_load_arm_normalises_the_delta_column_under_either_header(tmp_path):
    """src/token_steer.py now writes `tgt_minus_top_delta`; the twelve files on disk
    still carry `readout_delta`. Both must load to the same column name or every
    downstream lookup breaks on whichever generation it was not written for."""
    old = tmp_path / "old.csv"
    old.write_text("direction,stmt,frac,scale,hit_target,frac_margin,readout_delta\n"
                   "A,1,1.0,1.0,0,0.01,0.13\n")
    new = tmp_path / "new.csv"
    new.write_text("direction,stmt,frac,scale,hit_target,frac_margin,tgt_minus_top_delta\n"
                   "A,1,1.0,1.0,0,0.01,0.13\n")
    assert "tgt_minus_top_delta" in tc.load_arm(str(old)).columns
    assert "readout_delta" not in tc.load_arm(str(old)).columns
    assert "tgt_minus_top_delta" in tc.load_arm(str(new)).columns


def test_the_alias_map_matches_token_steer():
    """token_steer.LEGACY_COLUMN_ALIASES is the source of truth and is duplicated in
    token_conclusions to avoid a 4.2s torch import for a one-entry dict. Read the
    literal out of the source text rather than importing, so this test stays fast and
    still fails loudly if the two ever drift."""
    src = os.path.join(os.path.dirname(__file__), "..", "src", "token_steer.py")
    with open(src) as f:
        line = next(l for l in f if l.startswith("LEGACY_COLUMN_ALIASES"))
    assert ast.literal_eval(line.split("=", 1)[1].strip()) == tc.LEGACY_COLUMN_ALIASES
```

Add `import ast` to the test file's imports.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'token_conclusions'`.

- [ ] **Step 3: Write the module**

Create `src/token_conclusions.py`:

```python
"""token_conclusions.py: the computations behind every number in
docs/TOKEN_SPACE_FINDINGS.md.

Read-only. Nothing here writes, regenerates or overwrites an artifact.

Design note on coverage. `jtw_legacy` ran on 90 of 200 statements on common_claim,
because reach_margins was computed on a 2,000-row stratified subsample of a 4,450-row
dataset while token_geom sampled the full set. Every cross-direction statistic in this
module therefore restricts to the statements all directions ran. src/viz_token.py
fig_steer already does this for the figure and tests/test_viz_token.py pins it; this is
the same rule applied to the tables.
"""
import os

import numpy as np
import pandas as pd

# Mirrors token_steer.LEGACY_COLUMN_ALIASES, which is the source of truth. Duplicated
# rather than imported because token_steer imports torch.
LEGACY_COLUMN_ALIASES = {"readout_delta": "tgt_minus_top_delta"}


def load_arm(path):
    """One token_steer_*.csv, with empty completions preserved as empty strings.

    `keep_default_na=False` is load-bearing. With pandas defaults every empty
    `completion` field becomes NaN, and the 740 empty completions in the cities
    post-norm arm are md_full's failure mode (argmax is a newline, generation stops),
    not missing data. Verified: 4,660 non-empty + 740 empty = 5,400 rows, so no row was
    ever left ungenerated.
    """
    Also normalises the delta column. src/token_steer.py now writes it as
    `tgt_minus_top_delta`; the twelve files on disk from the finished cluster runs were
    not regenerated and still carry the old `readout_delta` header, so both spellings
    are live. token_steer.load_steer_csv does the same rename, but importing that module
    pulls in torch (4.2s) for a one-entry dict, so the map is duplicated here and pinned
    against the original by test_the_alias_map_matches_token_steer.
    """
    return (pd.read_csv(path, keep_default_na=False, na_values=[],
                        dtype={"completion": str})
            .rename(columns=LEGACY_COLUMN_ALIASES))


def shared_statements(df):
    """Statements every direction in `df` ran, sorted."""
    sets = [set(g.stmt.unique()) for _, g in df.groupby("direction")]
    return sorted(set.intersection(*sets)) if sets else []


def restrict_to_shared(df):
    """`df` restricted to the statements every direction ran."""
    return df[df.stmt.isin(shared_statements(df))].copy()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Verify against the real artifact**

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import token_conclusions as tc
d = tc.load_arm('token_steer_common_claim_true_false_postnorm_all_rp1.csv')
print('rows', len(d), 'shared', len(tc.shared_statements(d)))
c = tc.load_arm('token_steer_cities_postnorm_all_rp1.csv')
print('cities empty completions', (c.completion == '').sum())
"
```

Expected exactly:
```
rows 4410 shared 90
cities empty completions 740
```

- [ ] **Step 6: Commit**

```bash
git add src/token_conclusions.py tests/test_token_conclusions.py && git commit -m "feat(conclusions): arm loading and shared-statement restriction"
```

---

### Task 2: The legacy readout reconstruction and the 2x2

This is A1 and it is the headline result. On cities at frac 1.0, the legacy truth readout is pushed a median of 2.42 times past its own decision threshold on 88.1% of true statements, and the target token flips on 0.0% of them. That is the readout-versus-behavior dissociation inside a single experiment, where the existing audit had to assemble it across two runs and a judge.

Two things make this delicate. The readout axis is reconstructed, not measured, because `token_steer` never logs the truth readout (see the `readout_delta` warning in the spec §0). And `eps*` does not exist on disk in any file; it must be rebuilt from `g` and `m`.

**Files:**
- Modify: `src/token_conclusions.py`
- Test: `tests/test_token_conclusions.py`

**Interfaces:**
- Consumes: `load_arm`, `restrict_to_shared` from Task 1.
- Produces: `legacy_eps_star(ds, root=".") -> (dict[int, float], dict[int, int])` mapping dataset row index to `eps*` and to label; `crossed_flags(df, eps_by_row, labels_by_row, label=None) -> pd.DataFrame` adding `eps_legacy` and `crossed` columns; `cross_tab(df) -> dict` with the four cell counts and `phi`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_token_conclusions.py`:

```python
def test_crossed_uses_the_legacy_budget_not_the_token_budget():
    """The two budgets share the symbol eps and differ by orders of magnitude.

    Here eps*_legacy is 0.5 and the swept scale is 50, so the readout crossed its
    threshold 100x over. A helper that mistakenly compared against a token-space budget
    (delta_cone / alpha, ~500 for this statement) would report not-crossed, and the
    dissociation would vanish. The assertion is on the eps_legacy column itself so the
    error cannot hide inside a boolean.
    """
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5}, {7: 1})
    assert out.eps_legacy.iloc[0] == pytest.approx(0.5)
    assert bool(out.crossed.iloc[0]) is True


def test_crossed_flags_drops_statements_outside_the_subsample():
    """jtw_legacy covers 90 of 200 on common_claim. A statement with no reach_margins
    row has no eps* and must be dropped, not treated as eps*=0 and therefore always
    crossed."""
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 9, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5}, {7: 1})
    assert list(out.stmt) == [7]


def test_crossed_flags_can_restrict_to_one_label():
    """g <= 0 means the statement already sits inside the target halfspace, so eps* is
    0 and `crossed` is trivially true. Measured on cities: 96 of the 99 label-0
    statements have g <= 0 and 0 of the 101 label-1 statements do. Pooling them would
    report a crossed rate inflated by statements that never had to move."""
    df = _arm([("jtw_legacy", 7, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 8, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {7: 0.5, 8: 0.0}, {7: 1, 8: 0}, label=1)
    assert list(out.stmt) == [7]


def test_cross_tab_handles_an_empty_column_without_raising():
    """jtw_legacy is expected to produce an all-zero flipped column. phi's denominator
    is then 0 and must come back as NaN rather than a ZeroDivisionError or, worse, a
    silent 0.0 that reads as 'no association measured' instead of 'not measurable'."""
    df = _arm([("jtw_legacy", 1, 1.0, 50.0, 0, 0.02, "x"),
               ("jtw_legacy", 2, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {1: 0.5, 2: 100.0}, {1: 1, 2: 1})
    t = tc.cross_tab(out)
    assert t["crossed_not_flipped"] == 1
    assert t["not_crossed_not_flipped"] == 1
    assert t["crossed_flipped"] == 0
    assert np.isnan(t["phi"])


def test_the_join_key_is_the_dataset_row_index_not_a_position():
    """token_steer.stmt and token_geom.idx carry original dataset row indices running
    to 1495 on cities, not 0-based positions running to 199. A merge that assumes
    positions lands on the wrong statements while still producing a full-looking join,
    so the error is invisible downstream."""
    df = _arm([("jtw_legacy", 1246, 1.0, 50.0, 0, 0.02, "x")])
    out = tc.crossed_flags(df, {1246: 0.5, 0: 999.0}, {1246: 1, 0: 1})
    assert out.eps_legacy.iloc[0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v -k "crossed or cross_tab or join_key"
```

Expected: FAIL with `AttributeError: module 'token_conclusions' has no attribute 'crossed_flags'`.

- [ ] **Step 3: Write the implementation**

Append to `src/token_conclusions.py`:

```python
LEGACY_NAME = "mean_diff_tgt"     # the readout every prior reach_steer run used
UNREACHABLE = 1e12                # sentinel eps* when the margin vanishes


def legacy_eps_star(ds, root="."):
    """Per-statement eps* for the legacy truth readout, keyed by dataset row index.

    eps* is NOT stored anywhere on disk. `margins` in reach_margins_<ds>.npz holds
    m = ||J^T w||, and eps* = g/m with g = w.h_tgt - t02, exactly the definition in
    reach_analyze.required_eps. Reconstructing it here rather than importing keeps this
    module free of the reach pipeline's torch imports.

    Returns (eps_by_row, labels_by_row). Statements outside the reach_margins subsample
    are absent from both dicts, which is what makes crossed_flags drop them.

    Every npz member is bound to a local array once. Indexing an NpzFile re-inflates
    the whole array on each access and `jtw` alone is 1496 x 14 x 2304.
    """
    dirs = np.load(os.path.join(root, f"reach_dirs_{ds}.npz"), allow_pickle=True)
    acts = np.load(os.path.join(root, f"reach_acts_{ds}.npz"), allow_pickle=True)
    mz = np.load(os.path.join(root, f"reach_margins_{ds}.npz"), allow_pickle=True)

    names = [str(x) for x in dirs["names"]]
    k = names.index(LEGACY_NAME)
    w = np.asarray(dirs["W"], np.float64)[k]
    t02 = float(np.asarray(dirs["thresh02"], np.float64)[k])
    m = np.asarray(mz["margins"], np.float64)[:, k]

    n = m.shape[0]                       # reach_margins may be shorter on a --limit run
    h = np.asarray(acts["h_tgt"], np.float64)[:n]
    row_index = np.asarray(acts["row_index"])[:n]
    labels = np.asarray(acts["labels"]).astype(int)[:n]

    g = h @ w - t02
    eps = np.where(g <= 0, 0.0,
                   np.where(m > 1e-12, g / np.maximum(m, 1e-12), UNREACHABLE))
    return ({int(r): float(e) for r, e in zip(row_index, eps)},
            {int(r): int(v) for r, v in zip(row_index, labels)})


def crossed_flags(df, eps_by_row, labels_by_row, label=None):
    """Add `eps_legacy` and `crossed` to `df`, keyed on the dataset row index in `stmt`.

    `crossed` is |scale| >= eps*_legacy, which is a FIRST-ORDER PREDICTION and not a
    measurement. token_steer logs no truth readout at any scale (its `readout_delta`
    column is the token margin under a misleading name). Because jtw_legacy is the unit
    vector along J^T w, the readout moves by scale * ||J^T w|| to first order, and
    eps*_legacy = g / ||J^T w|| is defined precisely so the readout reaches its
    threshold at scale = eps*_legacy. The linear step is licensed by an already
    measured per-statement R^2 of 0.9991 on cities, not assumed here. Any table built
    from this column must say so.

    `label=1` restricts to true statements. Use it for any pooled crossed rate: g <= 0
    means the statement already sits in the target halfspace, so eps* is 0 and
    `crossed` is trivially true. On cities that is 96 of 99 label-0 statements and 0 of
    101 label-1 statements.
    """
    e = df.stmt.map(eps_by_row)
    y = df.stmt.map(labels_by_row)
    keep = e.notna() & y.notna()
    if label is not None:
        keep &= (y == label)
    out = df[keep].copy()
    out["eps_legacy"] = e[keep].to_numpy(dtype=float)
    out["label"] = y[keep].to_numpy(dtype=int)
    out["crossed"] = out.scale.abs().to_numpy() >= out.eps_legacy.to_numpy()
    return out


def cross_tab(df):
    """2x2 counts and the phi coefficient for `crossed` against `hit_target`.

    phi is NaN when any margin of the table is zero. jtw_legacy is expected to produce
    an all-zero flipped column, so this is the normal case, not an error case. Returning
    0.0 there would read as "measured no association" when the truth is "association is
    not measurable from a degenerate table".
    """
    c = df.crossed.to_numpy().astype(bool)
    f = df.hit_target.to_numpy().astype(bool)
    n11, n10 = int((c & f).sum()), int((c & ~f).sum())
    n01, n00 = int((~c & f).sum()), int((~c & ~f).sum())
    den = float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = float("nan") if den <= 0 else (n11 * n00 - n10 * n01) / np.sqrt(den)
    return {"n": int(len(df)), "crossed_flipped": n11, "crossed_not_flipped": n10,
            "not_crossed_flipped": n01, "not_crossed_not_flipped": n00, "phi": phi}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Verify against the real artifacts**

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import numpy as np, token_conclusions as tc
for ds, arm in [('cities','postnorm_all_rp1'),
                ('common_claim_true_false','postnorm_all_rp1')]:
    eps, lab = tc.legacy_eps_star(ds)
    d = tc.load_arm(f'token_steer_{ds}_{arm}.csv')
    d = d[(d.direction=='jtw_legacy') & (d.frac==1.0)]
    o = tc.crossed_flags(d, eps, lab, label=1)
    t = tc.cross_tab(o)
    print(ds, 'n', t['n'], 'crossed', round(o.crossed.mean(),3),
          'flipped', round(o.hit_target.mean(),3),
          'med scale/eps*', round(float(np.median(o.scale.abs()/np.maximum(o.eps_legacy,1e-12))),3))
"
```

Expected exactly:
```
cities n 101 crossed 0.881 flipped 0.0 med scale/eps* 2.416
common_claim_true_false n 44 crossed 0.068 flipped 0.0 med scale/eps* 0.172
```

If cities does not read `crossed 0.881 flipped 0.0`, stop. The reconstruction is wrong and nothing downstream is readable.

- [ ] **Step 6: Commit**

```bash
git add src/token_conclusions.py tests/test_token_conclusions.py && git commit -m "feat(conclusions): legacy readout reconstruction and the readout-behavior 2x2"
```

---

### Task 3: Margin consumption and the duplicate-column detector

This is A2. `frac_margin` is the fraction of the logit margin consumed, the unit the PI explicitly asked for, and it has never appeared in a results table in this project.

The numbers separate two failure modes that have been reported as one. `jtw_legacy` consumes 0.0174 of the margin at its full certified budget. `md_full` consumes 1.088 at the same site and still fails to flip, because consuming the target-versus-top margin is necessary and not sufficient: at `md_full`'s scale some third token wins, and the empty-completion evidence in Task 4 shows which one.

The detector exists because the finding that `readout_delta` duplicates `frac_margin` is what forced A1 into a reconstruction. If `src/token_steer.py` is ever corrected to log the real truth readout, this test must fail loudly so A1 and A2 get revisited rather than silently continuing to call a measurement a prediction.

**Files:**
- Modify: `src/token_conclusions.py`
- Test: `tests/test_token_conclusions.py`

**Interfaces:**
- Consumes: `load_arm` from Task 1.
- Produces: `margin_consumption(df) -> pd.DataFrame` with columns `direction, frac, median_abs, p90_abs, n`; `proportional_per_statement(df, a, b, rtol=1e-3) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_token_conclusions.py`:

```python
def test_margin_consumption_reports_median_and_p90_per_direction_and_frac():
    df = _arm([("A", 1, 1.0, 1.0, 0, 0.01, "x"),
               ("A", 2, 1.0, 1.0, 0, -0.03, "x"),
               ("A", 3, 1.0, 1.0, 0, 0.05, "x"),
               ("A", 1, 2.0, 2.0, 0, 1.00, "x")])
    out = tc.margin_consumption(df).set_index(["direction", "frac"])
    assert out.loc[("A", 1.0), "median_abs"] == pytest.approx(0.03)
    assert out.loc[("A", 1.0), "n"] == 3
    assert out.loc[("A", 2.0), "median_abs"] == pytest.approx(1.00)


def test_proportional_detector_fires_on_a_per_statement_rescale():
    """readout_delta = frac_margin * |m0| with |m0| constant within a statement and
    different across statements. This is what token_steer actually writes, verified:
    the per-statement std of the ratio is 2.3e-5 against a mean of 13.8."""
    df = pd.DataFrame({
        "direction": ["A"] * 4,
        "stmt": [1, 1, 2, 2],
        "frac_margin": [0.01, 0.02, 0.05, 0.10],
        "tgt_minus_top_delta": [0.13, 0.26, 1.00, 2.00],  # m0 = 13 then 20
    })
    assert tc.proportional_per_statement(df) is True


def test_proportional_detector_stays_silent_on_independent_columns():
    """If token_steer is corrected to log the real truth readout, the ratio stops being
    constant and this returns False, which is the notification that A1 and A2 must be
    revisited."""
    df = pd.DataFrame({
        "direction": ["A"] * 4,
        "stmt": [1, 1, 2, 2],
        "frac_margin": [0.01, 0.02, 0.05, 0.10],
        "readout_delta": [0.13, 0.90, 1.00, 0.20],
    })
    assert tc.proportional_per_statement(df) is False


def test_proportional_detector_ignores_the_frac_zero_rows():
    """frac == 0 makes frac_margin exactly 0 by construction, so the ratio is undefined
    there. Including those rows would make the detector return False on data that is in
    fact proportional."""
    df = pd.DataFrame({
        "direction": ["A"] * 3,
        "stmt": [1, 1, 1],
        "frac_margin": [0.0, 0.01, 0.02],
        "readout_delta": [0.0, 0.13, 0.26],
    })
    assert tc.proportional_per_statement(df) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v -k "margin_consumption or proportional"
```

Expected: FAIL with `AttributeError: module 'token_conclusions' has no attribute 'margin_consumption'`.

- [ ] **Step 3: Write the implementation**

Append to `src/token_conclusions.py`:

```python
def margin_consumption(df):
    """Median and p90 of |frac_margin| per direction per frac.

    frac_margin is the fraction of the logit margin consumed. Report THIS and never
    `readout_delta`: the two are one measurement rescaled per statement by |m0|, so
    printing both would present a single number twice and inflate the apparent weight
    of the evidence. frac_margin is the dimensionless one and the unit the PI asked
    for; |m0| is separately available as `margin` in token_geom_<ds>.csv if an absolute
    figure is ever wanted.
    """
    g = df.groupby(["direction", "frac"]).frac_margin
    return pd.DataFrame({
        "median_abs": g.apply(lambda s: float(s.abs().median())),
        "p90_abs": g.apply(lambda s: float(s.abs().quantile(0.9))),
        "n": g.size(),
    }).reset_index()


def proportional_per_statement(df, a="tgt_minus_top_delta", b="frac_margin", rtol=1e-3):
    """True when a/b is constant within each statement, i.e. the columns are one
    measurement rescaled. Pass frames through load_arm first: it normalises the old
    `readout_delta` header to `tgt_minus_top_delta`, which is this default.

    This pins the finding that forced A1 into a closed-form reconstruction:
    src/token_steer.py writes `readout_delta = r - r0` where `probe` returns
    r = logit[j_tgt] - logit[j_top], which is the token margin and not the truth probe
    readout. A False here means token_steer was corrected and A1's reconstruction must
    be revisited before any table built on it is trusted.

    Rows with |b| below 1e-9 are dropped: frac == 0 makes frac_margin exactly zero by
    construction and the ratio is undefined there.
    """
    s = df[df[b].abs() > 1e-9].copy()
    if s.empty:
        return False
    s["_r"] = s[a] / s[b]
    rel = s.groupby(["direction", "stmt"])._r.agg(
        lambda x: 0.0 if len(x) < 2 else float(x.std(ddof=0) / max(abs(x.mean()), 1e-12)))
    return bool(rel.max() <= rtol)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Verify against the real artifacts**

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import token_conclusions as tc
d = tc.load_arm('token_steer_cities_postnorm_all_rp1.csv')
m = tc.margin_consumption(d).set_index(['direction','frac'])
for k in ['jtw_legacy','md_full','oracle']:
    print(k, round(m.loc[(k,1.0),'median_abs'],4))
print('duplicate columns:', tc.proportional_per_statement(d))
"
```

Expected exactly:
```
jtw_legacy 0.0174
md_full 1.0877
oracle 1.001
duplicate columns: True
```

- [ ] **Step 6: Commit**

```bash
git add src/token_conclusions.py tests/test_token_conclusions.py && git commit -m "feat(conclusions): margin consumption and the duplicate-column detector"
```

---

### Task 4: Completion forensics

This is A3, the claim 2 test. Claim 2 asserts that if direct final-layer steering only degrades text, the final-layer truth feature is not behaviorally linear. The oracle at frac 1.0 flips the target token on 100% of statements, so it is the exact test case.

**Pre-registered expectation, recorded before the systematic measurement so the analysis cannot be quietly re-aimed: claim 2 will be refuted.** Three oracle completions at frac 1.0 read "North East China, and is the capital", "the northern part of Hebei Province,", "South Odisha, India. It is located". Fluent and grammatical. If the systematic measurement disagrees, it is reported as-is with this expectation quoted.

The target token on cities is the first subword of a false country name (`first_token` of a country from the pool, `src/token_geom.py:310-336`), so `tok_tgt` values are fragments like "North", "South", "the". The certificate controls that first token and nothing after it, which makes the country question the interesting one: does the completion still name the correct country even when the first token flipped?

**Files:**
- Modify: `src/token_conclusions.py`
- Test: `tests/test_token_conclusions.py`

**Interfaces:**
- Consumes: `load_arm` from Task 1.
- Produces: `is_degenerate(completion) -> bool`; `edit_ratio(a, b) -> float`; `country_outcome(completion, correct_country, target_countries) -> str` returning one of `"correct"`, `"target"`, `"both"`, `"none"`; `target_country_set(tok_tgt, countries) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_token_conclusions.py`:

```python
def test_an_empty_completion_is_degenerate_not_missing():
    """All 740 empty completions in the cities post-norm arm belong to md_full, 739 of
    them at negative fracs where |scale| runs to 1.6e3, and their argmax_tok is a
    newline or <eos>. The model emitted a newline and stopped. 4,660 non-empty plus 740
    empty is exactly the 5,400 rows, so nothing was left ungenerated. Dropping them as
    missing would delete the loudest evidence of md_full's failure mode."""
    assert tc.is_degenerate("") is True
    assert tc.is_degenerate("   ") is True


def test_degeneracy_catches_repetition_and_non_alphabetic_output():
    assert tc.is_degenerate("the the the city") is True
    assert tc.is_degenerate("... , . ; -") is True
    assert tc.is_degenerate("the northern part of Hebei Province,") is False


def test_edit_ratio_is_zero_for_identical_and_one_for_disjoint():
    assert tc.edit_ratio("in China", "in China") == pytest.approx(0.0)
    assert tc.edit_ratio("aaaa", "bbbb") == pytest.approx(1.0)
    assert 0.0 < tc.edit_ratio("in China. It is", "in Japan. It is") < 1.0


def test_country_outcome_is_three_way_not_binary():
    """The third case, naming some country that is neither the correct one nor the
    steering target, is the interesting one. A plain flipped/not-flipped binary would
    hide it."""
    assert tc.country_outcome("in China. It is", "China", ("North Korea",)) == "correct"
    assert tc.country_outcome("in North Korea", "China", ("North Korea",)) == "target"
    assert tc.country_outcome("in Peru, a nation", "China", ("North Korea",)) == "none"
    assert tc.country_outcome("China and North Korea", "China", ("North Korea",)) == "both"


def test_target_country_set_is_every_country_sharing_the_target_token():
    """tok_tgt is the FIRST SUBWORD of a country name, so it does not identify one
    country. 'South' is shared by South Africa and South Korea. Treating it as a single
    country would either miss a hit or claim a specific one the data cannot support."""
    countries = ["China", "North Korea", "South Africa", "South Korea"]
    assert tc.target_country_set("South", countries) == ("South Africa", "South Korea")
    assert tc.target_country_set("North", countries) == ("North Korea",)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v -k "degenerate or degeneracy or edit_ratio or country"
```

Expected: FAIL with `AttributeError: module 'token_conclusions' has no attribute 'is_degenerate'`.

- [ ] **Step 3: Write the implementation**

Add `import difflib` to the imports at the top of `src/token_conclusions.py`, then append:

```python
def is_degenerate(completion):
    """Empty, non-alphabetic, or a token repeated three or more times in a row.

    An empty completion counts as degenerate. See load_arm: nothing in these files was
    left ungenerated, so an empty string is a real observation of the model emitting a
    newline and stopping.
    """
    s = str(completion)
    if not s.strip():
        return True
    if not any(ch.isalpha() for ch in s):
        return True
    t = s.split()
    return any(t[i] == t[i + 1] == t[i + 2] for i in range(len(t) - 2))


def edit_ratio(a, b):
    """1 - difflib.SequenceMatcher ratio. 0.0 identical, 1.0 nothing in common.

    difflib is standard library, so this adds no dependency for a single distance.
    """
    return 1.0 - difflib.SequenceMatcher(None, str(a), str(b)).ratio()


def target_country_set(tok_tgt, countries):
    """Every country whose name starts with `tok_tgt`, as a tuple.

    token_geom builds the cities target as first_token(tokenizer, country), so tok_tgt
    is a subword fragment and several countries can share one. Returning the set keeps
    the analysis from claiming a specific country the token does not identify.
    """
    t = str(tok_tgt).strip()
    if not t:
        return ()
    return tuple(c for c in countries if str(c).startswith(t))


def country_outcome(completion, correct_country, target_countries):
    """'correct', 'target', 'both' or 'none' by substring match, case insensitive."""
    s = str(completion).lower()
    hit_c = str(correct_country).strip().lower() in s if str(correct_country).strip() else False
    hit_t = any(str(c).strip().lower() in s for c in target_countries if str(c).strip())
    if hit_c and hit_t:
        return "both"
    if hit_c:
        return "correct"
    if hit_t:
        return "target"
    return "none"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest tests/test_token_conclusions.py -v
```

Expected: 19 passed.

- [ ] **Step 5: Verify against the real artifact**

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import token_conclusions as tc
d = tc.load_arm('token_steer_cities_postnorm_all_rp1.csv')
for k in ['oracle','md_full','jtw_legacy']:
    s = d[(d.direction==k) & (d.frac==1.0)]
    print(k, 'n', len(s), 'degenerate', round(s.completion.map(tc.is_degenerate).mean(),3))
n = d[d.completion=='']
print('empty by direction', n.groupby('direction').size().to_dict())
print('empty at negative frac', int((n.frac<0).sum()), 'of', len(n))
"
```

Expected: `oracle` and `jtw_legacy` degenerate rates at or near 0.0 with n 200 each, `md_full` above them, `empty by direction {'md_full': 740}`, `empty at negative frac 739 of 740`.

- [ ] **Step 6: Commit**

```bash
git add src/token_conclusions.py tests/test_token_conclusions.py && git commit -m "feat(conclusions): completion forensics for the claim-2 test"
```

---

### Task 5: Pass 1 extraction and the raw findings dump

The five analyses whose value is still unknown stay in the scratchpad and are not tested, per the spec. The four confirmed ones call the module.

`docs/TOKEN_SPACE_RAW_FINDINGS.md` is deliberately unpolished. It carries every number produced, including nulls, including numbers that turn out to mean nothing. Its audience is the researcher reading it cold to decide what matters, and its header says so.

**Files:**
- Create: `<scratchpad>/a4_alpha.py`, `<scratchpad>/a5_broadcast.py`, `<scratchpad>/a6_sae.py`, `<scratchpad>/a7_split.py`, `<scratchpad>/a8_prenorm.py`, `<scratchpad>/extract_all.py`
- Create: `docs/TOKEN_SPACE_RAW_FINDINGS.md`

**Interfaces:**
- Consumes: every public function from Tasks 1 to 4.
- Produces: `docs/TOKEN_SPACE_RAW_FINDINGS.md`, read by Task 6.

- [ ] **Step 1: Write `a4_alpha.py`, does alpha predict behavior**

`alpha_<name>` columns live in `token_geom_<ds>.csv` keyed by `idx`, which equals `token_steer.stmt`. Available: `alpha_md_full`, `alpha_mean_diff_tgt_asis`, `alpha_probe_grad_tgt_asis`. Only `md_full` has both an alpha and a steered arm, so this is a one-direction test.

```python
"""A4: does the geometric predictor alpha predict realized hit rate?

eps(u) = eps*/alpha is the centre of the token-space framing and has never been checked
against behaviour. POWER WARNING recorded before running: md_full's hit rate is 0.005
on cities and 0.200 on common_claim, so within-direction variance may be too small to
correlate. If so, report the test as underpowered rather than reporting a null
correlation as evidence of anything.
"""
import sys
sys.path.insert(0, "/Users/vaibhav.wudaru/llm-activation-steering-research/src")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import token_conclusions as tc

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"

for ds in ("cities", "common_claim_true_false"):
    g = pd.read_csv(f"{ROOT}/token_geom_{ds}.csv")[["idx", "alpha_md_full"]]
    d = tc.load_arm(f"{ROOT}/token_steer_{ds}_postnorm_all_rp1.csv")
    d = d[(d.direction == "md_full") & (d.frac > 0)]
    hr = d.groupby("stmt").hit_target.mean().rename("hit_rate").reset_index()
    m = hr.merge(g, left_on="stmt", right_on="idx", how="inner")
    print(f"\n### A4 {ds}  n={len(m)}  hit_rate variance={m.hit_rate.var():.6f}")
    if m.hit_rate.nunique() < 2:
        print("  UNDERPOWERED: hit rate is constant, no correlation is computable")
        continue
    rho, p = spearmanr(m.alpha_md_full, m.hit_rate)
    print(f"  spearman rho={rho:.4f} p={p:.4g}")
    m["q"] = pd.qcut(m.alpha_md_full, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    print(m.groupby("q", observed=True).agg(
        n=("hit_rate", "size"), hit_rate=("hit_rate", "mean"),
        alpha=("alpha_md_full", "median")).round(4).to_string())
```

Run it:

```bash
./.venv/bin/python "/private/tmp/claude-501/-Users-vaibhav-wudaru-llm-activation-steering-research/c8145135-7f79-4eff-81b2-f40f140cba80/scratchpad/a4_alpha.py"
```

If `scipy` is not installed, replace `spearmanr` with a rank correlation computed from `pd.Series.rank()` and `np.corrcoef`. Do not add a dependency for one statistic.

- [ ] **Step 2: Write `a5_broadcast.py`, the E7 resolution**

```python
"""A5: broadcast versus per-position at depth, the one unresolved item in the program.

Observed: cities layer 16 jtw_token reaches 0.340 broadcast against 0.135 per-position
at frac 1.0, despite each spending its own separately certified budget. common_claim
layer 8 shows no gap, 0.815 both ways.

broadcast_gain from token_jac is 1.355 at cities layer 16 and 1.720 at layer 8. Note
this contradicts the 1.55 figure in the project record, which is wrong.
"""
import sys
sys.path.insert(0, "/Users/vaibhav.wudaru/llm-activation-steering-research/src")
import pandas as pd
import token_conclusions as tc

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"
ARMS = {"cities": ("layer16", 16), "common_claim_true_false": ("layer8", 8)}

for ds, (site, layer) in ARMS.items():
    j = pd.read_csv(f"{ROOT}/token_jac_{ds}.csv")
    bg = float(j.loc[j.layer == layer, "broadcast_gain"].iloc[0])
    ratio = float(j.loc[j.layer == layer, "m_all_median"].iloc[0] /
                  j.loc[j.layer == layer, "m_last_median"].iloc[0])
    print(f"\n### A5 {ds} layer {layer}: broadcast_gain={bg:.3f} m_all/m_last={ratio:.3f}")
    for pos in ("all", "last"):
        d = tc.load_arm(f"{ROOT}/token_steer_{ds}_{site}_{pos}_rp1.csv")
        d = tc.restrict_to_shared(d)
        r = d[d.frac.isin([1.0, 2.0])].groupby(["direction", "frac"]).hit_target.mean()
        print(f"  positions={pos}: {r.round(3).to_dict()}")
    stems = pd.read_csv(f"{ROOT}/token_geom_{ds}.csv").stem.astype(str)
    print(f"  mean stem words={stems.str.split().str.len().mean():.2f} "
          f"mean stem chars={stems.str.len().mean():.1f}")
```

Run it with the same `./.venv/bin/python` prefix.

- [ ] **Step 3: Write `a6_sae.py`, claim 5**

```python
"""A6: what the SAE decomposition actually supports.

CONSTRAINT established during planning: sae_features_<ds>.csv decomposes
w_mean_diff_tgt at layer 20 and every jtw_* vector at layer 11. GemmaScope layer-11 and
layer-20 dictionaries are unrelated, so a feature-ID overlap BETWEEN the truth readout
and the steering vectors is not computable from these files. Overlap is reported only
within a layer, and the cross-layer question is reported as not answerable rather than
answered with a meaningless number.
"""
import pandas as pd

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"

for ds in ("cities", "common_claim_true_false"):
    s = pd.read_csv(f"{ROOT}/sae_features_{ds}.csv")
    print(f"\n### A6 {ds}")
    print("  vector x layer:", s.groupby(["vector", "layer"]).size().to_dict())
    for (v, l), g in s.groupby(["vector", "layer"]):
        g = g.sort_values("rank")
        at = {r: (round(float(g[g["rank"] <= r].cumulative_explained.max()), 3)
                  if (g["rank"] <= r).any() else None) for r in (1, 5, 10)}
        top = g.head(5)[["feature", "coef"]].round(4).values.tolist()
        print(f"  {v:24s} L{l}: r1={at[1]} r5={at[5]} r10={at[10]} top5={top}")
    for l, g in s.groupby("layer"):
        vs = sorted(g.vector.unique())
        if len(vs) < 2:
            print(f"  L{l}: only {vs}, no within-layer overlap computable")
            continue
        print(f"  L{l} top-10 feature Jaccard:")
        for i in range(len(vs)):
            for k in range(i + 1, len(vs)):
                a = set(g[(g.vector == vs[i]) & (g["rank"] <= 10)].feature)
                b = set(g[(g.vector == vs[k]) & (g["rank"] <= 10)].feature)
                jac = len(a & b) / max(len(a | b), 1)
                print(f"    {vs[i]} vs {vs[k]}: {jac:.3f}")
    print("  CROSS-LAYER overlap (w_mean_diff_tgt L20 vs jtw_* L11): NOT COMPUTABLE, "
          "different SAE dictionaries")
```

- [ ] **Step 4: Write `a7_split.py`, the failure-mode split**

```python
"""A7: cities fails by inertness, common_claim is partly the Tan anti-steerability
regime. The project record is explicit that these must never be pooled.

Confound to state rather than resolve: common_claim's target is the runner-up token
(token_geom target_mode='runnerup') and some statements are near-ties, so generic
disruption can land on target there. cities uses target_mode='countries', a semantic
false-country target, and is the discriminating dataset.
"""
import pandas as pd

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"
COLS = ["delta_cone", "delta_rel_z", "rmsnorm_penalty", "z_norm", "margin",
        "alpha_md_full", "alpha_mean_diff_tgt_asis", "alpha_probe_grad_tgt_asis"]

for ds in ("cities", "common_claim_true_false"):
    s = pd.read_csv(f"{ROOT}/signed_steer_summary_{ds}.csv")
    g = pd.read_csv(f"{ROOT}/token_geom_{ds}.csv")
    print(f"\n### A7 {ds}")
    print("  signed_steer:", s.iloc[0].to_dict())
    print("  token_geom medians:",
          {c: round(float(g[c].median()), 5) for c in COLS if c in g.columns})
    print("  min delta_cone:", round(float(g.delta_cone.min()), 5))
```

- [ ] **Step 5: Write `a8_prenorm.py`, the arithmetic closure**

```python
"""A8: the pre-norm null is arithmetic, not failure.

A post-norm-certified displacement must survive RMSNorm, which costs a factor of
rmsnorm_penalty. The sweep stops at frac 2.0 and the penalty is 4.38 (cities) and 5.18
(common_claim), so the oracle CANNOT recover inside the swept range.

token_sens carries eight random* directions. Their gain is the chance baseline: if the
candidates land on it, pre-norm gain is a property of the norm layer and not of the
direction, which is a stronger statement than the candidates merely matching 1/penalty.

FALSIFIABLE PREDICTION, stated but not run (needs the cluster): sweeping the pre-norm
arm to frac 6 should recover the oracle to near 1.0.
"""
import sys
sys.path.insert(0, "/Users/vaibhav.wudaru/llm-activation-steering-research/src")
import pandas as pd
import token_conclusions as tc

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"

for ds in ("cities", "common_claim_true_false"):
    g = pd.read_csv(f"{ROOT}/token_geom_{ds}.csv")
    pen = float(g.rmsnorm_penalty.median())
    t = pd.read_csv(f"{ROOT}/token_sens_{ds}_prenorm_all.csv")
    med = t.groupby("direction").gain.median()
    rnd = med[[c for c in med.index if c.startswith("random")]]
    print(f"\n### A8 {ds}: rmsnorm_penalty={pen:.3f} 1/penalty={1/pen:.4f}")
    print(f"  random baseline: median={rnd.median():.4f} "
          f"range=[{rnd.min():.4f}, {rnd.max():.4f}] n={len(rnd)}")
    print("  candidate gains:",
          {k: round(float(v), 4) for k, v in med.items() if not k.startswith("random")})
    d = tc.load_arm(f"{ROOT}/token_steer_{ds}_prenorm_all_rp1.csv")
    r = d[d.frac.isin([1.0, 2.0])].groupby(["direction", "frac"]).hit_target.mean()
    print("  prenorm hit rates:", r.round(3).to_dict())
    print(f"  frac needed for a post-norm-certified displacement to survive: ~{pen:.1f}; "
          f"sweep stops at 2.0")
```

- [ ] **Step 6: Write `extract_all.py` and generate the dump**

```python
"""Drives every Pass 1 analysis and writes docs/TOKEN_SPACE_RAW_FINDINGS.md."""
import io
import os
import runpy
import sys
import contextlib

ROOT = "/Users/vaibhav.wudaru/llm-activation-steering-research"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import token_conclusions as tc                                      # noqa: E402

ARMS = {
    "cities": ["postnorm_all_rp1", "postnorm_all_rp1.3", "postnorm_last_rp1",
               "prenorm_all_rp1", "layer16_all_rp1", "layer16_last_rp1"],
    "common_claim_true_false": ["postnorm_all_rp1", "postnorm_all_rp1.3",
                                "postnorm_last_rp1", "prenorm_all_rp1",
                                "layer8_all_rp1", "layer8_last_rp1"],
}


def a1():
    print("## A1 readout-behaviour cross-tab (RECONSTRUCTED readout axis)")
    print("The readout axis is a first-order PREDICTION, not a measurement. token_steer")
    print("logs no truth readout. crossed = |scale| >= eps*_legacy, licensed by the")
    print("already measured per-statement R^2 of 0.9991 for readout against scale.")
    print("jtw_legacy's scale is frac*delta_cone at EVERY site (no alpha rescale), so")
    print("the crossed column is identical across arms and only `flipped` varies.")
    for ds in ARMS:
        eps, lab = tc.legacy_eps_star(ds)
        for arm in ARMS[ds]:
            p = f"{ROOT}/token_steer_{ds}_{arm}.csv"
            if not os.path.exists(p):
                print(f"  MISSING {os.path.basename(p)}")
                continue
            d = tc.load_arm(p)
            d = d[d.direction.isin(["jtw_legacy", "jtw_token", "md_full", "oracle"])]
            for k in sorted(d.direction.unique()):
                for fr in (0.25, 0.5, 1.0, 2.0):
                    s = d[(d.direction == k) & (d.frac == fr)]
                    if s.empty:
                        continue
                    o = tc.crossed_flags(s, eps, lab, label=1)
                    if o.empty:
                        continue
                    t = tc.cross_tab(o)
                    print(f"  {ds:24s} {arm:18s} {k:11s} frac={fr:<5} {t}")


def a2():
    print("\n## A2 margin consumption (frac_margin ONLY; readout_delta is the same")
    print("measurement rescaled per statement by |m0| and is never reported beside it)")
    for ds in ARMS:
        for arm in ARMS[ds]:
            p = f"{ROOT}/token_steer_{ds}_{arm}.csv"
            if not os.path.exists(p):
                continue
            d = tc.load_arm(p)
            m = tc.margin_consumption(tc.restrict_to_shared(d))
            m = m[m.frac.isin([1.0, 2.0])]
            print(f"  {ds} {arm} (duplicate columns: "
                  f"{tc.proportional_per_statement(d)})")
            print(m.round(4).to_string(index=False))


def a3():
    print("\n## A3 completion forensics (the claim 2 test)")
    print("PRE-REGISTERED EXPECTATION: claim 2 will be REFUTED. Recorded before the")
    print("systematic measurement. If the measurement disagrees, it is reported as-is.")
    cities = pd.read_csv(f"{ROOT}/got_datasets/cities.csv")
    countries = sorted(cities.correct_country.astype(str).unique())
    for ds in ARMS:
        geom = pd.read_csv(f"{ROOT}/token_geom_{ds}.csv")[["idx", "tok_tgt"]]
        for arm in ARMS[ds]:
            p = f"{ROOT}/token_steer_{ds}_{arm}.csv"
            if not os.path.exists(p):
                continue
            d = tc.load_arm(p)
            base = (d[d.frac == 0.0].set_index(["direction", "stmt"]).completion
                    .to_dict())
            d["degenerate"] = d.completion.map(tc.is_degenerate)
            d["edit"] = [tc.edit_ratio(c, base.get((k, s), c))
                         for c, k, s in zip(d.completion, d.direction, d.stmt)]
            print(f"\n  {ds} {arm}")
            print(f"    empty by direction: "
                  f"{d[d.completion == ''].groupby('direction').size().to_dict()}")
            print(d.groupby(["direction", "frac"])
                  .agg(degen=("degenerate", "mean"), edit=("edit", "mean"),
                       n=("degenerate", "size")).round(3).to_string())
            if ds != "cities":
                continue
            m = d.merge(geom, left_on="stmt", right_on="idx", how="left")
            m = m.merge(cities[["correct_country"]].assign(idx=cities.index),
                        on="idx", how="left")
            m["outcome"] = [
                tc.country_outcome(c, cc, tc.target_country_set(tt, countries))
                for c, cc, tt in zip(m.completion, m.correct_country, m.tok_tgt)]
            print("    country outcome:")
            print(m.groupby(["direction", "frac"]).outcome
                  .value_counts(normalize=True).unstack(fill_value=0)
                  .round(3).to_string())


def a9():
    print("\n## A9 validity ledger (no computation)")
    for line in [
        "n = 200 statements per dataset, subsampled from 1,496 and 4,450",
        "jtw_legacy covers 90 of 200 on common_claim; restricted to 44 at label 1",
        "one model, google/gemma-2-2b, fp32",
        "two datasets, both English, both short declarative statements",
        "targets are FIRST TOKEN only; the certificate makes no claim about the rest",
        "temperature 0 only; nothing transfers to sampling without re-derivation",
        "the _asis alphas are cross-layer carryovers, fitted at the target layer and "
        "read in post-norm coordinates",
        "A1's readout axis is reconstructed, not measured",
        "common_claim's target is the runner-up token, so generic disruption can land "
        "on target there; cities is the discriminating dataset",
    ]:
        print(f"  - {line}")


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a1(); a2(); a3()
    for name in ("a4_alpha", "a5_broadcast", "a6_sae", "a7_split", "a8_prenorm"):
        print(f"\n## {name}")
        try:
            runpy.run_path(os.path.join(HERE, f"{name}.py"), run_name="__main__")
        except Exception as e:                                     # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
    a9()

header = """# Token-space raw findings (Pass 1, unpolished)

**This document is not for the PI.** It is the raw extraction dump: every number the
Pass 1 analyses produced, including nulls, including numbers that turn out to mean
nothing. Its audience is the researcher reading it cold to decide what is worth
promoting into `docs/TOKEN_SPACE_FINDINGS.md`.

Generated by the scratchpad scripts described in
`docs/superpowers/plans/2026-08-04-token-space-conclusions.md` Task 5. Nothing here
wrote, regenerated or overwrote an artifact.

"""
with open(f"{ROOT}/docs/TOKEN_SPACE_RAW_FINDINGS.md", "w") as f:
    f.write(header + "```\n" + buf.getvalue() + "\n```\n")
print(buf.getvalue())
print("\nwrote docs/TOKEN_SPACE_RAW_FINDINGS.md")
```

Run it:

```bash
./.venv/bin/python "/private/tmp/claude-501/-Users-vaibhav-wudaru-llm-activation-steering-research/c8145135-7f79-4eff-81b2-f40f140cba80/scratchpad/extract_all.py"
```

Expected: every section prints, `docs/TOKEN_SPACE_RAW_FINDINGS.md` is written, and the A1 cities post-norm `jtw_legacy` frac=1.0 line reads `n: 101, crossed_flipped: 0, crossed_not_flipped: 89`.

- [ ] **Step 7: Confirm no artifact was modified**

```bash
git status --short | grep -E "^ M (token_|reach_|sae_|signed_|plot_)" ; echo "exit=$?"
```

Expected: no matching lines, `exit=1`. If any artifact shows as modified, stop and investigate before committing.

- [ ] **Step 8: Commit**

```bash
git add docs/TOKEN_SPACE_RAW_FINDINGS.md && git commit -m "docs(token-space): pass 1 raw extraction dump"
```

---

### Task 6: The verdict document

Read `docs/TOKEN_SPACE_RAW_FINDINGS.md` cold and write `docs/TOKEN_SPACE_FINDINGS.md` in the structure of `docs/PROJECT_PROGRESS_TO_DATE.md`. Read that file first for house style before writing a line.

**Files:**
- Create: `docs/TOKEN_SPACE_FINDINGS.md`
- Read for style: `docs/PROJECT_PROGRESS_TO_DATE.md`
- Read for the claims: `docs/NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md` Part II

- [ ] **Step 1: Read the house-style source**

```bash
sed -n '1,80p' docs/PROJECT_PROGRESS_TO_DATE.md
```

The structure to mirror: `## 0. The one-paragraph version`, a question-chain table, numbered chapters, `## 13. The quantified ledger, wins stated as numbers`, `## 14. What is not established (the honest column)`, `## 15. Where it goes next`.

- [ ] **Step 2: Write the document**

Required sections, in order:

1. **One-paragraph version.**
2. **Claim table.** The PI's eight claims, one row each, columns: claim, experiment, verdict, section. Verdicts must be one of REFUTED, CONFIRMED, ANSWERED, UNRESOLVED, NOT ANSWERABLE FROM THESE FILES.
3. **One numbered section per claim.** Each states method, number, figure reference, and reading. Where a pre-registered expectation exists, quote it and say whether the result matched.
4. **The quantified ledger.** Every win as a number with its n.
5. **The honest column.** The A9 ledger plus anything Pass 1 surfaced.
6. **Where it goes next.** Lead with the frac-6 pre-norm prediction from A8, which is cheap and falsifiable.

Binding requirements on content:

- **Claim 2 gets whatever verdict A3 produced**, with the pre-registered expectation ("claim 2 will be refuted") quoted beside it. If A3 disagreed with the expectation, say so plainly.
- **Every A1 number carries the reconstruction caveat.** The readout axis is a first-order prediction licensed by an already measured R^2 of 0.9991, not an observation. A reader must not be able to mistake it for a measurement.
- **Never report the delta column beside `frac_margin`.** They are one measurement. The column is `tgt_minus_top_delta` in current `token_steer.py` output and `readout_delta` in the files on disk; neither spelling belongs in a results table.
- **Never pool cities and common_claim.** They fail by different mechanisms (inertness against the Tan anti-steerability regime) and their targets are different (semantic false country against runner-up token). Lead with cities.
- **State the `jtw_legacy` coverage caveat** wherever a common_claim cross-direction number appears: 90 of 200 statements, 44 at label 1.
- **Claim 5 is reported as not answerable from these files**, with the reason (the truth readout is decomposed at layer 20 and the steering vectors at layer 11, so their feature IDs come from unrelated dictionaries). Do not report a cross-layer overlap number.
- **Claim 8 uses `broadcast_gain` 1.355 at cities layer 16**, not the 1.55 in the project record, and notes the correction.
- No em dashes.

- [ ] **Step 3: Check the constraints**

```bash
grep -c "—" docs/TOKEN_SPACE_FINDINGS.md; echo "em dashes above (want 0)"
grep -nE "readout_delta|tgt_minus_top_delta" docs/TOKEN_SPACE_FINDINGS.md || echo "delta column absent: good"
grep -n "1\.55" docs/TOKEN_SPACE_FINDINGS.md || echo "stale broadcast_gain absent: good"
```

Expected: `0` em dashes, `readout_delta` absent unless it appears only inside the explanation of why it is not reported, no bare `1.55`.

- [ ] **Step 4: Run the full test suite**

```bash
./.venv/bin/python -m pytest tests/ -q
```

Expected: the pre-existing count plus 19, no failures.

- [ ] **Step 5: Confirm the staging set**

```bash
git status --short
```

Confirm `src/spectrum_utils.py`, `src/viz_spectrum.py`, `tests/test_spectrum_utils.py`, `tests/test_viz_spectrum.py`, `docs/RESULTS_SINCE_LAST_MEETING_PART2.md`, `docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md` and `plot_mag_linearity_v2.png` are still unstaged.

- [ ] **Step 6: Commit**

```bash
git add docs/TOKEN_SPACE_FINDINGS.md && git commit -m "docs(token-space): claim-by-claim verdict document"
```

---

## Notes for the implementer

**The `readout_delta` rename has already landed.** Task `task_427f63e2` finished during planning. `src/token_steer.py` now writes the column as `tgt_minus_top_delta` and exposes `LEGACY_COLUMN_ALIASES` plus `load_steer_csv`; `docs/superpowers/specs/2026-08-04-token-space-conclusions-design.md` carries a "Follow-up done" note recording it, and that spec edit is still unstaged. The twelve CSVs on disk were correctly not regenerated and still carry `readout_delta`, which `load_arm` normalises. Nothing in the spec's reasoning changed: the column is still the same measurement as `frac_margin`, so A1 remains a reconstruction and A2 still reports only one of the two.

If `proportional_per_statement` ever returns `False` on these files, the CSVs were regenerated, which was out of scope, and A1 and A2 must be revisited before anything is reported.

**What is out of scope.** Any cluster job including the frac-6 pre-norm test (stated as a prediction only), any model load or generation or judge run, regenerating any artifact, the refusal positive control, and editing `REACH_AUDIT_FINDINGS.md`, `RESULTS_SINCE_LAST_MEETING_PART3.md` or `PROJECT_PROGRESS_TO_DATE.md`.

**No new figures.** The five per dataset from `src/viz_token.py` are sufficient. If an analysis surfaces something they do not show, add it to `viz_token.py` rather than creating a new module.
