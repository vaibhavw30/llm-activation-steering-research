# Design Spec — MAG (Mining via Activation Geometry) Battery, Adapted to Truth

*Status: approved design, ready for implementation-plan. Author: session 2026-07-16.*

**One-line goal:** implement a second, independent unsupervised feature-mining method — **MAG**
(LeVi, David & Fomin, *Unsupervised Features Mining via Activation Geometry*, arXiv:2607.04222) —
adapted to our **truth** question on **gemma-2-2b base**, and run it **head-to-head against DCT** so
we can say which method better *recovers* and *steers* the truth direction, and whether our
symmetric-degradation steering finding survives MAG's principled magnitude calibration.

Companion docs: `PIPELINE_AND_JUDGE_SINCE_LAST_MEETING.md` (the DCT arm + judge this compares to),
`INVESTIGATION_steering_validity.md` (our steering statistics), `DCT_VS_XGBOOST_FINDINGS.md`
(linear/non-linear decodability). Findings will land in a new `docs/DCT_VS_MAG_ON_TRUTH.md`.

---

## 0. Why this exists (the scientific question)

Our funnel established, with DCT, that **truth is decodable but not causally dominant** in
gemma-2-2b: the supervised truth direction is not among DCT's most causally-salient directions, and
steering it degrades the model symmetrically rather than making it lie. A fair objection: *that's one
unsupervised method's verdict.* MAG is a **different** unsupervised method — it mines a feature from
the **prefix-induced activation shift** `Δ^Q(p) = m(Q‖p) − m(p)` induced by prepending a fixed
natural-language instruction `Q`, rather than from gradient-optimized causal directions. Running MAG
on the same model and the same truth datasets gives us a **second, methodologically-independent read**
on the same claim, plus one thing DCT gave us no principled handle on: **norm-calibrated steering**
(`α(τ) = τ·‖A_prefix‖/‖d‖`), which directly tests whether our degradation result is an artifact of
un-normalized ±120 magnitudes.

**The three questions the battery answers:**
1. *Decoding* — does MAG's prefix-shift feature decode truth (and the model's *believed* truth `y^M`)
   better than raw activations and better than DCT's causal subspace? (**E1, E2, E3**)
2. *Steering* — does MAG's class-mean direction `u_Q` steer truth better than the supervised
   direction or DCT's top vector, under one calibrated magnitude and one judge — and does our
   degradation finding survive calibration? (**E4**)
3. *Utility* — can MAG geometry rank which truth dataset best trains a transferable probe? (**§4**)

---

## 1. Locked decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| **Framing** | Adapt MAG to our truth question (not reproduce the paper on its datasets) | Keeps it inside the funnel; the deliverable is DCT-vs-MAG-on-truth. |
| **Model** | `google/gemma-2-2b` **base** (same as DCT arm) | Apples-to-apples with all existing DCT/probe/steering results; zero re-runs. |
| **`y^M`** | Next-token yes/no probability, MAG Eq. 1: `y^M(p) = 1[Pr(yes\|Q‖p) > Pr(no\|Q‖p)]` | Faithful to MAG's own definition; a base model can produce it (noisily). Cross-checked vs gold + OLMo judge. |
| **Scope** | All 8 operators + probes **E1–E4 + §4 transfer**. Skip E5–E7. | E5–E7 are specific to MAG's prompt-injection/concept datasets; no truth analog without inventing mappings the paper doesn't define. |
| **Readout layer** | Compute directions at **both** final-block and **truth-peak** layer; **lead the DCT head-to-head at peak layer** | Peak layer matches our existing DCT/probe results; final-block is MAG-faithful and reported alongside. |
| **`u_Q` labelling** | Build the class-mean contrast **both ways**: `y^M`-labeled (MAG-faithful) and gold-labeled | The divergence is exactly what E2 measures — "believed truth vs actual truth." |

**Out of scope (explicitly):** E5 (prompt-injection-isn't-one-feature), E6 (desert/ocean semantic
axes), E7 (Bob-context composition); any instruct-tuned model; reproducing the paper's numbers on its
own datasets. If base-model `y^M` turns out near-random (a real risk, see §11), that is a *reportable
finding*, not a blocker — the gold-labeled arm still yields the full DCT-vs-MAG comparison.

---

## 2. Background — MAG in the terms we need

MAG applies a fixed instruction `Q` to every input `p` and reads the residual stream at the **last
token of a chosen block**. Writing `Q‖p` for concatenation and `A(x) = m(x)` for the readout:

- **Prefix-induced shift:** `Δ^Q(p) := m(Q‖p) − m(p)`.
- **Feature direction (Eq. 2):** `v_Q = E_p[ m(Q‖p) − m(p) ]` — the average shift; a single linear
  direction that should reproduce the prefix's effect.
- **Reconstruction error (Eq. 3):** `ε_Q = E_p[‖m(Q‖p) − m̂(p)‖] / E_p[‖m(Q‖p) − m(p)‖]`, where
  `m̂(p) = m(p) + v_Q`. `ε_Q = 0` exact; `ε_Q = 1` no better than not steering; `ε_Q > 1` overshoot.
- **Self-label (Eq. 1):** `y^M(p) = 1[Pr_θ(yes | Q‖p) > Pr_θ(no | Q‖p)]`.
- **Steering direction:** `u_Q = v_Q^- − v_Q^+`, the difference of class-mean feature vectors over the
  `y^M`-negative vs `y^M`-positive statements (`v_Q^± = |P^±|^{-1} Σ_{p∈P^±} φ(p)`).
- **Calibrated strength:** `α(τ) = τ · ‖A_prefix‖ / ‖d‖` for direction `d`, so the injected vector
  `α(τ)·d` has norm `τ·‖A_prefix‖` — a fixed fraction `τ` of the prefixed-readout magnitude,
  independent of model or direction. Sweep `τ ∈ {0, 0.3, 1.0}`.

**The eight operators (Table 1)** — each maps a statement `p` to one feature vector `φ(p)` at the
readout (`A(x)=m(x)`; `E` = a fixed few-shot preamble; `∅` = empty input):

| operator | definition | shape |
|---|---|---|
| `φ_Direct` | `A(p)` | per-`p` vector (= our existing acts) |
| `φ_Prefixed` | `A(Q‖p)` | per-`p` |
| `φ_Answered` | `A(Q‖p‖y^M(p))` | per-`p` (needs `y^M` first) |
| `φ_Verdict` | `A(y^M(p))` | per-`p`, but only 2 distinct values (near-degenerate; kept for completeness) |
| `φ_InputDelta` | `A(Q‖p) − A(p)` | per-`p` (the canonical MAG feature) |
| `φ_QuestionDelta` | `A(Q‖p) − A(Q)` | per-`p` |
| `φ_Interaction` | `A(Q‖p) − A(p) + A(∅)` | per-`p` |
| `φ_FewShot` | `A(E‖Q‖p)` | per-`p` |

`A(Q)`, `A(∅)` are constants (one forward each). `φ_Answered` and `φ_FewShot` add the verdict token /
preamble before the readout.

---

## 3. The fixed instruction `Q` and few-shot preamble `E`

Base gemma-2-2b, so these are plain completions, not chat turns. Exact strings (constants in
`mag/config.py`, so a reviewer can see and change them):

```python
Q_TRUTH = "Answer in yes or no only. Is the following statement true?\nStatement: "
# readout / y^M taken at the last token of  Q_TRUTH + statement
# e.g.  "Answer in yes or no only. Is the following statement true?\nStatement: The city of Paris is in France.\nAnswer:"
Q_SUFFIX = "\nAnswer:"      # appended after the statement so the next token is the verdict

E_FEWSHOT = (               # a fixed 2-shot preamble for φ_FewShot (one true, one false)
    "Answer in yes or no only. Is the following statement true?\n"
    "Statement: The sky is blue.\nAnswer: yes\n"
    "Statement: Fish can fly.\nAnswer: no\n"
)
```

Design notes:
- We will **test 1–2 `Q` phrasings** in the smoke run and keep the one whose `y^M` best agrees with
  gold on cities (report the agreement; this is an instrument choice, logged not hidden).
- `y^M` is read at the **first generated token position** after `Q_TRUTH + p + Q_SUFFIX`.

---

## 4. `y^M` computation (exact)

In `mag/verdict.py`:
1. Build the yes/no token-id sets from the tokenizer, summing case/space variants:
   `YES = {" yes","yes","Yes"," Yes"}`, `NO = {" no","no","No"," No"}` → their first-token ids.
2. One forward pass on `Q_TRUTH + p + Q_SUFFIX`; take logits at the final position; softmax.
3. `p_yes = Σ softmax over YES ids`, `p_no = Σ over NO ids`. `y^M = 1[p_yes > p_no]`.
4. Also store the **margin** `p_yes − p_no` and `max(p_yes, p_no)` (confidence) — used to report how
   decisive the base model's verdict is (feeds the §11 "is `y^M` meaningful?" check).

Cross-checks recorded once: `agreement(y^M, gold)` and `agreement(y^M, OLMo-judge verdict)` on the
unsteered statements, per dataset.

---

## 5. Module architecture

```
src/mag/
  __init__.py
  config.py       # Q_TRUTH, Q_SUFFIX, E_FEWSHOT, YES/NO variants, TAUS=[-1.0,-0.3,0,0.3,1.0]
                  #   (MAG's {0,0.3,1.0} plus negatives, so the lie-asymmetry test still applies)
  verdict.py      # y^M (Eq.1): p_yes/p_no, margin, confidence
  extract.py      # forward passes → mag_acts_<ds>.npz  (all operator ingredients + y^M)
  operators.py    # the 8 operators as pure functions over the cached readouts
  directions.py   # v_Q (Eq.2), u_Q (both y^M- and gold-labeled) → mag_dir_<ds>.npz
  probes.py       # E1_readability, E2_disagreement, E3_linearity, transfer_rank — pure, testable
  steer.py        # E4: α(τ) calibration + generation via existing Steerer; writes steer CSV
src/run_mag.py    # CLI: --dataset --probe {extract,e1,e2,e3,e4,transfer,all} --layer --limit --device
src/viz_mag.py    # one plot per probe (mirrors viz_steer/viz_funnel)
tests/
  test_mag_operators.py   test_mag_verdict.py   test_mag_directions.py
  test_mag_probes.py      test_mag_steer.py
docs/DCT_VS_MAG_ON_TRUTH.md   # findings writeup (written after the run)
```

**Reuse (do not reinvent):** `funnel_utils` (`unit`, `resolve_layer`, `load_acts`, `mean_diff_dir`,
`grad_dir`, `load_dct`, `top_k_by_potency`); `dct_steer_utils` (`load_model`, `Steerer`, `generate`);
`judge_results` (`STEER_SYS`, `run_steer`/`run_steer_local`, `extract_json`, `_steer_summary_and_plot`)
for E4 scoring. New code is only the MAG-specific math and the extraction of prefixed readouts.

**Data flow:**
```
mag/extract.py  (gemma-2-2b base, fwd-only)  →  mag_acts_<ds>.npz
      │   contains, per statement, at every layer's last-token readout:
      │     A_p, A_Qp, A_Qpv (answered), A_verdict, A_EQp (fewshot),  ymL, margin, conf
      │   plus constants A_Q, A_empty
      ▼
directions.py → mag_dir_<ds>.npz  (v_Q, u_Q_yM, u_Q_gold, A_prefix_norm, layer, per operator)
      ▼
probes.py → E1/E2/E3/transfer  → mag_<probe>_<ds>.csv
steer.py  → E4 generations       → mag_steer_<ds>.csv → OLMo judge → judge_mag_steer_<ds>.csv + plot
      ▼
viz_mag.py → plot_mag_<probe>_<ds>.png
```

`mag_acts_<ds>.npz` stores **all layers** (like `extract.py`'s `(n_layers+1, n, d)`) so E3 layer-wise
and the both-layers direction comparison need no re-extraction.

---

## 6. Directions (`directions.py`)

For a chosen readout layer `L` (computed for both `L = final` and `L = peak`):
- `v_Q = mean_p( A_Qp[L] − A_p[L] )` (Eq. 2), stored raw and unit.
- For each operator `φ`, `v_φ^± = mean over {p : label=±} of φ(p)[L]`, and `u_Q^φ = v_φ^- − v_φ^+`,
  built twice: labels = `y^M` and labels = gold. Primary steering direction uses `φ = φ_InputDelta`
  (the canonical MAG feature) but we store `u_Q` for `Direct` and `InputDelta` at minimum.
- `A_prefix_norm[L] = mean_p ‖A_Qp[L]‖` — the calibration constant for `α(τ)`.
- Also store cosines to the funnel's directions: `cos(v_Q, mean_diff)`, `cos(u_Q_gold, mean_diff)`,
  `cos(u_Q_gold, grad)`, `cos(u_Q, DCT_top_V)` — this extends our existing directions-comparison table
  to include MAG.

`mag_dir_<ds>.npz` keys: `v_Q, u_Q_yM, u_Q_gold, A_prefix_norm, layer_final, layer_peak`, plus the
cosine scalars. Mirrors `truth_dir_<ds>.npz` so cluster/laptop handoff is identical.

---

## 7. Probe E1 — Readability

**Question:** does a MAG operator's feature decode truth (or `y^M`) better than raw `A(p)` and better
than DCT's causal subspace?

**Procedure** (`probes.E1_readability`, at the peak layer; repeat at final block):
- For each operator `φ ∈` the 8: features `X_φ = [φ(p) for p]`; standardize (fit on train).
- Two targets per operator: `y^M` and `gold`.
- Classifier: `LogisticRegression(max_iter=2000)`, **5-fold stratified CV** (seed 42) — matches
  `analyze.py`. Record mean accuracy + ROC-AUC.
- **Baselines in the same table:** `φ_Direct` (raw `A(p)`); **DCT-top-k** features `A(p) @ V_topk`
  for `k ∈ {10, 50}` (reuse `funnel_utils.load_dct` + `top_k_by_potency`); **random-k** projection
  (control, seed 42).

**Reading:** MAG's claim is `φ_Prefixed`/`φ_InputDelta` > `φ_Direct` for predicting `y^M`. Our added
value: the `gold` column tells us whether the prefix helps decode *actual* truth or only *believed*
truth; the DCT-top-k rows tell us whether MAG's feature beats DCT's subspace at the same task.

**Output** `mag_readability_<ds>.csv`: `operator, target(yM|gold), acc, roc, n`.

---

## 8. Probe E2 — Disagreement

**Question:** on statements where the model's verdict `y^M` differs from the gold label, does a
classifier trained on `y^M` side with the **model** or the **label**? I.e. is gemma-2-2b's *belief*
(even when wrong) linearly encoded — quantifying "believed truth vs actual truth."

**Procedure** (`probes.E2_disagreement`):
- Disagreement subset `D = {p : y^M(p) ≠ gold(p)}`. Report `|D|` and its share.
- Train `LogisticRegression` on the best E1 operator's features to predict `y^M` on the full set
  (LODO-style: fit on the agree-set, evaluate on `D`), and measure `match_yM_rate` = fraction of `D`
  where the classifier's prediction matches `y^M` (not gold). Wilson 95% CI.
- Report the same for a raw-`A(p)` classifier, to show the effect is the prefix's (MAG's E2 claim:
  the MAG classifier follows `y^M`, ~69–74% on the disagreement subset).

**Reading:** `match_yM_rate` well above 0.5 ⇒ the model's *belief* is what's encoded, not the dataset
truth — an interpretability result our funnel currently can't state. If base-model `y^M` is too rare
in disagreement or too noisy, report that honestly (small `|D|` caveat).

**Output** `mag_disagreement_<ds>.csv`: `feature(mag|raw), n_disagree, match_yM_rate, ci_lo, ci_hi`.

---

## 9. Probe E3 — Linearity / reconstruction

**Question:** is the prefix-induced shift captured by a **single** linear direction `v_Q`?

**Procedure** (`probes.E3_linearity`):
- **Final-layer (primary):** `ε_Q` per Eq. 3 with `m̂(p) = A_p[L] + v_Q`. Also `cos` between the
  per-prompt shift `Δ^Q(p)` and `v_Q`, averaged.
- **Layer-wise (secondary):** inject the per-block marginal `v_{ℓ,Q} = mean_p(A_ℓ(Q‖p) − A_{ℓ}(p))`
  at every block and recompute `ε_Q`; MAG finds this often overshoots (`ε_Q > 1`). Uses the all-layer
  readouts already in `mag_acts`.
- **Comparison rows:** the same reconstruction framing applied to `mean_diff` and the DCT top vector
  as the injected direction, so `ε_Q` is comparable across methods.

**Reading:** low `ε_Q` (≈ MAG's 0.6–0.9 final-layer) ⇒ the truth-prefix effect is approximately a
single linear direction — MAG's complement to our linear-probe / XGBoost decodability result.

**Output** `mag_linearity_<ds>.csv`: `direction(v_Q|mean_diff|dct), mode(final|layerwise), eps_Q, cos`.

---

## 10. Probe E4 — Verdict steering (the head-to-head + calibration fix)

**Question (two-in-one):** (a) does MAG's `u_Q` steer truth better than the supervised direction or
DCT's top vector? (b) does our **symmetric-degradation** finding survive **norm-calibrated** magnitudes?

**Directions compared (all at their native injection layer):**
| direction | source | injection layer |
|---|---|---|
| `u_Q_gold`, `u_Q_yM` | this spec (§6) | peak layer |
| `mean_diff`, `grad` | `truth_dir_<ds>.npz` (existing) | peak layer |
| DCT top-`V` | `dct_V_<ds>.pt` (existing) | DCT source layer 13 |

**Strength:** for each direction `d`, inject `α(τ)·d̂` where `d̂ = unit(d)` and
`α(τ) = τ · A_prefix_norm[L_inject]`, for `τ ∈ {0, 0.3, 1.0}` **and their negatives** `τ ∈ {−1.0,
−0.3}` (we keep the signed sweep so the lie-vs-degradation asymmetry test still applies). `‖A_prefix‖`
is measured at each direction's own injection layer.

**Generation (reuse `dct_steer_utils`):** with `Steerer(model, L_inject)` set to the injected vector,
generate on two prompt sets:
- **matched-format yes/no** (MAG's E4): `Q_TRUTH + p + Q_SUFFIX`, `max_new_tokens ≈ 3`, count verdict
  **flips** vs `τ=0` over ~24 neutral factual statements.
- **free-form factual** (our Test-2 style): the 32 factual stems from the corrected rerun,
  `max_new_tokens = 8` → scored by the **OLMo judge** (TRUE/FALSE/INCOHERENT) via the existing
  `run_steer`/`run_steer_local` path.

**CSV schema** (so `judge_results.run_steer` consumes it unchanged): `direction, scale, prompt,
completion` where `scale` encodes `τ` (e.g. `-1.0, -0.3, 0, 0.3, 1.0`). Judge writes
`judge_mag_steer_<ds>.csv` + `plot_judge_mag_steering_<ds>.png`.

**Analysis:** feed `judge_mag_steer_<ds>.csv` through the **existing** `investigate_steer.py` battery
(McNemar lie-asymmetry, TOST, Cochran-Armitage degradation trend) so MAG steering gets the *same*
statistical treatment as DCT steering, at calibrated magnitudes. **Key comparison:** MAG's verdict-flip
rate vs the others; and whether P(TRUE) still falls symmetrically with `|τ|` (degradation) with **no**
`−vs+` FALSE asymmetry — if it does, our finding is robust to calibration; if calibrated steering
*does* flip truth where ±120 only degraded, that is a genuine correction to report.

**Output:** `mag_steer_<ds>.csv`, `judge_mag_steer_<ds>.csv`, `plot_judge_mag_steering_<ds>.png`,
`mag_verdict_flips_<ds>.csv` (direction, tau, flip_rate).

---

## 11. §4 — Transfer prediction (scaled to 4 datasets)

**Question:** does MAG geometry predict which truth dataset best trains a transferable probe, beating
raw centroid-cosine?

**Procedure** (`probes.transfer_rank`) — honest small-`n` adaptation:
- Datasets `{cities, sp_en_trans, companies_true_false, common_claim_true_false}`.
- For each **target** `T` (leave-one-out) and each **candidate** `C ≠ T`: realized transfer
  `Δ(C,T) = Acc(train probe on C, test on T)` (logistic probe on the peak-layer `φ_InputDelta`
  features; also raw `A(p)` as a baseline feature space).
- Geometric predictor: rank candidates by a MAG-operator **centroid cosine** (and, as the paper's
  stronger variant, a small combination incl. class-conditional centroid cosine and CKA) between `C`
  and `T`. Compare its ranking to the realized `Δ` ranking.
- Metrics: **Top-1 accuracy** (does the geom-top candidate == realized-top?) over the 4 targets, and
  Spearman ρ over all 12 `(C,T)` pairs; baseline = raw-centroid-cosine.

**Reading + honesty:** with only 4 datasets this is **illustrative, not powered** (4 targets, 12
pairs) — state that plainly. A positive signal (geom Top-1 > raw Top-1) is suggestive that MAG's
model-relative geometry carries transfer information; a null is expected-given-`n` and reported as
such.

**Output** `mag_transfer.csv`: `target, candidate, realized_delta, geom_score, raw_cos, geom_rank,
realized_rank` + a one-line Top-1/Spearman summary.

---

## 12. Compute plan

- **Extraction + E1/E2/E3 + transfer:** laptop, `--device mps` (fallback cpu). gemma-2-2b fp32 eager
  ≈ 9.7 GB (per project memory); ~3–4 forward passes/statement; cities (1,496) is the largest → tens
  of minutes on MPS. No cluster needed.
- **E4 generation:** laptop (`Steerer` + short generations) — cheap.
- **OLMo judge scoring of E4:** the **existing GH200 path** (`.venv-judge-gpu`, `run_judge.slurm`), or
  local if the judge fits; reuse `submit_rerun.sh`-style chaining. The judge is unchanged — it already
  validated at 0.970.
- **Optional GH200 extraction** (`.venv-dct-gpu`) if MPS is slow: a thin `run_mag.slurm` mirroring
  `run_dct.slurm`. Not required for correctness.

**Determinism:** greedy decoding (`do_sample=False`), seed 42 everywhere, `repetition_penalty=1.3`
(matches `dct_steer_utils.generate`) so E4 completions match the DCT-arm generation settings exactly.

---

## 13. Testing strategy (TDD — write tests first per component)

Pure functions are separated from I/O precisely so they're unit-testable without loading gemma:

- **`test_mag_operators.py`** — with synthetic readout arrays: `InputDelta == Prefixed − Direct`;
  `Interaction == Prefixed − Direct + Empty`; `QuestionDelta == Prefixed − A_Q`; shapes and dtypes;
  operators are pure (no mutation of inputs).
- **`test_mag_verdict.py`** — feed synthetic logits favoring a yes-token → `y^M == 1`; favoring no →
  `0`; tie handling; margin sign matches; multi-variant id summation works.
- **`test_mag_directions.py`** — `v_Q` shape = `d` and equals mean of shifts; `u_Q` = class-mean diff;
  unit vectors are unit-norm; `A_prefix_norm` > 0; cosines in `[−1, 1]`.
- **`test_mag_steer.py`** — `α(0) == 0` ⇒ injected vector is zero ⇒ steered == unsteered (mock model);
  `α` linear in `τ`; injected-vector norm ≈ `|τ|·A_prefix_norm` (calibration invariant); signed `τ`
  flips vector sign.
- **`test_mag_probes.py`** — `ε_Q == 0` when `v_Q` perfectly reconstructs a synthetic shift; `ε_Q ≈ 1`
  when `v_Q == 0`; E1/E2/transfer return the documented dict/CSV keys; disagreement subset logic
  correct on a hand-built label pair.
- **Smoke:** `run_mag.py --dataset cities --probe all --limit 20 --device cpu` completes end-to-end and
  writes every CSV with sane shapes.

Existing test suite must stay green (`pytest -q`).

---

## 14. Outputs (what the run produces)

| file | from | content |
|---|---|---|
| `mag_acts_<ds>.npz` | extract | all-layer readouts for 8 operators + `y^M`, margin, conf |
| `mag_dir_<ds>.npz` | directions | `v_Q`, `u_Q_yM/gold`, `A_prefix_norm`, cosines-to-funnel-dirs |
| `mag_readability_<ds>.csv` | E1 | operator × target decoding acc/roc, incl. DCT-top-k & random-k |
| `mag_disagreement_<ds>.csv` | E2 | model-vs-label match rate on the disagreement subset |
| `mag_linearity_<ds>.csv` | E3 | `ε_Q` + cos, final & layer-wise, across directions |
| `mag_steer_<ds>.csv` + `judge_mag_steer_<ds>.csv` | E4 | calibrated steering completions + judged verdicts |
| `mag_verdict_flips_<ds>.csv` | E4 | matched-format yes/no flip rate per direction × τ |
| `mag_transfer.csv` | §4 | geom-vs-realized transfer ranking across the 4 datasets |
| `plot_mag_*_<ds>.png`, `plot_judge_mag_steering_<ds>.png` | viz_mag | one figure per probe |
| `docs/DCT_VS_MAG_ON_TRUTH.md` | (post-run) | the findings writeup |

Gitignore the large regenerable `mag_acts_*.npz` (like the DCT acts); commit code, tests, small CSVs,
plots, and the doc.

---

## 15. Risks & open questions (resolve during implementation, not now)

1. **Base-model `y^M` may be near-random.** Mitigation: report `agreement(y^M, gold)` and verdict
   confidence/margin up front; if low, the **gold-labeled** `u_Q` arm still delivers the full E1/E3/E4
   DCT-vs-MAG comparison, and "the base model has no decisive self-verdict" becomes a finding (it
   parallels our steering-baseline 0.81 story).
2. **Injection-layer mismatch in E4** (peak 11 vs DCT source 13). Primary run injects each direction at
   its native layer (how each method is meant to be used); **optional robustness variant** injects all
   at a common layer. Flagged, not silently resolved.
3. **`φ_Verdict` is near-degenerate** (2 distinct values). Kept for completeness; expect it to be
   uninformative — that's the correct MAG-faithful result, not a bug.
4. **Transfer §4 is small-`n`** (4 datasets). Reported as illustrative with the caveat stated; not a
   headline claim.
5. **`Q` phrasing sensitivity.** We test 1–2 phrasings and log the choice; not tuned for effect.

---

## 16. Success criteria

- All 8 operators + E1–E4 + §4 implemented, unit-tested, and green; existing suite still green.
- A single `run_mag.py --dataset <ds> --probe all` reproduces every CSV/plot from cached acts.
- E4 judged through the **same** OLMo judge and **same** `investigate_steer.py` battery as the DCT arm,
  at calibrated `α(τ)`.
- `docs/DCT_VS_MAG_ON_TRUTH.md` states, with numbers: (i) whether MAG decodes truth/`y^M` better than
  raw and than DCT (E1), (ii) the believed-vs-actual-truth divergence (E2), (iii) whether the truth
  prefix is one linear direction (E3), (iv) **whether our symmetric-degradation steering finding
  survives MAG's magnitude calibration and how MAG's `u_Q` compares to `mean_diff`/DCT (E4)**, and
  (v) the transfer signal (§4, caveated).
- Every cross-method claim ties back to a direction cosine or a judged rate, never a hand-read.
