# Token-space conclusions: design

**Date:** 2026-08-04
**Status:** design approved, not yet implemented
**Predecessors:** `docs/TOKEN_SPACE_PROGRAM.md` (what was run),
`docs/NEXT_STEPS_BRIEF_AND_RESEARCH_PROMPT.md` Part II (the PI's eight claims),
`docs/PROJECT_PROGRESS_TO_DATE.md` (house style for the output doc).

---

## 0. What this builds and why

Every cluster job in the token-space program has finished. Twelve `token_steer_*.csv`
arms, two `token_geom_*.csv`, two `token_jac_*.csv`, eight `token_sens_*.csv`, plus the
older `signed_steer_*` and `sae_features_*` artifacts are on the local disk. What has
been extracted from them so far is a single number per arm: the hit rate on the target
token.

That is a small fraction of what is there. Three columns that decide the PI's questions
have never been read:

| Column | File family | Rows | What it decides |
|---|---|---|---|
| `frac_margin` | `token_steer_*` | 5,400 per arm | The unit the PI explicitly asked for, never tabulated |
| `readout_delta` | `token_steer_*` | 5,400 per arm | The same quantity in absolute units, see the warning below |
| `completion` | `token_steer_*` | 4,660 non-null per arm | Claim 2 is literally a claim about text coherence, and not one string has been read |

**Warning about `readout_delta`, established during this spec's own review.** The column
name is misleading. `src/token_steer.py:322` writes `r - r0` where `probe` returns
`r = logit[j_tgt] - logit[j_top]`. That is the **token margin**, not the legacy truth
probe readout. It is therefore identical to `frac_margin` up to the per-statement constant
`|m0|`, which was verified numerically: `readout_delta / frac_margin` has a per-statement
standard deviation of 2.4e-5 against values around 19.

Two consequences bind the analyses below. The two columns must never be presented as
independent evidence, or the verdict document double-counts a single measurement. And the
readout-versus-behavior dissociation is **not** directly measurable from these files,
which changes A1 from a measurement into a closed-form reconstruction. A separate fix to
the column name in `src/token_steer.py` is out of scope here and is left as a follow-up.

This project builds two documents and one tested module: an unpolished findings dump for
the researcher to react to, a claim-by-claim verdict document for the PI, and a module
holding the computations behind every number that survives into the verdict document.

**Scope boundary.** CPU only, local only. No cluster jobs, no model loads, no judge runs,
no regeneration of any artifact. Every input file already exists on disk.

---

## 1. The organizing principle: pre-registered readings

The single most important design decision here is that **each analysis states, before it
runs, what each possible outcome would mean.** Section 3 carries a "Reading" block for
every analysis, and those blocks are written into the spec so they cannot be adjusted
after seeing the number.

The reason is specific to this project rather than general good practice. The program has
5,400 rows per arm across twelve arms, nine analysis axes, two datasets, and three
directions. That is enough surface area to find a satisfying story in noise. The
conclusions go to a PI who will reasonably ask whether the same conclusion would have been
drawn had the number gone the other way. A reading fixed in advance is the only answer to
that question that holds up.

A corollary that binds the implementation: **an analysis whose result contradicts its
pre-registered expectation is reported as-is, in both documents, with the expectation
quoted.** Two assertions in this program have already cried wolf on working code (the E6
fp32 tolerance and the `oracle_line` site gate). Both were caught because the expectation
was written down first.

---

## 2. Architecture: two passes

### Pass 1, extraction

Nine analysis scripts in the scratchpad directory
(`/private/tmp/claude-501/-Users-vaibhav-wudaru-llm-activation-steering-research/c8145135-7f79-4eff-81b2-f40f140cba80/scratchpad`).
Each reads CSVs with pandas, prints a labelled block to stdout, and exits. Output is
appended to `docs/TOKEN_SPACE_RAW_FINDINGS.md`.

The raw findings file is deliberately unpolished. It contains every number produced,
including nulls, including numbers that turn out to mean nothing, including anything
unexpected the extraction surfaces. Its audience is the researcher reading it cold to
decide what matters. It is not written for the PI and says so in its header.

Scratchpad scripts are not tested. They are exploratory and most will be discarded.

### Pass 2, writeup and promotion

Read `docs/TOKEN_SPACE_RAW_FINDINGS.md` cold. Write `docs/TOKEN_SPACE_FINDINGS.md` in the
structure of `PROJECT_PROGRESS_TO_DATE.md`:

1. One-paragraph version
2. Claim table: the PI's eight claims, each with a verdict and a pointer to its section
3. One numbered section per claim, each stating method, number, figure, and reading
4. A quantified ledger, wins stated as numbers
5. An honest column, what is not established
6. Where this goes next

Then promote into `src/token_conclusions.py` **only** the computations whose numbers
appear in the verdict document, with `tests/test_token_conclusions.py` covering them.
Analyses that produced nothing worth reporting are not promoted and not tested. This is
the reason for the two-pass split: writing tests for nine analyses when perhaps five
produce reportable numbers is waste, and the repo's 299-test norm is worth keeping honest
rather than padded.

### Why this split rather than one pass

A single claim-driven pass would only find what it went looking for. The three most
promising leads in this analysis (the within-run dissociation, the coherence of steered
completions, and whether alpha predicts behavior) were all discovered by looking at column
schemas rather than by working down the claim list. Open-ended extraction feeding a
disciplined writeup keeps both properties.

---

## 3. The nine analyses

Common facts used throughout:

- **Datasets:** `cities` and `common_claim_true_false`, 200 statements each.
- **Join key:** `token_steer.stmt` equals `token_geom.idx` exactly, verified. Both are
  original dataset row indices, not 0-based positions. Values run 1 to 1495 on cities.
- **Arms:** twelve `token_steer_*.csv` files. Naming is
  `token_steer_<ds>_<site>_<positions>_rp<penalty>.csv` with sites `postnorm`, `prenorm`,
  `layer16` (cities) and `layer8` (common_claim), positions `all` and `last`, penalties
  `1` and `1.3`.
- **Directions:** `oracle`, `md_full`, `jtw_legacy` in the post-norm and pre-norm arms;
  `oracle`, `md_full`, `jtw_token` in the layer arms.
- **Fracs:** -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0.
- **Coverage caveat that binds every cross-direction comparison:** `jtw_legacy` on
  common_claim covers only 90 of 200 statements, because `reach_margins` was computed on a
  2,000-row subsample of a 4,450-row dataset. Any statistic comparing `jtw_legacy` against
  another direction must first restrict to the statements both ran. `src/viz_token.py`
  `fig_steer` already does this and `tests/test_viz_token.py` pins the behavior.

### A1. Readout-behavior cross-tab, by closed-form reconstruction

**Input:** all twelve `token_steer_*.csv` (columns `scale`, `hit_target`, `stmt`) plus
`reach_margins_<ds>.npz` (arrays `jtw` of shape (1496, 14, 2304), `margins` of shape
(1496, 78), and `store_names`, whose entry `mean_diff_tgt` is the legacy readout).

**Why this is a reconstruction and not a measurement.** As established above, the
`token_steer` CSVs do not log the legacy truth readout at any scale. What they do log is
the absolute `scale` in activation-norm units. Because `jtw_legacy` is by construction the
unit vector along `J^T w`, the legacy readout displacement under a scale `sc` is exactly
`sc * ‖J^T w‖` to first order, and the legacy certificate defines
`eps*_legacy = g / ‖J^T w‖` precisely so that the readout crosses its threshold at
`sc = eps*_legacy`. So `crossed` reduces to `scale / eps*_legacy >= 1`, with no activation
data required.

**What licenses the first-order step.** The project has already measured the readout
against scale directly: per-statement R² of 0.9991 on cities, with 97% of statements above
0.99. The linear prediction is not an assumption being introduced here, it is a previously
validated result being reused. This must be stated wherever the A1 number appears, because
a reader is entitled to know the readout axis of the 2x2 is predicted rather than observed.

**Computation.** Read `eps*_legacy` per statement from `margins` for the `mean_diff_tgt`
entry, matching on the same dataset row index used by `token_steer.stmt`. For each row
define `crossed = |scale| / eps*_legacy >= 1` and `flipped = hit_target`. Build the 2x2
contingency table per direction per arm, restricted to shared statements. Report counts and
the phi coefficient.

**Reading.** High `crossed` with zero `flipped` for `jtw_legacy` is the
certified-reachable-but-behaviorally-inert dissociation, expressed within a single
experiment rather than assembled across two runs and a judge, which is the weakest link in
the existing audit. If `crossed` and `flipped` correlate for `jtw_token` at the same site
and budget, the dissociation is established as a property of the legacy direction
specifically rather than of the model or the harness.

If `jtw_legacy` never reaches `crossed` at any swept frac, the dissociation is not
demonstrated by this run at all, and the correct report is that the token-space budget
`eps*/alpha` and the legacy budget `eps*_legacy` are different scales that happen to share
a symbol. That outcome is entirely possible and would be a useful negative result about
the sweep design, not a failure of the analysis.

### A2. Margin consumption

**Input:** all twelve `token_steer_*.csv`, column `frac_margin`.

**Computation.** Per direction per arm per frac, report median and p90 of `|frac_margin|`.
Highlight the value at frac 1.0 and 2.0.

**Report `frac_margin` only, not `readout_delta`.** The two are the same measurement
rescaled per statement by `|m0|`. Reporting both would present one number twice and inflate
the apparent weight of the evidence. `frac_margin` is the one to keep because it is
dimensionless and is the unit the PI asked for; `|m0|` is separately available in
`token_geom` as `margin` if an absolute figure is ever wanted.

**Reading.** `frac_margin` is the fraction of the logit margin consumed, which is the unit
the PI asked for and which has never appeared in a results table in this project. If
`jtw_legacy` consumes on the order of 0.01 at twice its certified budget, then every
historical behavioral null in this project was a null about budget denomination rather
than about truth, and that sentence is the headline of the whole program.

A value near 1.0 for `jtw_legacy` would falsify that diagnosis outright and would mean the
direction did consume the margin and behavior still did not move, which is a different and
stronger claim. The number decides which.

### A3. Completion forensics, the claim 2 test

**Input:** the `completion` column across all twelve arms, 4,660 non-null rows in the
cities post-norm arm.

**Known wrinkle to handle and explain, not paper over.** In
`token_steer_cities_postnorm_all_rp1.csv`, all 740 null completions belong to `md_full`,
while `oracle` and `jtw_legacy` have none. The analysis must report this asymmetry, check
whether it holds across arms, and state what it implies for any `md_full` coherence
statistic before computing one.

**Computation.** Per direction per frac:

1. **Degeneracy rate.** Fraction of completions that are empty, that repeat a token three
   or more times consecutively, or that contain no alphabetic character.
2. **Normalized edit distance** from the same statement's `frac == 0` completion, using
   `difflib.SequenceMatcher` ratio, so no new dependency.
3. **Country swap.** For cities only, whether the country named in the completion differs
   from the country named at `frac == 0`. `got_datasets/cities.csv` carries `country` and
   `correct_country` columns per row, so this is an exact per-statement lookup joined on
   `stmt`, not a match against a scraped vocabulary. The three-way outcome is recorded:
   the completion names the correct country, names the false target country, or names some
   third country. The third case is the interesting one and a plain binary would hide it.

**Reading.** Claim 2 asserts that if direct final-layer steering only degrades text, the
truth feature is not behaviorally linear at the final layer. The oracle at frac 1.0 flips
the target token on 100% of statements, so it is the exact test case.

- If oracle completions at frac 1.0 are fluent with low degeneracy, **claim 2 is
  refuted.** The final layer is behaviorally linear at the certified magnitude, the text
  stays coherent, and the flip is real.
- If the first token flips but the named country reverts within the completion, that is a
  distinct and reportable result: first-token control without semantic control. It would
  mean the argmax cone certificate controls exactly what it claims to control and nothing
  downstream of it, which is a genuine limitation of the token-space formulation rather
  than a failure.
- If degeneracy is high, claim 2 is confirmed and the E5 and E6 branch the brief specified
  becomes live again.

An informal look at three oracle completions at frac 1.0 shows fluent, grammatical text
("the northern part of Hebei Province,"). The pre-registered expectation is therefore that
claim 2 will be refuted. That expectation is recorded here so the analysis cannot quietly
be re-aimed if the systematic measurement disagrees.

### A4. Does alpha predict behavior

**Input:** `token_geom_<ds>.csv` columns `alpha_md_full`, `alpha_mean_diff_tgt_asis`,
`alpha_probe_grad_tgt_asis`, merged onto per-statement hit rates from
`token_steer_<ds>_postnorm_all_rp1.csv` on `idx == stmt`.

**Computation.** Per statement, compute the hit rate across positive fracs for each
direction. Spearman correlation against that direction's alpha. Also report hit rate
bucketed by alpha quartile, because a monotone bucket table is more legible to a reader
than a single rho and does not assume linearity.

**Reading.** Alpha is the geometric predictor at the center of the whole token-space
framing: `eps(u) = eps* / alpha`. It has never been checked against realized behavior.

- If alpha predicts hit rate, the certificate framework is validated end to end, and the
  claim that these directions are uninformative about the token decision is supported by
  behavior rather than by geometry alone. That is a substantially stronger result than
  either piece separately.
- If alpha does not predict hit rate, the alpha framing is descriptive bookkeeping and
  must be labelled as such in the verdict document. This outcome would not invalidate the
  measured alphas, but it would remove their predictive claim.

The power limitation is real and must be stated: `md_full` and `jtw_legacy` hit rates are
near zero on cities, so within-direction variance may be too small to correlate. If so,
report that the test was underpowered rather than reporting a null correlation as
evidence.

### A5. E7 resolution, broadcast versus per-position

**Input:** `token_steer_<ds>_layer{16,8}_{all,last}_rp1.csv`, `token_jac_<ds>.csv` column
`broadcast_gain`.

**The open observation.** On cities layer 16, `jtw_token` broadcast reaches 0.340 at frac
1.0 against 0.135 per-position, despite each spending its own separately certified budget.
On common_claim layer 8 there is no such gap, 0.815 both ways. This is the one result in
the program currently marked unresolved.

**Computation.** Compare the observed broadcast advantage against `broadcast_gain` from
`token_jac` (independently computed, 1.55 at cities layer 16). Compute mean stem token
length per dataset, since broadcast energy scales as `sqrt(T)` and the two datasets have
different stem lengths.

**Reading.** Finding A in the brief flagged that broadcasting to every position including
BOS is a candidate confound for the entire negative result, because gemma-2 uses BOS as an
attention sink. If `broadcast_gain` and stem length together account for the cities gap
and for its absence on common_claim, that confound is quantified and closed. If they do
not, E7 remains open and is reported as the single unresolved item in the program rather
than omitted.

### A6. SAE decomposition, claim 5

**Input:** `sae_features_<ds>.csv`, columns `vector, layer, rank, feature, coef,
cumulative_explained`.

**Computation.** Per vector, report cumulative explained variance at ranks 1, 5, and 10,
and the top feature IDs with coefficients. State the overlap between the truth readout's
top features and the steering vector's top features.

**Reading.** Claim 5 proposed SAEs as an alternative way to define the target set. The
existing artifacts produced two plots and no stated numbers. If the truth readout
decomposes onto features that are not about truth, D3 (input-side concept misalignment) is
confirmed with interpretable evidence rather than by cosine similarity, which is what the
brief asked for. Prior expectation from the project record is an overlap near 0.05, which
was pre-warned before the run. Confirm or correct it, and either way give claim 5 a number.

The literature's negative prior on SAE features as steering directions is not tested here
and must not be implied. SAEs are used to diagnose the target set, not to actuate it.

### A7. Failure-mode split across datasets

**Input:** `signed_steer_<ds>.csv`, `signed_steer_summary_<ds>.csv`, cross-referenced
against per-dataset alpha from `token_geom_<ds>.csv` and `delta_rel_z`.

**The known split.** cities: 103 of 199 inert, 24 movers with the direction against 4
against, sign test p = 1.8e-4. common_claim: 8 of 197 inert, 35 with against 21 against,
p = 0.081. cities fails by inertness. common_claim is partly the Tan et al.
anti-steerability regime. The project record is explicit that these must never be pooled.

**Computation.** Tabulate the two failure modes beside each dataset's token-space
quantities: median `delta_rel_z` (0.034 cities, 0.011 common_claim), median alpha, and
`rmsnorm_penalty`.

**Reading.** If the dataset with the larger relative flip cost is the one that fails by
inertness, the two failure modes reduce to one mechanism expressed at two budget scales,
which is a simpler and more defensible story than two independent failures. If the
quantities do not line up that way, the two-mechanism account stands and is reported as
two mechanisms.

A confound to state rather than resolve: common_claim's target is the runner-up token and
some statements are near-ties (minimum `delta_cone` 0.0027 against a norm of 168), so
generic disruption can land on target there. cities, with a semantic false-country target,
is the discriminating dataset and should lead any presentation of this result.

### A8. Prenorm arithmetic closure

**Input:** `token_geom_<ds>.csv` column `rmsnorm_penalty`, `token_sens_<ds>_prenorm_all.csv`
column `gain`, and prenorm hit rates from `token_steer_<ds>_prenorm_all_rp1.csv`.

**Computation.** Tabulate predicted against observed. The measured pre-norm gain is
0.232 to 0.252 on cities against `1/4.38 = 0.228`, and 0.194 to 0.207 on common_claim
against `1/5.18 = 0.193`. Both computed independently, one from `token_sens` on the GPU and
one from `token_geom` in numpy.

**Reading.** The oracle reads 0.000 at every frac out to 2.0 at the pre-norm site on both
datasets. That is not a failure, it is arithmetic: a post-norm-certified displacement needs
roughly 4.4x (cities) or 5.2x (common_claim) the budget to survive RMSNorm, and the sweep
stops at 2x. Writing this as a table converts a null into a quantitative closure.

**The falsifiable prediction to state explicitly:** sweeping the pre-norm arm to frac 6
should recover the oracle to near 1.0. It is not run here (out of scope, needs the
cluster), but it is stated so the claim is refutable rather than merely consistent.

`token_sens` also carries eight `random*` directions and `a_unit`, giving a built-in chance
baseline for gain that should be reported alongside the candidate directions.

### A9. Validity ledger

**No computation.** A written section enumerating what the numbers do not support:

- n = 200 statements per dataset, a subsample of 1,496 and 4,450 respectively
- `jtw_legacy` covers only 90 of 200 on common_claim
- One model, `google/gemma-2-2b`, fp32
- Two datasets, both English, both short declarative statements
- Targets are first-token only. The certificate controls the first emitted token and makes
  no claim about the rest of the completion
- Temperature 0 only. Nothing here transfers to sampling without re-derivation
- The `_asis` alphas are cross-layer carryovers: those directions were fitted at the target
  layer and read in post-norm coordinates

---

## 4. Testing

`tests/test_token_conclusions.py` covers the promoted computations in `src/token_conclusions.py`.
Tests use small synthetic DataFrames constructed in the test file, never the real CSVs, so
they run in under a second and do not depend on artifacts that may be regenerated.

Required coverage, one test each:

1. **Shared-statement restriction.** A cross-direction statistic computed on unequal
   coverage returns a different answer than one restricted to the intersection. This is the
   same defect class already pinned in `tests/test_viz_token.py` and it applies to A1, A2,
   and A4 exactly as it applies to the figure.
2. **The intersection is the intersection.** Two directions each short in a different place
   must yield the shared set, not the smaller direction's set.
3. **Null completions are excluded, not counted as degenerate.** A missing completion and an
   empty completion are different things, and the 740 `md_full` nulls would otherwise inflate
   its degeneracy rate to a number that looks like a finding.
4. **The 2x2 handles an empty cell.** `jtw_legacy` is expected to produce a zero column, and
   phi must not raise or return NaN silently.
5. **The join key is dataset row index, not position.** A merge that assumes 0-based
   positions silently produces an empty or wrong join, and both frames use the same column
   values so the error would not be obvious.
6. **`frac_margin` and `readout_delta` are proportional per statement.** Construct a frame
   where the two are proportional and assert the helper detects it, and one where they are
   not and assert it does not. This pins the finding that drove A1's rewrite. If
   `src/token_steer.py` is later corrected to log the actual truth readout, this test fails
   loudly and A1 and A2 must be revisited, which is exactly the notification wanted. A
   silent change there would turn A1's closed-form reconstruction back into a measurement
   without anyone noticing that the doc still calls it a prediction.
7. **`crossed` is computed against the legacy budget, not the token budget.** Given a
   statement whose `eps*_legacy` and `eps*/alpha` differ by two orders of magnitude, assert
   that `crossed` uses the former. Swapping them is a one-character error that would make
   the dissociation appear or vanish entirely.

---

## 5. Deliverables

| Path | Contents | Tested |
|---|---|---|
| `docs/TOKEN_SPACE_RAW_FINDINGS.md` | Pass 1 output, unpolished, all nine analyses including nulls | no |
| `docs/TOKEN_SPACE_FINDINGS.md` | Claim-by-claim verdict document in `PROJECT_PROGRESS_TO_DATE.md` house style | no |
| `src/token_conclusions.py` | Computations behind every number in the verdict document | yes |
| `tests/test_token_conclusions.py` | The five tests in section 4 | n/a |

No new figures. The five per dataset produced by `src/viz_token.py` are sufficient unless an
analysis surfaces something they do not show, in which case it is added to `viz_token.py`
rather than to a new module.

---

## 6. Constraints carried from the project record

- No em dashes in prose.
- Never `git add -A` or `git add .`. Stage named files only.
- Never stage the other track's uncommitted files: `src/spectrum_utils.py`,
  `src/viz_spectrum.py`, `tests/test_spectrum_utils.py`, `tests/test_viz_spectrum.py`,
  `docs/RESULTS_SINCE_LAST_MEETING_PART2.md`, `docs/DEEP_RESEARCH_PROMPT_REACHABILITY.md`,
  `plot_mag_linearity_v2.png`.
- Never re-run or overwrite any existing truth artifact. Every script here reads only.
- Never import `xgboost` in the same process as `torch`.
- Never index into a loaded `.npz` inside a loop.
- `git push` is blocked by the permission classifier and is the researcher's to run.

---

## 7. Out of scope

- Any cluster job, including the frac-6 pre-norm test, which is stated as a prediction only
- Any model load, generation, or judge run
- Regenerating or overwriting any artifact
- The refusal positive control, which remains the publication gate and is unrun
- Restructuring or editing the existing findings documents. This adds one document and does
  not revise `REACH_AUDIT_FINDINGS.md`, `RESULTS_SINCE_LAST_MEETING_PART3.md`, or
  `PROJECT_PROGRESS_TO_DATE.md`
