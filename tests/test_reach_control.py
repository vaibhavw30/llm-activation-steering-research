# tests/test_reach_control.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import reach_control
from reach_control import (frac_of, align_stmt_rows, aggregate, readout_crossed,
                           behavior_delta, verdict, require_signal, mean_arm_rows,
                           stmt_crossing_fracs, near_boundary_check,
                           count_empty_completions)


def _j(scale, refused):
    return {"direction": "jtw_stmt", "scale": str(scale), "prompt": "p",
            "completion": "c", "refused": str(refused)}


def _m(scale, eps, g, stmt_index=0):
    return {"stmt_index": str(stmt_index), "label": "1", "eps_star": str(eps),
            "scale": str(scale), "g_read": str(g)}


def _stmt_rows(index, eps_star, baseline_g, treated_g=None, refused_treated=1.0):
    """(judged, meta) rows contributed by one per-statement-arm subject: always a
    baseline row (scale=0); a treatment row at frac=-1 (scale=-eps_star) only if
    the statement was actually steered (eps_star>0) -- a never-steered statement
    (reach_steer.py:167 -- eps_i=0 when g_i<=0) contributes ONLY the lone
    baseline row, since scale_grid returns [0.0] in that case."""
    judged = [_j(0.0, 0)]
    meta = [_m(0.0, eps_star, baseline_g, stmt_index=index)]
    if eps_star > 0.0:
        judged.append(_j(-eps_star, refused_treated))
        meta.append(_m(-eps_star, eps_star, treated_g, stmt_index=index))
    return judged, meta


def test_frac_of_divides_by_eps_star():
    assert frac_of(4.0, 2.0) == 2.0
    assert frac_of(-2.0, 2.0) == -1.0


def test_frac_of_is_zero_when_eps_star_is_zero():
    assert frac_of(5.0, 0.0) == 0.0


def test_align_stmt_rows_pairs_by_row_order():
    out, n_never_steered = align_stmt_rows(
        [_j(0.0, 0), _j(3.0, 1)],
        [_m(0.0, 3.0, 5.0, stmt_index=7), _m(3.0, 3.0, -1.0, stmt_index=7)])
    assert n_never_steered == 0
    assert [r["frac"] for r in out] == [0.0, 1.0]
    assert [r["refused"] for r in out] == [0.0, 1.0]
    assert [r["g_read"] for r in out] == [5.0, -1.0]
    assert [r["stmt_index"] for r in out] == [7, 7]


def test_align_stmt_rows_rejects_length_mismatch():
    with pytest.raises(SystemExit, match="row-count"):
        align_stmt_rows([_j(0.0, 0)], [_m(0.0, 3.0, 1.0), _m(3.0, 3.0, -1.0)])


def test_align_stmt_rows_rejects_scale_mismatch():
    with pytest.raises(SystemExit, match="scale mismatch"):
        align_stmt_rows([_j(0.0, 0), _j(3.0, 1)],
                        [_m(0.0, 3.0, 5.0), _m(9.0, 3.0, -1.0)])


def test_aggregate_buckets_by_rounded_frac():
    rows = [{"frac": 1.0001, "g_read": -1.0, "refused": 1.0},
            {"frac": 0.9999, "g_read": -3.0, "refused": 0.0},
            {"frac": 0.0, "g_read": 5.0, "refused": 0.0}]
    t = aggregate(rows)
    assert set(t) == {0.0, 1.0}
    assert t[1.0]["n"] == 2
    assert t[1.0]["g_read"] == -2.0
    assert t[1.0]["frac_refused"] == 0.5


def test_readout_crossed_flags_nonpositive_g():
    t = {0.0: {"g_read": 5.0}, 1.0: {"g_read": -2.0}, -1.0: {"g_read": 9.0}}
    assert readout_crossed(t) == {0.0: False, 1.0: True, -1.0: False}


def test_behavior_delta_is_relative_to_the_zero_baseline():
    t = {0.0: {"frac_refused": 0.05}, 1.0: {"frac_refused": 0.55},
         -1.0: {"frac_refused": 0.02}}
    d = behavior_delta(t)
    assert abs(d[1.0] - 0.50) < 1e-12
    assert abs(d[-1.0] + 0.03) < 1e-12
    assert d[0.0] == 0.0


def test_behavior_delta_requires_a_baseline():
    with pytest.raises(SystemExit, match="baseline"):
        behavior_delta({1.0: {"frac_refused": 0.5}})


def test_verdict_actuatable_when_crossing_moves_behavior():
    assert verdict({0.0: False, 1.0: True}, {0.0: 0.0, 1.0: 0.5}) == "actuatable"


def test_verdict_readout_only_when_crossing_moves_nothing():
    assert verdict({0.0: False, 1.0: True}, {0.0: 0.0, 1.0: 0.01}) == "readout-only"


def test_verdict_inert_when_behavior_moves_without_a_crossing():
    assert verdict({0.0: False, 1.0: False}, {0.0: 0.0, 1.0: 0.5}) == "inert"


def test_verdict_no_crossing_when_nothing_crosses_and_nothing_moves():
    assert verdict({0.0: False, 1.0: False}, {0.0: 0.0, 1.0: 0.0}) == "no-crossing"


def test_verdict_requires_crossing_and_movement_at_the_same_frac():
    # Crossing only at frac=-1; movement only at frac=+2. These are DIFFERENT fracs,
    # so this must NOT be "actuatable" -- it is the off-target "inert"-shaped result
    # the module's own docstring calls out (crossings can only occur at negative
    # fracs per reach_steer.py:167; a large +2*eps* shift can degrade generations
    # with no crossing at all). Independent "any_cross"/"any_move" quantifiers over
    # different fracs would wrongly report "actuatable" here.
    crossed = {0.0: False, -1.0: True, 2.0: False}
    delta = {0.0: 0.0, -1.0: 0.0, 2.0: 0.5}
    assert verdict(crossed, delta) != "actuatable"
    assert verdict(crossed, delta) == "readout-only"


def test_verdict_min_delta_boundary_is_inclusive():
    # Exactly MIN_DELTA (0.10) at the SAME frac as the crossing must count as moved.
    crossed = {0.0: False, 1.0: True}
    delta = {0.0: 0.0, 1.0: 0.10}
    assert verdict(crossed, delta) == "actuatable"


def test_verdict_negative_delta_counts_as_moved():
    # A steered DROP in refusal rate is evidence of actuation, not of nothing --
    # abs() must be applied, not a signed comparison.
    crossed = {0.0: False, 1.0: True}
    delta = {0.0: 0.0, 1.0: -0.5}
    assert verdict(crossed, delta) == "actuatable"


def test_require_signal_raises_when_grid_fully_clamped_into_baseline():
    # A fully-clamped scale grid (eps* far exceeds 1.5*input_scale) rounds every
    # steered row's frac into the frac=0 baseline bucket. Reporting "no-crossing"
    # off a table with no non-baseline bucket would be a contaminated baseline
    # silently read as "underpowered" rather than "broken sweep".
    table = {0.0: {"n": 40, "g_read": 1.0, "frac_refused": 0.1}}
    with pytest.raises(SystemExit, match="clamp"):
        require_signal(table)


def test_mean_arm_rows_filters_direction_and_drops_unmatched_scale(capsys):
    judged = [
        {"direction": "jtw_mean_diff_tgt", "scale": "0.0", "refused": "0"},
        {"direction": "jtw_mean_diff_tgt", "scale": "5.0", "refused": "1"},
        {"direction": "jtw_other", "scale": "5.0", "refused": "1"},
    ]
    readout = [
        {"direction": "jtw_mean_diff_tgt", "scale": "0.0", "g_read": "3.0"},
        # no readout entry at scale=5.0 -- that judged row must be dropped, not
        # silently coerced into some other bucket.
    ]
    summ = {"directions": {"mean_diff_tgt": {"median_eps_star": 2.5}}}
    out, dropped = mean_arm_rows(judged, readout, summ, direction="jtw_mean_diff_tgt")
    assert out == [{"frac": 0.0, "g_read": 3.0, "refused": 0.0}]
    assert dropped == 1
    captured = capsys.readouterr()
    assert "dropped 1" in captured.out


def test_readout_crossed_requires_baseline_sign_change_not_membership():
    # Baseline g_read is ALREADY inside the halfspace (-1.5 <= 0): a "crossing" here
    # is vacuous -- the readout never had to move. Reachable by construction: eps*
    # is the dataset-wide median_eps_star (reach_steer.py:123) while the baseline
    # g_read is measured on a DIFFERENT population (FACTUAL_PROMPTS / the refusal
    # holdout, reach_steer.py:111,133) that t02 was never fit on, so nothing
    # guarantees that population's unsteered readout sits outside the boundary.
    t = {0.0: {"g_read": -1.5, "frac_refused": 0.05},
         -1.0: {"g_read": -4.0, "frac_refused": 0.55}}
    c = readout_crossed(t)
    assert c == {0.0: False, -1.0: False}
    assert verdict(c, behavior_delta(t)) != "actuatable"


def test_readout_crossed_detects_genuine_sign_change_baseline_never_flagged():
    # A real crossing occurs at a NEGATIVE frac: eps* is positive only where g>0
    # (reach_steer.py:167), so crossings occur at negative fracs; a positive-frac
    # "crossing" would mean the linearization sign is wrong or the readout is
    # non-monotone (see the positive-frac warning in run()). The baseline frac
    # itself must never be reported as crossed.
    t = {0.0: {"g_read": 3.0, "frac_refused": 0.0},
         -1.0: {"g_read": -1.0, "frac_refused": 0.4}}
    c = readout_crossed(t)
    assert c[0.0] is False
    assert c[-1.0] is True


def test_verdict_refuses_a_baseline_only_table():
    with pytest.raises(SystemExit, match="baseline"):
        verdict({0.0: False}, {0.0: 0.0})


def test_named_errors_instead_of_bare_exceptions():
    # frac_of on an unparseable eps_star must not leak a bare ValueError.
    with pytest.raises(SystemExit, match="frac_of"):
        frac_of(5.0, "")
    # _read_json on a missing file must not leak a bare FileNotFoundError (or a
    # leaked file handle from a raw json.load(open(...))).
    with pytest.raises(SystemExit, match="missing"):
        reach_control._read_json("/nonexistent/path/reach_summary_doesnotexist.json")


def test_per_stmt_never_steered_rows_do_not_contaminate_frac0_baseline():
    # Reproduces the reviewer's exact scenario: 100 statements with textbook
    # crossings (g: +2 -> -3 at frac -1, refusal 0 -> 1.0) plus 40 NEVER-STEERED
    # statements (eps_star=0, only a lone baseline row at g=-6, no treatment row).
    # frac_of(0.0, 0.0)==0.0 lands those 40 rows in the SAME frac=0 bucket as the
    # 100 genuine baselines, biasing bucket 0 toward g<=0 while the treatment
    # bucket stays g>0-only -- this flips a textbook actuatable result into
    # "inert", the worst possible failure direction for a positive control.
    judged, meta = [], []
    for i in range(100):
        j, m = _stmt_rows(i, eps_star=3.0, baseline_g=2.0, treated_g=-3.0,
                          refused_treated=1.0)
        judged += j; meta += m
    for i in range(100, 140):
        j, m = _stmt_rows(i, eps_star=0.0, baseline_g=-6.0)
        judged += j; meta += m

    rows, n_never_steered = align_stmt_rows(judged, meta)
    assert n_never_steered == 40
    table = aggregate(rows)
    require_signal(table)
    assert table[0.0]["g_read"] > 0.0    # baseline no longer contaminated
    crossed, delta = readout_crossed(table), behavior_delta(table)
    assert verdict(crossed, delta) == "actuatable"


def test_align_stmt_rows_carries_stmt_index_and_per_statement_crossing_fraction():
    # stmt 7 individually crosses (base g=+2 -> g=-1 at frac -1); stmt 9 does NOT
    # (base g=+5 -> stays positive, g=+1, at frac -1). The bucket MEAN at frac -1
    # is (-1+1)/2 = 0.0 <= 0 -- readout_crossed would call the whole bucket
    # "crossed" even though only ONE of the two statements individually did.
    # stmt_crossing_fracs is strictly more informative: it must report 0.5, not a
    # binary "crossed".
    judged = [_j(0.0, 0), _j(-2.0, 1),
              _j(0.0, 0), _j(-5.0, 0)]
    meta = [_m(0.0, 2.0, 2.0, stmt_index=7), _m(-2.0, 2.0, -1.0, stmt_index=7),
            _m(0.0, 5.0, 5.0, stmt_index=9), _m(-5.0, 5.0, 1.0, stmt_index=9)]
    rows, n_never_steered = align_stmt_rows(judged, meta)
    assert n_never_steered == 0
    assert {r["stmt_index"] for r in rows} == {7, 9}

    table = aggregate(rows)
    assert table[-1.0]["g_read"] == 0.0            # the bucket mean says "crossed"
    assert readout_crossed(table)[-1.0] is True    # (bucket-level view)

    frac_map = stmt_crossing_fracs(rows)
    assert frac_map[-1.0] == 0.5                   # per-statement view: only half did


def test_near_boundary_baseline_warns_but_still_returns_actuatable():
    # Baseline g0 is tiny but still positive (1e-9); the SMALLEST swept |frac|
    # (-1) already crosses, so the certificate's magnitude claim (~eps* of
    # perturbation needed) is untested for this population. This must WARN via
    # the returned flag, not abort -- the verdict computed from the table is
    # unaffected.
    table = {0.0: {"g_read": 1e-9, "frac_refused": 0.0},
             -1.0: {"g_read": -4.0, "frac_refused": 0.6},
             1.0: {"g_read": 9.0, "frac_refused": 0.0}}
    diag = near_boundary_check(table)
    assert diag["near_boundary"] is True
    assert diag["baseline_g_read"] == 1e-9
    assert diag["g_read_swing_at_pm1"] == {"+1": 9.0, "-1": -4.0}

    crossed, delta = readout_crossed(table), behavior_delta(table)
    assert verdict(crossed, delta) == "actuatable"


def test_count_empty_completions_handles_none_value_without_raising():
    # csv.DictReader yields the literal None (not a missing key) for a short row;
    # `.get("completion", "")` alone still returns None in that case, and
    # None.strip() raises AttributeError. Must not crash, and must filter by
    # direction like every other sidecar field.
    rows = [
        {"direction": "jtw_mean_diff_tgt", "completion": None},
        {"direction": "jtw_mean_diff_tgt", "completion": "  "},
        {"direction": "jtw_mean_diff_tgt", "completion": "ok"},
        {"direction": "jtw_other", "completion": None},
    ]
    assert count_empty_completions(rows, "jtw_mean_diff_tgt") == 2


def test_an_n_of_1_bucket_cannot_decide_actuatable():
    """I5: `actuatable` is the branch's headline number ("the instrument is valid") and
    must not rest on a single observation.

    Reachable by construction: scale_grid's `min(f*eps_i, 1.5*input_scale)` clamp fires
    per statement, so a clamped subset gets idiosyncratic frac = cap/eps_i values that
    round into their own n=1 buckets while the unclamped statements pool at +/-1, +/-2.
    require_signal only fires when there is NO non-baseline bucket, so it does not
    catch this."""
    from reach_control import MIN_BUCKET_N
    assert MIN_BUCKET_N > 1
    counts = {0.0: 40, -1.37: 1}
    crossed = {0.0: False, -1.37: True}
    delta = {0.0: 0.0, -1.37: 1.0}
    # without bucket sizes this is the legacy behaviour (unchanged for callers that
    # have no counts to give)
    assert verdict(crossed, delta) == "actuatable"
    # with them, a single observation must not buy the headline verdict
    assert verdict(crossed, delta, counts=counts) == "readout-only"


def test_a_full_size_bucket_still_decides_actuatable():
    from reach_control import MIN_BUCKET_N
    counts = {0.0: 40, -1.0: MIN_BUCKET_N}
    crossed = {0.0: False, -1.0: True}
    delta = {0.0: 0.0, -1.0: 1.0}
    assert verdict(crossed, delta, counts=counts) == "actuatable"


def test_undersized_buckets_are_named_not_dropped():
    from reach_control import undersized_fracs
    table = {0.0: {"n": 32}, -1.0: {"n": 32}, -1.37: {"n": 1}, 2.0: {"n": 2}}
    assert undersized_fracs(table) == [-1.37, 2.0]
    # the baseline bucket is never reported as undersized-for-adjudication
    assert 0.0 not in undersized_fracs({0.0: {"n": 1}, -1.0: {"n": 32}})


def test_aggregate_records_the_refusal_count_alongside_the_rate():
    rows = [{"frac": 1.0, "g_read": -1.0, "refused": 1.0},
            {"frac": 1.0, "g_read": -3.0, "refused": 0.0},
            {"frac": 1.0, "g_read": -2.0, "refused": 1.0},
            {"frac": 0.0, "g_read": 5.0, "refused": 0.0}]
    t = aggregate(rows)
    assert t[1.0]["n"] == 3 and t[1.0]["n_refused"] == 2
    assert abs(t[1.0]["frac_refused"] - 2 / 3) < 1e-12
    assert t[0.0]["n_refused"] == 0


def test_wilson_interval_on_a_known_case():
    """I6: MIN_DELTA is a hard threshold on a point estimate; the mean arm has n=32, so
    4 of 32 prompts crosses it. The sidecar must carry an interval.

    Wilson 95% for k=2, n=10 is (0.056682, 0.509838) at z=1.959964 — cross-checked
    against an independent scipy.stats.norm-based reference implementation."""
    from reach_control import wilson_interval
    lo, hi = wilson_interval(2, 10)
    assert abs(lo - 0.056682) < 1e-5 and abs(hi - 0.509838) < 1e-5
    # degenerate cases
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.25
    lo1, hi1 = wilson_interval(20, 20)
    assert hi1 == 1.0 and 0.75 < lo1 < 1.0
    # 4 of 32 -- the exact scenario the review flags
    lo4, hi4 = wilson_interval(4, 32)
    assert lo4 < 0.125 < hi4 and hi4 > 0.28


def test_mean_arm_rows_distinguishes_a_missing_median_eps_star_from_a_zero_one():
    # M13: `if not eps` reported "no median_eps_star" for a legitimate 0.0, sending the
    # operator to hunt the summary file instead of the real condition (every label-1 row
    # already sits inside the halfspace).
    judged = [{"direction": "jtw_mean_diff_tgt", "scale": "0.0", "refused": "0"}]
    readout = [{"direction": "jtw_mean_diff_tgt", "scale": "0.0", "g_read": "3.0"}]
    with pytest.raises(SystemExit, match="no median_eps_star"):
        mean_arm_rows(judged, readout, {"directions": {}}, "jtw_mean_diff_tgt")
    with pytest.raises(SystemExit, match="is 0"):
        mean_arm_rows(judged, readout,
                      {"directions": {"mean_diff_tgt": {"median_eps_star": 0.0}}},
                      "jtw_mean_diff_tgt")


def test_run_sidecar_carries_counts_intervals_and_undersized_buckets(tmp_path,
                                                                    monkeypatch):
    """End-to-end over the mean arm: the sidecar must carry the exact k/n, a Wilson
    interval, and the undersized-bucket accounting — and an n=1 crossing+moved bucket
    must not be enough to return `actuatable`."""
    import csv as _csv
    import json as _json
    monkeypatch.chdir(tmp_path)
    ds, arm = "sentinel_control", "mean"
    eps = 2.0
    # frac 0 (baseline, n=8, never refuses), frac -1 (n=8, crosses, all refuse),
    # frac -1.37 (n=1, a clamp artefact that also crosses and refuses)
    grid = [(0.0, 8, 3.0, 0), (-2.0, 8, -4.0, 1), (-2.74, 1, -5.0, 1)]
    judged = [("direction", "scale", "prompt", "completion", "refused")]
    readout = [("direction", "scale", "prompt", "g_read")]
    for scale, n, g, ref in grid:
        for i in range(n):
            judged.append(("jtw_mean_diff_tgt", scale, f"p{i}", "I cannot", ref))
            readout.append(("jtw_mean_diff_tgt", scale, f"p{i}", g))
    with open(f"judge_refusal_{ds}_{arm}.csv", "w", newline="") as f:
        _csv.writer(f).writerows(judged)
    with open(f"reach_steer_readout_{ds}.csv", "w", newline="") as f:
        _csv.writer(f).writerows(readout)
    with open(f"reach_summary_{ds}.json", "w") as f:
        _json.dump({"directions": {"mean_diff_tgt": {"median_eps_star": eps}}}, f)

    v = reach_control.run(ds, arm)
    side = _json.load(open(f"reach_control_{ds}_{arm}.json"))
    assert v == "actuatable"                       # the n=8 bucket at -1 earns it
    assert side["min_bucket_n"] == reach_control.MIN_BUCKET_N
    assert side["undersized_fracs"] == [-1.37]
    t = side["table"]
    assert t["-1"]["n"] == 8 and t["-1"]["n_refused"] == 8
    assert t["-1"]["refused_ci95_lo"] > 0.6 and t["-1"]["refused_ci95_hi"] == 1.0
    assert t["0"]["n_refused"] == 0
    # the undersized bucket is REPORTED, not dropped
    assert t["-1.37"]["n"] == 1 and t["-1.37"]["too_small_to_adjudicate"] == 1
    assert t["-1"]["too_small_to_adjudicate"] == 0
    header = next(iter(_csv.reader(open(f"reach_control_{ds}_{arm}.csv"))))
    for col in ("n_refused", "refused_ci95_lo", "refused_ci95_hi",
                "too_small_to_adjudicate"):
        assert col in header and col in reach_control.COLUMN_SEMANTICS


def test_run_will_not_call_a_lone_undersized_crossing_actuatable(tmp_path, monkeypatch):
    import csv as _csv
    import json as _json
    monkeypatch.chdir(tmp_path)
    ds, arm = "sentinel_control_small", "mean"
    # ONLY an n=1 treatment bucket, which crosses and refuses. Before MIN_BUCKET_N this
    # returned `actuatable` -- "the instrument is valid" -- off one observation.
    grid = [(0.0, 8, 3.0, 0), (-2.74, 1, -5.0, 1)]
    judged = [("direction", "scale", "prompt", "completion", "refused")]
    readout = [("direction", "scale", "prompt", "g_read")]
    for scale, n, g, ref in grid:
        for i in range(n):
            judged.append(("jtw_mean_diff_tgt", scale, f"p{i}", "I cannot", ref))
            readout.append(("jtw_mean_diff_tgt", scale, f"p{i}", g))
    with open(f"judge_refusal_{ds}_{arm}.csv", "w", newline="") as f:
        _csv.writer(f).writerows(judged)
    with open(f"reach_steer_readout_{ds}.csv", "w", newline="") as f:
        _csv.writer(f).writerows(readout)
    with open(f"reach_summary_{ds}.json", "w") as f:
        _json.dump({"directions": {"mean_diff_tgt": {"median_eps_star": 2.0}}}, f)
    assert reach_control.run(ds, arm) == "readout-only"
