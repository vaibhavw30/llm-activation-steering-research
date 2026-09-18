"""Tests for src/tqa_discovery.py: the pure pieces of J-B, and the CPU stages on fakes."""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tqa_discovery as td  # noqa: E402


def _unit_cols(M):
    return M / np.linalg.norm(M, axis=0, keepdims=True)


# ------------------------------------------------------------------ geometry
def test_potency_order_ranks_by_column_norm():
    U = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    assert td.potency_order(U) == [2, 1, 0]


def test_factor_alignment_finds_the_planted_factor_and_keeps_its_sign():
    rng = np.random.default_rng(0)
    d, k = 64, 16
    V = _unit_cols(rng.standard_normal((d, k)))
    t = -V[:, 5]
    R = _unit_cols(rng.standard_normal((d, k)))
    got = td.factor_alignment(V, t, R)
    assert got["best_factor"] == 5
    assert got["best_cos"] == pytest.approx(-1.0)
    assert got["ratio_vs_random"] > 1.5


def test_span_fraction_is_one_inside_and_near_k_over_d_for_random():
    rng = np.random.default_rng(1)
    d, k = 200, 40
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    assert td.span_fraction(Q, Q[:, 3]) == pytest.approx(1.0)
    fr = np.mean([td.span_fraction(Q, td.unit(rng.standard_normal(d))) for _ in range(200)])
    assert fr == pytest.approx(k / d, abs=0.03)


def test_principal_cosines_identical_subspaces_are_all_one():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((50, 8))
    B = A @ rng.standard_normal((8, 8))          # same span, different basis
    assert np.allclose(td.principal_cosines(A, B), 1.0)


def test_stability_same_fit_is_perfect_and_beats_random():
    rng = np.random.default_rng(3)
    V = rng.standard_normal((64, 16))
    U = rng.standard_normal((32, 16))
    rows = td.stability_rows(V, U, V, k_top=4)
    fac = [r for r in rows if r["kind"] == "factor"]
    assert len(fac) == 4 and all(r["value"] == pytest.approx(1.0) for r in fac)
    mean = next(r for r in rows if r["stat"] == "mean principal cosine")
    assert mean["value"] == pytest.approx(1.0) and mean["random"] < 0.9


def test_geometry_rows_compare_V_at_source_and_U_at_target():
    rng = np.random.default_rng(4)
    V, U = rng.standard_normal((32, 8)), rng.standard_normal((32, 8))
    src = {"mean_diff": td.unit(V[:, 2])}
    tgt = {"mean_diff": td.unit(U[:, 6])}
    rows = td.geometry_rows("s1", V, U, src, tgt)
    by = {r["space"]: r for r in rows}
    assert by["V@src"]["best_factor"] == 2 and by["U@tgt"]["best_factor"] == 6
    assert by["V@src"]["span_frac"] == pytest.approx(1.0)


# ------------------------------------------------------------------ screen
def _fake_df(n_q=300):
    rows = []
    for i in range(n_q):
        rows += [{"statement": f"Q: q{i}?\nA: yes", "label": 0, "question": f"q{i}?"},
                 {"statement": f"Q: q{i}?\nA: no", "label": 1, "question": f"q{i}?"}]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_dct_fit_questions_matches_run_dct_data_sampling():
    df = _fake_df()
    got = td.dct_fit_questions(df, 64, seed=325)
    per = 32
    pos = df[df["label"] == 1].sample(per, random_state=325)
    neg = df[df["label"] == 0].sample(per, random_state=325)
    want = set(pd.concat([pos, neg]).sample(frac=1, random_state=325)["question"])
    assert got == want


def test_screen_questions_excludes_and_is_deterministic():
    qs = [f"q{i}" for i in range(200)]
    ex = {f"q{i}" for i in range(50)}
    a = td.screen_questions(qs, ex, n=96)
    assert a == td.screen_questions(qs, ex, n=96)
    assert len(set(a)) == 96 and not set(a) & ex


def test_screen_questions_refuses_a_short_pool():
    with pytest.raises(ValueError):
        td.screen_questions(["a", "b"], set(), n=3)


def test_split_halves_are_disjoint_and_cover():
    qs = [f"q{i}" for i in range(96)]
    a, b = td.split_halves(qs)
    assert len(a) == len(b) == 48 and not a & b and a | b == set(qs)


def test_screen_directions_order_signs_and_count():
    rng = np.random.default_rng(5)
    V = rng.standard_normal((16, 10))
    U = rng.standard_normal((16, 10)) * np.arange(1, 11)
    q2 = rng.standard_normal(16)
    dirs = td.screen_directions(V, U, q2, k=3, n_rand=2)
    names = [n for n, _ in dirs]
    top = td.potency_order(U)[:3]
    assert names[:2] == ["baseline", "q2_mean_diff"]
    assert names[2:8] == [f"dct_{i}_{s}" for i in top for s in ("pos", "neg")]
    assert names[-2:] == ["rand_0", "rand_1"]
    vec = dict(dirs)
    assert vec["baseline"] is None
    assert np.allclose(vec[f"dct_{top[0]}_neg"], -vec[f"dct_{top[0]}_pos"])
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for n, v in dirs if v is not None)


def test_paired_gains_pairs_by_question():
    rows = [{"direction": "baseline", "question": "a", "t": "0"},
            {"direction": "baseline", "question": "b", "t": "1"},
            {"direction": "x", "question": "a", "t": "1"},
            {"direction": "x", "question": "b", "t": "1"},
            {"direction": "y", "question": "a", "t": "0"}]
    g = td.paired_gains(rows, "t", {"a", "b"})
    assert g == {"x": 0.5, "y": 0.0}
    assert td.paired_gains(rows, "t", {"b"}) == {"x": 0.0}


def test_select_sbeh_needs_top_on_both_halves_and_positive():
    cands = ["d1", "d2", "d3", "d4", "d5"]
    ga = {"d1": .3, "d2": .2, "d3": .1, "d4": 0., "d5": -.1}
    gb = {"d1": .0, "d2": .25, "d3": .2, "d4": .3, "d5": -.1}
    win, both = td.select_sbeh(ga, gb, cands, top=3)
    assert set(both) == {"d2", "d3"} and win == "d2"
    # d1 is top on A only; d4 on B only: neither may win
    win, both = td.select_sbeh({"d1": .3, "d4": 0.}, {"d1": 0., "d4": .3}, ["d1", "d4"], top=1)
    assert win is None and both == []


def test_select_sbeh_refuses_a_nonpositive_gain():
    win, _ = td.select_sbeh({"d1": 0.0}, {"d1": 0.0}, ["d1"], top=1)
    assert win is None


def test_parse_dct_name():
    assert td.parse_dct_name("dct_17_neg") == (17, -1)
    assert td.parse_dct_name("dct_3_pos") == (3, 1)


def test_info_judge_passes_uses_the_registered_bar():
    rows = [{"kind": k, "informative": v, "expected_informative": e}
            for k, v, e in (("correct", 1, 1), ("no_comment", 0, 0), ("restated", 0, 0))]
    assert td.info_judge_passes(rows)
    rows[1]["informative"] = 1
    assert not td.info_judge_passes(rows)


# ------------------------------------------------------------------ G0
def test_mag_g0_row_dead_channel_fails():
    truth = np.array([1, 0] * 50)
    r = td.mag_g0_row("true", np.ones(100, int), truth)
    assert r["share_no"] == 0.0 and r["agree"] == 0.5 and r["passes"] == 0
    assert r["balanced_acc"] == 0.5


def test_mag_g0_row_live_channel_passes():
    truth = np.array([1, 0] * 50)
    ym = truth.copy()
    ym[:10] = 1 - ym[:10]
    r = td.mag_g0_row("true", ym, truth)
    assert r["passes"] == 1 and r["agree"] == pytest.approx(0.9)


def test_choose_phrasing_prefers_passing_then_agreement():
    rows = [{"phrasing": "true", "passes": "0", "agree": "0.9"},
            {"phrasing": "correct", "passes": "1", "agree": "0.7"}]
    assert td.choose_phrasing(rows) == "correct"
    assert td.choose_phrasing([{"phrasing": "correct", "passes": "0", "agree": ".9"}]) == "true"


def test_mag_phrasings_end_in_the_same_suffix_and_differ():
    pre = [p for p, _ in td.MAG_PHRASINGS.values()]
    assert len(set(pre)) == len(pre) == 2


# ------------------------------------------------------------------ sign conventions
def test_truthful_sign_flips_truthfulqa_only():
    assert td.TRUTHFUL_SIGN["truthfulqa"] == -1
    assert td.TRUTHFUL_SIGN["cities"] == 1
    assert td.Q2_TRUTHFUL_SIGN == -1 and td.MAG_TRUTHFUL_SIGN == 1


def test_truth_dirs_signs_truthfulqa_toward_truthful(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    md = np.array([3.0, 0.0, 4.0])
    np.savez("truth_dir_truthfulqa.npz", mean_diff=md, grad=md, layer=11)
    np.savez("truth_dir_cities.npz", mean_diff=md, grad=md, layer=11)
    assert np.allclose(td.truth_dirs("truthfulqa")["mean_diff"], -md / 5)
    assert np.allclose(td.truth_dirs("cities")["mean_diff"], md / 5)


def test_mag_u_q_points_at_true_through_the_input_delta_operator():
    """MAG_TRUTHFUL_SIGN rests on BOTH signs: class_mean_diff is (label 0) - (label 1), and
    the InputDelta operator is A_Qp - A_p. With the truth signal in A_p (the bare
    statement), the two minus signs cancel and u_Q points at label 1, true."""
    from mag.directions import build_directions
    n, d = 8, 3
    y = np.array([1, 0] * (n // 2))
    A_p = np.zeros((1, n, d))
    A_p[0, y == 1, 0] = 1.0                           # true statements sit at +e0
    A_Qp = np.zeros((1, n, d))                        # the verdict context carries no label
    cache = {"A_p": A_p, "A_Qp": A_Qp, "A_Qpv": A_Qp, "A_verdict": A_Qp, "A_EQp": A_Qp,
             "A_Q": np.zeros((1, d)), "A_empty": np.zeros((1, d))}
    u = build_directions(cache, 0, y, y)["u_Q_gold"]
    assert td.MAG_TRUTHFUL_SIGN * u[0] > 0


def test_cosine_rows_every_pair_once():
    dirs = {"a": np.array([1.0, 0]), "b": np.array([0, 2.0]), "c": np.array([1.0, 1.0])}
    rows = td.cosine_rows(dirs)
    assert [(r["a"], r["b"]) for r in rows] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert rows[0]["cos"] == pytest.approx(0.0)
    assert rows[1]["cos"] == pytest.approx(np.sqrt(0.5))


# ------------------------------------------------------------------ CPU stages on fakes
@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Two tiny DCT fits, truth dirs, and a judged screen, in a scratch cwd."""
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(td, "PREFIX", "")
    rng = np.random.default_rng(6)
    d, k = 24, 20
    V = rng.standard_normal((d, k))
    U = rng.standard_normal((d, k)) * np.linspace(1, 3, k)
    os.makedirs("dct_s2_truthfulqa")
    for where in (".", "dct_s2_truthfulqa"):
        torch.save(torch.tensor(V, dtype=torch.float32), f"{where}/dct_V_truthfulqa.pt")
        torch.save(torch.tensor(U, dtype=torch.float32), f"{where}/dct_U_truthfulqa.pt")
    md = rng.standard_normal(d)
    for name in ("truth_dir_truthfulqa.npz", "truth_dir_tgt_truthfulqa.npz"):
        np.savez(name, mean_diff=md, grad=md, layer=11)
    return V, U, md


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_stage_geometry_writes_both_files_and_refuses_rerun(fake_repo):
    td.stage_geometry()
    geo = pd.read_csv("dct_geometry_truthfulqa.csv")
    assert set(geo["fit"]) == {"s1", "s2"} and set(geo["space"]) == {"V@src", "U@tgt"}
    st = pd.read_csv("dct_stability_truthfulqa.csv")
    assert (st[st["kind"] == "factor"]["value"] > 0.999).all()      # identical fits
    with pytest.raises(SystemExit):
        td.stage_geometry()


def _screen(V, U, winner, gain_rows=None):
    """A judged screen where `winner` (a dct_* name) gains on both halves."""
    names = [n for n, _ in td.screen_directions(V, U, np.ones(V.shape[0]), k=4, n_rand=2)]
    qs = [(f"q{i}", "A" if i % 2 else "B") for i in range(8)]
    rows = []
    for n in names:
        for q, h in qs:
            t = 1 if (n in (winner, "q2_mean_diff") or (n != "baseline" and q == "q0")) else 0
            rows.append({"direction": n, "question": q, "half": h, "truthful": t,
                         "truthful_and_informative": t})
    return rows


def test_stage_select_picks_the_planted_winner_and_flags_stability(fake_repo, monkeypatch):
    V, U, md = fake_repo
    top = td.potency_order(U)[1]
    monkeypatch.setattr(td, "K_SCREEN", 4)
    _write("dct_screen_judged_truthfulqa.csv", _screen(V, U, f"dct_{top}_neg"))
    td.stage_select()
    sel = json.load(open("dct_selection_truthfulqa.json"))
    assert sel["score_col"] == "truthful"                 # no gold file: truthful-only
    assert sel["sbeh_status"] == "winner"
    b = sel["rules"]["S-beh"][0]
    assert (b["factor"], b["sign"]) == (top, -1)
    assert b["cross_seed_cos"] == pytest.approx(1.0, abs=1e-5) and b["seed_specific"] is False
    assert b["beats_every_random"] is True
    assert [r["factor"] for r in sel["rules"]["S-none"]] == td.potency_order(U)[:td.K_SNONE]
    g = sel["rules"]["S-geo"][0]
    c = td.unit(V).T @ (-td.unit(md))
    assert g["factor"] == int(np.argmax(np.abs(c))) and g["sign"] == int(np.sign(c[g["factor"]]))
    assert sel["positive_control_gain"] > 0


def test_stage_select_no_stable_winner(fake_repo, monkeypatch):
    V, U, _ = fake_repo
    monkeypatch.setattr(td, "K_SCREEN", 4)
    rows = _screen(V, U, winner="none")
    _write("dct_screen_judged_truthfulqa.csv", rows)
    td.stage_select()
    sel = json.load(open("dct_selection_truthfulqa.json"))
    assert sel["sbeh_status"] == "no stable winner" and sel["rules"]["S-beh"] == []


def test_stage_select_uses_t_and_i_when_the_info_judge_passed(fake_repo, monkeypatch):
    V, U, _ = fake_repo
    monkeypatch.setattr(td, "K_SCREEN", 4)
    _write("dct_screen_judged_truthfulqa.csv", _screen(V, U, "none"))
    _write(td.GOLD, [{"kind": "correct", "informative": 1, "expected_informative": 1},
                     {"kind": "no_comment", "informative": 0, "expected_informative": 0}])
    td.stage_select()
    assert json.load(open("dct_selection_truthfulqa.json"))["score_col"] == \
        "truthful_and_informative"


def test_main_refuses_limit_without_prefix():
    with pytest.raises(SystemExit):
        td.main(["--stage", "geometry", "--limit", "4"])


def test_main_skips_a_finished_stage(fake_repo, capsys):
    open("dct_geometry_truthfulqa.csv", "w").write("x\n1\n")
    td.main(["--stage", "geometry"])
    assert "stage done, skipping" in capsys.readouterr().out


def test_prefix_reaches_every_output(fake_repo, monkeypatch):
    monkeypatch.setattr(td, "PREFIX", "smoke_")
    td.stage_geometry()
    assert os.path.exists("smoke_dct_geometry_truthfulqa.csv")
    assert not os.path.exists("dct_geometry_truthfulqa.csv")
    assert td.mag_dir_path() == os.path.join(td.WORK, "smoke_mag_dir_truthfulqa.npz")
