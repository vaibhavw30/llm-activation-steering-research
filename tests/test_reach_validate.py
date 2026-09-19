"""Tests for reach_validate, the V1 closed check.

Everything here is synthetic. The point of V1 is that a number produced by the GPU
stage is only meaningful if the arithmetic around it is right, so these tests pin the
arithmetic: the certificate solves the equation it claims to solve, a ratio of 1 means
what it says, and a measurement that could not be made reads as missing rather than as
a failure or a pass.
"""
import numpy as np
import pytest

import reach_validate as rv


# --- certificate: the minimum-norm perturbation that lands the readout on zero ---

def test_certificate_moves_the_readout_by_exactly_minus_g():
    """The defining property. Every ratio in this module is measured against the
    promise -g, so a certificate that does not first-order deliver -g would make
    ratio_A something other than 1 for a reason that has nothing to do with the
    model."""
    jtw = np.array([[3.0, 4.0], [0.0, -2.0], [1.0, 1.0]])
    g = np.array([10.0, -6.0, 0.5])
    _, delta, _ = rv.certificate(g, jtw)
    assert np.allclose(np.sum(jtw * delta, axis=1), -g)


def test_certificate_norm_is_eps_star_and_the_margin_is_the_pullback_norm():
    jtw = np.array([[3.0, 4.0], [0.0, -2.0]])
    g = np.array([10.0, -6.0])
    eps, delta, m = rv.certificate(g, jtw)
    assert np.allclose(m, [5.0, 2.0])
    assert np.allclose(eps, [2.0, 3.0])
    assert np.allclose(np.linalg.norm(delta, axis=1), eps)


def test_certificate_pushes_against_the_sign_of_g():
    """A statement on the far side of the boundary has to be pushed the other way.
    The caller never picks a direction, so the sign has to live in here."""
    jtw = np.array([[1.0, 0.0], [1.0, 0.0]])
    _, delta, _ = rv.certificate(np.array([4.0, -4.0]), jtw)
    assert delta[0, 0] < 0 < delta[1, 0]
    assert np.allclose(delta[0], -delta[1])


def test_certificate_broadcasts_over_layers():
    """compute() calls this once per context with g as (n, 1) and the pullback as
    (n, L, d), so the per-layer certificate has to fall out of one call."""
    rng = np.random.default_rng(0)
    jtw = rng.normal(size=(5, 4, 3))
    g = rng.normal(size=5)
    eps, delta, m = rv.certificate(g[:, None], jtw)
    assert eps.shape == m.shape == (5, 4) and delta.shape == (5, 4, 3)
    assert np.allclose(np.sum(jtw * delta, axis=-1), -g[:, None])


def test_certificate_survives_a_dead_layer_without_dividing_by_zero():
    """||J^T a|| == 0 means the layer cannot move the readout at all. That is a real
    outcome on late layers, not an error, and it must not produce a nan that then
    poisons the median of every other row."""
    eps, delta, m = rv.certificate(np.array([1.0]), np.zeros((1, 3)))
    assert m[0] == 0.0
    assert np.all(np.isfinite(delta))


# --- realized_ratio: what actually happened over what was promised ---

def test_realized_ratio_is_one_when_the_promise_is_kept():
    g0 = np.array([2.0, -3.0])
    assert np.allclose(rv.realized_ratio(g0, np.zeros(2), -g0), [1.0, 1.0])


def test_realized_ratio_is_the_fraction_of_the_promised_move():
    assert np.allclose(rv.realized_ratio(np.array([10.0]), np.array([9.3]),
                                         np.array([-10.0])), [0.07])


def test_a_statement_already_on_the_boundary_is_nan_not_zero():
    """Predicting nothing and delivering nothing is not a failure to steer. Scoring
    it as 0.0 would drag the median toward a collapse that did not happen."""
    r = rv.realized_ratio(np.array([0.0, 4.0]), np.array([0.0, 0.0]),
                          np.array([0.0, -4.0]))
    assert np.isnan(r[0]) and r[1] == 1.0


# --- the behavioural closure: is the readout really the decision? ---

def test_decision_sign_is_yes_minus_no():
    assert list(rv.decision_sign([1.0, 0.0], [0.0, 1.0])) == [1.0, -1.0]


def test_dissociation_counts_the_off_diagonal_cells():
    """The cells this design exists to leave empty. A crossing that does not flip the
    argmax, or a flip with no crossing, is the readout coming apart from behaviour,
    which is exactly what an output-layer contrastive readout is supposed to make
    impossible."""
    g0 = np.array([1.0, 1.0, 1.0, 1.0])
    g1 = np.array([-1.0, -1.0, 1.0, 1.0])       # rows 0, 1 crossed
    d0 = np.array([1.0, 1.0, 1.0, 1.0])
    d1 = np.array([-1.0, 1.0, -1.0, 1.0])       # rows 0, 2 flipped
    assert rv.dissociation(g0, g1, d0, d1) == {
        "crossed_and_flipped": 1, "crossed_not_flipped": 1,
        "flipped_not_crossed": 1, "neither": 1, "n": 4}


# --- summarize ---

def test_summarize_drops_the_rows_that_could_not_be_measured():
    s = rv.summarize([1.0, np.nan, 3.0, np.inf])
    assert s["n"] == 2 and s["median"] == 2.0


def test_summarize_of_nothing_measurable_is_empty_not_zero():
    s = rv.summarize([np.nan, np.nan])
    assert s["n"] == 0 and np.isnan(s["median"])


# --- analyze: the gate that decides whether the other columns may be read ---

def _fake_npz(path, n=4, layers=2, g_decl=None, after_A=None):
    """A minimal reach_validate_<ds>.npz. Defaults describe a perfect run: every
    steered readout lands exactly on zero, so every ratio is 1."""
    g = np.array([2.0, -3.0, 4.0, -1.0])[:n] if g_decl is None else np.asarray(g_decl)
    zeros = np.zeros((layers, n))
    np.savez(path, stmt_index=np.arange(n), layers=np.arange(layers),
             readout=np.zeros(3), g_decl=g, g_stem=g, g_quest=g,
             m_decl=np.ones((n, layers)), m_stem=np.ones((n, layers)),
             m_quest=np.ones((n, layers)), eps_decl=np.ones((n, layers)),
             eps_stem=np.ones((n, layers)), eps_quest=np.ones((n, layers)),
             after_A=zeros if after_A is None else np.asarray(after_A),
             after_B=zeros, after_B_own=zeros, after_Q_own=zeros,
             after_Q_yes=np.ones((layers, n)), after_Q_no=np.zeros((layers, n)),
             q_yes0=np.zeros(n), q_no0=np.ones(n))


def test_analyze_reports_the_arithmetic_as_sound_when_every_ratio_is_one(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _fake_npz(tmp_path / "reach_validate_fake.npz")
    rv.analyze("fake")
    out = capsys.readouterr().out
    assert "pullback arithmetic is sound" in out
    assert "!!!!" not in out


def test_analyze_refuses_to_pass_a_layer_whose_ratio_could_not_be_measured(
        tmp_path, monkeypatch, capsys):
    """Every statement at layer 0 sits on the boundary, so ratio_A there is nan. A
    naive `max(...) > tol` is False on nan and would print the reassuring line about
    a check that never ran."""
    monkeypatch.chdir(tmp_path)
    _fake_npz(tmp_path / "reach_validate_fake.npz",
              g_decl=np.zeros(4), after_A=np.zeros((2, 4)))
    rv.analyze("fake")
    out = capsys.readouterr().out
    assert "!!!!" in out
    assert "pullback arithmetic is sound" not in out


def test_analyze_writes_a_row_per_source_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_npz(tmp_path / "reach_validate_fake.npz", layers=3)
    rv.analyze("fake")
    import csv as _csv
    rows = list(_csv.DictReader(open(tmp_path / "reach_validate_summary_fake.csv")))
    assert [r["layer"] for r in rows] == ["0", "1", "2"]
    assert all(float(r["ratio_A"]) == 1.0 for r in rows)
