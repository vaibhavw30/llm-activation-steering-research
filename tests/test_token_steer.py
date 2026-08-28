"""E8, part 2: the actuator injects what we think, where we think.

The PI's claim 3 in its sharpest form. Every steering result the project has ever
produced depends on a hook adding the right vector at the right tensor, and that has
never been an automated test. It is now. A fake two-layer model stands in for gemma so
these run in a second with no weights.

The specific failure this guards against is Finding A: `Steerer._hook` adds a (d,)
vector to a (batch, seq, d) tensor, so torch broadcasts it to EVERY position including
BOS. That is not a bug to fix silently, it is a choice with a measurable cost, so the
tests below pin down both conventions and the energy ratio sqrt(T) between them.
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import token_steer as ts


class Recorder(nn.Module):
    """Records what it was called with and what it returned."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.seen_in, self.seen_out = None, None

    def forward(self, hidden_states, **kw):
        self.seen_in = hidden_states.detach().clone()
        out = hidden_states * 2.0
        self.seen_out = out.detach().clone()
        return out


class FakeInner(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([Recorder(d) for _ in range(n_layers)])
        self.norm = Recorder(d)


class FakeModel(nn.Module):
    def __init__(self, d=8, n_layers=3):
        super().__init__()
        self.model = FakeInner(d, n_layers)


def _run(model, x):
    h = x
    for lyr in model.model.layers:
        h = lyr(h)
    return model.model.norm(h)


# ------------------------------------------------------------------ site targeting

def test_postnorm_adds_after_the_norm_and_leaves_its_input_untouched():
    m = FakeModel()
    x = torch.zeros(1, 4, 8)
    v = torch.arange(8, dtype=torch.float32)
    with ts.SiteSteerer(m, "postnorm") as st:
        st.set(v)
        out = _run(m, x)
    assert torch.allclose(m.model.norm.seen_in, torch.zeros(1, 4, 8))
    assert torch.allclose(out, m.model.norm.seen_out + v)


def test_prenorm_adds_before_the_norm_so_the_norm_sees_the_perturbation():
    m = FakeModel()
    x = torch.zeros(1, 4, 8)
    v = torch.arange(8, dtype=torch.float32)
    with ts.SiteSteerer(m, "prenorm") as st:
        st.set(v)
        _run(m, x)
    assert torch.allclose(m.model.norm.seen_in, v.expand(1, 4, 8))


@pytest.mark.parametrize("L", [0, 1, 2])
def test_layer_site_fires_on_exactly_the_requested_layer(L):
    """An off-by-one here silently relabels every per-layer result."""
    m = FakeModel(n_layers=3)
    x = torch.zeros(1, 4, 8)
    v = torch.ones(8)
    with ts.SiteSteerer(m, f"layer:{L}") as st:
        st.set(v)
        _run(m, x)
    for k, lyr in enumerate(m.model.layers):
        got = float(lyr.seen_in.abs().max())
        if k < L:
            assert got == 0.0                       # untouched, upstream of the hook
        elif k == L:
            assert torch.allclose(lyr.seen_in, v.expand(1, 4, 8))
        else:
            assert got > 0.0                        # downstream, carries the effect


def test_unknown_site_is_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError):
        ts.SiteSteerer(FakeModel(), "layer13")


def test_the_hook_is_removed_on_exit():
    m = FakeModel()
    with ts.SiteSteerer(m, "postnorm") as st:
        st.set(torch.ones(8))
    out = _run(m, torch.zeros(1, 4, 8))
    assert torch.allclose(out, torch.zeros(1, 4, 8))


def test_a_none_vector_is_an_exact_no_op():
    m = FakeModel()
    x = torch.randn(1, 4, 8)
    clean = _run(FakeModel(), x)
    with ts.SiteSteerer(m, "postnorm") as st:
        st.set(None)
        out = _run(m, x)
    assert torch.allclose(out, clean)


# --------------------------------------------------------------- position semantics

def test_positions_last_perturbs_only_the_final_position():
    m = FakeModel()
    v = torch.ones(8)
    with ts.SiteSteerer(m, "prenorm", positions="last") as st:
        st.set(v)
        _run(m, torch.zeros(1, 5, 8))
    seen = m.model.norm.seen_in[0]
    assert float(seen[:-1].abs().max()) == 0.0
    assert torch.allclose(seen[-1], v)


def test_positions_all_perturbs_every_position_including_the_first():
    """gemma-2 uses BOS as a strong attention sink, so 'every position' includes the
    one token you least want to move. Pinned so the cost is never invisible again."""
    m = FakeModel()
    v = torch.ones(8)
    with ts.SiteSteerer(m, "prenorm", positions="all") as st:
        st.set(v)
        _run(m, torch.zeros(1, 5, 8))
    seen = m.model.norm.seen_in[0]
    assert torch.allclose(seen, v.expand(5, 8))
    assert float(seen[0].abs().max()) > 0.0          # BOS row is perturbed too


def test_broadcast_spends_sqrt_T_times_the_energy_of_a_single_position():
    """E8's arithmetic assertion: a broadcast of ||v|| over T positions has total
    perturbation norm ||v||*sqrt(T). Reporting it as a budget of ||v|| understates the
    spend by that factor, which is what every previous run did."""
    m = FakeModel()
    v = torch.randn(8)
    T = 7
    with ts.SiteSteerer(m, "prenorm", positions="all") as st:
        st.set(v)
        _run(m, torch.zeros(1, T, 8))
    total = float(m.model.norm.seen_in.norm())
    assert np.isclose(total, float(v.norm()) * np.sqrt(T), rtol=1e-5)


def test_single_token_decode_steps_are_perturbed_under_either_convention():
    """During generation the sequence length is 1. 'last' and 'all' must agree there,
    otherwise the two arms differ during prefill only and the comparison is confounded."""
    for pos in ("all", "last"):
        m = FakeModel()
        v = torch.ones(8)
        with ts.SiteSteerer(m, "prenorm", positions=pos) as st:
            st.set(v)
            _run(m, torch.zeros(1, 1, 8))
        assert torch.allclose(m.model.norm.seen_in[0, 0], v)


# ------------------------------------------------------------------- budget algebra

def test_required_scale_is_the_certified_budget_for_post_norm_directions():
    dc = np.array([1.5, 2.5, 3.5])
    assert ts.required_scale("oracle", 1, dc, None, None) == 2.5
    assert ts.required_scale("md_full", 2, dc, None, None) == 3.5


def test_required_scale_uses_the_per_layer_epsilon_for_the_jacobian_actuator():
    dc = np.array([1.5, 2.5])
    eps_all = np.array([[10.0, 11.0], [20.0, 21.0]])
    assert ts.required_scale("jtw_token", 1, dc, eps_all, 0) == 20.0
    assert ts.required_scale("jtw_token", 1, dc, eps_all, 1) == 21.0


def test_jtw_actuator_matches_the_injection_convention(tmp_path, monkeypatch):
    """The pullback and the budget must both follow --positions. `jtw` is the gradient
    summed over all positions (what a broadcast actuates); `jtw_last` is the gradient at
    the last position alone. Crossing them injects one convention while certifying the
    other, and they differ by the broadcast gain, up to 3.3x at the early layers."""
    monkeypatch.chdir(tmp_path)
    n, L, d = 4, 3, 6
    allv = np.tile(np.array([1.0, 0, 0, 0, 0, 0]), (n, L, 1))
    lastv = np.tile(np.array([0, 1.0, 0, 0, 0, 0]), (n, L, 1))
    np.savez("token_jac_toy.npz", jtw=allv, jtw_last=lastv,
             eps_all=np.full((n, L), 2.0), eps_last=np.full((n, L), 7.0))
    jac = np.load("token_jac_toy.npz")
    geom = {"delta_cone": np.ones(n)}

    D = ts.load_dirs("toy", ["jtw_token"], "layer:1", geom, jac, 1, positions="all")
    assert np.allclose(D["jtw_token"][0], [1, 0, 0, 0, 0, 0])

    D = ts.load_dirs("toy", ["jtw_token"], "layer:1", geom, jac, 1, positions="last")
    assert np.allclose(D["jtw_token"][0], [0, 1, 0, 0, 0, 0])


def test_jtw_at_one_position_refuses_to_run_without_the_matching_pullback():
    """Better a clear stop than a silent fall back to the broadcast actuator."""
    n, L, d = 2, 2, 4

    class OnlyBroadcast:
        files = ["jtw", "eps_all"]

        def __getitem__(self, k):
            return np.ones((n, L, d))

    with pytest.raises(SystemExit, match="jtw_last"):
        ts.load_dirs("toy", ["jtw_token"], "layer:1", {"delta_cone": np.ones(n)},
                     OnlyBroadcast(), 1, positions="last")


def test_legacy_positions_is_the_identity_when_no_subsample_cap_applied(tmp_path,
                                                                       monkeypatch):
    """`cities` runs the reach pipeline uncapped, so reach_margins row k is csv row k."""
    monkeypatch.chdir(tmp_path)
    np.savez("reach_acts_toy.npz", row_index=np.arange(10))
    got = ts.legacy_positions("toy", np.array([0, 3, 9]), 10)
    assert list(got) == [0, 3, 9]


def test_legacy_positions_follows_the_subsample_instead_of_assuming_identity(
        tmp_path, monkeypatch):
    """`common_claim` keeps a stratified 2000 of 4450, so reach_margins row k is csv row
    idx[k]. Indexing by the csv position pairs each statement with another row's
    actuator, which is silent and wrong wherever it lands in bounds."""
    monkeypatch.chdir(tmp_path)
    np.savez("reach_acts_toy.npz", row_index=np.array([5, 11, 40, 41]))
    got = ts.legacy_positions("toy", np.array([40, 5, 41, 11]), 4)
    assert list(got) == [2, 0, 3, 1]


def test_legacy_positions_reports_minus_one_for_rows_outside_the_subsample():
    """Absent is not zero. A statement the reach run never saw has no legacy actuator,
    and must be skipped rather than paired with row 0."""
    got = ts.legacy_positions("no_such_dataset", np.array([0, 1, 7]), 3)
    assert list(got) == [0, 1, -1]


def test_legacy_positions_ignores_reach_rows_beyond_the_margins_array():
    """reach_acts can hold more rows than reach_margins when the vjp stage was limited."""
    import tempfile, os as _os
    d = tempfile.mkdtemp()
    cwd = _os.getcwd()
    try:
        _os.chdir(d)
        np.savez("reach_acts_toy.npz", row_index=np.arange(10))
        got = ts.legacy_positions("toy", np.array([2, 8]), 5)
        assert list(got) == [2, -1]
    finally:
        _os.chdir(cwd)


def test_unit_rows_normalizes_each_row_and_survives_a_zero_row():
    v = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]])
    u = ts.unit_rows(v)
    assert np.isclose(np.linalg.norm(u[0]), 1.0)
    assert np.isclose(np.linalg.norm(u[2]), 1.0)
    assert np.all(np.isfinite(u))                    # no NaN from the zero row


# ------------------------------------------------------- the oracle verdict's scope

def test_the_oracle_is_a_harness_gate_only_at_the_postnorm_site():
    """A zero oracle at post-norm is a broken injection. A zero oracle at prenorm or a
    layer site is RMSNorm annihilating the radial component of a post-norm vector, which
    is the physics the arm was run to measure. The first run of this printed HARNESS
    BROKEN on a correct prenorm arm, which is a false alarm that costs a debugging
    session on working code."""
    assert "HARNESS OK" in ts.oracle_line("postnorm", 1.0)
    assert "HARNESS BROKEN" in ts.oracle_line("postnorm", 0.0)

    for site in ("prenorm", "layer:8", "layer:16"):
        line = ts.oracle_line(site, 0.0)
        assert "BROKEN" not in line
        assert "OK" not in line
        assert "NOT a harness check" in line


def test_the_verdict_still_reports_the_number_at_every_site():
    """Downgrading the label must not hide the measurement; the prenorm hit rate is how
    the norm layer's cost gets quantified."""
    for site in ("postnorm", "prenorm", "layer:8"):
        assert "0.375" in ts.oracle_line(site, 0.375)


def test_alpha_rescaling_is_what_turns_a_null_into_a_fair_test():
    """eps(u) = eps* / alpha. With alpha ~ 0.013 the fair budget is ~77x the certified
    one; sweeping the certified budget instead is the units error that produced the
    original behavioral null. This pins the arithmetic that fixes it."""
    delta_cone, alpha = 6.37, 0.0133
    assert np.isclose(delta_cone / alpha, 478.9, rtol=1e-3)


# ------------------------------------------------- probe()'s sign convention (E8, p3)
#
# probe() returns two floats that are one number with opposite signs. The third was
# logged as `readout_delta`, a name that reads as the legacy mean_diff_tgt readout
# (w . h) from reach_margins. It is not that. It is Delta(logit[j_tgt] - logit[j_top]),
# which equals frac_margin * |m0| — the same measurement in absolute units. An analysis
# was designed on the wrong reading (a readout-vs-behavior 2x2 these files cannot
# support) before the algebra was checked. These tests make the convention explicit so
# the next reader does not have to re-derive it, and so it cannot silently flip.

class FakeEnc(dict):
    """Tokenizer output: unpacks with ** and answers .to(dev)."""

    def to(self, dev):
        return self


class FakeTok:
    def __call__(self, prompt, **kw):
        return FakeEnc(input_ids=torch.tensor([[1, 2, 3]]))


class FakeOut:
    def __init__(self, z, logits):
        self.hidden_states = [z.reshape(1, 1, -1)]
        self.logits = logits.reshape(1, 1, -1)


class FakeLM:
    """logits = E z exactly — the post-norm regime probe() is stated in, so the test
    controls the residual stream and the head at once."""

    def __init__(self, E, z):
        self.E, self.z = E, z

    def __call__(self, **kw):
        return FakeOut(self.z, self.E @ self.z)


def _head(d=4, vocab=6, j_top=2, j_tgt=5, top_row=None, tgt_row=None):
    E = torch.zeros(vocab, d)
    E[j_top] = torch.tensor(top_row)
    E[j_tgt] = torch.tensor(tgt_row)
    return E, E[[j_top, j_tgt]], j_top, j_tgt


def test_probe_returns_the_top_minus_target_margin_second():
    """[1] is logit[j_top] - logit[j_tgt], positive while the incumbent still wins."""
    E, Wr, j_top, j_tgt = _head(top_row=[3.0, 0, 0, 0], tgt_row=[0, 1.0, 0, 0])
    z = torch.tensor([1.0, 1.0, 0.0, 0.0])
    am, m, r = ts.probe(FakeLM(E, z), FakeTok(), "x", Wr, j_top, j_tgt, "cpu")
    assert am == j_top                               # 3.0 beats 1.0
    assert np.isclose(m, 3.0 - 1.0)


def test_probe_third_return_is_the_negated_margin_not_a_probe_readout():
    """[2] = logit[j_tgt] - logit[j_top] = -[1]. It is NOT w . h for any direction: it
    is fixed by the two unembedding rows alone and carries no probe in it."""
    E, Wr, j_top, j_tgt = _head(top_row=[3.0, 0, 0, 0], tgt_row=[0, 1.0, 0, 0])
    z = torch.tensor([1.0, 1.0, 0.0, 0.0])
    _, m, r = ts.probe(FakeLM(E, z), FakeTok(), "x", Wr, j_top, j_tgt, "cpu")
    assert np.isclose(r, 1.0 - 3.0)
    assert np.isclose(r, -m)


def test_the_sign_flips_together_once_the_target_overtakes_the_incumbent():
    """A flipped statement has m < 0 and r > 0. Pinning both ends stops a future edit
    from swapping the return order and leaving every delta sign inverted."""
    E, Wr, j_top, j_tgt = _head(top_row=[1.0, 0, 0, 0], tgt_row=[0, 4.0, 0, 0])
    z = torch.tensor([1.0, 1.0, 0.0, 0.0])
    am, m, r = ts.probe(FakeLM(E, z), FakeTok(), "x", Wr, j_top, j_tgt, "cpu")
    assert am == j_tgt
    assert m < 0 < r
    assert np.isclose(r, -m)


def test_the_logged_delta_is_positive_exactly_when_steering_helps():
    """The row logs r - r0. That is m0 - m: positive when the budget moved probability
    TOWARD the target. This is the whole reason probe() returns the negated copy."""
    E, Wr, j_top, j_tgt = _head(top_row=[3.0, 0, 0, 0], tgt_row=[0, 1.0, 0, 0])
    tok = FakeTok()
    _, m0, r0 = ts.probe(FakeLM(E, torch.tensor([1.0, 1.0, 0, 0])), tok, "x",
                         Wr, j_top, j_tgt, "cpu")
    # z moved along the target's row: the margin must fall and the delta must rise.
    _, m, r = ts.probe(FakeLM(E, torch.tensor([1.0, 2.0, 0, 0])), tok, "x",
                       Wr, j_top, j_tgt, "cpu")
    assert m < m0
    assert r - r0 > 0
    assert np.isclose(r - r0, m0 - m)


def test_the_delta_column_carries_nothing_frac_margin_lacks():
    """delta == frac_margin * |m0|, per statement. The two columns are one measurement,
    so no analysis can cross them as independent axes."""
    E, Wr, j_top, j_tgt = _head(top_row=[3.0, 0, 0, 0], tgt_row=[0, 1.0, 0, 0])
    tok = FakeTok()
    _, m0, r0 = ts.probe(FakeLM(E, torch.tensor([1.0, 1.0, 0, 0])), tok, "x",
                         Wr, j_top, j_tgt, "cpu")
    for zt in ([1.0, 2.0, 0, 0], [1.0, 0.0, 0, 0], [2.0, 1.0, 0, 0]):
        _, m, r = ts.probe(FakeLM(E, torch.tensor(zt)), tok, "x",
                           Wr, j_top, j_tgt, "cpu")
        frac_margin = (m0 - m) / abs(m0)
        assert np.isclose(r - r0, frac_margin * abs(m0))


# -------------------------------------------------------- the column name and its past

def test_the_delta_column_is_named_for_the_quantity_it_holds():
    assert "tgt_minus_top_delta" in ts.STEER_COLUMNS
    assert "readout_delta" not in ts.STEER_COLUMNS


def test_load_steer_csv_reads_both_header_generations(tmp_path):
    """Twelve token_steer_*.csv from finished cluster runs carry `readout_delta` and are
    deliberately not regenerated. A reader that only knows the new name would silently
    drop the column on every one of them."""
    old = tmp_path / "old.csv"
    old.write_text("direction,frac_margin,readout_delta\noracle,0.5,9.5\n")
    new = tmp_path / "new.csv"
    new.write_text("direction,frac_margin,tgt_minus_top_delta\noracle,0.5,9.5\n")
    for p in (old, new):
        d = ts.load_steer_csv(str(p))
        assert "readout_delta" not in d.columns
        assert float(d.tgt_minus_top_delta.iloc[0]) == 9.5


def test_load_steer_csv_leaves_every_other_column_alone():
    """The alias map must rename one column, not reshape the frame."""
    assert list(ts.LEGACY_COLUMN_ALIASES) == ["readout_delta"]
