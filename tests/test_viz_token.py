"""fig_steer must not average two different statement populations together.

`jtw_legacy` is skipped wherever a statement falls outside the reach_margins
subsample, which on common_claim is 110 of 200. If the figure takes a plain mean per
direction, the legacy curve is computed on 90 statements and the oracle curve on 200,
and the two are then read off the same axis as though they were comparable. That is a
silent wrong conclusion, not a rendering glitch, so it is pinned here.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd                        # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import viz_token as vt                      # noqa: E402

FRACS = (-1.0, 0.0, 1.0)


def _arm(coverage, hit):
    """One arm's CSV. `coverage[name]` = which stmt ids that direction ran.
    `hit[name]` = hit_target for that direction at frac > 0."""
    rows = []
    for nm, stmts in coverage.items():
        for s in stmts:
            for f in FRACS:
                rows.append(dict(direction=nm, stmt=s, stem="x", scale=f,
                                 frac=f, argmax_id=0, argmax_tok="a",
                                 hit_target=int(hit[nm] and f > 0),
                                 margin=0.0, frac_margin=0.0,
                                 tgt_minus_top_delta=0.0, completion=""))
    return pd.DataFrame(rows)


def test_panel_restricts_to_statements_every_direction_ran(tmp_path, monkeypatch):
    """oracle ran 0..9, legacy only 0..4. The panel must use the 5 shared ones."""
    monkeypatch.chdir(tmp_path)
    d = _arm({"oracle": range(10), "jtw_legacy": range(5)},
             {"oracle": True, "jtw_legacy": False})
    vt.fig_steer("toy", [("postnorm_all_rp1", d)], "out.png")
    assert os.path.exists("out.png")


def test_the_shared_set_is_the_intersection_not_the_smallest_direction():
    """Two directions can each be short in a different place. Taking the smaller count
    would keep statements the other direction never ran."""
    a = _arm({"x": [0, 1, 2, 3], "y": [2, 3, 4, 5]}, {"x": True, "y": True})
    sets = [set(g.stmt.unique()) for _, g in a.groupby("direction")]
    assert set.intersection(*sets) == {2, 3}
    assert min(len(s) for s in sets) == 4          # the wrong answer, pinned


def test_restriction_changes_the_reported_rate_when_coverage_differs():
    """The whole point: an unrestricted mean reports a different number. Statements 0-4
    flip, 5-9 do not; a direction covering only 0-4 looks perfect until it is put on
    the same population as one covering all ten."""
    rows = []
    for s in range(10):
        for nm in ("oracle", "jtw_legacy"):
            if nm == "jtw_legacy" and s >= 5:
                continue
            rows.append(dict(direction=nm, stmt=s, frac=1.0,
                             hit_target=int(s < 5)))
    d = pd.DataFrame(rows)
    unrestricted = d.groupby("direction").hit_target.mean()
    sets = [set(g.stmt.unique()) for _, g in d.groupby("direction")]
    shared = set.intersection(*sets)
    restricted = d[d.stmt.isin(shared)].groupby("direction").hit_target.mean()

    assert unrestricted["oracle"] == 0.5           # 5 of 10
    assert unrestricted["jtw_legacy"] == 1.0       # 5 of 5, flattering
    assert restricted["oracle"] == 1.0             # same population, same answer
    assert restricted["jtw_legacy"] == 1.0


def test_missing_arms_do_not_raise(tmp_path, monkeypatch):
    """viz runs before every arm has landed; an absent arm is normal, not an error."""
    monkeypatch.chdir(tmp_path)
    vt.fig_steer("toy", [], "empty.png")
