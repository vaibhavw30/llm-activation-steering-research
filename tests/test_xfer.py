"""Tests for src/xfer_common.py, src/xfer_cities.py and src/xfer_tqa.py (J-D1, J-D2)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tqa_discovery as td  # noqa: E402
import xfer_cities as xcit  # noqa: E402
import xfer_common as xc  # noqa: E402
import xfer_tqa as xtqa  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _fit(d=8, k=6, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((d, k))
    U = rng.standard_normal((d, k)) * np.arange(1, k + 1)     # potency rises with index
    return V, U


# ------------------------------------------------------------------ directions
def test_cities_picks_potency_and_signed_geo():
    V, _ = _fit()
    U = np.eye(8)[:, :6] * np.arange(1, 7)             # potency = index + 1
    md = -td.unit(V[:, 2])                    # factor 2 is exactly anti-aligned
    picks = xc.cities_picks(V, U, md, k=2)
    assert picks[:2] == [("S-none", 5, 1), ("S-none", 4, 1)]
    assert picks[2] == ("S-geo", 2, -1)


def test_selection_picks_reads_both_as_plus():
    sel = {"rules": {"S-none": [{"factor": 3, "sign": "both"}],
                     "S-beh": [{"factor": 7, "sign": -1}]}}
    assert xc.selection_picks(sel) == [("S-none", 3, 1), ("S-beh", 7, -1)]


def test_anchor_prefers_sbeh_then_sgeo():
    assert xc.anchor_pick([("S-none", 1, 1), ("S-geo", 2, 1), ("S-beh", 3, -1)])[1] == 3
    assert xc.anchor_pick([("S-none", 1, 1), ("S-geo", 2, 1)])[1] == 2
    assert xc.anchor_pick([("S-none", 1, 1)]) is None


def test_potency_matched_excludes_picks_and_matches_nearest():
    U = np.diag([1.0, 2.0, 2.1, 5.0, 2.05])
    i, s = xc.potency_matched(U, anchor=1, exclude={1, 2})
    assert i == 4 and s in (-1, 1)


def test_dct_directions_dedupes_signs_and_adds_control():
    V, U = _fit()
    picks = [("S-none", 5, 1), ("S-geo", 5, 1), ("S-beh", 3, -1)]
    out = dict(xc.dct_directions("tqa", V, picks, U))
    assert "tqa:dct_S-geo_5" not in out                          # same (factor, sign)
    assert np.allclose(out["tqa:dct_S-beh_3"], -td.unit(V[:, 3]))
    ctl = [n for n in out if ":dct_ctl_" in n]
    assert len(ctl) == 1 and int(ctl[0].split("_")[-1]) not in (3, 5)
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for v in out.values())


def test_dose_grid_both_signs_and_units():
    g = xc.dose_grid(120.0, 14.0)
    assert len(g) == 2 * (len(xc.NORM_FRACS) + len(xc.EPS_FRACS))
    assert ("norm", -0.25, -30.0) in g and ("eps", 2.0, 28.0) in g
    assert xc.READ_FRAC in xc.NORM_FRACS
    assert all(u == "norm" for u, _, _ in xc.dose_grid(60.0, None))


def test_beyond_null_floors_at_beating_every_random():
    assert xc.beyond_null(1 / 9, 8)                    # 8 randoms: beat all 8
    assert not xc.beyond_null(2 / 9, 8)
    assert xc.beyond_null(0.04, 32) and not xc.beyond_null(2 / 33, 32)
    assert not xc.beyond_null("", 8)


def test_perm_p():
    assert xc.perm_p(0.5, [0.1, 0.6, np.nan, None]) == pytest.approx(2 / 3)
    assert xc.perm_p(1.0, [0.0] * 32) == pytest.approx(1 / 33)


def test_steer_vec_shapes():
    assert tuple(xc.steer_vec(np.ones(4), 2.0).shape) == (4,)
    v = xc.steer_vec(np.ones((3, 4)), 2.0)
    assert tuple(v.shape) == (3, 1, 4) and float(v[0, 0, 0]) == 2.0


def test_word_count_and_incoherent():
    assert xc.word_count("Russia, in Europe.") == 3
    assert not xc.incoherent("Russia.")
    assert not xc.incoherent("")
    assert xc.incoherent("the the the city")
    assert xc.incoherent("!!! ### @@@")
    assert xc.incoherent("→→→→ ok")


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "reach_margins_truthfulqa.npz")),
                    reason="TQA reach files are cluster-only")
def test_sup_jtw_truthfulqa_is_the_q2_direction(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert np.allclose(xc.sup_jtw("truthfulqa"), td.q2_vector())


def test_every_direction_is_signed_toward_truthful_on_cities(monkeypatch):
    """The cities supervised and MAG directions must read truthful (label 1) higher on
    cities' own layer-11 activations. A sign slip here would invert a whole row of X."""
    p = os.path.join(ROOT, "activations", "acts_cities.npz")
    if not os.path.exists(p):
        pytest.skip("acts_cities.npz not here")
    monkeypatch.chdir(ROOT)
    z = np.load(p, allow_pickle=True)
    X, y = z["activations"][xc.LAYER].astype(np.float64), z["labels"].astype(int)
    dirs = {"cities:mean_diff": td.truth_dirs("cities")["mean_diff"],
            "cities:mag_uQ": td.MAG_TRUTHFUL_SIGN * td.unit(
                np.load("mag_dir_cities.npz")["u_Q_gold"]),
            "cities:sup_jtw": xc.sup_jtw("cities")}
    for name, v in dirs.items():
        s = X @ v
        assert s[y == 1].mean() > s[y == 0].mean(), name


# ------------------------------------------------------------------ cities readout
COUNTRIES = ["Russia", "South Africa", "South Korea", "Niger", "Nigeria", "Papua New Guinea",
             "Guinea"]


def test_first_country():
    fc = xcit.first_country
    assert fc(" Russia. It is also in South Africa.", COUNTRIES) == "Russia"
    assert fc(" the south of South Korea", COUNTRIES) == "South Korea"
    assert fc(" Nigeria, not Niger", COUNTRIES) == "Nigeria"
    assert fc(" Papua New Guinea", COUNTRIES) == "Papua New Guinea"
    assert fc(" a big country.\nRussia", COUNTRIES) is None       # after the sentence
    assert fc(" Russian Federation", COUNTRIES) is None            # not a whole word


def test_score_gen():
    s = xcit.score_gen(" Nigeria, a country in Africa. The", "Niger", COUNTRIES)
    assert (s["gen_correct"], s["gen_wrong_country"], s["words"]) == (0, 1, 5)
    s = xcit.score_gen(" Niger.", "Niger", COUNTRIES)
    assert (s["gen_correct"], s["gen_wrong_country"]) == (1, 0)


def test_clean_pool_drops_shared_first_tokens_and_bare_strips_the():
    pool = xcit.clean_pool({"South Africa": 7, "South Korea": 7, "Russia": 3, "Niger": 9,
                            "North Korea": 11, "Saudi Arabia": 12})
    assert pool == {"Russia": 3, "Niger": 9, "Saudi Arabia": 12}
    assert xcit.wrong_ids(pool, "Russia") == [9, 12]
    assert xcit.bare("the Philippines") == "Philippines" and xcit.bare("Thailand") == "Thailand"
    assert xcit.PROMPT.format(city="Weifang").endswith("is in the country of")


def _cities_frame(n=30, n_rand=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(name, unit, frac, dm, gc, words):
        for i in range(n):
            rows.append({"direction": name, "unit": unit, "frac": frac, "stmt": i,
                         "margin_correct": 5.0 + dm[i], "argmax_correct": gc[i],
                         "hit_tgt": 0, "gen_correct": gc[i],
                         "gen_wrong_country": 1 - gc[i], "words": words[i],
                         "incoherent": 0})
    ones, base_w = np.ones(n, int), np.full(n, 2.0)
    add("baseline", "none", 0.0, np.zeros(n), ones, base_w)
    for j in range(n_rand):
        add(f"rand_{j}", "norm", -xc.READ_FRAC, rng.normal(0, 0.1, n), ones, base_w)
    fact = (np.arange(n) >= n // 2).astype(int)
    add("tqa:dct_S-beh_3", "norm", -xc.READ_FRAC, -np.full(n, 3.0), fact, base_w)     # facts flip
    add("tqa:dct_S-geo_4", "norm", -xc.READ_FRAC, rng.normal(0, 0.1, n), ones,
        np.full(n, 9.0))                                                     # form only
    add("cities:sup_jtw", "norm", -xc.READ_FRAC, rng.normal(0, 0.1, n), ones, base_w)
    return pd.DataFrame(rows)


def test_cities_summary_classifies_the_three_outcomes():
    out = {r["direction"]: r for r in xcit.summarize(_cities_frame())}
    beh = out["tqa:dct_S-beh_3"]
    assert beh["d_margin"] == pytest.approx(-3.0) and beh["e_logit"] == pytest.approx(3.0)
    assert beh["lost"] == 15 and xcit.classify(beh) == "a"
    assert xcit.classify(out["tqa:dct_S-geo_4"]) == "b"
    assert xcit.classify(out["cities:sup_jtw"]) in ("c", "logit-only")
    assert out["rand_0"]["p_logit"] == ""
    assert xcit.p8_cell(out) == "tqa:dct_S-beh_3"
    assert xcit.p8_cell(["tqa:dct_S-geo_4", "rand_0"]) == "tqa:dct_S-geo_4"


def test_cities_summary_stage_writes_outcomes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _cities_frame()
    orc = df[df["direction"] == "baseline"].copy()
    for f in xcit.ORACLE_FRACS:
        o = orc.assign(direction="oracle", unit="oracle_eps", frac=f, hit_tgt=1)
        df = pd.concat([df, o])
    df.to_csv("xfer_cities_steer.csv", index=False)
    xcit.stage_summary()
    import json
    got = json.load(open("xfer_cities_outcomes.json"))
    assert got["oracle_ok"] and got["p8_cell"] == "tqa:dct_S-beh_3"
    assert got["outcomes"]["tqa:dct_S-beh_3"] == "a"
    with pytest.raises(SystemExit):
        xcit.stage_summary()                                   # never overwrites


# ------------------------------------------------------------------ TQA readout
def _tqa_frame(n=40, n_rand=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []

    def add(name, unit, frac, t, words):
        for i in range(n):
            rows.append({"direction": name, "unit": unit, "frac": frac,
                         "question": f"q{i}", "truthful": int(t[i]), "words": words[i]})
    add("baseline", "none", 0.0, np.zeros(n), np.full(n, 5.0))
    for j in range(n_rand):
        add(f"rand_{j}", "norm", xc.READ_FRAC, rng.random(n) < 0.05, np.full(n, 5.0))
    add("cities:sup_jtw", "norm", xc.READ_FRAC, np.arange(n) < 20, np.full(n, 5.0))
    add("tqa:sup_jtw", "norm", xc.READ_FRAC, np.zeros(n), np.full(n, 19.0))
    add("tqa:sup_jtw", "eps", 2.0, np.arange(n) < 20, np.full(n, 19.0))
    return pd.DataFrame(rows)


def test_tqa_summary_and_classify():
    out = {(r["direction"], r["unit"]): r for r in xtqa.summarize(_tqa_frame(), "truthful")}
    c = out[("cities:sup_jtw", "norm")]
    assert (c["gained"], c["lost"], c["rate"]) == (20, 0, 0.5)
    assert xtqa.classify(c) == "gain"
    assert xtqa.classify(out[("tqa:sup_jtw", "norm")]) == "form"
    assert out[("tqa:sup_jtw", "eps")]["p_gain"] == ""           # no randoms at eps here


def test_q2_interval_prefers_v2_and_flags_v1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = "arm,direction,frac,rate,wilson_lo,wilson_hi\nmean,jtw_mean_diff_tgt,-2.0,{r},0.3,0.6\n"
    open(xtqa.Q2_REFERENCE_V1, "w").write(row.format(r=0.5))
    got = xtqa.q2_interval("truthful_and_informative")
    assert got[0] == 0.5 and got[3].endswith("(v1 judge!)")
    assert xtqa.q2_interval("truthful") is None                  # v1 is not truthful-only
    open(xtqa.Q2_REFERENCE["truthful"], "w").write(row.format(r=0.45))
    assert xtqa.q2_interval("truthful")[0] == 0.45


def test_mains_refuse_limit_without_prefix():
    for m in (xcit, xtqa):
        with pytest.raises(SystemExit):
            m.main(["--stage", "steer", "--limit", "2"])
