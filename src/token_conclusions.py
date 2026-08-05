"""token_conclusions.py: the computations behind every number in
docs/TOKEN_SPACE_FINDINGS.md.

Read-only. Nothing here writes, regenerates or overwrites an artifact.

Design note on coverage. `jtw_legacy` ran on 90 of 200 statements on common_claim,
because reach_margins was computed on a 2,000-row stratified subsample of a 4,450-row
dataset while token_geom sampled the full set. Every cross-direction statistic in this
module therefore restricts to the statements all directions ran. src/viz_token.py
fig_steer already does this for the figure and tests/test_viz_token.py pins it, this is
the same rule applied to the tables.
"""
import difflib
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

    ASSUMES A POSITIVE SCALE. The comparison is `|scale| >= eps*_legacy`, so a row swept
    at a negative frac is marked crossed on the strength of its magnitude alone, even
    though a negative scale moves the readout AWAY from the FALSE threshold and the
    first-order argument above then predicts no crossing at any magnitude. Every caller
    in the published analysis iterates positive fracs only, so no reported number is
    affected, but the arms on disk carry fracs down to -2.0 and feeding them here would
    produce cells that are the bare inequality and not a readout prediction. The
    absolute value is deliberately left in place rather than made sign-aware: changing
    it would move published numbers. Filter to `scale > 0` before calling if negative
    fracs are ever needed. Pinned by
    test_crossed_is_sign_blind_and_marks_a_negative_scale_as_crossed.
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
    construction and the ratio is undefined there. Rows where either column is NaN are
    dropped too, and before the two-row gate rather than after: a group of two rows one
    of which has a NaN ratio scores `std(ddof=0) == 0` on its single surviving value and
    passes, which is the same silent leniency the two-row gate exists to prevent. All
    twelve arms on disk are NaN-free in both columns, so this changes no reported number.

    A (direction, stmt) group with fewer than two surviving rows carries no evidence
    about within-statement proportionality, so it is excluded from the comparison
    rather than scored as a pass. If no group has enough rows to be informative, the
    function returns False: no evidence is not the same as proven proportional.
    """
    s = df[df[b].abs() > 1e-9].copy()
    if s.empty:
        return False
    s["_r"] = s[a] / s[b]
    s = s[s["_r"].notna()]
    if s.empty:
        return False
    counts = s.groupby(["direction", "stmt"])._r.transform("size")
    s = s[counts >= 2]
    if s.empty:
        return False
    rel = s.groupby(["direction", "stmt"])._r.agg(
        lambda x: float(x.std(ddof=0) / max(abs(x.mean()), 1e-12)))
    return bool(rel.max() <= rtol)


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
