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
