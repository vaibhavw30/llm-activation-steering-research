"""sae_decompose.py — Horizon-1 1.2: which SAE features make up our directions?

Decomposes a vector into GemmaScope decoder atoms with orthogonal matching pursuit.
NOT naive encoder application: running the SAE encoder on a steering direction is the
mistake arXiv:2411.08790 warns about — the encoder is trained on activations, and a
direction is not an activation, so its encoding is not a faithful decomposition. That
paper's gradient pursuit approximates matching pursuit for streaming activations; we
decompose a handful of fixed vectors, so exact OMP with a least-squares refit on the
support is both affordable and strictly better.

Vectors decomposed (per dataset) — see collect_vectors (Task B3):
  w_mean_diff_tgt        the readout the certificate is defined against (target layer)
  jtw_mean               dataset-mean unit J^T w over label-1 rows: the input direction
                         Phase 3 actually steered along (source layer)
  jtw_full_matched_mean  the same quantity restricted to the statements stemjac also
                         measured, so the full-vs-stem comparison is like-for-like
  jtw_stem_mean          stem-context mean unit J^T w (source layer)
  common_v1              top right-singular vector of [full; stem]: what survives the
                         one-word context shift (the D1/D2 question)
  V64_common_j           top-4 right-singular vectors of the stacked per-statement
                         leading V64 columns: the high-gain input channel, pooled

This module (Task B2) is the pure-helper half: omp, explained, jaccard, K_ATOMS.
The CLI and collect_vectors are added by Task B3.

    PYTHONPATH=src .venv/bin/python src/sae_decompose.py --dataset cities
"""
import numpy as np

K_ATOMS = 32


def omp(v, D, k=K_ATOMS):
    """Orthogonal matching pursuit of v over unit-norm atoms D (F,d). Returns
    (support (k,), coefs (k,), residual_fraction) with residual_fraction =
    ||v - D_S^T c|| / ||v||."""
    v = np.asarray(v, np.float64)
    D = np.asarray(D, np.float64)
    if k > D.shape[0]:
        raise ValueError(f"k={k} exceeds dictionary size {D.shape[0]}")
    nv = float(np.linalg.norm(v))
    r = v.copy()
    support, coefs = [], np.zeros(0)
    for _ in range(int(k)):
        corr = D @ r
        corr[support] = 0.0                      # never repeat an atom
        support.append(int(np.argmax(np.abs(corr))))
        A = D[support].T                          # (d, |S|)
        coefs = np.linalg.lstsq(A, v, rcond=None)[0]
        r = v - A @ coefs
    resid = float(np.linalg.norm(r) / nv) if nv > 0 else 0.0
    return np.array(support, int), np.asarray(coefs, np.float64), resid


def explained(v, D, support, coefs):
    """Fraction of ||v||^2 captured by the reconstruction."""
    v = np.asarray(v, np.float64)
    rec = np.asarray(D, np.float64)[np.asarray(support, int)].T @ \
        np.asarray(coefs, np.float64)
    nv = float(v @ v)
    return float(1.0 - ((v - rec) @ (v - rec)) / nv) if nv > 0 else 1.0


def jaccard(a, b):
    """|A n B| / |A u B|; 0.0 when both are empty."""
    sa, sb = {int(x) for x in a}, {int(x) for x in b}
    u = sa | sb
    return float(len(sa & sb) / len(u)) if u else 0.0
