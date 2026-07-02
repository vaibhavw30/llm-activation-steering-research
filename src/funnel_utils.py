"""
funnel_utils.py — shared helpers for the DCT-interpretation funnel (local analysis).

Loads geometry-of-truth activations and DCT vectors, computes the two supervised truth
directions, and resolves the comparison layer. Used by subspace_top_k.py, cross_dataset.py,
export_truth_dir.py, viz_funnel.py.
"""

import json
import os
import numpy as np
import torch

# Truth-peak layers from the geometry-of-truth analysis (best XGBoost layer per dataset).
BEST_LAYER = {
    "cities": 11,
    "sp_en_trans": 7,
    "companies_true_false": 14,
    "common_claim_true_false": 13,
}


def unit(v, axis=0, eps=1e-8):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def resolve_layer(ds, override=None):
    """Comparison layer: explicit override > DCT meta source_layer > geometry best layer."""
    if override is not None:
        return override
    meta = f"dct_meta_{ds}.json"
    if os.path.exists(meta):
        return int(json.load(open(meta))["source_layer"])
    return BEST_LAYER.get(ds, 11)


def load_acts(ds, layer, acts_path=None):
    """Return (X, y) at a single layer. X is (n, d) float64, y is int {0,1}."""
    path = acts_path or f"activations/acts_{ds}.npz"
    data = np.load(path, allow_pickle=True)
    X = data["activations"][layer].astype(np.float64)
    y = data["labels"].astype(int)
    return X, y


def mean_diff_dir(X, y):
    """Contrastive mean-difference truth direction (unit)."""
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def grad_dir(X, y):
    """Logistic-regression gradient truth direction (unit). Needs scikit-learn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return unit(lr.coef_[0] / sc.scale_)


def load_dct(ds, v_path=None, u_path=None):
    """Return (V, U, meta) — V/U are (d, num_factors) float numpy; meta is dict or {}."""
    V = torch.load(v_path or f"dct_V_{ds}.pt", map_location="cpu").float().numpy()
    upath = u_path or f"dct_U_{ds}.pt"
    U = torch.load(upath, map_location="cpu").float().numpy() if os.path.exists(upath) else None
    mpath = f"dct_meta_{ds}.json"
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    return V, U, meta


def top_k_by_potency(V, U, k):
    """Indices of the top-k DCT vectors ranked by downstream-effect magnitude ||U_i||.
    Falls back to first-k if U is unavailable."""
    if U is None:
        return list(range(min(k, V.shape[1])))
    potency = np.linalg.norm(U, axis=0)
    return list(np.argsort(potency)[::-1][:k])
