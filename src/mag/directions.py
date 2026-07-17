"""MAG directions: v_Q (Eq.2 mean prefix shift), u_Q (class-mean contrast v^- - v^+),
and A_prefix_norm (the alpha(tau) calibration constant). Plus assembly into mag_dir_<ds>.npz.

E4 leads (behavioral steering candidates beyond the canonical InputDelta u_Q):
  * Lead #1 — the divergent operators (Prefixed/Answered/QuestionDelta/FewShot) read truth
    well yet their class contrast sits off the supervised mean_diff axis (cos ~0.09-0.13).
    build_lead_directions() emits u_op for each as distinct steering candidates.
  * Lead #3 — the shift is *rank-1* in its class-mean structure: residualising the InputDelta
    shifts against the primary truth axis and re-taking the class-mean-diff yields ~0
    (resid_meandiff_norm), so no second LINEAR truth axis exists. The dominant off-axis
    direction that DOES remain is residual-PC1 — not a truth direction, but a steerable probe
    for "is any large non-truth component of the prefix shift causal?".
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_utils import unit
from mag.operators import operator_features
from mag.probes import _cv_acc_roc

# read truth well (E1) but land off the supervised mean_diff axis (divergence check)
DIVERGENT_OPERATORS = ("Prefixed", "Answered", "QuestionDelta", "FewShot")


def v_Q(A_Qp_L, A_p_L):
    """Eq. 2 — average prefix-induced shift at a layer. (n,d),(n,d) -> (d,)."""
    return (np.asarray(A_Qp_L, np.float64) - np.asarray(A_p_L, np.float64)).mean(axis=0)


def class_mean_diff(feats, labels):
    """u_Q = v^- - v^+  (negative class mean minus positive class mean). (n,d),(n,) -> (d,)."""
    feats = np.asarray(feats, np.float64); labels = np.asarray(labels).astype(int)
    d = feats.shape[1]
    neg = feats[labels == 0].mean(axis=0) if (labels == 0).any() else np.zeros(d)
    pos = feats[labels == 1].mean(axis=0) if (labels == 1).any() else np.zeros(d)
    return neg - pos


def a_prefix_norm(A_Qp_L):
    """mean_p ||A(Q||p)[L]|| — the calibration magnitude for alpha(tau)."""
    return float(np.linalg.norm(np.asarray(A_Qp_L, np.float64), axis=1).mean())


def build_directions(cache, layer, labels_gold, labels_yM,
                     mean_diff=None, grad=None, dct_V_top=None):
    """Assemble the MAG directions at `layer` and their cosines to the funnel directions.
    Steering direction is built from the canonical MAG feature (InputDelta)."""
    A_p_L = np.asarray(cache["A_p"][layer], np.float64)
    A_Qp_L = np.asarray(cache["A_Qp"][layer], np.float64)
    feat = operator_features("InputDelta", cache, layer)     # (n,d)

    v = v_Q(A_Qp_L, A_p_L)
    u_gold = class_mean_diff(feat, labels_gold)
    u_yM = class_mean_diff(feat, labels_yM)
    apn = a_prefix_norm(A_Qp_L)

    out = {"v_Q": v.astype(np.float32), "v_Q_unit": unit(v).astype(np.float32),
           "u_Q_gold": u_gold.astype(np.float32), "u_Q_gold_unit": unit(u_gold).astype(np.float32),
           "u_Q_yM": u_yM.astype(np.float32), "u_Q_yM_unit": unit(u_yM).astype(np.float32),
           "A_prefix_norm": np.float32(apn), "layer": np.int64(layer)}

    def cos(a, b):
        return float(unit(a) @ unit(b)) if b is not None else float("nan")

    out["cos_vQ_mean_diff"] = np.float32(cos(v, mean_diff))
    out["cos_uGold_mean_diff"] = np.float32(cos(u_gold, mean_diff))
    out["cos_uGold_grad"] = np.float32(cos(u_gold, grad))
    out["cos_uGold_dctV"] = np.float32(cos(u_gold, dct_V_top))
    return out


def _dir_readout_acc(feats, direction, labels):
    """5-fold CV accuracy of a probe on the 1-D projection feats @ unit(direction).
    Stamps whether a steering DIRECTION (not the full operator) actually carries truth,
    so a dead candidate is visible before we spend any generation. nan if degenerate."""
    proj = (np.asarray(feats, np.float64) @ unit(np.asarray(direction, np.float64))).reshape(-1, 1)
    acc, _ = _cv_acc_roc(proj, labels)
    return float(acc)


def residual_pc1(cache, layer, primary_unit):
    """Top principal component (max-variance direction) of the InputDelta per-prompt shifts
    after the primary truth axis is projected out. Orthogonal to primary_unit by construction."""
    s = operator_features("InputDelta", cache, layer)                 # (n, d)
    u = unit(np.asarray(primary_unit, np.float64))
    R = s - np.outer(s @ u, u)                                        # strip the truth axis
    R = R - R.mean(axis=0, keepdims=True)                             # center for PCA
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[0]


def build_lead_directions(cache, layer, labels_gold):
    """E4 steering candidates for leads #1 (divergent operators) and #3 (residual-PC1),
    plus the rank-1 check. Merge into the mag_dir_<ds>.npz produced by build_directions()."""
    labels = np.asarray(labels_gold).astype(int)
    feat_id = operator_features("InputDelta", cache, layer)           # primary axis source
    u_primary = class_mean_diff(feat_id, labels)
    u_primary_unit = unit(u_primary)
    out = {"u_primary_norm": np.float32(np.linalg.norm(u_primary))}

    # Lead #1 — one contrast direction per divergent operator
    for op in DIVERGENT_OPERATORS:
        feats = operator_features(op, cache, layer)
        u_op = class_mean_diff(feats, labels)
        out[f"u_{op}"] = u_op.astype(np.float32)
        out[f"u_{op}_unit"] = unit(u_op).astype(np.float32)
        out[f"acc_{op}"] = np.float32(_dir_readout_acc(feats, u_op, labels))
        out[f"cos_{op}_primary"] = np.float32(float(unit(u_op) @ u_primary_unit))

    # Lead #3 — rank-1 check: class-mean-diff of the primary-residualised shift is ~0,
    # i.e. the primary axis already carries ALL the linear class-mean signal.
    R_check = feat_id - np.outer(feat_id @ u_primary_unit, u_primary_unit)
    out["resid_meandiff_norm"] = np.float32(np.linalg.norm(class_mean_diff(R_check, labels)))

    # Lead #3 — the steerable dominant off-axis direction (NOT a truth axis)
    pc1 = residual_pc1(cache, layer, u_primary_unit)
    out["resid_pc1_unit"] = unit(pc1).astype(np.float32)
    out["acc_resid_pc1"] = np.float32(_dir_readout_acc(feat_id, pc1, labels))
    out["cos_resid_pc1_primary"] = np.float32(float(unit(pc1) @ u_primary_unit))
    return out
