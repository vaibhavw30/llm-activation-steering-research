"""MAG directions: v_Q (Eq.2 mean prefix shift), u_Q (class-mean contrast v^- - v^+),
and A_prefix_norm (the alpha(tau) calibration constant). Plus assembly into mag_dir_<ds>.npz."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_utils import unit
from mag.operators import operator_features


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
