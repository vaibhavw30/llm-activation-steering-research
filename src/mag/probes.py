"""MAG probes E1 (readability), E2 (disagreement), E3 (linearity/reconstruction), and the
§4 transfer ranking. Pure functions over cached readouts so they unit-test without gemma."""
import math
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_utils import unit, top_k_by_potency
from mag.operators import operator_features
from mag.config import OPERATOR_NAMES

SEED = 42


def reconstruction_error(shifts, v):
    """Eq. 3 — mean||shift - v|| / mean||shift||. shifts (n,d), v (d,)."""
    shifts = np.asarray(shifts, np.float64); v = np.asarray(v, np.float64)
    denom = np.linalg.norm(shifts, axis=1).mean()
    if denom == 0:
        return 0.0
    num = np.linalg.norm(shifts - v, axis=1).mean()
    return float(num / denom)


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _cv_acc_roc(X, y):
    """5-fold stratified CV mean accuracy + ROC-AUC; scaler refit per-fold (no leakage)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    X = np.asarray(X, np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    pred = cross_val_predict(pipe, X, y, cv=skf)
    try:
        proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, proba))
    except Exception:
        roc = float("nan")
    return float((pred == y).mean()), roc


def E1_readability(cache, layer, y_gold, y_yM, dct=None):
    rows = []
    for name in OPERATOR_NAMES:
        X = operator_features(name, cache, layer)
        for tgt, y in (("gold", y_gold), ("yM", y_yM)):
            acc, roc = _cv_acc_roc(X, y)
            rows.append({"operator": name, "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    # DCT-top-k and random-k baselines on raw A(p)
    A_p = operator_features("Direct", cache, layer)
    if dct is not None:
        V, U, _ = dct
        for k in (10, 50):
            idx = top_k_by_potency(V, U, k)
            Xk = A_p @ V[:, idx]
            for tgt, y in (("gold", y_gold), ("yM", y_yM)):
                acc, roc = _cv_acc_roc(Xk, y)
                rows.append({"operator": f"DCT_top{k}", "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    rng = np.random.default_rng(SEED)
    Rk = A_p @ rng.standard_normal((A_p.shape[1], 10))
    for tgt, y in (("gold", y_gold), ("yM", y_yM)):
        acc, roc = _cv_acc_roc(Rk, y)
        rows.append({"operator": "random_10", "target": tgt, "acc": acc, "roc": roc, "n": len(y)})
    return rows


def E2_disagreement(feat_mag, feat_raw, y_gold, y_yM):
    """Fit on the agree-set (y^M==gold), predict y^M on the disagreement set D; report the
    fraction of D where the classifier sides with the MODEL (y^M) not the label."""
    y_gold = np.asarray(y_gold).astype(int); y_yM = np.asarray(y_yM).astype(int)
    agree = y_gold == y_yM
    D = ~agree
    rows = []
    for tag, feat in (("mag", feat_mag), ("raw", feat_raw)):
        n_dis = int(D.sum())
        if n_dis == 0 or len(np.unique(y_yM[agree])) < 2:
            rows.append({"feature": tag, "n_disagree": n_dis, "match_yM_rate": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan")})
            continue
        sc = StandardScaler().fit(feat[agree])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(feat[agree]), y_yM[agree])
        pred = clf.predict(sc.transform(feat[D]))
        k = int((pred == y_yM[D]).sum())
        lo, hi = wilson_ci(k, n_dis)
        rows.append({"feature": tag, "n_disagree": n_dis, "match_yM_rate": k / n_dis,
                     "ci_lo": lo, "ci_hi": hi})
    return rows


def E3_linearity(cache, layer, extra_dirs=None):
    """eps_Q for v_Q (final-layer mode) and, for comparison, for each direction in extra_dirs
    (e.g. {'mean_diff':vec,'dct':vec}). cos = mean cosine of per-prompt shift to the direction."""
    A_p = operator_features("Direct", cache, layer)
    A_Qp = operator_features("Prefixed", cache, layer)
    shifts = A_Qp - A_p
    v = shifts.mean(axis=0)
    rows = []
    def cos_to(d):
        d = unit(np.asarray(d, np.float64))
        s = shifts / (np.linalg.norm(shifts, axis=1, keepdims=True) + 1e-8)
        return float((s @ d).mean())
    rows.append({"direction": "v_Q", "mode": "final",
                 "eps_Q": reconstruction_error(shifts, v), "cos": cos_to(v)})
    for name, d in (extra_dirs or {}).items():
        # scale the comparison direction to v_Q's magnitude so eps_Q is comparable
        d = np.asarray(d, np.float64)
        d_scaled = unit(d) * np.linalg.norm(v)
        rows.append({"direction": name, "mode": "final",
                     "eps_Q": reconstruction_error(shifts, d_scaled), "cos": cos_to(d)})
    return rows


def transfer_rank(feats_by_ds, labels_by_ds):
    """Leave-one-out: for each target T, rank candidates C by centroid-cosine and by realized
    transfer accuracy (train probe on C, test on T). Report Top-1 match and Spearman rho."""
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score
    names = list(feats_by_ds)
    rows, geom_all, real_all = [], [], []
    top1 = 0
    for T in names:
        cands = [c for c in names if c != T]
        realized, geom = {}, {}
        cent_T = feats_by_ds[T].mean(axis=0)
        for C in cands:
            sc = StandardScaler().fit(feats_by_ds[C])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(feats_by_ds[C]), labels_by_ds[C])
            realized[C] = accuracy_score(labels_by_ds[T], clf.predict(sc.transform(feats_by_ds[T])))
            geom[C] = float(unit(feats_by_ds[C].mean(axis=0)) @ unit(cent_T))
        best_real = max(realized, key=realized.get)
        best_geom = max(geom, key=geom.get)
        top1 += int(best_real == best_geom)
        for C in cands:
            rows.append({"target": T, "candidate": C, "realized_delta": realized[C],
                         "geom_score": geom[C]})
            geom_all.append(geom[C]); real_all.append(realized[C])
    rho = float(spearmanr(geom_all, real_all).correlation) if len(geom_all) > 2 else float("nan")
    return {"top1": top1 / len(names), "spearman": rho, "rows": rows}
