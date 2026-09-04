"""reach_stemprobe.py — Horizon-0 item 0.2: probe refit on the generation population.

D2 said the probe boundary doesn't transfer to generation prefixes (cities stems
read g ~ -52 "deep FALSE" while the model completes truthfully). This quantifies
it. Labels are FREE: the scale-0 completions in judge_reach_steer_stmt_<ds>.csv
were already judged — TRUE means "the model completes this stem truthfully".

  --extract (GPU): target-layer activation at each stem's last token
                   -> reach_stemacts_<ds>.npz
  --fit     (CPU): (a) old-threshold transfer accuracy, (b) AUC of the old
                   readout, (c) recalibrated 1-D threshold on the same w,
                   (d) 5-fold CV refit LR probe, (e) 5-fold CV XGBoost
                   (nonlinear headroom on this population)
                   -> reach_d2_summary_<ds>.csv + reach_stemprobe_<ds>.csv
  --summary (CPU): the same statistics, writing only reach_d2_summary_<ds>.csv.
                   Use this when reach_stemprobe_<ds>.csv already exists; it is
                   cluster output and there is no reason to rewrite it.

    PYTHONPATH=src python src/reach_stemprobe.py --dataset cities --extract --device cuda
    PYTHONPATH=src python src/reach_stemprobe.py --dataset cities --summary
"""
import argparse
import csv

import numpy as np

ACTS_BATCH = 16
CV_FOLDS = 5
SEED = 42                                    # mirrors reach_steer.SEED
LOGIT_02 = float(np.log(0.2 / 0.8))          # mirrors reach_margins.LOGIT_02

# NOTE: the --fit path must stay torch-free. reach_steer/reach_margins import
# torch at module level, and torch's libomp + xgboost's libomp in one process
# segfault on macOS — so model-side helpers are imported lazily in extract().


def stem_labels(path):
    """{stem_prompt: 1/0} from scale-0 judge verdicts; TRUE->1, FALSE->0,
    anything else (INCOHERENT, ...) dropped."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["scale"]) == 0.0 and r["verdict"] in ("TRUE", "FALSE"):
                out[r["prompt"]] = 1 if r["verdict"] == "TRUE" else 0
    return out


def transfer_metrics(g, y):
    """(accuracy, balanced accuracy) of predicting y with sign(g) > 0."""
    g = np.asarray(g, np.float64)
    y = np.asarray(y, int)
    pred = (g > 0).astype(int)
    acc = float((pred == y).mean())
    per = [float((pred[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return acc, float(np.mean(per))


def _picks(acts):
    from reach_steer import N_PER_STMT
    y = np.asarray(acts["labels"]).astype(int)
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    return rng.permutation(idx1)[:N_PER_STMT]


def fit_threshold_1d(scores, y):
    """1-D logistic calibration (reach_margins.fit_threshold, torch-free copy).
    Returns (acc, slope_sign, t02)."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, np.float64).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000).fit(s, y)
    acc = float(lr.score(s, y))
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    sign = 1.0 if a > 0 else -1.0
    t02 = (LOGIT_02 - b) / a if a != 0 else float("nan")
    return acc, sign, t02


def extract(ds, device, limit=0):
    from reach_steer import stem_of
    from reach_hop import load_model_and_slice, forward_source_batch
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    stmts = acts["statements"]
    picks = _picks(acts)
    if limit:
        picks = picks[:limit]
    keep = [(int(i), stem_of(stmts[i])) for i in picks
            if stem_of(stmts[i]) is not None]
    tok, model, _, meta = load_model_and_slice(ds, device)
    hs = []
    for b0 in range(0, len(keep), ACTS_BATCH):
        batch = keep[b0:b0 + ACTS_BATCH]
        fb = forward_source_batch(model, tok, [s for _, s in batch],
                                  meta["src"], meta["tgt"], meta["device"])
        hs.append(fb["h_tgt"].cpu().numpy())
        print(f"[stemprobe] {ds} {b0 + len(batch)}/{len(keep)}", flush=True)
    np.savez(f"reach_stemacts_{ds}.npz",
             h_tgt_stem=np.concatenate(hs).astype(np.float32),
             stmt_index=np.array([i for i, _ in keep]),
             stems=np.array([s for _, s in keep], dtype=object))
    print(f"[stemprobe] wrote reach_stemacts_{ds}.npz  n={len(keep)}")


def _cv_acc(clf_factory, X, y, folds=CV_FOLDS):
    from sklearn.model_selection import StratifiedKFold
    k = min(folds, int(np.bincount(y).min()))
    if k < 2:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True,
                                  random_state=SEED).split(X, y):
        clf = clf_factory().fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def d2_metrics(ds):
    """Compute the D2 transfer statistics. Returns (rows, metrics).

    rows are the per-stem records (stmt_index, stem, label, g_old); metrics is
    the summary dict that reach_d2_summary_<ds>.csv is written from.
    """
    sa = np.load(f"reach_stemacts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    k = names.index("mean_diff_tgt")
    w = np.asarray(dirs["W"][k], np.float64)
    t02 = float(dirs["thresh02"][k])
    labels = stem_labels(f"judge_reach_steer_stmt_{ds}.csv")
    rows, X, y = [], [], []
    for i, stem, h in zip(sa["stmt_index"], sa["stems"], sa["h_tgt_stem"]):
        lab = labels.get(str(stem))
        if lab is None:
            continue
        g = float(np.asarray(h, np.float64) @ w - t02)
        rows.append((int(i), str(stem), lab, f"{g:.6g}"))
        X.append(np.asarray(h, np.float64))
        y.append(lab)
    X, y = np.stack(X), np.array(y, int)
    g_old = np.array([float(r[3]) for r in rows])
    acc_old, bal_old = transfer_metrics(g_old, y)
    m = {
        "dataset": ds,
        "n": len(y),
        "n_true": int((y == 1).sum()),
        "n_false": int((y == 0).sum()),
        "base_rate_true": float(y.mean()),
        "g_old_min": float(g_old.min()),
        "g_old_max": float(g_old.max()),
        "g_old_one_sided": bool((g_old > 0).all() or (g_old < 0).all()),
        "acc_old_threshold": acc_old,
        "balanced_acc_old_threshold": bal_old,
        "auc_g_old": float("nan"),
        "acc_recalibrated_1d": float("nan"),
        "t02_stem": float("nan"),
        "t02_stmt": t02,
        "acc_refit_lr_cv": float("nan"),
        "acc_xgboost_cv": float("nan"),
        "nonlinear_gap": float("nan"),
    }
    if len(np.unique(y)) > 1:
        from sklearn.metrics import roc_auc_score
        m["auc_g_old"] = float(roc_auc_score(y, g_old))
        acc_recal, _, t02_stem = fit_threshold_1d(X @ w, y)
        m["acc_recalibrated_1d"] = acc_recal
        m["t02_stem"] = t02_stem
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m["acc_refit_lr_cv"] = _cv_acc(lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)), X, y)
        try:
            from xgboost import XGBClassifier
            m["acc_xgboost_cv"] = _cv_acc(lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8), X, y)
            m["nonlinear_gap"] = m["acc_xgboost_cv"] - m["acc_refit_lr_cv"]
        except ImportError:
            pass
    return rows, m


def _report(m, y_medians):
    ds = m["dataset"]
    print(f"[stemprobe] {ds}: n={m['n']} labeled stems, "
          f"base rate P(TRUE)={m['base_rate_true']:.3f} "
          f"({m['n_true']} TRUE / {m['n_false']} FALSE)")
    print(f"  g_old range [{m['g_old_min']:+.1f}, {m['g_old_max']:+.1f}], "
          f"never changes sign: {m['g_old_one_sided']}")
    tail = (f"(median g: TRUE {y_medians[1]:+.1f}, FALSE {y_medians[0]:+.1f})"
            if y_medians[0] is not None else "(all labels TRUE)")
    print(f"  old threshold  acc {m['acc_old_threshold']:.3f}  "
          f"balanced {m['balanced_acc_old_threshold']:.3f}  {tail}")
    if not np.isnan(m["auc_g_old"]):
        print(f"  AUC of g_old on the stem population: {m['auc_g_old']:.3f}")
        print(f"  recalibrated 1-D threshold on same w: "
              f"acc {m['acc_recalibrated_1d']:.3f}, "
              f"t02_stem {m['t02_stem']:+.2f} (old t02 {m['t02_stmt']:+.2f})")
        print(f"  refit LR probe ({CV_FOLDS}-fold CV): "
              f"acc {m['acc_refit_lr_cv']:.3f}")
        if not np.isnan(m["acc_xgboost_cv"]):
            print(f"  XGBoost ({CV_FOLDS}-fold CV): acc {m['acc_xgboost_cv']:.3f}"
                  f"  (gap vs LR {m['nonlinear_gap']:+.3f})")
        else:
            print("  XGBoost not installed — nonlinear arm skipped "
                  "(run --summary locally)")
    else:
        print("  only one label class present — refits skipped")


def write_summary(ds, m):
    path = f"reach_d2_summary_{ds}.csv"
    with open(path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(m.keys())
        w_.writerow(m.values())
    print(f"[stemprobe] wrote {path}")


def fit(ds, write_rows=True):
    """Print the D2 statistics and write reach_d2_summary_<ds>.csv.

    write_rows=False leaves the existing reach_stemprobe_<ds>.csv alone; it is
    July cluster output and the summary does not need to touch it.
    """
    rows, m = d2_metrics(ds)
    y = np.array([r[2] for r in rows], int)
    g_old = np.array([float(r[3]) for r in rows])
    med = ({0: float(np.median(g_old[y == 0])), 1: float(np.median(g_old[y == 1]))}
           if (y == 0).any() else {0: None, 1: None})
    _report(m, med)
    write_summary(ds, m)
    if write_rows:
        with open(f"reach_stemprobe_{ds}.csv", "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(("stmt_index", "stem", "label", "g_old"))
            w_.writerows(rows)
        print(f"[stemprobe] wrote reach_stemprobe_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="statistics only; leaves reach_stemprobe_<ds>.csv alone")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap stems (smoke)")
    a = ap.parse_args()
    if a.extract:
        extract(a.dataset, a.device, a.limit)
    if a.fit or a.summary:
        fit(a.dataset, write_rows=a.fit)
    if not (a.extract or a.fit or a.summary):
        raise SystemExit("pass --extract, --fit and/or --summary")


if __name__ == "__main__":
    main()
