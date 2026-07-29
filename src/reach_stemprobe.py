"""reach_stemprobe.py — Horizon-0 item 0.2: probe refit on the generation population.

D2 said the probe boundary doesn't transfer to generation prefixes (cities stems
read g ~ -52 "deep FALSE" while the model completes truthfully). This quantifies
it. Labels are FREE: the scale-0 completions in judge_reach_steer_stmt_<ds>.csv
were already judged — TRUE means "the model completes this stem truthfully".

  --extract (GPU): target-layer activation at each stem's last token
                   -> reach_stemacts_<ds>.npz
  --fit     (CPU): (a) old-threshold transfer accuracy, (b) recalibrated 1-D
                   threshold on the same w, (c) 5-fold CV refit LR probe,
                   (d) 5-fold CV XGBoost (nonlinear headroom on this population)
                   -> reach_stemprobe_<ds>.csv + printed summary

    PYTHONPATH=src python src/reach_stemprobe.py --dataset cities --extract --device cuda
    PYTHONPATH=src python src/reach_stemprobe.py --dataset cities --fit
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


def fit(ds):
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
    base = float(y.mean())
    acc_old, bal_old = transfer_metrics(g_old, y)
    print(f"[stemprobe] {ds}: n={len(y)} labeled stems, base rate P(TRUE)={base:.3f}")
    print(f"  old threshold  acc {acc_old:.3f}  balanced {bal_old:.3f}  "
          f"(median g: TRUE {np.median(g_old[y == 1]):+.1f}, "
          f"FALSE {np.median(g_old[y == 0]):+.1f})" if (y == 0).any() else
          f"  old threshold  acc {acc_old:.3f}  (all labels TRUE)")
    if len(np.unique(y)) > 1:
        s = X @ w
        acc_recal, _, t02_stem = fit_threshold_1d(s, y)
        print(f"  recalibrated 1-D threshold on same w: acc {acc_recal:.3f}, "
              f"t02_stem {t02_stem:+.2f} (old t02 {t02:+.2f})")
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        acc_lr = _cv_acc(lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)), X, y)
        print(f"  refit LR probe ({CV_FOLDS}-fold CV): acc {acc_lr:.3f}")
        try:
            from xgboost import XGBClassifier
            acc_xgb = _cv_acc(lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8), X, y)
            print(f"  XGBoost ({CV_FOLDS}-fold CV): acc {acc_xgb:.3f}  "
                  f"(gap vs LR {acc_xgb - acc_lr:+.3f})")
        except ImportError:
            print("  XGBoost not installed — nonlinear arm skipped (run --fit locally)")
    else:
        print("  only one label class present — refits skipped")
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap stems (smoke)")
    a = ap.parse_args()
    if a.extract:
        extract(a.dataset, a.device, a.limit)
    if a.fit:
        fit(a.dataset)
    if not (a.extract or a.fit):
        raise SystemExit("pass --extract and/or --fit")


if __name__ == "__main__":
    main()
