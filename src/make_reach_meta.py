"""make_reach_meta.py — Horizon-1 1.1 Task A4: dct_meta_<ds>.json without DCT training.

The reach pipeline reads source/target layers, the model id, and input_scale from
dct_meta_<ds>.json. The truth datasets got theirs from run_dct_data.py as a side
effect of fitting DCT factors. The minimal refusal control skips that fit, so this
writes the same file from a probe-accuracy sweep, and calibrate_scale.py fills in
input_scale with the same SteeringCalibrator run_dct_data.py would have used.

Layer rule (pre-registered): source_layer = the EARLIEST layer whose linear-probe
accuracy is within ACC_TOL of the best, among layers with MIN_SOURCE_LAYER <= L and
L + HOP_DEPTH + SLICE_MARGIN <= max hidden-state index. Plain argmax is wrong here:
harmful-vs-harmless instructions are lexically separable, so accuracy can saturate at
layer 0 and argmax-with-earliest-tiebreak would pick the embedding layer.
Earliest-within-tolerance also matches Horizon-0 D4, which found controllability peaks
at layers 5-9.

The SLICE_MARGIN term exists because dct.SlicedModel.forward does
`self.model.model.layers = self.L[start_layer:end_layer+2]` (src/dct.py:238): if
`end_layer+2` exceeds `len(self.L)` the slice is silently shorter than the config
claims, and the Jacobian describes the wrong map. `end_layer` here is `target_layer`,
so the eligibility bound must keep `target_layer + 2 <= max hidden-state index`, not
just `target_layer <= max hidden-state index`.

  target_layer = source_layer + 9   (cities 11->20, common_claim 13->22)

    PYTHONPATH=src python src/make_reach_meta.py --dataset refusal \\
        --model google/gemma-2-2b-it
"""
import argparse
import csv
import json
import os

HOP_DEPTH = 9
MIN_SOURCE_LAYER = 5
ACC_TOL = 0.005
NUM_SAMPLES = 64          # matches run_dct_data.py's calibration population
TOKEN_IDXS = "-3:"
SLICE_MARGIN = 2          # dct.SlicedModel.forward slices self.L[start:end+2]
                          # (src/dct.py:238); target_layer+2 must stay in bounds.


def layer_sweep(acts, labels):
    """Per-layer linear-probe test accuracy. acts (L, n, d), labels (n,) ->
    [{"layer": int, "linear_acc": float}, ...]. Mirrors analyze.py's linear arm
    (StandardScaler, LogisticRegression(max_iter=2000), 80/20 stratified split,
    random_state=42) but skips the XGBoost arm, which layer choice does not use and
    which would drag xgboost into the cluster environment. Uses acts in their native
    dtype (no float64 upcast) to match analyze.py:46's `X = acts[layer]` exactly,
    since ACC_TOL is a threshold comparison and the two arms must agree bit-for-bit
    on what "the same accuracy" means."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(labels).astype(int)
    out = []
    for L in range(np.asarray(acts).shape[0]):
        X = np.asarray(acts[L])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        out.append({"layer": L, "linear_acc": float(lr.score(sc.transform(Xte), yte))})
        print(f"[meta] layer {L:2d}: linear acc {out[-1]['linear_acc']:.3f}", flush=True)
    return out


def pick_source_layer(rows, max_layer, hop=HOP_DEPTH, min_layer=MIN_SOURCE_LAYER,
                      tol=ACC_TOL, slice_margin=SLICE_MARGIN):
    """rows: dicts with 'layer' and 'linear_acc'. See the module docstring for why
    this is earliest-within-tolerance and not argmax, and for why eligibility needs
    `+ slice_margin` on top of the hop (src/dct.py:238)."""
    elig = [(int(r["layer"]), float(r["linear_acc"])) for r in rows
            if min_layer <= int(r["layer"])
            and int(r["layer"]) + hop + slice_margin <= int(max_layer)]
    if not elig:
        raise SystemExit(
            f"[meta] no layer in [{min_layer}, {int(max_layer) - hop - slice_margin}] "
            f"leaves room for a {hop}-layer hop within the {slice_margin}-layer "
            f"SlicedModel margin (src/dct.py:238)")
    best = max(a for _, a in elig)
    return min(L for L, a in elig if a >= best - tol)


def meta_dict(ds, model, src, hop=HOP_DEPTH):
    """The dct_meta schema run_dct_data.py writes; input_scale is None until
    calibrate_scale.py fills it."""
    return {"dataset": ds, "model": model, "source_layer": int(src),
            "target_layer": int(src) + hop, "num_factors": None, "num_iters": None,
            "num_samples": NUM_SAMPLES, "input_scale": None,
            "token_idxs": TOKEN_IDXS, "balanced": True}


def _acts_path(ds):
    """analyze.py writes/reads acts_<ds>.npz in the cwd; funnel_utils.load_acts reads
    it from activations/. Accept either, so this runs before or after the file moves."""
    for p in (f"activations/acts_{ds}.npz", f"acts_{ds}.npz"):
        if os.path.exists(p):
            return p
    raise SystemExit(f"[meta] no acts_{ds}.npz in ./ or ./activations/")


def run(ds, model, results_path=None, force=False):
    meta_path = f"dct_meta_{ds}.json"
    # Never re-run or overwrite an existing truth artifact (plan Global Constraints):
    # dct_meta_cities.json / dct_meta_common_claim_true_false.json carry an
    # input_scale that took a GPU calibration run to produce and cannot be
    # regenerated from this script. Check this before touching anything else.
    if os.path.exists(meta_path) and not force:
        raise SystemExit(
            f"[meta] refusing to overwrite existing {meta_path} — it may be an "
            f"irreplaceable truth artifact (e.g. a calibrated input_scale). "
            f"Pass --force if you really mean to regenerate it.")
    import numpy as np
    acts_path = _acts_path(ds)
    z = np.load(acts_path, allow_pickle=True)
    # `extract --model` and `make_reach_meta --model` are two independently
    # operator-typed literals. extract.py records the checkpoint it ran on in the npz
    # (extract.py:152), so cross-check it: the meta is AUTHORITATIVE for every
    # downstream reach stage, and a meta describing a different model than the
    # activations the source layer was chosen from silently invalidates the whole run.
    if "model" in getattr(z, "files", ()):
        acts_model = str(z["model"])
        if acts_model != str(model):
            raise SystemExit(
                f"[meta] model mismatch: {acts_path} was extracted with "
                f"{acts_model!r} but --model says {str(model)!r} — the meta is "
                f"authoritative for every downstream reach stage, so writing it "
                f"would describe a different checkpoint than these activations. "
                f"Re-run extract.py and make_reach_meta.py with the SAME --model.")
    else:
        print(f"[meta] warning: {acts_path} has no 'model' member (pre-dates "
              f"extract.py's model key) — cannot cross-check against --model "
              f"{str(model)!r}")
    acts = z["activations"]
    max_layer = acts.shape[0] - 1
    path = results_path or f"results_{ds}.csv"
    # Which of the two sweep sources was used decides `source_layer`, and the model
    # cross-check above validates only the npz — a results CSV carries no model field, so
    # a leftover base-model results_<ds>.csv would pick the source layer from base-model
    # accuracies on an `-it` re-prep while the npz check passes clean. Latent today
    # (nothing in the refusal path writes results_refusal.csv), so print the provenance
    # rather than change the preference: the choice must be visible in the job log.
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"[meta] layer sweep: rows READ FROM EXISTING {path} ({len(rows)} rows), "
              f"NOT swept from {acts_path}. source_layer therefore comes from THAT "
              f"file's accuracies, and the --model cross-check above does NOT validate "
              f"it (a results CSV records no model) — delete {path} to re-sweep from "
              f"the activations if it may be from a different checkpoint")
    else:
        rows = layer_sweep(acts, z["labels"])
        print(f"[meta] layer sweep: rows COMPUTED from {acts_path} ({len(rows)} rows); "
              f"no {path} present")
    src = pick_source_layer(rows, max_layer)
    m = meta_dict(ds, model, src)
    with open(meta_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[meta] wrote {meta_path}: src={m['source_layer']} -> "
          f"tgt={m['target_layer']} (max hidden-state index {max_layer}), "
          f"model={model}")
    print("[meta] input_scale is null — run calibrate_scale.py next")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--results", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing dct_meta_<ds>.json")
    a = ap.parse_args()
    run(a.dataset, a.model, a.results, a.force)


if __name__ == "__main__":
    main()
