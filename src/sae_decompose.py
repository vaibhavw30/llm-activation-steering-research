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
import argparse
import csv
import json
import os

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


N_V64 = 4          # pooled high-gain input directions to decompose
D2_PAIR = "jtw_full_matched_mean|jtw_stem_mean"


def _unit(v):
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _renorm(M):
    M = np.asarray(M, np.float64)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def collect_vectors(ds):
    """{name: (vector, space)} where space is "src" (input side, source layer) or
    "tgt" (readout side, target layer). Optional artifacts are skipped when absent."""
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    store_names = [str(x) for x in mz["store_names"]]
    k, ks = names.index("mean_diff_tgt"), store_names.index("mean_diff_tgt")
    jtw_raw = mz["jtw"]                        # ONE decompression — never in a loop
    jtw = _renorm(jtw_raw[:, ks, :])           # float16 rows need re-normalizing
    y = np.asarray(acts["labels"]).astype(int)[:len(jtw)]
    out = {"w_mean_diff_tgt": (_unit(dirs["W"][k]), "tgt"),
           "jtw_mean": (_unit(jtw[y == 1].mean(axis=0)), "src")}
    sj = f"reach_stemjac_{ds}.npz"
    if os.path.exists(sj):
        z = np.load(sj, allow_pickle=True)
        # stemjac picks a RANDOM label-1 subset and stores stmt_index; aligning by
        # position would compare unrelated statements (reach_stemjac.py:81-83).
        idx = np.asarray(z["stmt_index"], int)
        stem = _renorm(z["jtw_stem"])
        full = jtw[idx]
        out["jtw_full_matched_mean"] = (_unit(full.mean(axis=0)), "src")
        out["jtw_stem_mean"] = (_unit(stem.mean(axis=0)), "src")
        out["common_v1"] = (_unit(np.linalg.svd(np.concatenate([full, stem]),
                                                full_matrices=False)[2][0]), "src")
    sdir = f"reach_svd_{ds}"
    if os.path.isdir(sdir):
        files = sorted(f for f in os.listdir(sdir) if f.endswith(".npz"))
        if files:
            # Pool the leading right-singular vector across statements. Averaging is
            # invalid (sign ambiguity); an SVD of the stack is sign-invariant.
            tops = np.stack([np.asarray(np.load(os.path.join(sdir, f))["V64"],
                                        np.float64)[:, 0] for f in files])
            Vh = np.linalg.svd(tops, full_matrices=False)[2]
            for j in range(min(N_V64, Vh.shape[0])):
                out[f"V64_common_{j}"] = (_unit(Vh[j]), "src")
    return out


SAE_MODEL = "google/gemma-2-2b"   # the only checkpoint sae_load.REPO has SAEs for


def check_sae_model(meta, ds):
    """Track B is gemma-2-2b-only: sae_load.REPO is hardcoded to
    google/gemma-scope-2b-pt-res, whose atoms live in the BASE model's residual basis.
    Pointing this at a run done on another checkpoint (e.g. a `refusal` run that fell
    back to google/gemma-2-2b-it) would decompose those directions in the wrong
    dictionary and report feature ids that mean nothing. Fail before any download."""
    from sae_load import REPO
    m = str(meta.get("model") or SAE_MODEL)
    if m != SAE_MODEL:
        raise SystemExit(
            f"[sae] dct_meta_{ds}.json says model={m!r}, but the SAEs come from "
            f"{REPO} and are only valid for {SAE_MODEL!r} — its decoder atoms are in "
            f"that model's residual basis. Refusing to decompose {m!r} directions in "
            f"{SAE_MODEL!r} SAE atoms.")
    return m


def run(ds, k=K_ATOMS, width="16k"):
    from sae_load import load_sae, decoder_unit
    meta = json.load(open(f"dct_meta_{ds}.json"))
    check_sae_model(meta, ds)
    layers = {"src": int(meta["source_layer"]), "tgt": int(meta["target_layer"])}
    vecs = collect_vectors(ds)
    D = {sp: decoder_unit(load_sae(layers[sp], width))
         for sp in sorted({sp for _, sp in vecs.values()})}
    rows = [("vector", "layer", "rank", "feature", "coef", "cumulative_explained")]
    supports = {}
    for name, (v, space) in sorted(vecs.items()):
        sup, coefs, resid = omp(v, D[space], k)
        supports[name] = sup.tolist()
        for r in range(len(sup)):
            c = np.linalg.lstsq(D[space][sup[:r + 1]].T, v, rcond=None)[0]
            rows.append((name, layers[space], r, int(sup[r]), f"{coefs[r]:.6g}",
                         f"{explained(v, D[space], sup[:r + 1], c):.6g}"))
        print(f"[sae] {name:>22s} @L{layers[space]:2d}: top-{k} OMP residual "
              f"{resid:.3f} (explained {1 - resid ** 2:.3f})")
    with open(f"sae_features_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    ordered = sorted(supports)
    ov = {f"{a}|{b}": jaccard(supports[a], supports[b])
          for i, a in enumerate(ordered) for b in ordered[i + 1:]
          if vecs[a][1] == vecs[b][1]}
    with open(f"sae_overlap_{ds}.json", "w") as f:
        json.dump({"k": k, "width": width, "layers": layers,
                   "supports": supports, "jaccard": ov}, f, indent=2)
    if D2_PAIR in ov:
        print(f"[sae] D2 mechanism: full-context vs stem-context J^T w share "
              f"{ov[D2_PAIR]:.3f} of their top-{k} features (Jaccard)")
    print(f"[sae] wrote sae_features_{ds}.csv and sae_overlap_{ds}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=K_ATOMS)
    ap.add_argument("--width", default="16k")
    a = ap.parse_args()
    run(a.dataset, a.k, a.width)


if __name__ == "__main__":
    main()
