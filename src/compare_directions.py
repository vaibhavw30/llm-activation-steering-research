"""
compare_directions.py — Stage 4: does unsupervised DCT rediscover the supervised
"truth" direction?

Compares, AT THE DCT SOURCE LAYER, four kinds of directions in gemma-2-2b's residual
stream (all unit-normalized; cosine similarity is the metric):

  - DCT steering vectors        : columns of dct_V_<dataset>.pt   (UNSUPERVISED)
  - mean-difference direction   : mean(true acts) - mean(false acts)   (supervised)
  - logistic-gradient direction : LogisticRegression coef / scaler.scale_   (supervised)
  - [optional] Gemma-Scope SAE features (--sae), reusing sae_comparison.py's approach

Key question: is there ANY DCT vector that aligns with the supervised truth direction
far above a random baseline? If so, DCT found the truth axis without using labels.

This is light (cosine on saved vectors) — run it locally in the geometry .venv after
bringing dct_V_<dataset>.pt back from the cluster:
    .venv/bin/python compare_directions.py --dataset cities
    .venv/bin/python compare_directions.py --dataset cities --dct-v dct_V.pt --layer 6   # smoke vectors

Layer alignment: acts_<dataset>.npz index L == hidden_states[L] == the residual stream
feeding decoder layer L. DCT's X = hidden_states[source_layer], so we compare at L=source_layer.
"""

import argparse
import json
import os
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Compare DCT vs supervised truth directions")
    p.add_argument("--dataset", required=True, help="name without .csv, e.g. cities")
    p.add_argument("--acts", default=None, help="default activations/acts_<dataset>.npz")
    p.add_argument("--dct-v", default=None, help="default dct_V_<dataset>.pt")
    p.add_argument("--layer", type=int, default=None,
                   help="residual-stream layer to compare at; default = DCT source_layer from meta, else 6")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sae", action="store_true", help="also compare against a Gemma-Scope SAE")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--sae-id", default=None, help="default layer_<layer>/width_16k/canonical")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def unit(v, axis=0, eps=1e-8):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    ds = args.dataset
    acts_path = args.acts or f"activations/acts_{ds}.npz"
    dct_path = args.dct_v or f"dct_V_{ds}.pt"

    # Resolve comparison layer: DCT source layer (from meta if present)
    layer = args.layer
    meta_path = f"dct_meta_{ds}.json"
    if layer is None and os.path.exists(meta_path):
        layer = json.load(open(meta_path))["source_layer"]
    if layer is None:
        layer = 6
    print(f"Dataset={ds} | comparing at residual-stream layer {layer}")

    # ---- Supervised directions from the geometry-of-truth activations ------
    data = np.load(acts_path, allow_pickle=True)
    X = data["activations"][layer].astype(np.float64)   # (n, d)
    y = data["labels"].astype(int)
    d = X.shape[1]
    print(f"  acts[{layer}]: {X.shape}, {int(y.sum())} true / {len(y)-int(y.sum())} false")

    v_mean = unit(X[y == 1].mean(0) - X[y == 0].mean(0))   # (d,)

    v_grad = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        v_grad = unit(lr.coef_[0] / sc.scale_)
    except Exception as e:
        print(f"  (sklearn unavailable — skipping gradient direction: {e})")

    # ---- DCT vectors -------------------------------------------------------
    V = torch.load(dct_path, map_location="cpu").float().numpy()   # (d, num_factors)
    V = unit(V, axis=0)
    num_factors = V.shape[1]
    print(f"  DCT V: {V.shape} from {dct_path}")
    assert V.shape[0] == d, f"dim mismatch: acts d={d} vs V d={V.shape[0]} (wrong --layer?)"

    # ---- Random baseline: max |cos| of a truth direction with num_factors random unit vectors
    def max_abs_cos(target, mat):  # mat: (d, k) columns
        c = mat.T @ target          # (k,)
        return c, np.abs(c)

    R = unit(np.random.randn(d, num_factors), axis=0)

    def summarize(name, target):
        cos_dct, abscos_dct = max_abs_cos(target, V)
        cos_rnd, abscos_rnd = max_abs_cos(target, R)
        best = int(np.argmax(abscos_dct))
        return {
            "target": name,
            "best_dct_vec": best,
            "best_cos": float(cos_dct[best]),          # signed
            "best_abs_cos": float(abscos_dct[best]),
            "random_max_abs_cos": float(abscos_rnd.max()),
            "ratio_vs_random": float(abscos_dct[best] / (abscos_rnd.max() + 1e-9)),
            "_per_vec_abscos": abscos_dct,
        }

    rows = [summarize("mean_diff", v_mean)]
    if v_grad is not None:
        rows.append(summarize("logistic_grad", v_grad))

    # Reference: how much do the two SUPERVISED directions agree?
    sup_pair = float(v_mean @ v_grad) if v_grad is not None else None

    # ---- Report ------------------------------------------------------------
    print("\n=== DCT vs supervised truth direction ===")
    print(f"{'target':14s} {'best DCT#':>9s} {'cos':>8s} {'|cos|':>8s} "
          f"{'rand|cos|':>10s} {'x random':>9s}")
    for r in rows:
        print(f"{r['target']:14s} {r['best_dct_vec']:>9d} {r['best_cos']:>+8.3f} "
              f"{r['best_abs_cos']:>8.3f} {r['random_max_abs_cos']:>10.3f} "
              f"{r['ratio_vs_random']:>8.1f}x")
    if sup_pair is not None:
        print(f"\nReference: cos(mean_diff, logistic_grad) = {sup_pair:+.3f} "
              f"(how much the two SUPERVISED directions agree)")

    # Save per-vector cosines
    import csv
    out_csv = f"compare_{ds}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["dct_vector", "abscos_mean_diff"] + (["abscos_logistic_grad"] if v_grad is not None else [])
        w.writerow(header)
        for i in range(num_factors):
            row = [i, f"{rows[0]['_per_vec_abscos'][i]:.4f}"]
            if v_grad is not None:
                row.append(f"{rows[1]['_per_vec_abscos'][i]:.4f}")
            w.writerow(row)
    print(f"\nSaved per-vector cosines -> {out_csv}")

    # ---- Subspace test: does truth live in the SPAN of the 512 DCT vectors? -
    # max-cosine asks "is truth ONE DCT vector?"; this asks "is truth a COMBINATION?"
    Q, _ = np.linalg.qr(V)               # orthonormal basis of the DCT column space
    k = V.shape[1]
    def proj_frac(t):                    # fraction of a unit vector's variance inside span(V)
        return float(((Q.T @ t) ** 2).sum())
    rand_fr = float(np.mean([proj_frac(unit(np.random.randn(d))) for _ in range(200)]))
    print("\n=== Subspace test: is truth in the span of the 512 DCT directions? ===")
    print(f"  chance level: a random unit vector lands {rand_fr:.3f} of its variance "
          f"in the span (~k/d = {k/d:.3f})")
    for r in rows:
        t = v_mean if r["target"] == "mean_diff" else v_grad
        pf = proj_frac(t)
        verdict = ("IN the DCT span" if pf >= 2 * rand_fr else
                   "NOT in the DCT span (~chance)" if pf <= 1.3 * rand_fr else
                   "partially in span")
        print(f"  {r['target']:14s}: {pf:.3f} of variance in span  "
              f"({pf/rand_fr:.1f}x chance) -> {verdict}")

    # ---- Interpretation ----------------------------------------------------
    print("\n=== Interpretation ===")
    for r in rows:
        verdict = ("STRONGLY aligned — DCT appears to rediscover this truth direction"
                   if r["ratio_vs_random"] >= 3 and r["best_abs_cos"] >= 0.2 else
                   "weakly/uncertainly aligned (near random baseline)"
                   if r["ratio_vs_random"] < 2 else
                   "moderately aligned")
        print(f"- vs {r['target']}: best DCT vector #{r['best_dct_vec']} has |cos|="
              f"{r['best_abs_cos']:.3f} ({r['ratio_vs_random']:.1f}x the random max of "
              f"{r['random_max_abs_cos']:.3f}) -> {verdict}.")

    # ---- Optional SAE comparison (reuses sae_comparison.py approach) --------
    if args.sae:
        compare_sae(args, layer, V, v_mean, v_grad, d, num_factors)


def compare_sae(args, layer, V, v_mean, v_grad, d, num_factors):
    """Compare DCT vectors and the truth direction against Gemma-Scope SAE features.
    Mirrors sae_comparison.py: normalize SAE decoder columns, take max cosine."""
    try:
        from sae_lens import SAE
    except ImportError:
        print("\n[--sae] sae_lens not installed. Install with: pip install sae-lens "
              "(needs GPU + internet to download Gemma-Scope). Skipping.")
        return
    sae_id = args.sae_id or f"layer_{layer}/width_16k/canonical"
    print(f"\n=== SAE comparison ({args.sae_release} / {sae_id}) ===")
    sae, _, _ = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id)
    import torch.nn.functional as F
    W = sae.W_dec.clone().detach().t().float()          # (d, n_features)
    W = F.normalize(W, dim=0).cpu().numpy()
    # best SAE feature aligned with the truth direction
    for name, t in [("mean_diff", v_mean)] + ([("logistic_grad", v_grad)] if v_grad is not None else []):
        c = np.abs(W.T @ t)
        print(f"  best SAE feature vs {name}: |cos|={c.max():.3f} (feature #{int(c.argmax())})")
    # best SAE feature aligned with each DCT vector (max over features), summary stat
    M = np.abs(V.T @ W)            # (num_factors, n_features)
    print(f"  DCT<->SAE: mean of per-DCT max |cos| = {M.max(axis=1).mean():.3f} "
          f"(how SAE-like the DCT vectors are)")


if __name__ == "__main__":
    main()
