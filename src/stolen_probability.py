"""Stolen-probability check for gemma-2-2b (Demeter et al. ACL 2020, arXiv:2005.02433).

A token j can be the argmax for SOME final activation iff its effective unembedding
row is a VERTEX of the convex hull of all rows. Formally, achievability is

    max_{||z||<=1} min_{k!=j} (W[j] - W[k]) . z   >  0

By minimax (compact convex), that value equals

    min_{lambda in simplex} || sum_k lambda_k (W[j] - W[k]) ||
  = dist(0, conv{W[j] - W[k]})
  = dist(W[j], conv{W[k] : k != j})

so the test is exactly "is W[j] outside the convex hull of the other rows".

Two rigorous certificates, both obtainable:
  ACHIEVABLE      we exhibit a witness z whose full-vocab argmax is j.
  NOT ACHIEVABLE  Frank-Wolfe drives dist(0, conv) to ~0, i.e. we exhibit convex
                  weights reconstructing W[j] from the other rows.

Effective unembedding: gemma applies final RMSNorm with gain gamma before the tied
unembedding, so logits_j = (E[j] * (1+gamma)) . (h/rms(h)). Hence W = E * (1+gamma).
Logit softcapping is a strictly increasing coordinatewise tanh, so it preserves the
argmax ordering and is irrelevant to this test.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer

import os

SNAP = (sys.argv[1] if len(sys.argv) > 1 else glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*"))[0])
FW_ITERS = 400


def load_W():
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    def get(key):
        with safe_open(f"{SNAP}/{idx[key]}", framework="np") as f:
            return f.get_tensor(key)
    E = np.asarray(get("model.embed_tokens.weight"), np.float32)
    gamma = np.asarray(get("model.norm.weight"), np.float32)
    print(f"[W] E {E.shape} gamma {gamma.shape} "
          f"gamma range [{gamma.min():.3f}, {gamma.max():.3f}]")
    return E * (1.0 + gamma)[None, :]


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def stage1_witness(W, cand):
    """z = unit(W[j]) is the natural witness. One matmul resolves most tokens."""
    Z = unit(W[cand].astype(np.float32))              # (J, d)
    U = W @ Z.T                                        # (V, J)
    am = U.argmax(axis=0)
    top2 = np.partition(U, -2, axis=0)[-2:, :]
    gap = top2[1] - top2[0]                            # top1 - top2 at this z
    return am == np.asarray(cand), am, gap


def frank_wolfe_batch(W, js, iters=FW_ITERS):
    """Batched: for every j in js, min_{lambda in simplex} ||sum_k lambda_k (W[j]-W[k])||.

    Each FW step needs argmin_k (W[j]-W[k]).p = argmax_k W[k].p, so one GEMM
    W @ P.T of shape (V, J) serves ALL candidates per iteration. That is the whole
    speedup over looping: gemm saturates BLAS threads, gemv does not.
    Returns P (J, d); ||P[i]|| is the distance, and P[i] doubles as the witness."""
    js = np.asarray(js, int)
    J = len(js)
    Wj = W[js].astype(np.float32)                       # (J, d)
    S = W @ unit(Wj).T                                  # (V, J)
    S[js, np.arange(J)] = -np.inf                       # never pick k == j
    P = Wj - W[S.argmax(axis=0)]                        # (J, d), a point in each hull
    for t in range(iters):
        S = W @ P.T                                     # (V, J)  <- the one GEMM
        S[js, np.arange(J)] = -np.inf
        A = Wj - W[S.argmax(axis=0)]                    # FW vertices
        D = A - P
        dd = np.einsum("ij,ij->i", D, D)
        g = np.clip(-np.einsum("ij,ij->i", P, D) / np.maximum(dd, 1e-30), 0.0, 1.0)
        P = P + g[:, None] * D
        if t % 50 == 0:
            nn = np.linalg.norm(P, axis=1)
            print(f"  [fw] iter {t:4d}  dist median {np.median(nn):.5f} "
                  f"min {nn.min():.5f} max {nn.max():.5f}", flush=True)
        if float(g.max()) <= 1e-12:
            print(f"  [fw] converged at iter {t}", flush=True)
            break
    return P


def first_token(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return int(ids[0])


def main():
    W = load_W()
    V, d = W.shape
    norms = np.linalg.norm(W, axis=1)
    print(f"[W] effective unembedding {W.shape}; "
          f"row norm median {np.median(norms):.4f} "
          f"p1 {np.percentile(norms,1):.4f} p99 {np.percentile(norms,99):.4f}")

    tok = AutoTokenizer.from_pretrained(SNAP)
    df = pd.read_csv("got_datasets/cities.csv")
    true_c = sorted(df["correct_country"].astype(str).unique())
    false_c = sorted(set(df.loc[df["label"] == 0, "country"].astype(str)) - set(true_c))
    print(f"[data] {len(true_c)} true countries, {len(false_c)} distinct false-only")

    groups = []
    seen = {}
    for name, words in (("true_country", true_c), ("false_country", false_c)):
        for wd in words:
            t = first_token(tok, wd)
            if t in seen:
                continue
            seen[t] = True
            groups.append((name, wd, t))
    rng = np.random.default_rng(0)
    for t in rng.choice(V, 64, replace=False):
        if int(t) not in seen:
            seen[int(t)] = True
            groups.append(("random_control", tok.decode([int(t)]), int(t)))
    for t in np.argsort(norms)[:16]:
        if int(t) not in seen:
            seen[int(t)] = True
            groups.append(("lowest_norm", tok.decode([int(t)]), int(t)))

    cand = [t for _, _, t in groups]
    print(f"[cand] {len(cand)} distinct first-tokens under test")

    ok, am, gap = stage1_witness(W, cand)
    print(f"[stage1] self-witness resolves {int(ok.sum())}/{len(cand)} as ACHIEVABLE",
          flush=True)

    todo = [i for i in range(len(groups)) if not ok[i]]
    P = frank_wolfe_batch(W, [groups[i][2] for i in todo]) if todo else None
    fw = {}
    if todo:
        # rigorous re-check: does the FW witness actually make the token the argmax?
        U = W @ unit(P).T.astype(np.float32)            # (V, len(todo))
        wins = U.argmax(axis=0)
        dists = np.linalg.norm(P, axis=1)
        for n, i in enumerate(todo):
            fw[i] = (float(dists[n]), int(wins[n]) == groups[i][2])

    rows = []
    for i, (grp, word, t) in enumerate(groups):
        pct = float((norms < norms[t]).mean() * 100.0)
        if ok[i]:
            rows.append((grp, word, t, "ACHIEVABLE", float(gap[i]), pct, "self-witness"))
            continue
        dist, win = fw[i]
        verdict = "ACHIEVABLE" if (dist > 1e-4 and win) else (
            "NOT_ACHIEVABLE" if dist <= 1e-4 else "UNRESOLVED")
        rows.append((grp, word, t, verdict, dist, pct, f"FW dist={dist:.5f} win={win}"))
        print(f"  [fw] {grp:15s} {word!r:22s} tok={t:6d} -> {verdict} "
              f"dist={dist:.5f} normpct={pct:.1f}", flush=True)

    out = pd.DataFrame(rows, columns=["group", "word", "token_id", "verdict",
                                      "margin_or_dist", "norm_percentile", "note"])
    out.to_csv("stolen_probability_cities.csv", index=False)
    print("\n=== VERDICT COUNTS BY GROUP ===")
    print(out.pivot_table(index="group", columns="verdict", values="token_id",
                          aggfunc="count", fill_value=0))
    print("\n=== NORM PERCENTILE BY GROUP ===")
    print(out.groupby("group")["norm_percentile"].describe()[["count", "mean", "min", "max"]])
    print("\nwrote stolen_probability_cities.csv")


if __name__ == "__main__":
    main()
