# src/audit_layer_sweep.py
"""S2 — layer sweep of readout against controllability.

The PI's question: sweep over input layers, do the same mapping from different earlier layers,
and check whether the same linear relationship holds and whether perturbation affects the
halfspace metric the same way at each depth.

Reads committed artifacts only. No GPU: all 27 layers are already cached in mag_acts_<ds>.npz.

    ./.venv/bin/python src/audit_layer_sweep.py --dataset cities

Two feature spaces are swept, because the project uses two:
  * Direct     = A_p         the raw last-token statement activation, the standard readout
  * InputDelta = A_Qp - A_p  the MAG feature the steering direction u_Q_gold was built from
                             (src/mag/directions.py:52), so it is the space in which the
                             cosine against the actual steering vector is meaningful

NEVER index into the loaded npz inside the layer loop. Both arrays are materialised once up
front; re-indexing the NpzFile re-inflates the whole (27, n, 2304) array on every access.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_FRAC = 0.2
# A layer counts as reading truth "as well as the best layer" if it is within this much of the
# dataset's peak accuracy. Peak-relative, because cities saturates near 0.99 and common_claim
# tops out near 0.73; any absolute threshold is right for one dataset and wrong for the other.
KNEE_TOL = 0.02

COL_DIRECT = "#4477aa"
COL_DELTA = "#cc6677"
COL_MARGIN = "#117733"
COL_EPS = "#999933"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def probe(X, y):
    """80/20 stratified split, standardize on train only, logistic probe. Returns acc, auc."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    Xte_s = sc.transform(Xte)
    return clf.score(Xte_s, yte), roc_auc_score(yte, clf.decision_function(Xte_s))


def class_mean_diff(feats, labels):
    """u = mean(negatives) - mean(positives). Matches src/mag/directions.py:33 exactly."""
    labels = np.asarray(labels).astype(int)
    return feats[labels == 0].mean(axis=0) - feats[labels == 1].mean(axis=0)


def run(ds):
    t0 = time.time()
    md = np.load(f"mag_dir_{ds}.npz")
    inject_layer = int(md["layer"])
    u_steer = unit(np.asarray(md["u_Q_gold_unit"], np.float64))
    print(f"[S2] {ds}: injection layer {inject_layer}, steering direction u_Q_gold")

    z = np.load(f"mag_acts_{ds}.npz", allow_pickle=True)
    # One decompression each. Everything below slices these in-memory arrays.
    A_p = np.asarray(z["A_p"])
    A_Qp = np.asarray(z["A_Qp"])
    y = np.asarray(z["labels"]).astype(int)
    n_layers = A_p.shape[0]
    print(f"[S2] loaded A_p {A_p.shape} and A_Qp {A_Qp.shape}, "
          f"{y.sum()} true / {len(y) - y.sum()} false, {time.time() - t0:.1f}s")

    rows, dirs = [], []
    for L in range(n_layers):
        direct = A_p[L].astype(np.float64)
        delta = A_Qp[L].astype(np.float64) - direct

        acc_d, auc_d = probe(direct, y)
        acc_i, auc_i = probe(delta, y)

        u_L = unit(class_mean_diff(delta, y))
        cos_steer = float(u_L @ u_steer)
        dirs.append(u_L)

        rows.append({
            "layer": L,
            "acc_direct": acc_d, "auc_direct": auc_d,
            "acc_inputdelta": acc_i, "auc_inputdelta": auc_i,
            "cos_meandiff_vs_steer": cos_steer,
            "abscos_meandiff_vs_steer": abs(cos_steer),
            "norm_direct_median": float(np.median(np.linalg.norm(direct, axis=1))),
            "norm_delta_median": float(np.median(np.linalg.norm(delta, axis=1))),
        })
        print(f"  L{L:02d}  direct acc={acc_d:.3f} auc={auc_d:.3f} | "
              f"inputdelta acc={acc_i:.3f} auc={auc_i:.3f} | "
              f"cos_to_steer={cos_steer:+.3f}  [{time.time() - t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)

    # token_jac_<ds>.csv covers layers 0..25; A_p covers 0..26. Left join keeps every probed
    # layer and leaves the controllability columns NaN where the Jacobian sweep did not reach.
    jac = pd.read_csv(f"token_jac_{ds}.csv")
    out = out.merge(
        jac[["layer", "m_all_median", "m_last_median", "eps_all_median", "eps_last_median",
             "broadcast_gain", "eps_all_p10", "eps_all_p90"]],
        on="layer", how="left")
    out["inject_layer"] = inject_layer

    path = f"audit_layers_{ds}.csv"
    out.to_csv(path, index=False)
    print(f"[S2] wrote {path}  ({time.time() - t0:.0f}s total)")

    # Cross-layer cosine matrix of the per-layer mean-difference directions. This is the
    # sharpest form of the PI's "same linear relationship" question: if every layer past the
    # readout knee decodes truth equally well AND does it along the same axis, this matrix is
    # near 1 in that block. If accuracy is flat while the matrix is not, the layers agree on
    # the label and disagree on the direction.
    D = np.stack(dirs)
    C = D @ D.T
    np.save(f"audit_layer_cos_{ds}.npy", C)
    print(f"[S2] wrote audit_layer_cos_{ds}.npy  {C.shape}")
    return out, inject_layer, C


def plot(ds, out, inject_layer):
    L = out["layer"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    def mark(ax):
        ax.axvline(inject_layer, color="#888888", ls=":", lw=1.4, zorder=0)
        ax.grid(lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    ax = axes[0]
    ax.plot(L, out["acc_direct"], "-o", ms=4, lw=2, color=COL_DIRECT, label="Direct (A_p) acc")
    ax.plot(L, out["auc_direct"], "--", lw=1.4, color=COL_DIRECT, alpha=0.7, label="Direct AUC")
    ax.plot(L, out["acc_inputdelta"], "-o", ms=4, lw=2, color=COL_DELTA,
            label="InputDelta acc (steering space)")
    ax.plot(L, out["auc_inputdelta"], "--", lw=1.4, color=COL_DELTA, alpha=0.7,
            label="InputDelta AUC")
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_ylabel("held-out readout")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"S2 layer sweep — {ds}   (dotted line = layer {inject_layer}, "
                 f"where both steering arms inject)")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    mark(ax)
    # Both curves usually peak at the same layer, so stagger the labels downward into the
    # empty lower half of the axis rather than upward into the title.
    for dy, (c, col) in zip((-16, -30), (("acc_direct", COL_DIRECT),
                                         ("acc_inputdelta", COL_DELTA))):
        k = int(out[c].idxmax())
        ax.annotate(f"peak L{out.layer[k]} = {out[c][k]:.3f}",
                    (out.layer[k], out[c][k]), textcoords="offset points",
                    xytext=(0, dy), ha="center", va="top", fontsize=8, color=col)

    ax = axes[1]
    ax.plot(L, out["cos_meandiff_vs_steer"], "-o", ms=4, lw=2, color="#882255")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(1.0, color="#888888", ls="--", lw=0.7)
    ax.set_ylabel("cos(layer mean-diff, steering dir)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("is the truth direction the same object at every depth?", fontsize=10)
    mark(ax)

    ax = axes[2]
    ax.plot(L, out["m_all_median"], "-o", ms=4, lw=2, color=COL_MARGIN,
            label="controllability margin m (median)")
    ax.set_ylabel("margin m", color=COL_MARGIN)
    ax.tick_params(axis="y", labelcolor=COL_MARGIN)
    ax2 = ax.twinx()
    ax2.plot(L, out["eps_all_median"], "-s", ms=4, lw=2, color=COL_EPS,
             label="certified budget eps (median)")
    ax2.fill_between(L, out["eps_all_p10"], out["eps_all_p90"], color=COL_EPS, alpha=0.15)
    ax2.set_ylabel("budget eps (p10 to p90 shaded)", color=COL_EPS)
    ax2.tick_params(axis="y", labelcolor=COL_EPS)
    ax.set_xlabel("layer")
    ax.set_title("cost of actuating the TOKEN DECISION from this layer "
                 "(token_jac, layers 0 to 25; broadcast convention)", fontsize=10)
    mark(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")

    fig.tight_layout()
    p = f"plot_s2_layers_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"[S2] wrote {p}")


def plot_cosine_matrix(ds, out, inject_layer, C):
    """Cross-layer agreement of the per-layer truth directions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.axhline(inject_layer, color="k", ls=":", lw=1.0)
    ax.axvline(inject_layer, color="k", ls=":", lw=1.0)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title("cos between per-layer truth directions", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine")

    # The diagnostic slice: among layers that all decode truth at ceiling, how much do their
    # directions agree with each other?
    ax = axes[1]
    # "Layers that decode truth as well as the best layer does." A fixed 0.95 threshold is
    # wrong here: common_claim never reaches it, and a boolean idxmax on an all-False series
    # silently returns layer 0. Anchor to the dataset's own peak instead. The argmax layer
    # always satisfies this, so the knee always exists.
    peak = float(out["acc_direct"].max())
    thresh = peak - KNEE_TOL
    knee = int(out.index[out["acc_direct"] >= thresh][0])
    block = C[knee:, knee:]
    off = block[~np.eye(len(block), dtype=bool)]

    # Cosine against layer separation. A flat line near 1 would mean these layers share one
    # truth axis. A decay means the direction rotates smoothly with depth while the accuracy
    # does not move, which is a different claim from "the layers disagree at random".
    seps = list(range(1, len(block)))
    meds = [float(np.median(np.diagonal(block, offset=d))) for d in seps]
    los = [float(np.percentile(np.diagonal(block, offset=d), 10)) for d in seps]
    his = [float(np.percentile(np.diagonal(block, offset=d), 90)) for d in seps]
    ax.plot(seps, meds, "-o", ms=4, lw=2, color="#4477aa", label="median")
    ax.fill_between(seps, los, his, color="#4477aa", alpha=0.18, label="p10 to p90")
    ax.axhline(1.0, color="#888888", ls="--", lw=0.9, label="identical")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlabel("separation in layers")
    ax.set_ylabel("cos between the two layers' truth directions")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(f"layers {knee} to {len(C) - 1}, all at acc >= {thresh:.3f}; "
                 f"median over all pairs {np.median(off):.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    fig.suptitle(f"S2 direction drift — {ds}   "
                 f"(same accuracy at every depth does not mean the same direction)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"plot_s2_layer_cosine_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"[S2] wrote {p}")
    return knee, off, thresh, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    out, inject_layer, C = run(a.dataset)
    plot(a.dataset, out, inject_layer)
    knee, off, thresh, peak = plot_cosine_matrix(a.dataset, out, inject_layer, C)

    print(f"\n=== S2 {a.dataset} ===")
    best_d = int(out["acc_direct"].idxmax())
    best_i = int(out["acc_inputdelta"].idxmax())
    print(f"readout peak, Direct:      layer {out.layer[best_d]} at "
          f"{out.acc_direct[best_d]:.3f} (AUC {out.auc_direct[best_d]:.3f})")
    print(f"readout peak, InputDelta:  layer {out.layer[best_i]} at "
          f"{out.acc_inputdelta[best_i]:.3f} (AUC {out.auc_inputdelta[best_i]:.3f})")
    print(f"injection layer used by both steering arms: {inject_layer} "
          f"(Direct acc {out.acc_direct[inject_layer]:.3f}, "
          f"InputDelta acc {out.acc_inputdelta[inject_layer]:.3f})")

    jac = out.dropna(subset=["m_all_median"])
    bm = int(jac["m_all_median"].idxmax())
    be = int(jac["eps_all_median"].idxmin())
    print(f"controllability margin peak: layer {out.layer[bm]} at m={out.m_all_median[bm]:.3f}")
    print(f"cheapest certified budget:   layer {out.layer[be]} at "
          f"eps={out.eps_all_median[be]:.3f}")

    print(f"readout knee (first layer within {KNEE_TOL} of the peak {peak:.3f}, "
          f"so acc >= {thresh:.3f}): layer {knee}; among layers {knee}..{len(C)-1} "
          f"the pairwise direction cosine has median {np.median(off):.3f}, "
          f"min {off.min():.3f}, max {off.max():.3f}")

    c = out["abscos_meandiff_vs_steer"]
    print(f"|cos| to the steering direction: min {c.min():.3f} (L{out.layer[int(c.idxmin())]}), "
          f"max {c.max():.3f} (L{out.layer[int(c.idxmax())]}), "
          f"at injection layer {c[inject_layer]:.3f}")


if __name__ == "__main__":
    main()
