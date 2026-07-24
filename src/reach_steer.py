"""reach_steer.py — Phase 3: steer along J^T w and behaviorally judge (spec §4 P3).

Arm "mean":     dataset-mean unit J^T w for w in {mean_diff_tgt, best truth-subspace
                member}, swept at ±{0.5,1,1.5,2} x median eps*(w), capped at
                1.5 x input_scale, on the 32 FACTUAL_PROMPTS.
Arm "per_stmt": per-statement unit J^T w (w = mean_diff_tgt) on 200 label-1
                statements, prompts = the statement minus its final word, swept at
                ±{1,2} x that statement's own eps*.

Every steered generation also logs the target-layer probe readout
g_read = w.h_tgt(last prompt token) - t02(w), so the deliverable 2x2
(readout moved x behavior moved) is directly tabulable. "Readout flips, model
still won't lie" is a first-class outcome (LiSeCo activation->behavior gap).

    PYTHONPATH=src python src/reach_steer.py --dataset cities --device cuda --arm mean
    PYTHONPATH=src python src/reach_steer.py --dataset cities --device cpu --arm mean --limit 2   # smoke
"""
import argparse
import csv
import json

import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from steer_supervised import FACTUAL_PROMPTS
from reach_hop import load_meta, last_nonpad_index

MEAN_FRACS = [0.5, 1.0, 1.5, 2.0]
STMT_FRACS = [1.0, 2.0]
N_PER_STMT, SEED = 200, 42
MIN_STEM_WORDS = 4
MAX_NEW_TOKENS = 8


def scale_grid(eps_star, input_scale, fracs):
    """Signed steering scales bracketing the reachability boundary, capped at
    1.5 x input_scale; includes the 0 baseline."""
    cap = 1.5 * float(input_scale)
    mags = sorted({min(f * float(eps_star), cap) for f in fracs if eps_star > 0})
    return [0.0] + [s * m for m in mags for s in (+1.0, -1.0)]


def stem_of(statement):
    words = str(statement).rstrip(" .").split()
    if len(words) < MIN_STEM_WORDS:
        return None
    return " ".join(words[:-1])


def read_g(model, tok, prompt, tgt_layer, w_t, t02, dev):
    """Target-layer probe reading at the last prompt token (steering hook active)."""
    enc = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    last = last_nonpad_index(enc["attention_mask"])[0]
    h = hs[tgt_layer][0, last]
    return float(h @ w_t) - float(t02)


def _load_common(ds):
    src, tgt, input_scale = load_meta(ds)
    summ = json.load(open(f"reach_summary_{ds}.json"))
    dirs = np.load(f"reach_dirs_{ds}.npz", allow_pickle=True)
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    names = [str(x) for x in dirs["names"]]
    store_names = [str(x) for x in mz["store_names"]]
    return src, tgt, input_scale, summ, dirs, mz, acts, names, store_names


def arm_mean(ds, device, limit=0):
    src, tgt, input_scale, summ, dirs, mz, acts, names, store_names = _load_common(ds)
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    lab1 = y == 1
    w_names = ["mean_diff_tgt"]
    best = summ.get("best_sub_name", "")
    if best and best != "mean_diff_tgt":
        w_names.append(best)
    prompts = FACTUAL_PROMPTS[:limit] if limit else FACTUAL_PROMPTS
    tok, model, dev = su.load_model(device)
    rows = [("direction", "scale", "prompt", "completion")]
    readout = [("direction", "scale", "prompt", "g_read")]
    for wn in w_names:
        k = names.index(wn)
        ks = store_names.index(wn)
        jtw = np.asarray(mz["jtw"], np.float64)[lab1, ks, :]      # unit rows
        mean_dir = unit(jtw.mean(axis=0))
        w_vec = torch.tensor(np.asarray(dirs["W"][k], np.float32))
        t02 = float(dirs["thresh02"][k])
        eps_star = summ["directions"][wn]["median_eps_star"]
        vec64 = mean_dir
        dname = f"jtw_{wn}"
        with su.Steerer(model, src) as st:
            for s in scale_grid(eps_star, input_scale, MEAN_FRACS):
                st.set(None if s == 0.0 else torch.tensor(
                    s * vec64, dtype=torch.float32))
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((dname, s, p, c))
                    g = read_g(model, tok, p, tgt, w_vec.to(dev), t02, dev)
                    readout.append((dname, s, p, f"{g:.6g}"))
                print(f"  {dname} scale={s:+.3g} done", flush=True)
    with open(f"reach_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(f"reach_steer_readout_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(readout)
    print(f"[steer] wrote reach_steer_{ds}.csv and reach_steer_readout_{ds}.csv")


def arm_per_stmt(ds, device, limit=0):
    src, tgt, input_scale, summ, dirs, mz, acts, names, store_names = _load_common(ds)
    stmts = acts["statements"]
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    k = names.index("mean_diff_tgt")
    ks = store_names.index("mean_diff_tgt")
    t02 = float(dirs["thresh02"][k])
    w_vec = torch.tensor(np.asarray(dirs["W"][k], np.float32))
    h_tgt = np.asarray(acts["h_tgt"], np.float64)[:len(y)]
    g_all = h_tgt @ np.asarray(dirs["W"][k], np.float64) - t02
    m_all = np.asarray(mz["margins"], np.float64)[:, k]
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)
    picks = rng.permutation(idx1)[:N_PER_STMT]
    if limit:
        picks = picks[:limit]
    tok, model, dev = su.load_model(device)
    rows = [("direction", "scale", "prompt", "completion")]
    meta_rows = [("stmt_index", "label", "eps_star", "scale", "g_read")]
    with su.Steerer(model, src) as st:
        for i in picks:
            stem = stem_of(stmts[i])
            if stem is None:
                continue
            eps_i = g_all[i] / max(m_all[i], 1e-12) if g_all[i] > 0 else 0.0
            jtw_i = unit(np.asarray(mz["jtw"], np.float64)[i, ks, :])
            for s in scale_grid(eps_i, input_scale, STMT_FRACS):
                st.set(None if s == 0.0 else torch.tensor(
                    s * jtw_i, dtype=torch.float32))
                c = su.generate(model, tok, stem, MAX_NEW_TOKENS)
                g = read_g(model, tok, stem, tgt, w_vec.to(dev), t02, dev)
                rows.append(("jtw_stmt", s, stem, c))
                meta_rows.append((int(i), int(y[i]), f"{eps_i:.6g}", s, f"{g:.6g}"))
            print(f"  stmt {int(i)} done", flush=True)
    with open(f"reach_steer_stmt_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(f"reach_steer_stmt_meta_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(meta_rows)
    print(f"[steer] wrote reach_steer_stmt_{ds}.csv and reach_steer_stmt_meta_{ds}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", required=True, choices=["mean", "per_stmt"])
    ap.add_argument("--limit", type=int, default=0, help="cap prompts/statements (smoke)")
    a = ap.parse_args()
    if a.arm == "mean":
        arm_mean(a.dataset, a.device, a.limit)
    else:
        arm_per_stmt(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
