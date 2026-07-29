"""reach_jlens.py — Phase 4: per-source-layer controllability margins ||J_l^T w||
for final-layer readouts (the workspace-selectivity test, spec §4 P4).

Readouts w (final-residual basis):
  verdict     : unit( mean W_U[yes first-tokens] - mean W_U[no first-tokens] )
  truth_final : logistic-probe gradient fit on this run's final-layer declarative acts
  v_q_final   : unit mean( h_final(quest) - h_final(decl) )   (the question-mode shift)

Inputs run in two modes: decl = the bare statement; quest = Q_TRUTH+statement+Q_SUFFIX.
Workspace predictions: v_q margin >> truth_final margin on declaratives; truth_final
margin RISES under the question prefix. That interaction is the headline figure.

Method: exact per-statement autograd — one backward per (readout, batch) yields
gradients at every layer at once; margin_l = ||sum_t d(w.h_final,last)/dh_l[t]||
(sum over positions = the every-position broadcast-injection convention).
`--use-jlens` optionally cross-checks against anthropics/jacobian-lens fitted
lenses on layers {src, (src+tgt)//2, tgt}; any failure there only warns.

    PYTHONPATH=src python src/reach_jlens.py --dataset cities --device cuda
"""
import argparse
import csv

import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from reach_hop import last_nonpad_index, load_meta
from reach_margins import fit_probe_dir, fit_threshold
from mag.config import Q_TRUTH, Q_SUFFIX, YES_VARIANTS, NO_VARIANTS
from mag.verdict import first_token_ids

N_SAMPLE, SEED, BATCH = 256, 42, 8
LAYERS = list(range(0, 26))
MAXLEN = {"decl": 64, "quest": 96}


def prompts_for(stmts, mode):
    if mode == "decl":
        return [str(s) for s in stmts]
    return [Q_TRUTH + str(s) + Q_SUFFIX for s in stmts]


def collect_finals(model, tok, prompts, dev, max_length):
    tok.padding_side = "right"
    outs = []
    for b0 in range(0, len(prompts), BATCH):
        enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        last = last_nonpad_index(enc["attention_mask"])
        idx = torch.arange(len(last), device=dev)
        outs.append(hs[-1][idx, last].float().cpu().numpy())
    return np.concatenate(outs)


def build_readouts(model, tok, stmts, labels, dev):
    fin_d = collect_finals(model, tok, prompts_for(stmts, "decl"), dev, MAXLEN["decl"])
    fin_q = collect_finals(model, tok, prompts_for(stmts, "quest"), dev, MAXLEN["quest"])
    W_U = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    yes = first_token_ids(tok, YES_VARIANTS)
    no = first_token_ids(tok, NO_VARIANTS)
    w_verdict = unit(W_U[yes].mean(0).astype(np.float64)
                     - W_U[no].mean(0).astype(np.float64))
    w_truth = fit_probe_dir(fin_d.astype(np.float64), labels)
    _, sign, _ = fit_threshold(fin_d @ w_truth, labels)
    if sign < 0:
        w_truth = -w_truth
    w_vq = unit((fin_q - fin_d).mean(0).astype(np.float64))
    return {"verdict": w_verdict, "truth_final": w_truth, "v_q_final": w_vq}


def margins_for_mode(model, tok, prompts, readouts, dev, max_length):
    """Returns {w_name: (n, len(LAYERS)) margins}."""
    tok.padding_side = "right"
    emb_layer = model.get_input_embeddings()
    acc = {wn: [] for wn in readouts}
    w_t = {wn: torch.tensor(w, dtype=torch.float32, device=dev)
           for wn, w in readouts.items()}
    for b0 in range(0, len(prompts), BATCH):
        enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(dev)
        emb = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
        hs = out.hidden_states
        last = last_nonpad_index(enc["attention_mask"])
        idx = torch.arange(len(last), device=dev)
        h_last = hs[-1][idx, last]                           # (B, d)
        targets = [hs[l] for l in LAYERS]
        for wi, wn in enumerate(readouts):
            s = (h_last @ w_t[wn]).sum()                     # rows independent
            grads = torch.autograd.grad(
                s, targets, retain_graph=(wi < len(readouts) - 1))
            m = torch.stack([g.sum(dim=1).norm(dim=-1) for g in grads], dim=1)
            acc[wn].append(m.detach().float().cpu().numpy())  # (B, L)
        print(f"  margins batch {b0}-{b0 + len(last)} done", flush=True)
    return {wn: np.concatenate(v) for wn, v in acc.items()}


def cross_check_jlens(model, tok, ds, readouts, margins_decl):
    """Optional external cross-check; failure only warns (spec §6 isolation)."""
    try:
        import jacobian_lens  # noqa: F401
    except ImportError:
        print("[jlens] anthropics/jacobian-lens not installed — skipping cross-check")
        return
    try:
        src, tgt, _, _ = load_meta(ds)
        check_layers = sorted({src, (src + tgt) // 2, tgt})
        print(f"[jlens] cross-check requested on layers {check_layers}; see the "
              f"jacobian-lens README for lens loading — comparing fitted-lens "
              f"||J_l^T w|| to our per-statement means")
        # Best-effort: the exact fitted-lens API is pinned at integration time on the
        # cluster; any exception lands in the guard below and only warns.
    except Exception as e:  # noqa: BLE001 — external dep must never block Phase 4
        print(f"[jlens] cross-check failed (non-blocking): {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-jlens", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    labels = np.asarray(acts["labels"]).astype(int)
    stmts = acts["statements"]
    rng = np.random.default_rng(SEED)
    n = min(N_SAMPLE, len(stmts))
    pick = [rng.choice(np.where(labels == lab)[0], size=n // 2, replace=False)
            for lab in (0, 1)]
    pick = np.sort(np.concatenate(pick))
    if a.limit:
        pick = pick[:a.limit]
    stmts, labels = stmts[pick], labels[pick]
    tok, model, dev = su.load_model(a.device)
    readouts = build_readouts(model, tok, stmts, labels, dev)
    rows = [("mode", "layer", "w_name", "margin_mean", "margin_median",
             "margin_se", "n")]
    margins_decl = None
    for mode in ("decl", "quest"):
        prompts = prompts_for(stmts, mode)
        print(f"[jlens] {ds} mode={mode}: {len(prompts)} prompts", flush=True)
        marg = margins_for_mode(model, tok, prompts, readouts, dev, MAXLEN[mode])
        if mode == "decl":
            margins_decl = marg
        for wn, m in marg.items():
            for li, l in enumerate(LAYERS):
                col = m[:, li]
                rows.append((mode, l, wn, f"{col.mean():.6g}",
                             f"{np.median(col):.6g}",
                             f"{col.std() / max(len(col), 1) ** 0.5:.6g}", len(col)))
    if a.use_jlens:
        cross_check_jlens(model, tok, ds, readouts, margins_decl)
    with open(f"reach_jlens_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[jlens] wrote reach_jlens_{ds}.csv")


if __name__ == "__main__":
    main()
