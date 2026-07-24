"""reach_hop.py — shared machinery for the backward-reachability audit (Phases 1-5).

Hop map, per statement i:  F_i : Delta in R^d  ->  h_tgt(last non-pad token) in R^d,
where Delta is added to the residual stream entering block `source_layer` at EVERY
token position — the exact convention of dct_steer_utils.Steerer (forward_pre_hook on
model.model.layers[src]) — and blocks src..tgt run via dct.SlicedModel, so that
F_i(0) == hidden_states[tgt] (run_dct_minimal's SlicedModel sanity check).

J_i = dF_i/dDelta at 0 is a d x d per-statement matrix, never materialized here:
    vjp_rows(f, delta0, W)   -> per-row J_i^T w      (one backward per direction w)
    jvp_cols(f, delta0, T)   -> per-row J_i v        (one forward-mode call per chunk)
dct.py already drives torch.func.vjp/jvp through SlicedModel on the real model
(dct.py lines ~419/~729), so these transforms are proven on this architecture.
"""
import json
import os

import numpy as np
import torch
from torch.func import vjp, jvp

import dct
import dct_steer_utils as su
import funnel_utils as fu
from funnel_utils import unit


def last_nonpad_index(attn_mask):
    """(B, T) 0/1 right-padded mask -> (B,) index of the last real token."""
    am = attn_mask
    return am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)


def make_hop(sliced, h_src_seq, attn_mask):
    """Return f(delta_rows (B,d)) -> (B,d): target-layer activation at the last
    non-pad token, with row b's delta broadcast over ALL of row b's positions.
    h_src_seq (B,T,d) is treated as a constant (detached)."""
    last = last_nonpad_index(attn_mask)
    idx = torch.arange(h_src_seq.shape[0], device=h_src_seq.device)
    h0 = h_src_seq.detach()

    def f(delta_rows):
        out = sliced(h0 + delta_rows[:, None, :])     # (B,T,d) = hidden_states[tgt]
        return out[idx, last, :]
    return f


def vjp_rows(f, delta0, W):
    """W (K,d) target-space directions. Returns (F0 (B,d), G (K,B,d)) with
    G[k,b] = J_b^T W[k]. One saved forward, K backward passes. Rows are independent
    because output row b depends only on delta row b."""
    B = delta0.shape[0]
    F0, pull = vjp(f, delta0)
    grads = []
    for w in W:
        cot = w.to(F0.dtype).to(F0.device).expand(B, -1).contiguous()
        grads.append(pull(cot)[0])
    return F0, torch.stack(grads)


def jvp_cols(f, delta0, T_rows):
    """T_rows (B,d) per-row tangents. Returns (F0, JT (B,d)) with JT[b] = J_b T_rows[b]."""
    return jvp(f, (delta0,), (T_rows,))


def load_meta(ds):
    m = json.load(open(f"dct_meta_{ds}.json"))
    return int(m["source_layer"]), int(m["target_layer"]), float(m["input_scale"])


def load_model_and_slice(ds, device):
    src, tgt, scale = load_meta(ds)
    tok, model, dev = su.load_model(device)
    sliced = dct.SlicedModel(model, start_layer=src, end_layer=tgt,
                             layers_name="model.layers")
    return tok, model, sliced, {"src": src, "tgt": tgt, "input_scale": scale,
                                "device": dev}


def forward_source_batch(model, tok, statements, src, tgt, device, max_length=64):
    """One no-grad full forward. Returns h_src_seq (B,T,d) + attn (B,T) for the hop,
    and last-token rows h_src / h_tgt / h_final (each (B,d), detached)."""
    tok.padding_side = "right"
    enc = tok(list(statements), return_tensors="pt", padding=True,
              truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    am = enc["attention_mask"]
    last = last_nonpad_index(am)
    idx = torch.arange(am.shape[0], device=am.device)
    return {"h_src_seq": hs[src].detach(), "attn": am,
            "h_src": hs[src][idx, last].detach(),
            "h_tgt": hs[tgt][idx, last].detach(),
            "h_final": hs[-1][idx, last].detach()}


def load_landmarks(ds):
    """Source-layer landmark unit vectors for cosine bookkeeping (spec §3)."""
    td = np.load(f"truth_dir_{ds}.npz")
    md_src = unit(np.asarray(td["mean_diff"], np.float64))
    mg = np.load(f"mag_dir_{ds}.npz")
    v_q = unit(np.asarray(mg["v_Q_unit"], np.float64))
    V, U, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(V, U, 1)[0]
    dct_v = unit(V[:, top].astype(np.float64))
    return {"md_src": md_src, "v_q": v_q, "dct_v": dct_v}


def validate_inputs(ds):
    """Fail-fast startup validation (spec §6): die in seconds on a config error,
    not after an hour of forwards."""
    problems = []
    checks = [
        (f"dct_meta_{ds}.json", None),
        (f"truth_dir_{ds}.npz", ("mean_diff", "grad", "layer")),
        (f"truth_dir_tgt_{ds}.npz", ("mean_diff", "grad", "layer")),
        (f"mag_dir_{ds}.npz", ("v_Q_unit",)),
        (f"dct_V_{ds}.pt", None),
        (f"dct_U_{ds}.pt", None),
        (os.path.join("got_datasets", f"{ds}.csv"), None),
    ]
    for path, keys in checks:
        if not os.path.exists(path):
            problems.append(f"missing {path}")
        elif keys is not None:
            z = np.load(path)
            missing = [k for k in keys if k not in z.files]
            if missing:
                problems.append(f"{path} lacks keys {missing}")
    if problems:
        raise SystemExit("[reach] input validation FAILED:\n  " + "\n  ".join(problems))
    src, tgt, _ = load_meta(ds)
    src_l = int(np.load(f"truth_dir_{ds}.npz")["layer"])
    tgt_l = int(np.load(f"truth_dir_tgt_{ds}.npz")["layer"])
    if src_l != src or tgt_l != tgt:
        raise SystemExit(f"[reach] layer mismatch: truth_dir layer {src_l} vs dct src {src}; "
                         f"truth_dir_tgt layer {tgt_l} vs dct tgt {tgt}")
