"""token_jac.py — E2/E3: per-layer controllability of the TOKEN readout.

The existing certificate hopped one chosen pair of layers (src -> tgt) and read out a
probe halfspace. This does the same algebra for every layer against a readout that
cannot dissociate from behavior, because it IS the decision:

    r_i(h) = a_i . z_i ,   a_i = W[j_target] - W[j_argmax]

r_i > 0 is *exactly* the statement "the model emits j_target instead of j_argmax" at
temperature 0. So

    m_i(l) = || J_l^T a_i ||       controllability margin of layer l
    eps*_token(l, i) = M_i / m_i(l)   certified budget, same g/m form as before with
                                      w := a_i and t02 := 0.

Two injection conventions, computed side by side, because they are different
experiments and the existing runs used the first without saying so:

    m_all(l)  = || sum_t  d r / d h_t^(l) ||     vector added at EVERY position
    m_last(l) = ||        d r / d h_last^(l) ||  vector added at the last position only

Their ratio is the broadcast gain. ActAdd injects at one aligned position; the
Steerer in dct_steer_utils broadcasts. If m_all / m_last is large, then every
"budget epsilon" reported so far was really spending closer to epsilon * that ratio,
and the honest per-position comparison is m_last.

    PYTHONPATH=src python src/token_jac.py --dataset cities --device cuda
"""
import argparse

import numpy as np
import pandas as pd
import torch

import dct_steer_utils as su

BATCH = 8
MAX_LEN = 64


def build_readout(ds):
    """a_i (unit) and M_i from token_geom, re-paired with their stems."""
    g = np.load(f"token_geom_{ds}.npz", allow_pickle=True)
    a = np.load(f"token_acts_{ds}.npz", allow_pickle=True)
    if not np.array_equal(np.asarray(g["row_index"]), np.asarray(a["row_index"])):
        raise SystemExit("token_geom and token_acts row_index disagree; re-run "
                         "token_geom.py --stage geom against the current token_acts")
    return (np.asarray(a["stems"], object), np.asarray(g["a_unit"], np.float32),
            np.asarray(g["margin"], np.float64), np.asarray(g["delta_cone"], np.float64),
            np.asarray(g["row_index"]))


def margins(model, tok, stems, A, layers, dev, save_dirs=False):
    """||J_l^T a|| for every layer, under both injection conventions.

    Batch rows never interact, so grad of the summed scalar gives each row its own
    Jacobian-vector product — one backward per batch covers all layers.

    With save_dirs, the pullback VECTORS J_l^T a are kept too (float16, (n, L, d)),
    since those are the actuators token_steer injects for the certified sweep."""
    emb_layer = model.get_input_embeddings()
    m_all, m_last, v_all, v_last = [], [], [], []
    for b0 in range(0, len(stems), BATCH):
        chunk = list(stems[b0:b0 + BATCH])
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(dev)
        # load_model freezes every parameter, so entering through input_ids leaves NO
        # tensor in the graph requiring grad and autograd.grad raises. Enter through
        # inputs_embeds with requires_grad instead, exactly as reach_jlens.py does.
        emb = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
        hs = out.hidden_states
        am = enc["attention_mask"]
        # gemma left-pads; mask.sum(1)-1 would read a pad row (reach_hop.last_nonpad_index)
        last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
        ar = torch.arange(len(last), device=dev)
        z = hs[-1][ar, last]                                     # (B, d) post-norm
        av = torch.tensor(A[b0:b0 + len(chunk)], dtype=z.dtype, device=dev)
        s = (z * av).sum()
        grads = torch.autograd.grad(s, [hs[l] for l in layers])
        g_all = torch.stack([g.sum(dim=1) for g in grads], 1)      # (B, L, d)
        g_last = torch.stack([g[ar, last] for g in grads], 1)      # (B, L, d)
        m_all.append(g_all.norm(dim=-1).detach().float().cpu().numpy())
        m_last.append(g_last.norm(dim=-1).detach().float().cpu().numpy())
        if save_dirs:
            v_all.append(g_all.detach().half().cpu().numpy())
            v_last.append(g_last.detach().half().cpu().numpy())
        print(f"  jac batch {b0}-{b0 + len(chunk)} done", flush=True)
    cat = np.concatenate
    return (cat(m_all), cat(m_last),
            cat(v_all) if save_dirs else None,
            cat(v_last) if save_dirs else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-dirs", action="store_true",
                   help="also store the pullback vectors J_l^T a (needed by "
                        "token_steer --dirs jtw_token); ~24MB per convention")
    a = p.parse_args()

    stems, A, M, delta, ridx = build_readout(a.dataset)
    if a.limit:
        stems, A, M, delta, ridx = (x[:a.limit] for x in (stems, A, M, delta, ridx))
    tok, model, dev = su.load_model(a.device, model_name=a.model)
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers))                 # hidden_states[l] = input of layer l
    print(f"[jac] {len(stems)} stems x {len(layers)} layers on {dev}")

    m_all, m_last, v_all, v_last = margins(model, tok, stems, A, layers, dev,
                                           save_dirs=a.save_dirs)
    eps_all = M[:, None] / np.maximum(m_all, 1e-12)
    eps_last = M[:, None] / np.maximum(m_last, 1e-12)

    extra = {"jtw": v_all, "jtw_last": v_last} if a.save_dirs else {}
    np.savez_compressed(f"token_jac_{a.dataset}.npz",
                        m_all=m_all, m_last=m_last, eps_all=eps_all,
                        eps_last=eps_last, margin=M, delta_cone=delta,
                        row_index=ridx, layers=np.asarray(layers), **extra)

    rows = []
    for li, l in enumerate(layers):
        rows.append(dict(
            layer=l,
            m_all_median=float(np.median(m_all[:, li])),
            m_last_median=float(np.median(m_last[:, li])),
            broadcast_gain=float(np.median(m_all[:, li] / np.maximum(m_last[:, li], 1e-12))),
            eps_all_median=float(np.median(eps_all[:, li])),
            eps_last_median=float(np.median(eps_last[:, li])),
            eps_all_p10=float(np.percentile(eps_all[:, li], 10)),
            eps_all_p90=float(np.percentile(eps_all[:, li], 90)),
            n=int(len(m_all))))
    out = pd.DataFrame(rows)
    out.to_csv(f"token_jac_{a.dataset}.csv", index=False)

    best_all = int(out.loc[out.eps_all_median.idxmin(), "layer"])
    best_last = int(out.loc[out.eps_last_median.idxmin(), "layer"])
    print(f"\n=== token_jac {a.dataset} ===")
    print(out.to_string(index=False))
    print(f"\ncheapest layer (broadcast injection)      : {best_all}  "
          f"eps*={out.eps_all_median.min():.4f}")
    print(f"cheapest layer (last-position injection)  : {best_last}  "
          f"eps*={out.eps_last_median.min():.4f}")
    print(f"median broadcast gain across layers       : "
          f"{out.broadcast_gain.median():.2f}x")
    print(f"optimal POST-NORM budget for the same flip: "
          f"{np.median(delta):.4f}  (token_geom delta_cone)")
    print(f"wrote token_jac_{a.dataset}.csv and .npz")


if __name__ == "__main__":
    main()
