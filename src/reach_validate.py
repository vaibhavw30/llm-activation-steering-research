"""reach_validate.py — V1: what the certificate promises against what steering delivers.

PLAN_ADVISOR_NOTES_2026-09.md section 2, V1.

Every reachability number this project has produced is a PREDICTION. The certificate
eps* = g / ||J_l^T w|| says "perturb the residual stream entering layer l by this much
along this direction and the readout lands on zero". Nothing has ever measured the
realized landing against it. Q2 is where that bill came due: reach_control's `crossed`
column is 0 at every dose on TruthfulQA, so the certificate was issued and never tested.

THE READOUT IS THE DECISION, BY CONSTRUCTION. The target sits at the OUTPUT layer and
the readout is contrastive,

    a = unit( mean W_U[yes first-tokens] - mean W_U[no first-tokens] )

so a . h_final is the yes-minus-no logit gap (gemma-2's final softcap is monotone and
preserves its sign). A readout-vs-behavior dissociation of the kind cities produced is
therefore not available to it. That is the point of the design: a failure here cannot
be blamed on a bad probe.

FOUR MEASUREMENTS, THREE CONTEXTS. Every statement is evaluated at all three, so every
comparison below is paired on the same statement.

  decl   the bare statement, the linearization point
  stem   the statement minus its final word, the generation context
  quest  Q_TRUTH + statement + Q_SUFFIX, the only context where the next token really
         is the verdict

  ratio_A       decl's own certificate, measured at decl.  Tests the pullback
                arithmetic alone. Expected near 1. Anything else is a bug in the
                vjp/injection convention, not a fact about the model.
  ratio_B       decl's certificate applied at stem.  The plan's point B: does the
                promise survive the one-word context shift? D1 says the gain collapses
                8-35x, so expected near 0.07.
  ratio_B_own   stem's OWN certificate, measured at stem.  NOT in the plan's two-point
                design, added because without it ratio_B is ambiguous: a low ratio_B
                could mean the certificate does not transport OR that the
                linearization is simply broken at the stem. This separates them. Near
                1 with a low ratio_B localizes the failure to transport.
  ratio_Q_own   quest's own certificate, measured at quest, WITH the behavioural
                closure: when a . z crosses zero, does argmax over the yes/no tokens
                actually flip? The no-dissociation claim above is checkable, so this
                checks it rather than assuming it.

WHAT MAKES THE RATIOS COMPARABLE. The margin is the every-position broadcast
convention, m_l = ||sum_t d(a.h_final)/d h_l[t]||, and the injection is
dct_steer_utils.Steerer, a forward_pre_hook adding one vector to layer l's input at
every position. Those are the same operator; if they ever drift apart ratio_A moves
off 1 and this module is measuring its own bug. That is why ratio_A is reported first
and why the analyze stage refuses to interpret the rest when it is far from 1.

  --compute (GPU): reach_validate_<ds>.npz
  --analyze (CPU): reach_validate_summary_<ds>.csv + printed per-layer table

    PYTHONPATH=src python src/reach_validate.py --dataset truthfulqa --compute --device cuda
    PYTHONPATH=src python src/reach_validate.py --dataset truthfulqa --analyze
"""
import argparse
import csv

import numpy as np

N_STMT, SEED, BATCH = 64, 42, 8
# ratio_A this far from 1 means the pullback and the injection disagree, which makes
# every other column a measurement of that disagreement rather than of the model.
ARITHMETIC_TOL = 0.10


# ---------------------------------------------------------------- pure helpers

def certificate(g, jtw):
    """Minimum-norm source perturbation that zeroes the readout to first order.

    `g` is the SIGNED readout value (n,) and `jtw` the raw pullback rows (n, d), not
    unit rows. Solving for the smallest Delta with (J^T a) . Delta = -g gives

        Delta = -(g / m^2) J^T a,     ||Delta|| = |g| / m = eps*

    so the direction carries the sign and the caller never has to reason about which
    way to push. Returns (eps_star, delta, m).
    """
    g = np.asarray(g, np.float64)
    jtw = np.asarray(jtw, np.float64)
    m = np.linalg.norm(jtw, axis=-1)
    safe = np.maximum(m, 1e-12)
    return np.abs(g) / safe, -(g / safe ** 2)[..., None] * jtw, m


def realized_ratio(g_before, g_after, predicted):
    """(realized change in the readout) / (predicted change), NaN where degenerate.

    NaN and not 0: a statement already on the boundary has nothing to predict, and
    folding it in as a zero would drag the median toward a failure that did not happen.
    """
    realized = np.asarray(g_after, np.float64) - np.asarray(g_before, np.float64)
    pred = np.asarray(predicted, np.float64)
    ok = np.abs(pred) > 1e-12
    return np.where(ok, realized / np.where(ok, pred, 1.0), np.nan)


def decision_sign(yes_logit, no_logit):
    """+1 when the model prefers yes, -1 when no. The behaviour a . z claims to be."""
    return np.sign(np.asarray(yes_logit, np.float64)
                   - np.asarray(no_logit, np.float64))


def dissociation(g_before, g_after, dec_before, dec_after):
    """Did the readout crossing and the yes/no decision move together?

    The 2x2 that the whole design rests on. `crossed` is a sign change in a . z,
    `flipped` a sign change in the argmax over the yes/no token sets. If the readout
    really is the decision, the off-diagonal cells are empty.
    """
    crossed = np.sign(np.asarray(g_before, np.float64)) != \
        np.sign(np.asarray(g_after, np.float64))
    flipped = np.asarray(dec_before) != np.asarray(dec_after)
    return {"crossed_and_flipped": int(np.sum(crossed & flipped)),
            "crossed_not_flipped": int(np.sum(crossed & ~flipped)),
            "flipped_not_crossed": int(np.sum(~crossed & flipped)),
            "neither": int(np.sum(~crossed & ~flipped)),
            "n": int(len(crossed))}


def summarize(ratios):
    """Median and IQR of a ratio column, ignoring the degenerate rows."""
    r = np.asarray(ratios, np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "median": np.nan, "p25": np.nan, "p75": np.nan}
    return {"n": int(r.size), "median": float(np.median(r)),
            "p25": float(np.percentile(r, 25)),
            "p75": float(np.percentile(r, 75))}


# --------------------------------------------------------------------- stages

def _contexts(stmts):
    """The three evaluation contexts, on the SAME statements so every row is paired.

    Statements too short for a stem are dropped here rather than at the stem context,
    so `decl` and `quest` never quietly contain rows that `stem` does not.
    """
    from reach_steer import stem_of
    from mag.config import Q_TRUTH, Q_SUFFIX
    keep = [(i, str(s), stem_of(s)) for i, s in enumerate(stmts)
            if stem_of(s) is not None]
    idx = np.array([i for i, _, _ in keep], int)
    return idx, {"decl": [d for _, d, _ in keep],
                 "stem": [t for _, _, t in keep],
                 "quest": [Q_TRUTH + d + Q_SUFFIX for _, d, _ in keep]}


def _pullback(model, tok, prompts, a, dev, max_length):
    """(g, jtw) for one context: the readout value and J_l^T a at every source layer.

    One backward per batch yields every layer at once, the same trick reach_jlens uses.
    jtw is the every-position sum, which is the broadcast-injection convention the
    steered forward below has to match for ratio_A to mean anything.
    """
    import torch
    from reach_hop import last_nonpad_index
    tok.padding_side = "right"
    emb_layer = model.get_input_embeddings()
    a_t = torch.tensor(a, dtype=torch.float32, device=dev)
    g_all, j_all = [], []
    for b0 in range(0, len(prompts), BATCH):
        enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(dev)
        emb = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
        hs = out.hidden_states
        last = last_nonpad_index(enc["attention_mask"])
        rows = torch.arange(len(last), device=dev)
        g = hs[-1][rows, last] @ a_t                       # (B,)
        targets = [hs[l] for l in range(len(hs) - 1)]
        grads = torch.autograd.grad(g.sum(), targets)      # rows are independent
        j_all.append(torch.stack([q.sum(dim=1) for q in grads], dim=1)
                     .detach().float().cpu().numpy())      # (B, L, d)
        g_all.append(g.detach().float().cpu().numpy())
        print(f"  [pullback] {b0 + len(last)}/{len(prompts)}", flush=True)
    return np.concatenate(g_all), np.concatenate(j_all)


def _steered(model, tok, prompts, a, layer, delta, dev, max_length, yes_no=None):
    """Readout after injecting a per-statement `delta` at `layer`, every position.

    `yes_no` is (yes_ids, no_ids); when given, also returns the max yes-token and
    max no-token logit at the last real position, which is the behaviour the readout
    claims to be.
    """
    import torch
    import dct_steer_utils as su
    from reach_hop import last_nonpad_index
    tok.padding_side = "right"
    a_t = torch.tensor(a, dtype=torch.float32, device=dev)
    g_all, y_all, n_all = [], [], []
    with su.Steerer(model, layer) as st:
        for b0 in range(0, len(prompts), BATCH):
            enc = tok(prompts[b0:b0 + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(dev)
            d = torch.tensor(delta[b0:b0 + BATCH], dtype=torch.float32,
                             device=dev)[:, None, :]       # (B, 1, d) broadcasts
            st.set(d)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            last = last_nonpad_index(enc["attention_mask"])
            rows = torch.arange(len(last), device=dev)
            h = out.hidden_states[-1][rows, last].float()
            g_all.append((h @ a_t).cpu().numpy())
            if yes_no is not None:
                lg = out.logits[rows, last].float()
                y_all.append(lg[:, yes_no[0]].max(dim=-1).values.cpu().numpy())
                n_all.append(lg[:, yes_no[1]].max(dim=-1).values.cpu().numpy())
    g = np.concatenate(g_all)
    if yes_no is None:
        return g, None, None
    return g, np.concatenate(y_all), np.concatenate(n_all)


MAXLEN = {"decl": 64, "stem": 64, "quest": 96}


def compute(ds, device, limit=0, only_layers=None):
    import dct_steer_utils as su
    from funnel_utils import unit
    from reach_hop import load_meta
    from mag.config import YES_VARIANTS, NO_VARIANTS
    from mag.verdict import first_token_ids

    _, _, _, model_name = load_meta(ds)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    stmts = acts["statements"]
    rng = np.random.default_rng(SEED)
    # Labels are deliberately NOT used to pick. The certificate always steers toward
    # the boundary from whichever side the statement starts on, so a label-balanced
    # sample would only add a nuisance factor to a question that does not have one.
    pick = rng.permutation(len(stmts))[:(limit or N_STMT)]
    sub_idx, ctx = _contexts(stmts[pick])
    stmt_index = pick[sub_idx]
    print(f"[validate] {ds}: {len(stmt_index)} statements "
          f"({len(pick) - len(stmt_index)} dropped for having no stem)", flush=True)

    tok, model, dev = su.load_model(device, model_name=model_name)
    W_U = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    yes = first_token_ids(tok, YES_VARIANTS)
    no = first_token_ids(tok, NO_VARIANTS)
    a = unit(W_U[yes].mean(0).astype(np.float64)
             - W_U[no].mean(0).astype(np.float64)).astype(np.float32)
    del W_U

    g, mnorm, eps, delta = {}, {}, {}, {}
    for name, prompts in ctx.items():
        print(f"[validate] pullback at context {name}", flush=True)
        # The (n, L, d) pullback block is the largest thing in this function and is
        # not needed past the certificate, so only its per-layer norms are kept.
        g[name], rows = _pullback(model, tok, prompts, a, dev, MAXLEN[name])
        eps[name], delta[name], mnorm[name] = certificate(g[name][:, None], rows)
        del rows
    # `only_layers` exists so this stage can be smoke-tested end to end on a CPU
    # before it is given a GPU: the pullback is one backward whatever happens, and
    # restricting the sweep is what turns 4 x 26 steered forwards into 4 x 2.
    all_layers = list(range(delta["decl"].shape[1]))
    layers = all_layers if only_layers is None else [l for l in only_layers
                                                     if l in set(all_layers)]
    if not layers:
        raise SystemExit(f"[validate] --layers selected nothing; this model has "
                         f"{len(all_layers)} source layers")
    print(f"[validate] sweeping {len(layers)} of {len(all_layers)} source layers, "
          f"readout at the output layer", flush=True)

    zero = np.zeros_like(delta["decl"][:, 0])
    _, y0, n0 = _steered(model, tok, ctx["quest"], a, 0, zero, dev,
                         MAXLEN["quest"], yes_no=(yes, no))

    out = {k: [] for k in ("A", "B", "B_own", "Q_own", "Q_yes", "Q_no")}
    for l in layers:
        gA, _, _ = _steered(model, tok, ctx["decl"], a, l, delta["decl"][:, l],
                            dev, MAXLEN["decl"])
        gB, _, _ = _steered(model, tok, ctx["stem"], a, l, delta["decl"][:, l],
                            dev, MAXLEN["stem"])
        gBo, _, _ = _steered(model, tok, ctx["stem"], a, l, delta["stem"][:, l],
                             dev, MAXLEN["stem"])
        gQo, yq, nq = _steered(model, tok, ctx["quest"], a, l, delta["quest"][:, l],
                               dev, MAXLEN["quest"], yes_no=(yes, no))
        for k, v in (("A", gA), ("B", gB), ("B_own", gBo), ("Q_own", gQo),
                     ("Q_yes", yq), ("Q_no", nq)):
            out[k].append(v)
        print(f"[validate] layer {l}: ratio_A median "
              f"{np.nanmedian(realized_ratio(g['decl'], gA, -g['decl'])):.3f}",
              flush=True)

    np.savez(f"reach_validate_{ds}.npz",
             stmt_index=stmt_index, layers=np.array(layers),
             readout=a.astype(np.float32),
             **{f"g_{k}": v.astype(np.float32) for k, v in g.items()},
             # Sliced to the swept layers so every saved array is indexed by
             # position in `layers`, whether or not the sweep was partial.
             **{f"m_{k}": v[:, layers].astype(np.float32)
                for k, v in mnorm.items()},
             **{f"eps_{k}": v[:, layers].astype(np.float32)
                for k, v in eps.items()},
             **{f"after_{k}": np.asarray(v, np.float32) for k, v in out.items()},
             q_yes0=y0.astype(np.float32), q_no0=n0.astype(np.float32))
    print(f"[validate] wrote reach_validate_{ds}.npz  n={len(stmt_index)}")


def analyze(ds):
    z = np.load(f"reach_validate_{ds}.npz", allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    g_d, g_s, g_q = z["g_decl"], z["g_stem"], z["g_quest"]
    dec0 = decision_sign(z["q_yes0"], z["q_no0"])
    rows = []
    for i, l in enumerate(layers):
        rA = realized_ratio(g_d, z["after_A"][i], -g_d)
        rB = realized_ratio(g_s, z["after_B"][i], -g_d)
        rBo = realized_ratio(g_s, z["after_B_own"][i], -g_s)
        rQo = realized_ratio(g_q, z["after_Q_own"][i], -g_q)
        dis = dissociation(g_q, z["after_Q_own"][i],
                           dec0, decision_sign(z["after_Q_yes"][i],
                                               z["after_Q_no"][i]))
        r = {"layer": l,
             "m_decl_median": round(float(np.median(z["m_decl"][:, i])), 4),
             "eps_decl_median": round(float(np.median(z["eps_decl"][:, i])), 4)}
        for name, vals in (("ratio_A", rA), ("ratio_B", rB),
                           ("ratio_B_own", rBo), ("ratio_Q_own", rQo)):
            s = summarize(vals)
            r[name] = round(s["median"], 4)
            r[f"{name}_p25"] = round(s["p25"], 4)
            r[f"{name}_p75"] = round(s["p75"], 4)
        r.update({f"q_{k}": v for k, v in dis.items()})
        rows.append(r)
    out = f"reach_validate_summary_{ds}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # nan, not a small number: if ratio_A could not be computed at some layer the
    # arithmetic check did not pass, it did not run, and the comparison below has to
    # say so rather than fall through to the reassuring branch.
    dev_a = [abs(r["ratio_A"] - 1.0) for r in rows]
    worst = np.nan if any(np.isnan(d) for d in dev_a) else max(dev_a)
    print(f"[validate] {ds}: n={len(g_d)} statements, {len(rows)} source layers")
    print(f"[validate] wrote {out}")
    hdr = f"{'layer':>5} {'eps*':>9} {'ratio_A':>8} {'ratio_B':>8} " \
          f"{'B_own':>8} {'Q_own':>8} {'x&flip':>7} {'x!flip':>7} {'flip!x':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['layer']:>5} {r['eps_decl_median']:>9.3f} {r['ratio_A']:>8.3f} "
              f"{r['ratio_B']:>8.3f} {r['ratio_B_own']:>8.3f} "
              f"{r['ratio_Q_own']:>8.3f} {r['q_crossed_and_flipped']:>7} "
              f"{r['q_crossed_not_flipped']:>7} {r['q_flipped_not_crossed']:>7}")
    if not (worst <= ARITHMETIC_TOL):
        print(f"\n[validate] !!!! ratio_A is off 1 by up to {worst:.3f} at some layer. "
              "The pullback and the injection are not the same operator, so every "
              "other column measures that disagreement. Fix this before reading them.")
    else:
        print(f"\n[validate] ratio_A within {worst:.3f} of 1 at every layer: the "
              "pullback arithmetic is sound, so the other columns are about the model.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default="",
                    help="comma-separated source layers; default every layer")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.layers.split(",") if x.strip()] or None
    if not (a.compute or a.analyze):
        ap.error("pick --compute (GPU) or --analyze (CPU)")
    if a.compute:
        compute(a.dataset, a.device, a.limit, only)
    if a.analyze:
        analyze(a.dataset)


if __name__ == "__main__":
    main()
