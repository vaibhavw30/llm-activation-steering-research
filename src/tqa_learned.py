"""tqa_learned.py: the best single layer-11 steering vector, trained (J-E, stage train).

Spec: docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md.

The ceiling every steering number is a fraction of. For each norm r = frac x the median
||h_11|| of the holdout prompts, a vector v with ||v|| = r is trained by Adam to maximize

    mean over pairs of logsigmoid( mean_lp(true answer) - mean_lp(false answer) )

on 600 of the 744 got_datasets/truthfulqa.csv questions (one true and one false answer
each), projected back to the sphere after every step, and the step with the best
objective on the other 144 is kept. Per-token means, so it cannot win on length. None of
the 744 questions is in the 64-question holdout every other stage is scored on.

Three random starts per frac: cosines near 1 between them mean one optimal direction;
low cosines at equal objectives mean many (non-identifiability in miniature).
"""
import math
import os

import numpy as np

import xfer_common as xc

SPLIT_SEED, N_VAL = 37, 144
SEEDS = (0, 1, 2)
LR_PER_R, BATCH_PAIRS, MAX_EPOCHS, EVAL_EVERY = 1e-2, 16, 4, 10


# ------------------------------------------------------------------ pure
def train_pairs():
    import pandas as pd

    from prep_truthfulqa import prompt_of
    df = pd.read_csv("got_datasets/truthfulqa.csv")
    out = []
    for q, g in df.groupby("question", sort=True):
        t, f = g[g["label"] == 1]["answer"], g[g["label"] == 0]["answer"]
        if len(t) != 1 or len(f) != 1:          # every question has one of each (2026-09-18)
            continue
        q = str(q).strip()
        out.append({"question": q, "prompt": prompt_of(q),
                    "true": str(t.iloc[0]).strip(), "false": str(f.iloc[0]).strip()})
    return out


def split_questions(qs, n_val=N_VAL, seed=SPLIT_SEED):
    qs = sorted(set(qs))
    perm = np.random.default_rng(seed).permutation(len(qs))
    val = {qs[i] for i in perm[:n_val]}
    return [q for q in qs if q not in val], sorted(val)


def project(v, r):
    return v * (r / v.norm().clamp_min(1e-12))


# ------------------------------------------------------------------ training
def pair_objective(model, tok, pairs, grad):
    import torch

    import tqa_mc as mc
    k = len(pairs)
    _, mean, _ = mc.answer_logprob(
        model, tok, [p["prompt"] for p in pairs] * 2,
        [p["true"] for p in pairs] + [p["false"] for p in pairs], grad=grad, batch=2 * k)
    d = mean[:k] - mean[k:]
    return torch.nn.functional.logsigmoid(d).mean(), (d > 0).float().mean()


def evaluate(model, tok, pairs):
    obj = acc = 0.0
    for b0 in range(0, len(pairs), BATCH_PAIRS):
        chunk = pairs[b0:b0 + BATCH_PAIRS]
        o, a = pair_objective(model, tok, chunk, grad=False)
        obj += float(o) * len(chunk)
        acc += float(a) * len(chunk)
    return obj / len(pairs), acc / len(pairs)


def train_one(model, tok, st, tr, va, r, seed, max_steps=None):
    """(unit vector, best validation objective, its accuracy). `st` is a Steerer at the
    steering layer; it holds v during training and is cleared before returning."""
    import torch
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    v = project(torch.randn(model.config.hidden_size, generator=g), r).to(dev)
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=LR_PER_R * r)
    rng = np.random.default_rng(seed)
    per_epoch = math.ceil(len(tr) / BATCH_PAIRS)
    total = max_steps or MAX_EPOCHS * per_epoch
    best = (-np.inf, 0.0, v.detach().clone())
    step = 0
    st.set(v)
    try:
        while step < total:
            for idx in np.array_split(rng.permutation(len(tr)), per_epoch):
                loss = -pair_objective(model, tok, [tr[i] for i in idx], grad=True)[0]
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v.copy_(project(v, r))
                step += 1
                if step % EVAL_EVERY == 0 or step == total:
                    with torch.no_grad():
                        o, a = evaluate(model, tok, va)
                    if o > best[0]:
                        best = (o, a, v.detach().clone())
                    print(f"  [train] r={r:.4g} seed={seed} step {step}/{total} "
                          f"val obj {o:.4f} acc {a:.3f}", flush=True)
                if step >= total:
                    break
    finally:
        st.set(None)
    u = (best[2] / r).cpu().double().numpy()
    return u, float(best[0]), float(best[1])


# ------------------------------------------------------------------ io
def load_learned(p):
    if not os.path.exists(p):
        return []
    z = np.load(p, allow_pickle=True)
    return [(f"learned_f{float(f):g}_s{int(s)}", np.asarray(v, np.float64), float(f))
            for v, f, s in zip(z["vecs"], z["frac"], z["seed"])]


def _save(p, rows, extra):
    """Atomic write: a job killed mid-write (e.g. at the wall-clock limit) must never leave
    `p` half-written, or every resume's np.load(p) would crash. np.savez appends ".npz" to
    a name that doesn't already end with it, so the temp name is spelled with it up front."""
    tmp = p + ".tmp.npz"
    np.savez(tmp, vecs=np.array([r[0] for r in rows]), frac=np.array([r[1] for r in rows]),
             seed=np.array([r[2] for r in rows]), val_obj=np.array([r[3] for r in rows]),
             val_acc=np.array([r[4] for r in rows]), **extra)
    os.replace(tmp, p)


def learned_cos(learned, dirs):
    """Seed-vs-seed cosines per frac, and every learned vector against every direction."""
    rows = []
    for i, (n1, v1, f1) in enumerate(learned):
        for n2, v2, f2 in learned[i + 1:]:
            if f1 == f2:
                rows.append({"a": n1, "b": n2, "kind": "seed_vs_seed",
                             "cos": float(v1 @ v2)})
        for n2, v2 in dirs:
            rows.append({"a": n1, "b": n2, "kind": "vs_direction",
                         "cos": float(v1 @ xc.unit(v2))})
    return rows


def stage_train(device, limit=0, steps=None, jb_prefix=""):
    import pandas as pd

    import tqa_mc as mc
    import xfer_tqa as xtqa
    from prep_truthfulqa import prompt_of
    out = mc.path("mc_learned.npz")
    su, tok, model = mc.load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, [prompt_of(q)
                                                for q in xtqa.questions(limit)])
    pairs = train_pairs()
    tr_q, va_q = split_questions([p["question"] for p in pairs])
    tr = [p for p in pairs if p["question"] in set(tr_q)]
    va = [p for p in pairs if p["question"] in set(va_q)]
    if limit:
        tr, va = tr[:4 * limit], va[:limit]
    rows = []
    if os.path.exists(out):
        z = np.load(out)
        rows = [(v, float(f), int(s), float(o), float(a)) for v, f, s, o, a in
                zip(z["vecs"], z["frac"], z["seed"], z["val_obj"], z["val_acc"])]
    done = {(r[1], r[2]) for r in rows}
    base_o, base_a = evaluate(model, tok, va)
    extra = {"norm_med": norm_med, "base_val_obj": base_o, "base_val_acc": base_a}
    print(f"[train] {len(tr)} train / {len(va)} val pairs; median ||h_11|| {norm_med:.4g}; "
          f"unsteered val obj {base_o:.4f} acc {base_a:.3f}", flush=True)
    with su.Steerer(model, xc.LAYER) as st:
        for f in xc.NORM_FRACS:
            for s in SEEDS:
                if (f, s) in done:
                    continue
                u, o, a = train_one(model, tok, st, tr, va, f * norm_med, s, steps)
                rows.append((u, f, s, o, a))
                _save(out, rows, extra)
                print(f"[train] frac {f:g} seed {s}: val obj {o:.4f} acc {a:.3f} "
                      f"(unsteered {base_o:.4f} / {base_a:.3f})", flush=True)
    dirs, missing = xc.build_directions(0, 0, jb_prefix)
    for m in missing:
        print(f"[train] cosines leave out: {m}", flush=True)
    pd.DataFrame(learned_cos(load_learned(out), dirs)).to_csv(
        mc.path("mc_learned_cos.csv"), index=False)
    del model
