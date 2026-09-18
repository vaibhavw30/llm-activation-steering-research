"""tqa_mc.py: judge-free TruthfulQA and the dataset x format grid (J-E, round 3).

    PYTHONPATH=src python src/tqa_mc.py --stage all --device cuda                  # CLUSTER
    PYTHONPATH=src python src/tqa_mc.py --stage all --device cuda \\
        --limit 4 --steps 1 --prefix smoke_                                       # its smoke
    PYTHONPATH=src python src/tqa_mc.py --stage summary                            # LAPTOP

Spec: docs/superpowers/specs/2026-09-18-tqa-mc-learned-ceiling-design.md.

Every TruthfulQA number before this went through a judge, and the dataset card's
registered hypothesis is that TQA steers only because the judged score rewards form. Here
the score is the model's own log-probability of TruthfulQA's reference answers, so no
judge and no generated length is involved:

  margin  (PRIMARY) mean over correct answers of the per-token mean log-prob, minus the
          same over incorrect ones. Per token, so a direction cannot win by preferring
          shorter answers.
  mc1     1 if the answer with the highest per-token mean log-prob is a correct one.
  mc2     summed-log-prob probability mass on the correct answers (the standard MC2).

|            | short answer              | long answer               |
|------------|---------------------------|---------------------------|
| cities     | J-D1                      | stage cities_long (here)  |
| TruthfulQA | stage mc (here)           | J-D2                      |

The learned vectors (tqa_learned.py, stage train) are scored in every stage: they are
the ceiling every other direction is a fraction of.
"""
import argparse
import ast
import json
import os

import numpy as np

import tqa_discovery as td
import xfer_common as xc

DS = "truthfulqa"
N_RAND, RAND_SEED = 32, 41             # the MC null: a permutation p can reach 1/33
N_RAND_LONG, RAND_SEED_LONG = 8, 29    # the cities long-form null, J-D1's randoms
BATCH = 32
LONG_PROMPT = "Q: Where is the city of {city}?\nA:"
LONG_MAX_NEW, LONG_REP = 48, 1.0
PAD_TOL = 1e-3                         # batched vs single summed log-prob, fp32
PREFIX = ""


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure
def parse_answers(cell):
    """The holdout's answer lists are Python list literals."""
    return [str(a).strip() for a in ast.literal_eval(str(cell)) if str(a).strip()]


def encode_pairs(tok, prompts, answers):
    """Left-padded (ids, attention mask, answer mask).

    The prompt and the answer are tokenized SEPARATELY and concatenated. The prompt's
    ids are then exactly the ones generation starts from, and the answer's are the ones
    the model would have to emit after them; tokenizing the joined string could merge
    tokens across the seam and score a sequence generation never produces."""
    import torch
    seqs, masks = [], []
    for p, a in zip(prompts, answers):
        pi = list(tok(p)["input_ids"])
        ai = list(tok(" " + a, add_special_tokens=False)["input_ids"])
        seqs.append(pi + ai)
        masks.append([0] * len(pi) + [1] * len(ai))
    T, pad = max(map(len, seqs)), tok.pad_token_id
    ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs])
    att = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
    am = torch.tensor([[0] * (T - len(m)) + m for m in masks])
    return ids, att, am


def position_ids(att):
    """Positions counted from the first real token. A plain forward with left padding
    otherwise numbers the pads, shifting every real token's position."""
    return (att.cumsum(-1) - 1).clamp(min=0)


def token_logprobs(logits, ids, am):
    """(sum, mean, n) of log p(ids[t] | ids[<t]) over positions with am == 1.
    logits[:, t - 1] is the prediction for ids[:, t]."""
    import torch
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = am[:, 1:].to(tok_lp.dtype)
    s = (tok_lp * m).sum(-1)
    n = m.sum(-1)
    return s, s / n.clamp(min=1), n


def answer_logprob(model, tok, prompts, answers, grad=False, batch=BATCH):
    """Per-row (sum, mean, n) of the answer's log-prob given the prompt, under whatever
    steering the caller's Steerer holds. grad=True keeps the graph (training)."""
    import torch
    dev = next(model.parameters()).device
    outs = []
    for b0 in range(0, len(prompts), batch):
        ids, att, am = (t.to(dev) for t in encode_pairs(
            tok, prompts[b0:b0 + batch], answers[b0:b0 + batch]))
        with torch.set_grad_enabled(grad):
            logits = model(input_ids=ids, attention_mask=att,
                           position_ids=position_ids(att)).logits
            outs.append(token_logprobs(logits, ids, am))
    return tuple(torch.cat([o[i] for o in outs]) for i in range(3))


def mc_metrics(mean_lp, sum_lp, correct):
    mean_lp, sum_lp = np.asarray(mean_lp, float), np.asarray(sum_lp, float)
    c = np.asarray(correct).astype(bool)
    w = np.exp(sum_lp - sum_lp.max())
    return {"margin": float(mean_lp[c].mean() - mean_lp[~c].mean()),
            "mc1": int(c[int(np.argmax(mean_lp))]),
            "mc2": float(w[c].sum() / w.sum())}


def holdout_items(limit=0):
    """One item per (holdout question, reference answer), correct = 1 or 0."""
    import pandas as pd

    from prep_truthfulqa import prompt_of
    h = pd.read_csv(td.HOLDOUT)
    if limit:
        h = h.head(limit)
    out = []
    for _, r in h.iterrows():
        q = str(r["question"]).strip()
        for kind, col in ((1, "correct_answers"), (0, "incorrect_answers")):
            for a in parse_answers(r[col]):
                out.append({"question": q, "prompt": prompt_of(q), "answer": a,
                            "correct": kind})
    return out


# ------------------------------------------------------------------ blocks
def blocks(dirs, grid, learned, norm_med):
    """Baseline, every direction at every dose, then each learned vector at its own frac
    only (a vector trained at one norm is not claimed at another), toward truthful."""
    out = [("baseline", "none", 0.0, 0.0, None)]
    out += [(n, u, f, s, v) for n, v in dirs for u, f, s in grid]
    out += [(n, "norm", f, f * norm_med, v) for n, v, f in learned]
    return out


def smoke_dirs(dirs):
    return ([d for d in dirs if not d[0].startswith("rand_")][:3] +
            [d for d in dirs if d[0].startswith("rand_")][:2])


def load_model_left(device):
    import dct_steer_utils as su
    from reach_hop import load_meta
    src, _, _, model_name = load_meta(DS)
    assert src == xc.LAYER, f"TQA meta says source layer {src}, J-E assumes {xc.LAYER}"
    tok, model, _ = su.load_model(device, model_name=model_name)
    tok.padding_side = "left"
    return su, tok, model


def check_padding(model, tok, items):
    """Batched and one-at-a-time scores must agree: the left-padding and position-id
    handling is otherwise unverified on the real model."""
    it = items[:4]
    s_b = answer_logprob(model, tok, [i["prompt"] for i in it], [i["answer"] for i in it])[0]
    s_1 = [float(answer_logprob(model, tok, [i["prompt"]], [i["answer"]])[0][0]) for i in it]
    diff = float(np.max(np.abs(s_b.cpu().numpy() - np.array(s_1))))
    print(f"[mc] padding check: max |batched - single| summed log-prob {diff:.2e}",
          flush=True)
    if diff > PAD_TOL:
        raise SystemExit(f"[mc] !!!! padding changes scores by {diff:.2e} > {PAD_TOL}")


def score_blocks(model, tok, st, bl, items, out):
    done = xc.done_blocks(out)
    prompts, answers = [i["prompt"] for i in items], [i["answer"] for i in items]
    for name, u, f, s, v in bl:
        if done.get((name, u, float(f)), 0) >= len(items):
            continue
        st.set(None if v is None else xc.steer_vec(v, s))
        sm, mn, n = (t.cpu().numpy() for t in answer_logprob(model, tok, prompts, answers))
        st.set(None)
        xc.append_rows(out, [{"direction": name, "unit": u, "frac": f, "scale": s,
                              "question": i["question"], "answer": i["answer"],
                              "correct": i["correct"], "sum_lp": float(a),
                              "mean_lp": float(b), "n_tok": int(c)}
                             for i, a, b, c in zip(items, sm, mn, n)])
        print(f"[score] {os.path.basename(out)} {name} {u} {f:+g} done", flush=True)


# ------------------------------------------------------------------ stages
def stage_mc(device, limit=0, jb_prefix=""):
    import tqa_learned as tl
    items = holdout_items(limit)
    dirs, missing = xc.build_directions(N_RAND, RAND_SEED, jb_prefix)
    if limit:
        dirs = smoke_dirs(dirs)
    for m in missing:
        print(f"[mc] !!!! left out: {m}", flush=True)
    learned = tl.load_learned(path("mc_learned.npz"))
    if not learned:
        print("[mc] !!!! no learned vectors: stage train has not run", flush=True)
    su, tok, model = load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, sorted({i["prompt"] for i in items}))
    grid = xc.dose_grid(norm_med, xc.eps_star(DS))
    json.dump({"n_questions": len({i["question"] for i in items}), "n_items": len(items),
               "norm_median_layer11": norm_med, "directions": [d[0] for d in dirs],
               "learned": [n for n, _, _ in learned], "left_out": missing,
               "grid": [list(g) for g in grid], "n_rand": N_RAND},
              open(path("mc_tqa_meta.json"), "w"), indent=2)
    check_padding(model, tok, items)
    with su.Steerer(model, xc.LAYER) as st:
        score_blocks(model, tok, st, blocks(dirs, grid, learned, norm_med), items,
                     path("mc_tqa_scores.csv"))
        # Secondary: the 744 training pairs at the read dose. In-sample for every
        # TQA-sourced direction and (outside the validation split) every learned one.
        pairs = tl.train_pairs()[:limit or None]
        titems = [{"question": p["question"], "prompt": p["prompt"], "answer": p[k],
                   "correct": c} for p in pairs for k, c in (("true", 1), ("false", 0))]
        rgrid = [(xc.READ_UNIT, xc.READ_FRAC, xc.READ_FRAC * norm_med)]
        rlearned = [x for x in learned if x[2] == xc.READ_FRAC]
        score_blocks(model, tok, st, blocks(dirs, rgrid, rlearned, norm_med), titems,
                     path("mc_tqa_train_scores.csv"))
    del model


def stage_cities_long(device, limit=0, jb_prefix=""):
    import tqa_learned as tl
    import xfer_cities as xcit
    dirs, missing = xc.build_directions(N_RAND_LONG, RAND_SEED_LONG, jb_prefix)
    if limit:
        dirs = smoke_dirs(dirs)
    for m in missing:
        print(f"[long] !!!! left out: {m}", flush=True)
    learned = [x for x in tl.load_learned(path("mc_learned.npz")) if x[2] == xc.READ_FRAC]
    su, tok, model = load_model_left(device)
    countries = xcit.countries_of()
    pool = xcit.clean_pool({c: int(tok(" " + c, add_special_tokens=False)["input_ids"][0])
                            for c in countries})
    recs = xcit.targets(pool, limit)
    prompts = [LONG_PROMPT.format(city=r["city"]) for r in recs]
    norm_med = xc.median_last_norm(model, tok, prompts)
    grid = [("norm", s * xc.READ_FRAC, s * xc.READ_FRAC * norm_med) for s in (1.0, -1.0)]
    out = path("mc_cities_long.csv")
    done = xc.done_blocks(out)
    print(f"[long] {len(recs)} cities; median ||h_11|| {norm_med:.4g}", flush=True)
    with su.Steerer(model, xc.LAYER) as st:
        for name, u, f, s, v in blocks(dirs, grid, learned, norm_med):
            if done.get((name, u, float(f)), 0) >= len(recs):
                continue
            st.set(None if v is None else xc.steer_vec(v, s))
            raws = []
            for b0 in range(0, len(prompts), BATCH):
                raws += xc.gen_batch(model, tok, prompts[b0:b0 + BATCH], LONG_MAX_NEW,
                                     LONG_REP)
            st.set(None)
            xc.append_rows(out, [dict({"direction": name, "unit": u, "frac": f,
                                       "scale": s, "stmt": r["stmt"], "city": r["city"],
                                       "correct": r["correct"],
                                       "completion": raw.replace("\n", " ").strip()},
                                      **xcit.score_gen(raw, r["correct"], countries))
                                 for r, raw in zip(recs, raws)])
            print(f"[long] {name} {u} {f:+g} done", flush=True)
    del model


def stage_gen(device, limit=0):
    """The learned vectors, generated and later judged: the ceiling on the judged scale."""
    import tqa_learned as tl
    import xfer_tqa as xtqa
    from prep_truthfulqa import prompt_of
    learned = tl.load_learned(path("mc_learned.npz"))
    if not learned:
        print("[gen] !!!! no learned vectors, nothing to generate", flush=True)
        return
    qs = xtqa.questions(limit)
    su, tok, model = load_model_left(device)
    norm_med = xc.median_last_norm(model, tok, [prompt_of(q) for q in qs])
    out = path("mc_learned_gen.csv")
    done = xc.done_blocks(out)
    with su.Steerer(model, xc.LAYER) as st:
        for name, u, f, s, v in blocks([], [], learned, norm_med):
            if done.get((name, u, float(f)), 0) >= len(qs):
                continue
            xc.append_rows(out, xtqa.answer_rows(name, u, f, s, qs,
                                                 xtqa.generate(model, tok, st, v, s, qs)))
            print(f"[gen] {name} {u} {f:+g} done", flush=True)
    del model


def stage_judge(device):
    from judge_audit import judge_resumable
    from judges.local_hf import TruthJudge
    p = path("mc_learned_gen.csv")
    if not os.path.exists(p):
        print("[judge] nothing generated, skipping", flush=True)
        return
    judge_resumable(td.read_csv(p), TruthJudge(device).score, path("mc_learned_judged.csv"))


STAGES = ("train", "mc", "cities_long", "gen", "judge", "summary")


def main(argv=None):
    global PREFIX
    import tqa_learned as tl
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--jb-prefix", default="")
    a = ap.parse_args(argv)
    if a.limit and not a.prefix:
        ap.error("--limit writes partial files: give it a --prefix")
    PREFIX = a.prefix
    todo = STAGES if a.stage == "all" else (a.stage,)
    for s in todo:
        print(f"=== tqa_mc stage {s} ===", flush=True)
        if s == "train":
            tl.stage_train(a.device, a.limit, a.steps, a.jb_prefix)
        elif s == "mc":
            stage_mc(a.device, a.limit, a.jb_prefix)
        elif s == "cities_long":
            stage_cities_long(a.device, a.limit, a.jb_prefix)
        elif s == "gen":
            stage_gen(a.device, a.limit)
        elif s == "judge":
            stage_judge(a.device)
        else:
            stage_summary()


if __name__ == "__main__":
    # Run through the importable module, not this __main__ copy: tqa_learned imports
    # tqa_mc, and PREFIX must be the one it sees (else the smoke writes unprefixed files).
    import tqa_mc
    tqa_mc.main()
