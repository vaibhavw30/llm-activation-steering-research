"""xfer_tqa.py: every truth direction, from both datasets, steered on TruthfulQA (J-D2, X).

    PYTHONPATH=src python src/xfer_tqa.py --stage all --device cuda                 # CLUSTER
    PYTHONPATH=src python src/xfer_tqa.py --stage all --device cuda \\
        --limit 4 --prefix smoke_                                                  # its smoke
    PYTHONPATH=src python src/xfer_tqa.py --stage summary                           # LAPTOP

Written for docs/PLAN_PI_FEEDBACK_2026-09-18.md, section 8, job J-D2 (round 2). Same
direction list and dose grid as J-D1 (xfer_common); the target is the 64 TruthfulQA
holdout questions, answered in Q:/A: format and judged by the v2 allenai judges. J-C
already measured the TQA-sourced directions here in eps* units; this job adds the cities
directions, the norm unit, and its own random null, so every cell of the TQA column is
measured in one job against one baseline.

The reverse cell, cities -> TQA, is the one to read: the cities directions were inert on
cities, so a TQA gain from one would put the difference in the dataset, not the direction.

POSITIVE CONTROL. tqa:sup_jtw at eps +2 IS Q2's registered dose (Q2's frac -2; the sign is
flipped here so positive = toward truthful). Its rate must land inside Q2's own Wilson
interval from the v2 re-judged table, or no TQA number in this job is used.

GENERATION IS BATCHED (left padding), where Q1, Q2 and J-C generated one prompt at a
time. Greedy decoding on padded batches is not guaranteed byte-identical, so the steer
stage first writes xfer_truthfulqa_parity.csv: 16 questions under no steering and under
the positive control, batched vs one at a time. The positive control above is the check
that matters; the parity file says how much of any miss the batching could explain.
"""
import argparse
import json
import os

import numpy as np

import tqa_discovery as td
import xfer_common as xc

DS = "truthfulqa"
N_RAND, RAND_SEED = 8, 31
READ_UNIT, READ_FRAC = xc.READ_UNIT, xc.READ_FRAC      # toward truthful, where Q2 gained
POS_CONTROL = ("tqa:sup_jtw", "eps", 2.0)
MAX_NEW, REP_PENALTY, BATCH = td.MAX_NEW_TOKENS, 1.3, 32
N_PARITY = 16
Q2_REFERENCE = {"truthful_and_informative": "tqa_q2_summary_truthfulqa_judge_v2.csv",
                "truthful": "tqa_q2_summary_truthfulqa_judge_v2_truthful.csv"}
Q2_REFERENCE_V1 = "tqa_q2_summary_truthfulqa.csv"
PREFIX = ""


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure
def summarize(df, col):
    """One row per (direction, unit, frac): rate, Wilson, exact McNemar against the
    unsteered answer to the same question, and a permutation p among the random
    directions at the same dose on effect = sign(frac) x (rate - base rate)."""
    from tqa_baseline import wilson
    from tqa_q2_analyze import mcnemar_exact
    df = df.copy()
    df[col] = df[col].astype(int)
    base = df[df["direction"] == "baseline"].set_index("question")
    out = []
    for (name, u, f), g in df[df["direction"] != "baseline"].groupby(
            ["direction", "unit", "frac"]):
        g = g.set_index("question")
        b = base.loc[g.index]
        gained, lost, p, n = mcnemar_exact(b[col], g[col])
        k = int(g[col].sum())
        lo, hi = wilson(k, len(g))
        s = float(np.sign(f)) or 1.0
        out.append({"direction": name, "unit": u, "frac": float(f), "n": len(g),
                    "rate": k / len(g), "lo": lo, "hi": hi, "base_rate": float(b[col].mean()),
                    "gained": gained, "lost": lost, "mcnemar_p": p,
                    "words": float(g["words"].mean()), "base_words": float(b["words"].mean()),
                    "e_gain": s * (k / len(g) - float(b[col].mean())),
                    "e_words": abs(float(g["words"].mean() - b["words"].mean()))})
    rand = {}
    for r in out:
        if r["direction"].startswith("rand_"):
            rand.setdefault((r["unit"], r["frac"]), []).append(r)
    for r in out:
        rs = rand.get((r["unit"], r["frac"]), [])
        r["n_null"] = len(rs)
        for e in ("e_gain", "e_words"):
            r["p_" + e[2:]] = ("" if r["direction"].startswith("rand_") or not rs
                               else xc.perm_p(r[e], [x[e] for x in rs]))
    return sorted(out, key=lambda r: (r["direction"], r["unit"], r["frac"]))


def classify(r, alpha=xc.ALPHA):
    """gain: exact McNemar p <= alpha against the unsteered answers AND the gain clears the
    random null (xfer_common.beyond_null: with 8 randoms, beat all 8). form: only the word
    count clears the null. none: neither."""
    if r.get("p_gain", "") == "":
        return "no null"
    if r["mcnemar_p"] <= alpha and xc.beyond_null(r["p_gain"], r["n_null"], alpha):
        return "gain"
    return "form" if xc.beyond_null(r["p_words"], r["n_null"], alpha) else "none"


def q2_interval(col):
    """(rate, lo, hi, source file) of Q2's mean arm at frac -2, from the v2 table for
    this score column; the v1 table only as a flagged fallback."""
    import pandas as pd
    for p, flag in ((Q2_REFERENCE.get(col), ""), (Q2_REFERENCE_V1, " (v1 judge!)")):
        if p and os.path.exists(p) and (flag == "" or col == "truthful_and_informative"):
            d = pd.read_csv(p)
            r = d[(d["arm"] == "mean") & (d["direction"] == "jtw_mean_diff_tgt")
                  & (d["frac"].astype(float) == -2.0)]
            if len(r):
                r = r.iloc[0]
                return float(r["rate"]), float(r["wilson_lo"]), float(r["wilson_hi"]), p + flag
    return None


# ------------------------------------------------------------------ steer
def questions(limit=0):
    import pandas as pd
    qs = list(pd.read_csv(td.HOLDOUT)["question"].astype(str).str.strip())
    return qs[:limit] if limit else qs


def answer_rows(name, u, f, scale, qs, raws):
    from prep_truthfulqa import prompt_of
    from tqa_baseline import first_answer
    rows = []
    for q, raw in zip(qs, raws):
        a = first_answer(raw)
        rows.append({"direction": name, "unit": u, "frac": f, "scale": scale,
                     "question": q, "prompt": prompt_of(q),
                     "completion": raw.replace("\n", " ").strip(), "answer": a,
                     "words": xc.word_count(a)})
    return rows


def generate(model, tok, st, vec, scale, qs):
    from prep_truthfulqa import prompt_of
    st.set(None if vec is None else xc.steer_vec(vec, scale))
    raws = []
    for b0 in range(0, len(qs), BATCH):
        raws += xc.gen_batch(model, tok, [prompt_of(q) for q in qs[b0:b0 + BATCH]],
                             MAX_NEW, REP_PENALTY)
    st.set(None)
    return raws


def parity(model, tok, st, q2vec, eps, qs):
    """Batched vs one-at-a-time answers on N_PARITY questions, both unsteered and at the
    positive control. Written once; never fatal."""
    import dct_steer_utils as su
    from prep_truthfulqa import prompt_of
    from tqa_baseline import first_answer
    p = path(f"xfer_{DS}_parity.csv")
    if os.path.exists(p):
        return
    qs = qs[:N_PARITY]
    rows = []
    for name, vec, scale in (("baseline", None, 0.0),
                             (POS_CONTROL[0], q2vec, POS_CONTROL[2] * eps)):
        batched = generate(model, tok, st, vec, scale, qs)
        st.set(None if vec is None else xc.steer_vec(vec, scale))
        single = [su.generate_raw(model, tok, prompt_of(q), MAX_NEW) for q in qs]
        st.set(None)
        for q, a, b in zip(qs, batched, single):
            rows.append({"direction": name, "question": q, "batched": first_answer(a),
                         "single": first_answer(b),
                         "identical": int(first_answer(a) == first_answer(b))})
        same = np.mean([r["identical"] for r in rows if r["direction"] == name])
        print(f"[parity] {name}: batched == one-at-a-time on {same:.2f} of {len(qs)} "
              "answers", flush=True)
    td.write_new(p, rows)


def stage_steer(device, limit=0, jb_prefix=""):
    import torch

    import dct_steer_utils as su
    from prep_truthfulqa import prompt_of
    from reach_hop import load_meta

    out = path(f"xfer_{DS}_steer.csv")
    qs = questions(limit)
    dirs, missing = xc.build_directions(N_RAND, RAND_SEED, jb_prefix)
    if POS_CONTROL[0] not in dict(dirs):
        raise SystemExit(f"[xfer] the positive control {POS_CONTROL[0]} could not be built")
    for m in missing:
        print(f"[xfer] !!!! left out: {m}", flush=True)
    if limit:                           # smoke: the control, two others, two randoms
        other = [d for d in dirs if not d[0].startswith("rand_") and d[0] != POS_CONTROL[0]]
        dirs = [d for d in dirs if d[0] == POS_CONTROL[0]] + other[:2] + \
            [d for d in dirs if d[0].startswith("rand_")][:2]
    src, _, _, model_name = load_meta(DS)
    assert src == xc.LAYER, f"TQA meta says source layer {src}, X assumes {xc.LAYER}"
    tok, model, dev = su.load_model(device, model_name=model_name)
    tok.padding_side = "left"
    norm_med = xc.median_last_norm(model, tok, [prompt_of(q) for q in qs])
    eps = xc.eps_star(DS)
    grid = xc.dose_grid(norm_med, eps)
    json.dump({"n_questions": len(qs), "norm_median_layer11": norm_med, "eps_star": eps,
               "directions": [d[0] for d in dirs], "left_out": missing,
               "grid": [list(g) for g in grid], "n_rand": N_RAND},
              open(path(f"xfer_{DS}_meta.json"), "w"), indent=2)
    print(f"[xfer] {len(dirs)} directions x {len(grid)} doses x {len(qs)} questions; "
          f"median ||h_11|| {norm_med:.4g}, eps* {eps:.4g}", flush=True)
    done = xc.done_blocks(out)
    blocks = [("baseline", "none", 0.0, 0.0, None)]
    blocks += [(n, u, f, s, v) for n, v in dirs for u, f, s in grid]
    with su.Steerer(model, xc.LAYER) as st:
        parity(model, tok, st, dict(dirs)[POS_CONTROL[0]], eps, qs)
        for name, u, f, s, v in blocks:
            if done.get((name, u, float(f)), 0) >= len(qs):
                continue
            xc.append_rows(out, answer_rows(name, u, f, s, qs,
                                            generate(model, tok, st, v, s, qs)))
            print(f"[steer] {name} {u} {f:+g} done", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def stage_judge(device):
    from judge_audit import judge_resumable
    from judges.local_hf import TruthJudge
    judge_resumable(td.read_csv(path(f"xfer_{DS}_steer.csv")), TruthJudge(device).score,
                    path(f"xfer_{DS}_judged.csv"))


def stage_summary(jb_prefix=""):
    import pandas as pd
    sel_p = jb_prefix + f"dct_selection_{DS}.json"
    col = json.load(open(sel_p))["score_col"] if os.path.exists(sel_p) else "truthful"
    print(f"[summary] scored on `{col}`" + ("" if os.path.exists(sel_p) else
                                             f" ({sel_p} missing, so truthful alone)"),
          flush=True)
    rows = summarize(pd.read_csv(path(f"xfer_{DS}_judged.csv")), col)
    ctl = [r for r in rows if (r["direction"], r["unit"], r["frac"]) == POS_CONTROL]
    ref = q2_interval(col)
    ok = None
    if ctl and ref:
        ok = ref[1] <= ctl[0]["rate"] <= ref[2]
        print(f"[summary] POSITIVE CONTROL {POS_CONTROL[0]} eps +2: rate {ctl[0]['rate']:.3f}"
              f" vs Q2 {ref[0]:.3f} [{ref[1]:.3f}, {ref[2]:.3f}] from {ref[3]} -> "
              + ("REPRODUCED" if ok else "!!!! NOT reproduced: no TQA number in this job "
                 "is used (see the parity file)"), flush=True)
    else:
        print("[summary] !!!! positive control or Q2 reference missing; the control is "
              "UNCHECKED", flush=True)
    outcomes = {}
    for r in rows:
        r["outcome"] = "" if r["direction"].startswith("rand_") else classify(r)
        if r["unit"] == READ_UNIT and r["frac"] == READ_FRAC and r["outcome"]:
            outcomes[r["direction"]] = r["outcome"]
            print(f"[summary] {r['direction']:26s} {READ_UNIT} {READ_FRAC:+g}  rate "
                  f"{r['rate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] vs {r['base_rate']:.3f}  "
                  f"+{r['gained']}/-{r['lost']}  perm p {r['p_gain']:.3g}  words "
                  f"{r['words']:.1f} vs {r['base_words']:.1f}  -> {r['outcome']}", flush=True)
    td.write_new(path(f"xfer_{DS}_summary.csv"), rows)
    p = path(f"xfer_{DS}_outcomes.json")
    if os.path.exists(p):
        raise SystemExit(f"[xfer] {p} exists; this module never overwrites.")
    json.dump({"score_col": col, "read_at": [READ_UNIT, READ_FRAC],
               "positive_control_reproduced": ok, "q2_reference": ref,
               "outcomes": outcomes}, open(p, "w"), indent=2)


STAGES = ("steer", "judge", "summary")


def main(argv=None):
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--jb-prefix", default="", help="read J-B's outputs under this prefix")
    a = ap.parse_args(argv)
    PREFIX = a.prefix
    if a.limit and not a.prefix:
        raise SystemExit("--limit writes partial outputs; give it a --prefix")
    for st in STAGES if a.stage == "all" else (a.stage,):
        print(f"=== xfer_tqa: {st} ===", flush=True)
        if st == "steer":
            stage_steer(a.device, a.limit, a.jb_prefix)
        elif st == "judge":
            stage_judge(a.device)
        elif os.path.exists(path(f"xfer_{DS}_summary.csv")):
            print("[xfer] summary exists, skipping", flush=True)
        else:
            stage_summary(a.jb_prefix)


if __name__ == "__main__":
    main()
