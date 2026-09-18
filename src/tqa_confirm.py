"""tqa_confirm.py: do J-B's picks move TruthfulQA truth on questions nobody selected on? (D3, G1)

    PYTHONPATH=src python src/tqa_confirm.py --stage all --device cuda              # CLUSTER, J-C
    PYTHONPATH=src python src/tqa_confirm.py --stage all --device cuda \\
        --limit 4 --prefix smoke_                                                  # its smoke
    PYTHONPATH=src python src/tqa_confirm.py --stage summary                        # LAPTOP

Written for docs/PLAN_PI_FEEDBACK_2026-09-18.md, job J-C (round 1, afterok J-B). Reads
J-B's dct_selection_truthfulqa.json and jb_truthfulqa/mag_dir_truthfulqa.npz. Nothing here
chooses a direction: every direction was fixed by J-B on fit-set questions, and this job
only measures them on the 64 holdout questions, which no selection ever saw.

DIRECTIONS, every one a unit vector signed toward TRUTHFUL (tqa_discovery's convention):
  dct_<rule>_<i>   each factor a J-B rule picked. S-none's carry no sign ("both"): the grid
                   is symmetric, so both signs are measured anyway.
  q2_mean_diff     the Q2 direction. Positive control: at frac +2 here it is Q2's frac -2,
                   and it must reproduce Q2's gain or this job's numbers are not used.
  mag_uQ           MAG's gold-label direction, always (P5 asks for every MAG direction).
  mag_uyM          MAG's label-free direction, only if J-B's G0 found the channel alive.
  rand_0..7        norm-matched random directions: the null each gain is ranked against.

DOSES. frac in FRACS, both signs, times Q2's eps*, so every direction gets the same norm of
push that moved TruthfulQA for the supervised one. POSITIVE frac = toward truthful here,
the reverse of Q2's tables, where the untruthful-pointing vector took negative fracs.

STATISTICS per (direction, frac): the rate with a Wilson interval, the paired gain over the
unsteered answers with an exact McNemar p, and a permutation p: where the gain ranks among
the random directions' gains at the same frac.
"""
import argparse
import json
import os

import numpy as np

import tqa_discovery as td

DS = td.DS
FRACS = (1.0, 2.0)
N_RAND, RAND_SEED = 8, 13                  # up from Q2's 3, per the plan
Q2_CONTROL = "q2_mean_diff"
PREFIX = ""


def path(name):
    return PREFIX + name


def fracs_signed():
    return [s * f for f in FRACS for s in (1.0, -1.0)]


# ------------------------------------------------------------------ pure
def confirm_directions(selection, V, q2_vec, mag=None, n_rand=N_RAND, seed=RAND_SEED):
    """[(name, unit vector)] from J-B's selection. V: (d, k) DCT fit s1. mag: the
    mag_dir npz as a dict, or None. A factor picked by two rules is steered once."""
    V = td.unit(V)
    d = V.shape[0]
    out, seen = [], set()
    for rule, picks in selection["rules"].items():
        for r in picks:
            i = int(r["factor"])
            s = 1 if r["sign"] == "both" else int(r["sign"])
            if (i, s) in seen:
                continue
            seen.add((i, s))
            out.append((f"dct_{rule}_{i}", s * V[:, i]))
    out.append((Q2_CONTROL, td.unit(q2_vec)))
    if mag is not None:
        out.append(("mag_uQ", td.MAG_TRUTHFUL_SIGN * td.unit(mag["u_Q_gold"])))
        if bool(mag.get("g0_alive", False)):
            out.append(("mag_uyM", td.MAG_TRUTHFUL_SIGN * td.unit(mag["u_Q_yM"])))
    rng = np.random.default_rng(seed)
    out += [(f"rand_{j}", td.unit(rng.standard_normal(d))) for j in range(n_rand)]
    return out


def perm_p(gain, rand_gains):
    """One-sided: the share of random directions that do at least as well, with the +1
    that keeps it from ever being 0."""
    r = [g for g in rand_gains if g is not None]
    return (1 + sum(g >= gain for g in r)) / (1 + len(r))


def summarize(rows, col):
    """One row per (direction, frac) from the judged CSV's records."""
    import pandas as pd

    from tqa_baseline import wilson
    from tqa_q2_analyze import mcnemar_exact
    df = pd.DataFrame(rows)
    df["frac"] = df["frac"].astype(float)
    df[col] = df[col].astype(int)
    base = df[df["direction"] == "baseline"].set_index("question")[col]
    out = []
    for (name, frac), g in df[df["direction"] != "baseline"].groupby(["direction", "frac"]):
        arm = g.set_index("question")[col]
        gained, lost, p, n = mcnemar_exact(base, arm)
        k = int(arm.sum())
        lo, hi = wilson(k, len(arm))
        out.append({"direction": name, "frac": frac, "n": len(arm), "rate": k / len(arm),
                    "lo": lo, "hi": hi, "base_rate": float(base.mean()),
                    "gain": (gained - lost) / n if n else 0.0, "gained": gained,
                    "lost": lost, "mcnemar_p": p})
    rand = {}
    for r in out:
        if r["direction"].startswith("rand_"):
            rand.setdefault(r["frac"], []).append(r["gain"])
    for r in out:
        r["perm_p"] = (perm_p(r["gain"], rand.get(r["frac"], []))
                       if not r["direction"].startswith("rand_") else "")
    return sorted(out, key=lambda r: (r["direction"], r["frac"]))


# ------------------------------------------------------------------ stages
def load_selection():
    p = td.path(f"dct_selection_{DS}.json")
    if not os.path.exists(p):
        raise SystemExit(f"[confirm] {p} missing: J-B has not finished")
    return json.load(open(p))


def stage_steer(device, limit=0):
    import pandas as pd
    import torch

    import dct_steer_utils as su
    from reach_hop import load_meta

    out = path(f"d3_steer_{DS}.csv")
    fit = td.load_fit("s1")
    mp = td.mag_dir_path()
    mag = dict(np.load(mp)) if os.path.exists(mp) else None
    if mag is None:
        print(f"[confirm] {mp} missing: no MAG directions this run", flush=True)
    dirs = confirm_directions(load_selection(), fit[0], td.q2_vector(), mag)
    recs = [{"question": q} for q in
            pd.read_csv(td.HOLDOUT)["question"].astype(str).str.strip()]
    if limit:
        recs, dirs = recs[:limit], [d for d in dirs if not d[0].startswith("rand_")][:2] \
            + [d for d in dirs if d[0].startswith("rand_")][:1]
    eps = td.q2_eps_star()
    done = {}
    if os.path.exists(out):
        for r in td.read_csv(out):
            key = (r["direction"], float(r["frac"]))
            done[key] = done.get(key, 0) + 1
    src, _, _, model_name = load_meta(DS)
    tok, model, dev = su.load_model(device, model_name=model_name)
    print(f"[confirm] {len(dirs)} directions x {len(fracs_signed())} doses x {len(recs)} "
          f"holdout questions, eps* {eps:.4g}", flush=True)
    with su.Steerer(model, src) as st:
        if done.get(("baseline", 0.0), 0) < len(recs):
            td.steer_block(model, tok, st, "baseline", None, 0.0, recs, out, tag=0.0)
        for name, vec in dirs:
            for f in fracs_signed():
                if done.get((name, f), 0) >= len(recs):
                    continue
                td.steer_block(model, tok, st, name, vec, f * eps, recs, out, tag=f)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def stage_judge(device):
    from judge_audit import judge_resumable
    from judges.local_hf import TruthJudge
    tj = TruthJudge(device)
    judge_resumable(td.read_csv(path(f"d3_steer_{DS}.csv")), tj.score,
                    path(f"d3_judged_{DS}.csv"))


def stage_summary():
    col = load_selection()["score_col"]
    rows = summarize(td.read_csv(path(f"d3_judged_{DS}.csv")), col)
    td.write_new(path(f"d3_summary_{DS}.csv"), rows)
    print(f"[confirm] scored on `{col}`; frac > 0 is toward truthful", flush=True)
    for r in rows:
        if r["direction"].startswith("rand_"):
            continue
        print(f"[confirm] {r['direction']:18s} frac {r['frac']:+.0f}  rate {r['rate']:.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}]  base {r['base_rate']:.3f}  "
              f"+{r['gained']}/-{r['lost']}  McNemar p {r['mcnemar_p']:.3g}  "
              f"perm p {r['perm_p']:.3g}", flush=True)
    ctl = [r for r in rows if r["direction"] == Q2_CONTROL and r["frac"] == max(FRACS)]
    if ctl and ctl[0]["gained"] <= ctl[0]["lost"]:
        print(f"[confirm] !!!! the positive control does not gain at frac +{max(FRACS):g}. "
              "Q2's effect did not reproduce in this job, so no other row here is "
              "readable as a steering result.", flush=True)


STAGES = ("steer", "judge", "summary")


def main(argv=None):
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", default="",
                    help="prefix this job's outputs (smoke). J-B's inputs are read with "
                         "--jb-prefix, so a smoke can read J-B's smoke files")
    ap.add_argument("--jb-prefix", default="")
    a = ap.parse_args(argv)
    PREFIX, td.PREFIX = a.prefix, a.jb_prefix
    if a.limit and not a.prefix:
        raise SystemExit("--limit writes partial outputs; give it a --prefix")
    for st in STAGES if a.stage == "all" else (a.stage,):
        print(f"=== tqa_confirm: {st} ===", flush=True)
        if st == "summary" and os.path.exists(path(f"d3_summary_{DS}.csv")):
            print("[confirm] summary exists, skipping", flush=True)
            continue
        if st == "steer":
            stage_steer(a.device, a.limit)
        elif st == "judge":
            stage_judge(a.device)
        else:
            stage_summary()


if __name__ == "__main__":
    main()
