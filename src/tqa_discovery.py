"""tqa_discovery.py: which unsupervised directions move TruthfulQA truth, and how stable are they?

    PYTHONPATH=src python src/tqa_discovery.py --stage all --device cuda          # CLUSTER, J-B
    PYTHONPATH=src python src/tqa_discovery.py --stage all --device cuda \\
        --limit 16 --prefix smoke_                                                # its smoke

Written for docs/PLAN_PI_FEEDBACK_2026-09-18.md, job J-B (round 1). Tracks D (DCT on
TruthfulQA), G0 (is MAG's verdict channel alive here?) and the readout half of X. Every
stage writes only new files; a stage whose output exists is skipped, and the screen
resumes direction by direction, so a resubmit after a timeout loses one direction.

STAGES, in the order `all` runs them:
  geometry  D1 + D-R1. For each DCT fit on disk (s1 = U1's, s2 = the --seed=326 refit in
            dct_s2_truthfulqa/): the best single factor's |cos| to the supervised
            directions against a random basis of the same size, the share of each
            direction inside span(V) against chance k/d, and the cosine to the top-potency
            factor. Then the two fits against each other: principal cosines between the
            two 512-factor subspaces, and each top factor's best match in the other fit.
  screen    D2, S-beh's data. 96 fit-set questions (none of them in either DCT fit's
            64-statement draw, none in the holdout) answered under: no steering, the Q2
            direction at its working dose (the positive control), the top K_SCREEN
            factors by potency in both signs, and N_RAND_SCREEN norm-matched random
            directions. One dose for all: SCREEN_FRAC x Q2's eps*, the push that moved
            TruthfulQA for the supervised direction.
  judge     The screen, judged by the v2 allenai judges (J-A's fixed prompts).
  select    D2. The three registered rules, S-none / S-geo / S-beh, with the 48/48
            split-half stability gate on S-beh and D-R1's cross-seed cosine on every pick.
            Writes dct_selection_truthfulqa.json, which J-C (D3) and J-D (X) read.
  mag       G0. y^M on the TruthfulQA fit set under two verdict phrasings, then the full
            MAG extraction under the better one, and mag_dir_truthfulqa.npz from it.
  readout   X, readout half. Probes trained on one dataset read on another at layer 11,
            and the cosine between every truth direction the matrix will steer.

SIGN. TruthfulQA's label 1 is UNTRUTHFUL (prep_truthfulqa). Every direction this module
hands on is flipped to point toward TRUTHFUL, by TRUTHFUL_SIGN and the two constants
next to it, because the Q0 inversion already caused one misreading.
"""
import argparse
import csv
import json
import os
import random

import numpy as np

DS = "truthfulqa"
LAYER = 11                                   # DCT source layer = Q2's steering layer
FITS = {"s1": ".", "s2": "dct_s2_truthfulqa"}
WORK = "jb_truthfulqa"                       # MAG npz files and the card's analyze.py run
HOLDOUT = "got_datasets/truthfulqa_holdout.csv"
GOLD = "judge_audit_gold_truthfulqa.csv"     # J-A's output; decides the screen's score column

# The screen, registered 2026-09-18 before any of it ran.
N_SCREEN, SCREEN_SEED = 96, 20260919
K_SCREEN = 16                                # top factors by potency, both signs
K_SNONE = 4                                  # S-none: what the margins battery takes
TOP_BOTH = 3                                 # S-beh: top-3 on BOTH halves, or no winner
N_RAND_SCREEN, RAND_SEED = 4, 11
SCREEN_FRAC = 2.0                            # x Q2's eps*: frac -2, Q2's registered dose
MAX_NEW_TOKENS = 48                          # Q1/Q2's decoder budget
STABLE_COS = 0.5                             # D-R1: below this a pick is seed-specific
DCT_FIT_SAMPLES = 64                         # run_dct_data --num-samples, both fits
DCT_FIT_SEEDS = (325, 326)                   # s1 (the script default) and s2

# +1 if the dataset's label 1 means true. mean_diff = mean(label 1) - mean(label 0).
TRUTHFUL_SIGN = {"cities": 1, "common_claim_true_false": 1, "truthfulqa": -1}
# The Q2 vector is J^T w for w = mean_diff_tgt, which points at label 1 (untruthful), and
# Q2 steered it at NEGATIVE scales. Truthful is its negative.
Q2_TRUTHFUL_SIGN = -1
# u_Q = class_mean_diff(InputDelta) = (label 0) - (label 1) of A_Qp - A_p, with MAG labels
# 1 = true (TruthfulQA's flipped on extraction to match). The class signal lives in A_p, the
# bare statement (cities: cos +0.988 through A_p, -0.04 through A_Qp), so the operator's
# minus sign flips it back: u_Q points at TRUE. This was -1 until 2026-09-18, which read
# the operator as if it were A_p alone; tests/test_xfer.py checks it on cities' activations.
MAG_TRUTHFUL_SIGN = 1

# G0. Two phrasings, so a dead channel under one is not mistaken for a dead model.
MAG_PHRASINGS = {
    "true": ("Answer in yes or no only. Is the answer in the following exchange true?\n",
             "\nVerdict:"),
    "correct": ("Answer in yes or no only. Is the answer in the following exchange "
                "correct?\n", "\nVerdict:"),
}
MAG_MIN_NO, MAG_MIN_AGREE = 0.2, 0.6         # G0's registered pass bar
MAG_MAX_LENGTH = 192                         # the Q/A rows plus the verdict question

PREFIX = ""                                  # set by --prefix; smoke runs write smoke_*


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure: geometry
def unit(v, axis=0):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def potency_order(U):
    """Factor indices by ||U_i||, largest first: funnel_utils.top_k_by_potency's ranking."""
    return [int(i) for i in np.argsort(np.linalg.norm(np.asarray(U), axis=0))[::-1]]


def factor_alignment(V, t, R):
    """Best single factor for target t. V and R are (d, k) unit columns, R random."""
    c = V.T @ t
    best = int(np.argmax(np.abs(c)))
    rmax = float(np.abs(R.T @ t).max())
    return {"best_factor": best, "best_cos": float(c[best]),
            "best_abs_cos": float(abs(c[best])), "random_max_abs_cos": rmax,
            "ratio_vs_random": float(abs(c[best]) / (rmax + 1e-12))}


def span_fraction(Q, t):
    """Share of unit vector t's squared norm inside the column space of orthonormal Q."""
    return float(((Q.T @ t) ** 2).sum())


def principal_cosines(A, B):
    """Cosines of the principal angles between span(A) and span(B), largest first."""
    qa, _ = np.linalg.qr(np.asarray(A, np.float64))
    qb, _ = np.linalg.qr(np.asarray(B, np.float64))
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def best_cross_cos(v, V_other):
    """Largest |cos| between one unit vector and any unit column of another fit."""
    return float(np.abs(V_other.T @ v).max())


def geometry_rows(fit, V, U, src_dirs, tgt_dirs, seed=42):
    """D1 for one fit. src_dirs / tgt_dirs: {name: unit vector} at the source and target
    layers. V is compared at the source layer, U at the target layer, which is where
    each lives."""
    rng = np.random.default_rng(seed)
    out = []
    for space, M, dirs in (("V@src", unit(V), src_dirs), ("U@tgt", unit(U), tgt_dirs)):
        d, k = M.shape
        R = unit(rng.standard_normal((d, k)))
        Q, _ = np.linalg.qr(M)
        Qr, _ = np.linalg.qr(R)
        top = potency_order(U)[0]
        for name, t in dirs.items():
            row = {"fit": fit, "space": space, "target": name, "k": k, "d": d}
            row.update(factor_alignment(M, t, R))
            sf, chance = span_fraction(Q, t), span_fraction(Qr, t)
            row.update({"span_frac": sf, "span_chance": chance,
                        "span_x_chance": sf / (chance + 1e-12),
                        "top_potency_factor": top, "cos_top_potency": float(M[:, top] @ t)})
            out.append(row)
    return out


def stability_rows(V1, U1, V2, k_top=K_SCREEN, seed=42):
    """D-R1: fit s1 against fit s2, and both against a random pair of the same size."""
    V1, V2 = unit(V1), unit(V2)
    d, k = V1.shape
    rng = np.random.default_rng(seed)
    pc = principal_cosines(V1, V2)
    pr = principal_cosines(rng.standard_normal((d, k)), rng.standard_normal((d, k)))
    out = [{"kind": "subspace", "factor": "", "value": float(pc.mean()),
            "random": float(pr.mean()), "stat": "mean principal cosine"},
           {"kind": "subspace", "factor": "", "value": float(np.median(pc)),
            "random": float(np.median(pr)), "stat": "median principal cosine"},
           {"kind": "subspace", "factor": "", "value": float((pc > 0.9).mean()),
            "random": float((pr > 0.9).mean()), "stat": "share of principal cosines > 0.9"}]
    R2 = unit(rng.standard_normal((d, k)))
    for i in potency_order(U1)[:k_top]:
        out.append({"kind": "factor", "factor": i, "value": best_cross_cos(V1[:, i], V2),
                    "random": best_cross_cos(V1[:, i], R2),
                    "stat": "best |cos| in the other fit"})
    return out


# ------------------------------------------------------------------ pure: the screen
def dct_fit_questions(df, num_samples=DCT_FIT_SAMPLES, seed=325):
    """The questions run_dct_data.load_statements(balanced=True) draws for a fit. Same
    pandas calls in the same order, so the same rows; kept torch-free here."""
    import pandas as pd
    per = num_samples // 2
    pos = df[df["label"] == 1].sample(min(per, int((df["label"] == 1).sum())),
                                      random_state=seed)
    neg = df[df["label"] == 0].sample(min(per, int((df["label"] == 0).sum())),
                                      random_state=seed)
    got = pd.concat([pos, neg]).sample(frac=1, random_state=seed).head(num_samples)
    return set(got["question"].astype(str).str.strip())


def screen_questions(fit_questions, exclude, n=N_SCREEN, seed=SCREEN_SEED):
    """n questions from the fit set, none in `exclude`, deterministic."""
    pool = sorted(set(fit_questions) - set(exclude))
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} eligible screen questions, need {n}")
    return random.Random(seed).sample(pool, n)


def split_halves(questions, seed=SCREEN_SEED):
    """Two disjoint halves, A and B, as sets."""
    qs = sorted(questions)
    random.Random(seed + 1).shuffle(qs)
    h = len(qs) // 2
    return set(qs[:h]), set(qs[h:])


def screen_directions(V, U, q2_vec, k=K_SCREEN, n_rand=N_RAND_SCREEN, seed=RAND_SEED):
    """[(name, unit vector or None)] in the order the screen runs them. q2_vec is
    already signed toward truthful."""
    V = unit(V)
    d = V.shape[0]
    out = [("baseline", None), ("q2_mean_diff", unit(q2_vec))]
    for i in potency_order(U)[:k]:
        out += [(f"dct_{i}_pos", V[:, i]), (f"dct_{i}_neg", -V[:, i])]
    rng = np.random.default_rng(seed)
    out += [(f"rand_{j}", unit(rng.standard_normal(d))) for j in range(n_rand)]
    return out


def paired_gains(rows, col, questions):
    """{direction: mean over `questions` of (steered - baseline) on `col`}. Paired by
    question, so a question missing from either side is left out of that direction."""
    by = {}
    for r in rows:
        if r["question"] in questions:
            by.setdefault(r["direction"], {})[r["question"]] = int(r[col])
    base = by.get("baseline", {})
    out = {}
    for name, got in by.items():
        if name == "baseline":
            continue
        qs = [q for q in got if q in base]
        if qs:
            out[name] = sum(got[q] - base[q] for q in qs) / len(qs)
    return out


def top_names(gains, candidates, k):
    """The k best candidates by gain; ties broken by name, so the result is stable."""
    ranked = sorted((n for n in candidates if n in gains), key=lambda n: (-gains[n], n))
    return ranked[:k]


def select_sbeh(gains_a, gains_b, candidates, top=TOP_BOTH):
    """S-beh: a candidate must be in the top `top` on BOTH halves, and have a positive
    gain on both. Returns (winner or None, the names that were top on both)."""
    both = [n for n in top_names(gains_a, candidates, top)
            if n in top_names(gains_b, candidates, top)
            and gains_a[n] > 0 and gains_b[n] > 0]
    if not both:
        return None, []
    return max(both, key=lambda n: (gains_a[n] + gains_b[n], n)), both


def parse_dct_name(name):
    """'dct_17_neg' -> (17, -1)."""
    _, i, s = name.split("_")
    return int(i), (1 if s == "pos" else -1)


def info_judge_passes(gold_rows, bar=None):
    """J-A's registered bar, recomputed from its gold file: every informativeness side at
    or above INFO_PASS under the fixed Helpful: prompt."""
    from judge_audit import INFO_PASS, per_side_accuracy
    acc = per_side_accuracy(gold_rows, "informative", "expected_informative")
    return bool(acc) and all(a >= (INFO_PASS if bar is None else bar)
                             for _, a in acc.values())


def mag_g0_row(phrasing, ym, truth):
    """G0's numbers for one phrasing. ym and truth are 0/1 arrays, 1 = true."""
    ym, truth = np.asarray(ym, int), np.asarray(truth, int)
    tpr = float((ym[truth == 1] == 1).mean()) if (truth == 1).any() else float("nan")
    tnr = float((ym[truth == 0] == 0).mean()) if (truth == 0).any() else float("nan")
    share_no = float((ym == 0).mean())
    agree = float((ym == truth).mean())
    return {"phrasing": phrasing, "n": int(len(ym)), "share_no": share_no, "agree": agree,
            "balanced_acc": (tpr + tnr) / 2,
            "passes": int(share_no >= MAG_MIN_NO and agree >= MAG_MIN_AGREE)}


def choose_phrasing(g0_rows):
    """The passing phrasing with the best agreement, else the first registered one."""
    ok = [r for r in g0_rows if int(r["passes"])]
    if ok:
        return max(ok, key=lambda r: float(r["agree"]))["phrasing"]
    return next(iter(MAG_PHRASINGS))


def cosine_rows(dirs):
    """Long-format pairwise cosines, every unordered pair once."""
    names = sorted(dirs)
    return [{"a": a, "b": b, "cos": float(unit(dirs[a]) @ unit(dirs[b]))}
            for i, a in enumerate(names) for b in names[i + 1:]]


# ------------------------------------------------------------------ file helpers
def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def write_new(p, rows):
    if os.path.exists(p):
        raise SystemExit(f"[disc] {p} exists; this module never overwrites.")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[disc] wrote {p}  n={len(rows)}", flush=True)


def load_fit(fit):
    """(V, U) as (d, k) float64 arrays, or None if this fit is not complete on disk."""
    d = FITS[fit]
    pv, pu = os.path.join(d, f"dct_V_{DS}.pt"), os.path.join(d, f"dct_U_{DS}.pt")
    if not (os.path.exists(pv) and os.path.exists(pu)):
        return None
    import torch
    return (torch.load(pv, map_location="cpu").double().numpy(),
            torch.load(pu, map_location="cpu").double().numpy())


def truth_dirs(ds, tgt=False):
    """{mean_diff, grad} for a dataset, unit and signed toward TRUTHFUL."""
    z = np.load(f"truth_dir_{'tgt_' if tgt else ''}{ds}.npz")
    s = TRUTHFUL_SIGN[ds]
    return {k: s * unit(z[k]) for k in ("mean_diff", "grad")}


def q2_vector():
    """The Q2 steering direction, rebuilt exactly as reach_steer.arm_mean builds it, then
    signed toward truthful."""
    from reach_steer import _load_common
    (_, _, _, _, _, _, mz, acts, _, store_names) = _load_common(DS)
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    ks = store_names.index("mean_diff_tgt")
    jtw = np.asarray(mz["jtw"], np.float64)[y == 1, ks, :]
    return Q2_TRUTHFUL_SIGN * unit(jtw.mean(axis=0))


def mag_dir_path():
    return os.path.join(WORK, path(f"mag_dir_{DS}.npz"))


def q2_eps_star():
    summ = json.load(open(f"reach_summary_{DS}.json"))
    return float(summ["directions"]["mean_diff_tgt"]["median_eps_star"])


# ------------------------------------------------------------------ stages
def stage_geometry():
    src, tgt = truth_dirs(DS), truth_dirs(DS, tgt=True)
    rows = []
    fits = {f: load_fit(f) for f in FITS}
    for f, vu in fits.items():
        if vu is None:
            print(f"[geometry] fit {f} not on disk ({FITS[f]}), skipped", flush=True)
            continue
        rows += geometry_rows(f, vu[0], vu[1], src, tgt)
    for r in rows:
        print(f"[geometry] {r['fit']} {r['space']:6s} {r['target']:9s} best #{r['best_factor']}"
              f" |cos| {r['best_abs_cos']:.3f} vs random {r['random_max_abs_cos']:.3f} "
              f"({r['ratio_vs_random']:.1f}x)  span {r['span_frac']:.3f} vs chance "
              f"{r['span_chance']:.3f}  cos(top) {r['cos_top_potency']:+.3f}", flush=True)
    write_new(path(f"dct_geometry_{DS}.csv"), rows)
    if fits["s1"] is not None and fits["s2"] is not None:
        st = stability_rows(fits["s1"][0], fits["s1"][1], fits["s2"][0])
        for r in st:
            print(f"[stability] {r['stat']:34s} {r['factor']!s:>4s} {r['value']:.3f} "
                  f"(random {r['random']:.3f})", flush=True)
        write_new(path(f"dct_stability_{DS}.csv"), st)
    else:
        print("[stability] needs both fits; D-R1 NOT measured this run", flush=True)


def screen_frame(limit=0):
    """The screen's question set, with its A/B half, as records."""
    import pandas as pd
    df = pd.read_csv(f"got_datasets/{DS}.csv")
    fit_qs = set(df["question"].astype(str).str.strip())
    hold = set(pd.read_csv(HOLDOUT)["question"].astype(str).str.strip())
    assert not fit_qs & hold, "holdout questions leaked into the fit set"
    exclude = set().union(*(dct_fit_questions(df, seed=s) for s in DCT_FIT_SEEDS))
    qs = screen_questions(fit_qs, exclude | hold)
    a, _ = split_halves(qs)
    recs = [{"question": q, "half": "A" if q in a else "B"} for q in qs]
    return recs[:limit] if limit else recs


def stage_screen(device, limit=0):
    import torch

    import dct_steer_utils as su
    from reach_hop import load_meta

    out = path(f"dct_screen_{DS}.csv")
    fit = load_fit("s1")
    if fit is None:
        raise SystemExit("[screen] U1's fit (dct_V/U_truthfulqa.pt) is not here")
    dirs = screen_directions(fit[0], fit[1], q2_vector())
    if limit:
        dirs = dirs[:2] + dirs[2:4] + dirs[-1:]         # baseline, q2, one factor, one random
    qs = screen_frame(limit)
    scale = SCREEN_FRAC * q2_eps_star()
    done = {}
    if os.path.exists(out):
        for r in read_csv(out):
            done[r["direction"]] = done.get(r["direction"], 0) + 1
    src, _, _, model_name = load_meta(DS)
    assert src == LAYER, f"Q2 meta says source layer {src}, this module assumes {LAYER}"
    tok, model, dev = su.load_model(device, model_name=model_name)
    print(f"[screen] {len(dirs)} directions x {len(qs)} questions at scale {scale:.4g} "
          f"(= {SCREEN_FRAC} x Q2 eps*), layer {src}", flush=True)
    with su.Steerer(model, src) as st:
        for name, vec in dirs:
            if done.get(name, 0) >= len(qs):
                continue
            steer_block(model, tok, st, name, vec, scale, qs, out)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def steer_block(model, tok, st, name, vec, scale, recs, out, tag=None):
    """Generate one answer per record with `scale * vec` added at the steering layer (vec
    None = unsteered), and append the block to `out`. Records carry `question` and any
    extra columns to copy through (`half`, ...). `tag` is written as `frac` if given."""
    import torch

    import dct_steer_utils as su
    from prep_truthfulqa import prompt_of
    from tqa_baseline import first_answer

    st.set(None if vec is None or scale == 0 else
           torch.tensor(scale * np.asarray(vec), dtype=torch.float32))
    block = []
    for r in recs:
        p = prompt_of(r["question"])
        raw = su.generate_raw(model, tok, p, MAX_NEW_TOKENS)
        row = {"direction": name, "scale": 0.0 if vec is None else scale}
        if tag is not None:
            row["frac"] = tag
        row.update({k: v for k, v in r.items()})
        row.update(prompt=p, completion=raw.replace("\n", " ").strip(),
                   answer=first_answer(raw))
        block.append(row)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(block[0].keys()))
        if new:
            w.writeheader()
        w.writerows(block)
    print(f"[steer] {name} {'' if tag is None else f'frac {tag:+g} '}done", flush=True)


def stage_judge(device):
    from judge_audit import judge_resumable
    from judges.local_hf import TruthJudge
    rows = read_csv(path(f"dct_screen_{DS}.csv"))
    tj = TruthJudge(device)
    judge_resumable(rows, tj.score, path(f"dct_screen_judged_{DS}.csv"))


def stage_select():
    judged = read_csv(path(f"dct_screen_judged_{DS}.csv"))
    passed = os.path.exists(GOLD) and info_judge_passes(read_csv(GOLD))
    col = "truthful_and_informative" if passed else "truthful"
    print(f"[select] info judge {'passed' if passed else 'FAILED or unmeasured'} J-A's gold "
          f"bar, so the screen scores `{col}`", flush=True)
    qa = {r["question"] for r in judged if r["half"] == "A"}
    qb = {r["question"] for r in judged if r["half"] == "B"}
    g_all = paired_gains(judged, col, qa | qb)
    g_a, g_b = paired_gains(judged, col, qa), paired_gains(judged, col, qb)
    cands = sorted(n for n in g_all if n.startswith("dct_"))
    rand = {n: g_all[n] for n in g_all if n.startswith("rand_")}
    table = [{"direction": n, "gain_all": g_all[n], "gain_A": g_a.get(n), "gain_B": g_b.get(n)}
             for n in sorted(g_all, key=lambda n: -g_all[n])]
    def fmt(x):
        return "  n/a " if x is None else f"{x:+.3f}"
    for r in table:
        print(f"[select] {r['direction']:16s} all {fmt(r['gain_all'])}  A {fmt(r['gain_A'])}"
              f"  B {fmt(r['gain_B'])}", flush=True)
    pos = g_all.get("q2_mean_diff")
    if pos is None or pos <= 0:
        print(f"[select] !!!! the positive control (Q2's direction) gains {pos} here. The "
              "screen cannot see the effect it was built to find, so S-beh below is "
              "NOT trustworthy.", flush=True)

    fit1, fit2 = load_fit("s1"), load_fit("s2")
    V1 = unit(fit1[0])
    V2 = unit(fit2[0]) if fit2 is not None else None

    def pick(i, sign, **extra):
        cc = best_cross_cos(V1[:, i], V2) if V2 is not None else None
        return dict(factor=i, sign=sign, cross_seed_cos=cc,
                    seed_specific=None if cc is None else bool(cc < STABLE_COS), **extra)

    order = potency_order(fit1[1])
    s_none = [pick(i, "both", potency_rank=r) for r, i in enumerate(order[:K_SNONE])]
    t = truth_dirs(DS)["mean_diff"]
    c = V1.T @ t
    g = int(np.argmax(np.abs(c)))
    s_geo = [pick(g, int(np.sign(c[g])), cos_truthful_mean_diff=float(c[g]))]
    win, both = select_sbeh(g_a, g_b, cands)
    s_beh = []
    if win is not None:
        i, s = parse_dct_name(win)
        s_beh = [pick(i, s, gain_all=g_all[win], gain_A=g_a[win], gain_B=g_b[win],
                      cos_truthful_mean_diff=float(s * c[i]),
                      beats_every_random=bool(all(g_all[win] > v for v in rand.values())))]
    sel = {"score_col": col, "info_judge_passed": passed,
           "screen_scale_frac_of_q2_eps": SCREEN_FRAC, "positive_control_gain": pos,
           "random_gains": rand, "sbeh_top_on_both_halves": both,
           "sbeh_status": "winner" if s_beh else "no stable winner",
           "rules": {"S-none": s_none, "S-geo": s_geo, "S-beh": s_beh},
           "screen_table": table}
    p = path(f"dct_selection_{DS}.json")
    if os.path.exists(p):
        raise SystemExit(f"[disc] {p} exists; this module never overwrites.")
    json.dump(sel, open(p, "w"), indent=2)
    print(f"[select] S-none {[r['factor'] for r in s_none]}  S-geo {s_geo[0]['factor']}"
          f"{'+' if s_geo[0]['sign'] > 0 else '-'}  S-beh {sel['sbeh_status']}"
          f"{' ' + win if win else ''}  -> {p}", flush=True)


def stage_mag(device, limit=0):
    import pandas as pd
    import torch

    import dct_steer_utils as su
    from mag.directions import build_directions, build_lead_directions
    from mag.extract import extract_mag
    from mag.verdict import compute_verdicts

    os.makedirs(WORK, exist_ok=True)
    npz = os.path.join(WORK, path(f"mag_acts_{DS}.npz"))
    g0_path = path(f"mag_g0_{DS}.csv")
    df = pd.read_csv(f"got_datasets/{DS}.csv")
    if limit:
        df = df.head(limit)
    stmts = df["statement"].astype(str).tolist()
    truth = 1 - df["label"].to_numpy().astype(int)
    loaded = su.load_model(device)
    tok, model, dev = loaded
    worst = max(len(tok(MAG_PHRASINGS[k][0] + s + MAG_PHRASINGS[k][1])["input_ids"])
                for k in MAG_PHRASINGS for s in stmts)
    if worst > MAG_MAX_LENGTH:
        raise SystemExit(f"[mag] longest verdict prompt is {worst} tokens > "
                         f"MAG_MAX_LENGTH={MAG_MAX_LENGTH}; right truncation would cut "
                         "the verdict suffix")
    # Resumable in two pieces: G0's table, then the extraction. Each is skipped if its
    # file is already here, so a timeout in the extraction does not redo G0.
    g0 = read_csv(g0_path) if os.path.exists(g0_path) else []
    for k, (pre, suf) in MAG_PHRASINGS.items() if not g0 else ():
        v = compute_verdicts(model, tok, [pre + s + suf for s in stmts], dev,
                             max_length=MAG_MAX_LENGTH)
        g0.append(mag_g0_row(k, v["ymL"], truth))
        r = g0[-1]
        print(f"[G0] '{k}': share no {r['share_no']:.3f}  agree {r['agree']:.3f}  balanced "
              f"{r['balanced_acc']:.3f}  -> {'ALIVE' if r['passes'] else 'dead'}", flush=True)
    if not os.path.exists(g0_path):
        write_new(g0_path, g0)
    ph = choose_phrasing(g0)
    alive = any(int(r["passes"]) for r in g0)
    print(f"[G0] verdict channel {'ALIVE' if alive else 'DEAD under both phrasings'}; "
          f"extracting under '{ph}'", flush=True)
    pre, suf = MAG_PHRASINGS[ph]
    if os.path.exists(npz):
        print(f"[mag] {npz} exists, extraction skipped", flush=True)
    else:
        extract_mag(f"{DS}.csv", device, limit=limit or None, q_prefix=pre, q_suffix=suf,
                    flip_labels=True, out=npz, loaded=loaded, max_length=MAG_MAX_LENGTH)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    cache = dict(np.load(npz, allow_pickle=True))
    src = truth_dirs(DS)
    fit = load_fit("s1")
    top = unit(fit[0])[:, potency_order(fit[1])[0]] if fit is not None else None
    out = build_directions(cache, LAYER, cache["labels"], cache["ymL"],
                           src["mean_diff"], src["grad"], top)
    out.update(build_lead_directions(cache, LAYER, cache["labels"]))
    out.update(phrasing=ph, g0_alive=alive)
    dpath = mag_dir_path()
    np.savez(dpath, **out)
    print(f"[G0] wrote {dpath}  cos(u_Q_gold, truthful mean_diff) = "
          f"{float(out['cos_uGold_mean_diff']):+.3f} (u_Q points at FALSE, so about -1 "
          "means it IS the supervised axis)", flush=True)


def stage_readout():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    data = {}
    for ds in TRUTHFUL_SIGN:
        p = f"activations/acts_{ds}.npz"
        if not os.path.exists(p):
            print(f"[readout] {p} missing, {ds} left out", flush=True)
            continue
        z = np.load(p, allow_pickle=True)
        y = z["labels"].astype(int)
        data[ds] = (z["activations"][LAYER].astype(np.float64),
                    y if TRUTHFUL_SIGN[ds] > 0 else 1 - y)        # 1 = true, everywhere
    rows = []
    for a, (Xa, ya) in data.items():
        sc = StandardScaler().fit(Xa)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xa), ya)
        md = unit(Xa[ya == 1].mean(0) - Xa[ya == 0].mean(0))
        for b, (Xb, yb) in data.items():
            if a == b:
                acc = float(cross_val_score(LogisticRegression(max_iter=2000),
                                            sc.transform(Xa), ya, cv=5).mean())
            else:
                acc = float(clf.score(sc.transform(Xb), yb))
            rows.append({"train": a, "test": b, "probe_acc": acc,
                         "meandiff_auc": float(roc_auc_score(yb, Xb @ md)),
                         "in_distribution": int(a == b)})
            print(f"[readout] {a:24s} -> {b:24s} probe {acc:.3f}  1-D mean_diff AUC "
                  f"{rows[-1]['meandiff_auc']:.3f}", flush=True)
    write_new(path(f"readout_transfer_{DS}.csv"), rows)

    dirs = {}
    for ds in TRUTHFUL_SIGN:
        if os.path.exists(f"truth_dir_{ds}.npz"):
            for k, v in truth_dirs(ds).items():
                dirs[f"{ds}:{k}"] = v
    dirs[f"{DS}:q2_jtw"] = q2_vector()
    for ds, p in (("cities", "mag_dir_cities.npz"),
                  (DS, mag_dir_path())):
        if os.path.exists(p):
            dirs[f"{ds}:mag_u_Q"] = MAG_TRUTHFUL_SIGN * unit(np.load(p)["u_Q_gold"])
    sel_p = path(f"dct_selection_{DS}.json")
    fit = load_fit("s1")
    if os.path.exists(sel_p) and fit is not None:
        V = unit(fit[0])
        for rule, picks in json.load(open(sel_p))["rules"].items():
            for r in picks:
                s = 1 if r["sign"] == "both" else int(r["sign"])
                dirs[f"{DS}:dct_{rule}_{r['factor']}"] = s * V[:, r["factor"]]
    write_new(path(f"direction_cosines_{DS}.csv"), cosine_rows(dirs))


STAGES = ("geometry", "screen", "judge", "select", "mag", "readout")
# Single-output stages are skipped when their file exists. screen and judge resume inside.
# mag's done-marker is its last file, the directions npz; it resumes inside otherwise.
STAGE_OUTPUT = {"geometry": lambda: path(f"dct_geometry_{DS}.csv"),
                "select": lambda: path(f"dct_selection_{DS}.json"),
                "mag": mag_dir_path,
                "readout": lambda: path(f"readout_transfer_{DS}.csv")}


def main(argv=None):
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="questions / statements (smoke)")
    ap.add_argument("--prefix", default="", help="prefix every output name (smoke)")
    a = ap.parse_args(argv)
    PREFIX = a.prefix
    if a.limit and not a.prefix:
        raise SystemExit("--limit writes partial outputs; give it a --prefix so they "
                         "cannot be mistaken for the real run's")
    for st in STAGES if a.stage == "all" else (a.stage,):
        print(f"=== tqa_discovery: {st} ===", flush=True)
        if st in STAGE_OUTPUT and os.path.exists(STAGE_OUTPUT[st]()):
            print(f"[disc] {STAGE_OUTPUT[st]()} exists, stage done, skipping", flush=True)
            continue
        if st == "geometry":
            stage_geometry()
        elif st == "screen":
            stage_screen(a.device, a.limit)
        elif st == "judge":
            stage_judge(a.device)
        elif st == "select":
            stage_select()
        elif st == "mag":
            stage_mag(a.device, a.limit)
        elif st == "readout":
            stage_readout()


if __name__ == "__main__":
    main()
