"""xfer_cities.py: every truth direction, from both datasets, steered on cities (J-D1, X).

    PYTHONPATH=src python src/xfer_cities.py --stage all --device cuda              # CLUSTER
    PYTHONPATH=src python src/xfer_cities.py --stage all --device cuda \\
        --limit 8 --prefix smoke_                                                  # its smoke
    PYTHONPATH=src python src/xfer_cities.py --stage summary                        # LAPTOP

Written for docs/PLAN_PI_FEEDBACK_2026-09-18.md, section 8, job J-D1 (round 2). The P8
cell is `tqa:dct_S-beh_*` (backup `tqa:dct_S-geo_*`) on this target. The direction list
and the dose grid are xfer_common's; this module is the cities readout.

PROMPT. "The city of {city} is in the country of", so the next token IS a country. The
token-space stems ("The city of X is in") are NOT used: gemma continues 174 of 200 of them
with " the" ("... is in the north of Shandong Province"), so a next-token country readout
there measures filler words (the 2026-09-18 laptop smoke; see the plan).

TARGET. The distinct cities behind token_acts_cities.npz's rows (continuity with the
token-space set), keeping only those whose correct country is readable at one token:
country names are taken without a leading "the ", and a first token shared by two
countries (" South": South Korea, South Africa) is dropped from the pool, as is a
multi-word country whose first word is an ordinary word (" North": North Korea, where the
laptop check saw the oracle push "in the north of" instead), along with the cities whose
answer it is. 178 of the 192 cities remain before that last rule.

TWO READOUTS, so the result does not rest on one:
  logit   one forward, steered. margin_correct = logit[correct country's first token] -
          the best other country's, from the post-norm z against the plain unembedding
          (token_steer's convention, no softcap). Blind to answer form.
  gen     a greedy 12-token continuation (repetition penalty 1.0, as the token-space runs).
          Its first sentence is scored for the first country named (correct, wrong, none),
          its word count, and xfer_common.incoherent. Sees form.

POSITIVE CONTROL. An oracle built in this job on these prompts: for each city, the
cheapest wrong country j_tgt (highest logit among the pool's wrong countries), a =
unit(W[j_tgt] - W[j_top]), and the actuator unit(J_11^T a) at +1 and +2 of its certified
budget M / ||J_11^T a|| (broadcast convention, as the Steerer injects). Its target is a
real country by construction. A cities null is readable only if the oracle's hit rate at
+2 is at least ORACLE_MIN in this job.

NULL. N_RAND norm-matched random directions (32 here: no judge, so they are cheap, and
8 would floor a permutation p at 1/9). Every effect is ranked among them at the same dose.

OUTCOME of each direction, read at READ_UNIT / READ_FRAC (toward FALSE: cities sits near
ceiling, so the false side is where facts can move), by classify(). A readout "moves" when
its paired test (Wilcoxon on the margin, exact McNemar on gen_correct) has p <= 0.05 AND
the effect clears the random null (xfer_common.beyond_null):
  a            logit and gen both move: facts flip
  b            neither moves, but word count or incoherence does beyond random: form only
  c            nothing beyond random
  logit-only / gen-only   the two readouts disagree, which is itself evidence for C
"""
import argparse
import json
import os
import re

import numpy as np

import tqa_discovery as td
import xfer_common as xc

DS = "cities"
PROMPT = "The city of {city} is in the country of"
N_RAND, RAND_SEED = 32, 29
ORACLE_FRACS = (1.0, 2.0)
ORACLE_MIN = 0.2
READ_UNIT, READ_FRAC = xc.READ_UNIT, -xc.READ_FRAC     # toward false
MAX_NEW, REP_PENALTY, BATCH = 12, 1.0, 32
PREFIX = ""


def path(name):
    return PREFIX + name


# ------------------------------------------------------------------ pure
def bare(country):
    """The name as it follows "the country of": no leading "the "."""
    c = str(country).strip()
    return c[4:] if c.lower().startswith("the ") else c


def first_country(text, countries):
    """The first country named in the first sentence of `text`, or None. At equal start
    positions the longer name wins."""
    seg = str(text).split("\n")[0].split(".")[0]
    best = None
    for c in countries:
        m = re.search(r"(?<![A-Za-z])" + re.escape(c) + r"(?![A-Za-z])", seg)
        if m and (best is None or (m.start(), -len(c)) < (best[0], -len(best[1]))):
            best = (m.start(), c)
    return None if best is None else best[1]


def score_gen(text, correct, countries):
    seg = str(text).split("\n")[0].split(".")[0].strip()
    fc = first_country(text, countries)
    return {"segment": seg, "first_country": fc or "",
            "gen_correct": int(fc == correct), "gen_wrong_country": int(fc is not None and
                                                                      fc != correct),
            "words": xc.word_count(seg), "incoherent": int(xc.incoherent(text))}


# A multi-word country whose first word is an ordinary word: the token is the filler
# ("in the north of ...") as often as the country. North Korea is the one in cities.csv.
GENERIC_FIRST = {"North", "South", "East", "West", "New", "United", "Central", "Republic",
                 "Democratic", "Great"}


def clean_pool(first_ids):
    """{country: first token id} minus every country whose first token another country
    shares (" South" cannot say which country is meant) and every multi-word country whose
    first word is in GENERIC_FIRST."""
    count = {}
    for t in first_ids.values():
        count[t] = count.get(t, 0) + 1
    return {c: t for c, t in first_ids.items() if count[t] == 1
            and not (len(c.split()) > 1 and c.split()[0] in GENERIC_FIRST)}


def wrong_ids(pool, correct):
    """First-token ids of every OTHER country in the pool."""
    return sorted(t for k, t in pool.items() if k != correct)


def targets(pool, limit=0):
    """One record per distinct city behind token_acts_cities.npz's rows whose correct
    country is in the pool, in token_acts order."""
    import pandas as pd
    A = np.load(f"token_acts_{DS}.npz", allow_pickle=True)
    df = pd.read_csv(f"got_datasets/{DS}.csv")
    seen, recs, dropped = set(), [], 0
    for ri in A["row_index"]:
        row = df.iloc[int(ri)]
        city, c = str(row["city"]), bare(row["correct_country"])
        if city in seen:
            continue
        seen.add(city)
        if c not in pool:
            dropped += 1
            continue
        recs.append({"stmt": int(ri), "city": city, "stem": PROMPT.format(city=city),
                     "correct": c})
    print(f"[targets] {len(recs)} cities; {dropped} dropped because their country's "
          "first token is shared", flush=True)
    return recs[:limit] if limit else recs


def countries_of():
    import pandas as pd
    return sorted({bare(c) for c in
                   pd.read_csv(f"got_datasets/{DS}.csv")["correct_country"].astype(str)})


# ------------------------------------------------------------------ steer
def readout(model, tok, stems, dev):
    """(post-norm z as a (B, d) float tensor, model argmax ids). No steering is set here;
    the caller's Steerer state applies."""
    import torch
    enc = tok(stems, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return (out.hidden_states[-1][:, -1].float(),       # left-padded: -1 is the last token
            out.logits[:, -1].argmax(-1).cpu().numpy())


def oracle(model, tok, recs, pool, dev):
    """Per-city (unit actuator at LAYER, certified budget, j_tgt), cached. j_top is the
    model's own argmax on the full vocabulary, j_tgt the highest-logit WRONG country."""
    p = path(f"xfer_{DS}_oracle.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["vec"], z["eps"], z["j_tgt"]
    from token_jac import margins
    W = model.get_output_embeddings().weight.float()
    A, M, jt = [], [], []
    for b0 in range(0, len(recs), BATCH):
        br = recs[b0:b0 + BATCH]
        z, _ = readout(model, tok, [r["stem"] for r in br], dev)
        lg = (z @ W.T).cpu().numpy()                      # uncapped logits, full vocab
        for j, r in enumerate(br):
            top = int(lg[j].argmax())
            wr = [t for t in wrong_ids(pool, r["correct"]) if t != top]
            tgt = wr[int(np.argmax(lg[j, wr]))]
            a = (W[tgt] - W[top]).cpu().numpy().astype(np.float64)
            A.append(a / np.linalg.norm(a))
            M.append(float(lg[j, top] - lg[j, tgt]))
            jt.append(tgt)
    A, M, jt = np.asarray(A, np.float32), np.asarray(M), np.asarray(jt)
    m_all, _, v_all, _ = margins(model, tok, [r["stem"] for r in recs], A, [xc.LAYER],
                                 dev, save_dirs=True)
    vec = xc.unit(np.asarray(v_all[:, 0, :], np.float64).T).T
    eps = M / np.maximum(m_all[:, 0], 1e-12)
    np.savez_compressed(p, vec=vec, eps=eps, j_tgt=jt, margin=M)
    print(f"[oracle] layer {xc.LAYER} broadcast budget: median {np.median(eps):.3g} "
          f"(p10 {np.percentile(eps, 10):.3g}, p90 {np.percentile(eps, 90):.3g}); "
          f"targets: {len(set(jt.tolist()))} distinct countries", flush=True)
    return vec, eps, jt


def run_block(model, tok, st, name, unit_, frac, scale, recs, vec, pool, ids_all, jt,
              countries, dev):
    col = {t: i for i, t in enumerate(ids_all)}
    W = model.get_output_embeddings().weight
    rows = []
    for b0 in range(0, len(recs), BATCH):
        br = recs[b0:b0 + BATCH]
        if vec is None:
            st.set(None)
        elif np.ndim(vec) == 2:                                   # oracle: per row
            st.set(xc.steer_vec(vec[b0:b0 + BATCH] * scale[b0:b0 + BATCH, None], 1.0))
        else:
            st.set(xc.steer_vec(vec, scale))
        z, am = readout(model, tok, [r["stem"] for r in br], dev)
        lg = (z @ W[ids_all].float().T).cpu().numpy()
        texts = xc.gen_batch(model, tok, [r["stem"] for r in br], MAX_NEW, REP_PENALTY)
        for j, (r, t) in enumerate(zip(br, texts)):
            c = pool[r["correct"]]
            wr = [col[i] for i in wrong_ids(pool, r["correct"])]
            sc = scale if np.ndim(scale) == 0 else float(scale[b0 + j])
            rows.append({"direction": name, "unit": unit_, "frac": frac, "scale": sc,
                         "stmt": r["stmt"], "city": r["city"], "correct": r["correct"],
                         "margin_correct": float(lg[j, col[c]] - lg[j, wr].max()),
                         "argmax_id": int(am[j]), "argmax_correct": int(am[j] == c),
                         "hit_tgt": int(am[j] == int(jt[b0 + j])),
                         "completion": t.replace("\n", " ").strip(),
                         **score_gen(t, r["correct"], countries)})
    st.set(None)
    return rows


def stage_steer(device, limit=0, jb_prefix=""):
    import torch

    import dct_steer_utils as su
    from reach_hop import load_meta

    out = path(f"xfer_{DS}_steer.csv")
    dirs, missing = xc.build_directions(N_RAND, RAND_SEED, jb_prefix)
    if limit:
        dirs = [d for d in dirs if not d[0].startswith("rand_")][:3] + \
            [d for d in dirs if d[0].startswith("rand_")][:2]
    for m in missing:
        print(f"[xfer] !!!! left out: {m}", flush=True)
    src, _, _, model_name = load_meta(DS)
    assert src == xc.LAYER, f"cities meta says source layer {src}, X assumes {xc.LAYER}"
    tok, model, dev = su.load_model(device, model_name=model_name)
    tok.padding_side = "left"
    countries = countries_of()
    pool = clean_pool({c: int(tok(" " + c, add_special_tokens=False)["input_ids"][0])
                       for c in countries})
    recs = targets(pool, limit)
    ids_all = sorted(set(pool.values()))
    norm_med = xc.median_last_norm(model, tok, [r["stem"] for r in recs])
    eps = xc.eps_star(DS)
    grid = xc.dose_grid(norm_med, eps)
    vo, eo, jt = oracle(model, tok, recs, pool, dev)
    meta = {"prompt": PROMPT, "n_cities": len(recs), "pool_size": len(pool),
            "norm_median_layer11": norm_med, "eps_star": eps,
            "oracle_budget_median": float(np.median(eo)),
            "directions": [d[0] for d in dirs], "left_out": missing,
            "grid": [list(g) for g in grid], "n_rand": N_RAND}
    json.dump(meta, open(path(f"xfer_{DS}_meta.json"), "w"), indent=2)
    print(f"[xfer] {len(dirs)} directions x {len(grid)} doses x {len(recs)} cities; "
          f"median ||h_11|| {norm_med:.4g}, eps* {eps:.4g}", flush=True)

    done = xc.done_blocks(out)
    blocks = [("baseline", "none", 0.0, 0.0, None)]
    blocks += [("oracle", "oracle_eps", f, f * eo, vo) for f in ORACLE_FRACS]
    blocks += [(n, u, f, s, v) for n, v in dirs for u, f, s in grid]
    with su.Steerer(model, xc.LAYER) as st:
        for name, u, f, s, v in blocks:
            if done.get((name, u, float(f)), 0) >= len(recs):
                continue
            xc.append_rows(out, run_block(model, tok, st, name, u, f, s, recs, v, pool,
                                          ids_all, jt, countries, dev))
            print(f"[steer] {name} {u} {f:+g} done", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ summary
def summarize(df):
    """One row per (direction, unit, frac), each effect paired against the baseline on
    the same stem and signed so that larger = more toward truthful than the push asked
    for: effect = sign(frac) x change."""
    from scipy.stats import wilcoxon

    from tqa_baseline import wilson
    from tqa_q2_analyze import mcnemar_exact
    base = df[df["direction"] == "baseline"].set_index("stmt")
    out = []
    for (name, u, f), g in df[df["direction"] != "baseline"].groupby(
            ["direction", "unit", "frac"]):
        g = g.set_index("stmt")
        b = base.loc[g.index]
        s = float(np.sign(f)) or 1.0
        dm = g["margin_correct"] - b["margin_correct"]
        gained, lost, p_mc, n = mcnemar_exact(b["gen_correct"], g["gen_correct"])
        k = int(g["gen_correct"].sum())
        lo, hi = wilson(k, len(g))
        try:
            p_w = float(wilcoxon(dm).pvalue) if (dm != 0).any() else 1.0
        except ValueError:
            p_w = 1.0
        out.append({
            "direction": name, "unit": u, "frac": float(f), "n": len(g),
            "d_margin": float(dm.mean()), "wilcoxon_p": p_w,
            "argmax_correct": float(g["argmax_correct"].mean()),
            "base_argmax_correct": float(b["argmax_correct"].mean()),
            "hit_tgt": float(g["hit_tgt"].mean()),
            "gen_correct": k / len(g), "lo": lo, "hi": hi,
            "base_gen_correct": float(b["gen_correct"].mean()),
            "gen_wrong_country": float(g["gen_wrong_country"].mean()),
            "gained": gained, "lost": lost, "mcnemar_p": p_mc,
            "words": float(g["words"].mean()), "base_words": float(b["words"].mean()),
            "incoherent": float(g["incoherent"].mean()),
            "base_incoherent": float(b["incoherent"].mean()),
            "e_logit": s * float(dm.mean()),
            "e_gen": s * float(g["gen_correct"].mean() - b["gen_correct"].mean()),
            "e_words": abs(float(g["words"].mean() - b["words"].mean())),
            "e_incoh": float(g["incoherent"].mean() - b["incoherent"].mean())})
    rand = {}
    for r in out:
        if r["direction"].startswith("rand_"):
            rand.setdefault((r["unit"], r["frac"]), []).append(r)
    for r in out:
        rs = rand.get((r["unit"], r["frac"]), [])
        r["n_null"] = len(rs)
        for e in ("e_logit", "e_gen", "e_words", "e_incoh"):
            r["p_" + e[2:]] = ("" if r["direction"].startswith("rand_") or not rs
                               else xc.perm_p(r[e], [x[e] for x in rs]))
    return sorted(out, key=lambda r: (r["direction"], r["unit"], r["frac"]))


def classify(r, alpha=xc.ALPHA):
    """The registered outcome for one summary row (see the module docstring)."""
    if r.get("p_logit", "") == "":
        return "no null"
    n = r["n_null"]
    L = r["wilcoxon_p"] <= alpha and xc.beyond_null(r["p_logit"], n, alpha)
    G = r["mcnemar_p"] <= alpha and xc.beyond_null(r["p_gen"], n, alpha)
    F = xc.beyond_null(r["p_words"], n, alpha) or xc.beyond_null(r["p_incoh"], n, alpha)
    if L and G:
        return "a"
    if L:
        return "logit-only"
    if G:
        return "gen-only"
    return "b" if F else "c"


def p8_cell(names):
    """The P8 direction: TQA's S-beh DCT pick, else its S-geo backup, else None."""
    for rule in xc.DCT_ANCHOR_RULES:
        hit = sorted(n for n in names if n.startswith(f"tqa:dct_{rule}_"))
        if hit:
            return hit[0]
    return None


def stage_summary():
    import pandas as pd
    df = pd.read_csv(path(f"xfer_{DS}_steer.csv"))
    rows = summarize(df)
    orc = {r["frac"]: r["hit_tgt"] for r in rows if r["direction"] == "oracle"}
    ok = orc.get(max(ORACLE_FRACS), 0.0) >= ORACLE_MIN
    print(f"[summary] ORACLE hit rate " + "  ".join(f"+{f:g}: {h:.3f}" for f, h in
                                                    sorted(orc.items()))
          + f"  -> {'READABLE' if ok else '!!!! the oracle did not flip: every cities null in this file is UNREADABLE'}",
          flush=True)
    read = [r for r in rows if r["unit"] == READ_UNIT and r["frac"] == READ_FRAC
            and not r["direction"].startswith("rand_") and r["direction"] != "oracle"]
    outcomes = {}
    for r in read:
        outcomes[r["direction"]] = classify(r)
        print(f"[summary] {r['direction']:26s} {READ_UNIT} {READ_FRAC:+g}  "
              f"d_margin {r['d_margin']:+.3f} (perm p {r['p_logit']:.3g})  "
              f"gen_correct {r['gen_correct']:.3f} vs {r['base_gen_correct']:.3f} "
              f"(perm p {r['p_gen']:.3g})  words {r['words']:.1f} vs {r['base_words']:.1f}"
              f"  incoh {r['incoherent']:.2f}  -> {outcomes[r['direction']]}", flush=True)
    p8 = p8_cell(outcomes)
    print(f"[summary] P8 CELL {p8} -> "
          f"{outcomes.get(p8, 'not steered') if ok else 'UNREADABLE (oracle)'}", flush=True)
    for r in rows:
        r["outcome"] = (classify(r) if not r["direction"].startswith("rand_")
                        and r["direction"] != "oracle" else "")
    td.write_new(path(f"xfer_{DS}_summary.csv"), rows)
    p = path(f"xfer_{DS}_outcomes.json")
    if os.path.exists(p):
        raise SystemExit(f"[xfer] {p} exists; this module never overwrites.")
    json.dump({"read_at": [READ_UNIT, READ_FRAC], "oracle_hit": orc, "oracle_ok": ok,
               "p8_cell": p8, "outcomes": outcomes}, open(p, "w"), indent=2)


STAGES = ("steer", "summary")


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
        print(f"=== xfer_cities: {st} ===", flush=True)
        if st == "steer":
            stage_steer(a.device, a.limit, a.jb_prefix)
        elif os.path.exists(path(f"xfer_{DS}_summary.csv")):
            print("[xfer] summary exists, skipping", flush=True)
        else:
            stage_summary()


if __name__ == "__main__":
    main()
