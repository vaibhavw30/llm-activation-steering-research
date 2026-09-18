"""xfer_common.py: what J-D1 (xfer_cities) and J-D2 (xfer_tqa) share. Track X, round 2.

Written for docs/PLAN_PI_FEEDBACK_2026-09-18.md, section 8. Each direction, from each
source dataset, is steered on each target dataset. This module builds the one direction
list both jobs steer, the dose grid, and the statistics; the two job modules only differ
in how they read an answer out.

DIRECTIONS. Every one a unit vector at layer 11, signed toward TRUTHFUL
(tqa_discovery's constants), named `<source>:<family>`:
  <src>:sup_jtw        the supervised direction the reach pipeline steered: the mean of
                       J^T w over label-1 rows, w = mean_diff_tgt. On TQA this is Q2's.
  <src>:mean_diff      the layer-11 mean difference, from truth_dir_<ds>.npz.
  cities:dct_S-none_i  cities' top-K_SNONE factors by potency. No sign: the grid has both.
  cities:dct_S-geo_i   the cities factor with the largest |cos| to cities' mean_diff.
  tqa:dct_<rule>_i     J-B's picks, from dct_selection_truthfulqa.json.
  <src>:dct_ctl_i      a POTENCY-MATCHED random factor from the same fit: the unpicked
                       factor whose ||U_i|| is nearest the anchor pick's (TQA: S-beh, else
                       S-geo; cities: S-geo). "Any DCT factor does this" is a different
                       finding from "this one does".
  <src>:mag_uQ         MAG's gold-label direction. tqa:mag_uyM too if J-B's G0 was alive.
                       cities' u_yM is not steered: its verdict channel is dead (MAG runs).
  rand_j               norm-matched random directions, the null every effect is ranked in.

DOSES, two units, both signs:
  norm  frac x the target's median ||h_11|| at the last prompt token, measured in the job
        on the prompts being steered. The primary unit: a row means the same relative
        push on both targets.
  eps   frac x the target's eps* (reach_summary_<ds>.json, mean_diff_tgt).
"""
import json
import os
import re

import numpy as np

import tqa_discovery as td

LAYER = td.LAYER
SRC_TAG = {"cities": "cities", "truthfulqa": "tqa"}
# Measured 2026-09-18 on the laptop (bf16): median ||h_11|| at the last prompt token is ~115
# on the cities prompts and ~112 on the TQA holdout prompts. Q2's working push, 2 x eps* =
# 29.4, is 0.26 of that, and the cities oracle's budgets run 0.11-0.2 of it. The grid
# brackets both.
NORM_FRACS = (0.125, 0.25, 0.5)
EPS_FRACS = (2.0,)
K_SNONE = td.K_SNONE
CTL_SEED = 23
# The P8 cell is read at this dose; the others are reported beside it. norm 0.25 is the
# grid point nearest Q2's working push (29.4 against TQA's layer-11 norm ~112).
READ_UNIT, READ_FRAC = "norm", 0.25
ALPHA = 0.05
DCT_ANCHOR_RULES = ("S-beh", "S-geo")        # the P8 cell, then its registered backup


# ------------------------------------------------------------------ pure
def unit(v):
    return td.unit(v)


def cities_picks(V, U, md, k=K_SNONE):
    """[(rule, factor, sign)] for the cities fit: S-none (top-k potency, sign "both",
    stored as +1) and S-geo (largest |cos| to cities' mean_diff, signed toward it). There
    is no cities screen, so no S-beh (plan, section 8)."""
    Vn = unit(V)
    out = [("S-none", i, 1) for i in td.potency_order(U)[:k]]
    c = Vn.T @ unit(md)
    g = int(np.argmax(np.abs(c)))
    out.append(("S-geo", g, int(np.sign(c[g])) or 1))
    return out


def selection_picks(selection):
    """[(rule, factor, sign)] from J-B's selection json, S-none's "both" as +1."""
    out = []
    for rule, picks in selection["rules"].items():
        for r in picks:
            out.append((rule, int(r["factor"]), 1 if r["sign"] == "both" else int(r["sign"])))
    return out


def anchor_pick(picks):
    """The pick a potency-matched control is matched to: the first rule in
    DCT_ANCHOR_RULES that has one."""
    for rule in DCT_ANCHOR_RULES:
        for p in picks:
            if p[0] == rule:
                return p
    return None


def potency_matched(U, anchor, exclude, seed=CTL_SEED):
    """(factor, sign): the factor NOT in `exclude` whose potency ||U_i|| is nearest the
    anchor's, sign drawn at random. Ties broken by index."""
    pot = np.linalg.norm(np.asarray(U, np.float64), axis=0)
    cand = [i for i in range(len(pot)) if i not in set(exclude)]
    i = min(cand, key=lambda j: (abs(pot[j] - pot[anchor]), j))
    s = int(np.random.default_rng(seed).choice([-1, 1]))
    return i, s


def dct_directions(tag, V, picks, U, seed=CTL_SEED):
    """[(name, unit vec)] for one fit's picks, deduped on (factor, sign), plus the
    potency-matched control when there is an anchor."""
    Vn = unit(V)
    out, seen = [], set()
    for rule, i, s in picks:
        if (i, s) in seen:
            continue
        seen.add((i, s))
        out.append((f"{tag}:dct_{rule}_{i}", s * Vn[:, i]))
    a = anchor_pick(picks)
    if a is not None:
        j, s = potency_matched(U, a[1], {p[1] for p in picks}, seed)
        out.append((f"{tag}:dct_ctl_{j}", s * Vn[:, j]))
    return out


def random_directions(d, n, seed):
    rng = np.random.default_rng(seed)
    return [(f"rand_{j}", unit(rng.standard_normal(d))) for j in range(n)]


def dose_grid(norm_med, eps_star):
    """[(unit, frac, scale)], both signs, baseline NOT included."""
    out = []
    for f in NORM_FRACS:
        for s in (1.0, -1.0):
            out.append(("norm", s * f, s * f * norm_med))
    if eps_star is not None:
        for f in EPS_FRACS:
            for s in (1.0, -1.0):
                out.append(("eps", s * f, s * f * eps_star))
    return out


def perm_p(effect, rand_effects):
    """One-sided: the share of random directions at least as large, with the +1."""
    r = [e for e in rand_effects if e is not None and np.isfinite(e)]
    return (1 + sum(e >= effect for e in r)) / (1 + len(r))


def beyond_null(p_perm, n_null, alpha=ALPHA):
    """Does a permutation p clear the random null? With n_null randoms the smallest p is
    1/(n_null+1), which is 1/9 for 8: a bare p <= 0.05 rule could never fire there. So the
    bar is alpha, or beating EVERY random when the null is too small to reach alpha (J-C's
    beats_every_random). Always paired with a per-question test in the callers: beating
    eight randoms alone is not evidence of anything."""
    if p_perm == "" or p_perm is None:
        return False
    return float(p_perm) <= max(alpha, 1.0 / (n_null + 1)) + 1e-12


# ------------------------------------------------------------------ loaders
def sup_jtw(ds):
    """The supervised J^T w direction of dataset ds, rebuilt as reach_steer.arm_mean
    builds it, signed toward truthful. mean_diff_tgt points at label 1, so the sign is the
    dataset's TRUTHFUL_SIGN (on TQA that is Q2_TRUTHFUL_SIGN, -1)."""
    from reach_steer import _load_common
    (_, _, _, _, _, _, mz, acts, _, store_names) = _load_common(ds)
    y = np.asarray(acts["labels"]).astype(int)[:mz["margins"].shape[0]]
    ks = store_names.index("mean_diff_tgt")
    jtw = np.asarray(mz["jtw"], np.float64)[y == 1, ks, :]
    return td.TRUTHFUL_SIGN[ds] * unit(jtw.mean(axis=0))


def eps_star(ds):
    return float(json.load(open(f"reach_summary_{ds}.json"))
                 ["directions"]["mean_diff_tgt"]["median_eps_star"])


def _torch_load(p):
    import torch
    return torch.load(p, map_location="cpu").double().numpy()


def build_directions(n_rand, rand_seed, jb_prefix=""):
    """The full list, [(name, unit vec)], and a list of what was left out and why.
    Anything missing is left out LOUDLY, never silently."""
    out, missing = [], []
    for ds in ("cities", "truthfulqa"):
        tag = SRC_TAG[ds]
        for fam, fn in (("sup_jtw", lambda: sup_jtw(ds)),
                        ("mean_diff", lambda: td.truth_dirs(ds)["mean_diff"])):
            try:
                out.append((f"{tag}:{fam}", fn()))
            except FileNotFoundError as e:
                missing.append(f"{tag}:{fam} ({e.filename})")

    if os.path.exists("dct_V_cities.pt") and os.path.exists("dct_U_cities.pt"):
        V, U = _torch_load("dct_V_cities.pt"), _torch_load("dct_U_cities.pt")
        out += dct_directions("cities", V, cities_picks(V, U, td.truth_dirs("cities")
                                                        ["mean_diff"]), U)
    else:
        missing.append("cities DCT (dct_V/U_cities.pt)")

    sel_p = jb_prefix + f"dct_selection_{td.DS}.json"
    fit = td.load_fit("s1")
    if os.path.exists(sel_p) and fit is not None:
        out += dct_directions("tqa", fit[0], selection_picks(json.load(open(sel_p))), fit[1])
    else:
        missing.append(f"TQA DCT ({sel_p} or dct_V/U_truthfulqa.pt)")

    if os.path.exists("mag_dir_cities.npz"):
        m = np.load("mag_dir_cities.npz")
        out.append(("cities:mag_uQ", td.MAG_TRUTHFUL_SIGN * unit(m["u_Q_gold"])))
    else:
        missing.append("cities MAG (mag_dir_cities.npz)")
    mp = os.path.join(td.WORK, jb_prefix + f"mag_dir_{td.DS}.npz")
    if os.path.exists(mp):
        m = dict(np.load(mp))
        out.append(("tqa:mag_uQ", td.MAG_TRUTHFUL_SIGN * unit(m["u_Q_gold"])))
        if bool(m.get("g0_alive", False)):
            out.append(("tqa:mag_uyM", td.MAG_TRUTHFUL_SIGN * unit(m["u_Q_yM"])))
    else:
        missing.append(f"TQA MAG ({mp})")

    d = len(out[0][1])
    out += random_directions(d, n_rand, rand_seed)
    return out, missing


def median_last_norm(model, tok, prompts, layer=LAYER, batch=16):
    """Median ||h_layer|| at the last real token of each prompt, unsteered. hidden_states
    [layer] is the INPUT of decoder layer `layer`, which is where the Steerer adds."""
    import torch
    norms = []
    for b0 in range(0, len(prompts), batch):
        enc = tok(prompts[b0:b0 + batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        am = enc["attention_mask"]
        last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
        norms += hs[torch.arange(len(last)), last].float().norm(dim=-1).cpu().tolist()
    return float(np.median(norms))


def gen_batch(model, tok, prompts, max_new, rep_penalty):
    """Greedy continuations of a batch of prompts, raw (newlines kept). Left padding, so
    the last position of every row is its own last token; a steering vector the Steerer
    adds to pad positions never reaches a real token, since pads are masked."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=rep_penalty, pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True) for o in out]


def steer_vec(vec, scale):
    """A torch tensor for Steerer.set. vec is (d,) for one shared direction or (B, d) for
    per-row directions (the oracle), which becomes (B, 1, d) and broadcasts per row."""
    import torch
    v = torch.tensor(scale * np.asarray(vec, np.float64), dtype=torch.float32)
    return v[:, None, :] if v.ndim == 2 else v


def done_blocks(path, keys=("direction", "unit", "frac")):
    """{(direction, unit, frac): rows written} from a steering CSV, for resume."""
    import csv
    if not os.path.exists(path):
        return {}
    got = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["direction"], r["unit"], float(r["frac"]))
            got[k] = got.get(k, 0) + 1
    return got


def append_rows(path, rows):
    import csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ answer form
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def word_count(text):
    return len(_WORD.findall(str(text)))


def incoherent(text):
    """A cheap flag, not a judge. True when the text has no word at all, repeats one
    word three times running, or is more than 30% characters outside letters, digits,
    whitespace and ordinary punctuation."""
    t = str(text).strip()
    if not t:
        return False                   # an empty answer is empty, scored elsewhere
    words = [w.lower() for w in _WORD.findall(t)]
    if not words:
        return True
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return True
    odd = sum(not (c.isalnum() or c.isspace() or c in ".,;:!?'\"()-") for c in t)
    return odd / len(t) > 0.3
