"""E4 — calibrated steering. Inject alpha(tau)*unit(d) at a direction's native layer via the
existing Steerer, on both a matched-format yes/no set (verdict flips) and the free-form factual
set (OLMo-judged). CSV schemas match judge_results.run_steer so scoring is unchanged.

    python -m mag.steer --dataset cities --device mps
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dct_steer_utils as su
from funnel_utils import unit
from mag.config import TAUS, Q_TRUTH, Q_SUFFIX

# free-form factual set (same 32 stems as the corrected steer rerun) — imported to stay in sync
from steer_supervised import FACTUAL_PROMPTS

# 24 neutral factual statements for the matched-format yes/no flip test (true statements)
YESNO_STATEMENTS = [
    "The capital of Japan is Tokyo.", "Paris is the capital of France.",
    "Water is made of hydrogen and oxygen.", "The Earth orbits the Sun.",
    "Two plus two equals four.", "The Pacific is the largest ocean.",
    "Mount Everest is the tallest mountain.", "The sun rises in the east.",
    "Gold's chemical symbol is Au.", "Humans breathe out carbon dioxide.",
    "The heart pumps blood.", "Ten minus four equals six.",
    "The freezing point of water is zero Celsius.", "Rome is in Italy.",
    "The square root of nine is three.", "Plants absorb carbon dioxide.",
    "Sydney is in Australia.", "The opposite of hot is cold.",
    "A week has seven days.", "George Washington was the first US president.",
    "Cairo is in Egypt.", "Mercury is closest to the Sun.",
    "The sky is blue.", "Shakespeare wrote Romeo and Juliet.",
]


def alpha(tau, a_prefix_norm):
    return float(tau) * float(a_prefix_norm)


def injected_vector(tau, unit_dir, a_prefix_norm):
    return alpha(tau, a_prefix_norm) * np.asarray(unit_dir, np.float64)


def _yes_no_answer(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "?"


def run_e4(ds, directions, device, limit=0, taus=None):
    """directions: list of dicts {name, unit_dir (d,), layer, a_prefix_norm}.
    limit>0 caps both prompt lists (smoke); taus overrides the tau sweep."""
    taus = taus if taus is not None else TAUS
    factual = FACTUAL_PROMPTS[:limit] if limit else FACTUAL_PROMPTS
    yesno = YESNO_STATEMENTS[:limit] if limit else YESNO_STATEMENTS
    tok, model, dev = su.load_model(device)
    steer_rows = [("direction", "scale", "prompt", "completion")]
    flip_rows = [("direction", "tau", "flip_rate")]

    for d in directions:
        uvec = unit(np.asarray(d["unit_dir"], np.float64))
        with su.Steerer(model, int(d["layer"])) as st:
            # baseline (tau=0) yes/no answers for the flip comparison
            base = {}
            st.set(None)
            for s in yesno:
                base[s] = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
            for tau in taus:
                vec = torch.tensor(injected_vector(tau, uvec, d["a_prefix_norm"]), dtype=torch.float32)
                st.set(None if tau == 0 else vec)
                # free-form factual (judged)
                for p in factual:
                    c = su.generate(model, tok, p, 8)
                    steer_rows.append((d["name"], tau, p, c))
                # matched-format yes/no flips vs baseline
                flips = 0
                for s in yesno:
                    a = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
                    flips += int(a != base[s] and a != "?")
                flip_rows.append((d["name"], tau, flips / len(yesno)))
                print(f"  {d['name']} tau={tau:+.1f} done", flush=True)

    sp = f"mag_steer_{ds}.csv"
    with open(sp, "w", newline="") as f:
        csv.writer(f).writerows(steer_rows)
    fp = f"mag_verdict_flips_{ds}.csv"
    with open(fp, "w", newline="") as f:
        csv.writer(f).writerows(flip_rows)
    print(f"[mag/steer] wrote {sp} and {fp}")
    return sp, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap both prompt lists (smoke test)")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep only matching direction names")
    a = ap.parse_args()
    ds = a.dataset
    md = np.load(f"mag_dir_{ds}.npz")
    layer = int(md["layer"]); apn = float(md["A_prefix_norm"])
    directions = [
        {"name": "mag_u_gold", "unit_dir": md["u_Q_gold_unit"],
         "layer": layer, "a_prefix_norm": apn},
        {"name": "mag_u_yM", "unit_dir": md["u_Q_yM_unit"],
         "layer": layer, "a_prefix_norm": apn},
    ]
    # E4 lead #1 — the divergent operators (read truth off the mean_diff axis)
    for op in ("Prefixed", "Answered", "QuestionDelta", "FewShot"):
        if f"u_{op}_unit" in md:
            directions.append({"name": f"mag_u_{op}", "unit_dir": md[f"u_{op}_unit"],
                               "layer": layer, "a_prefix_norm": apn})
    # E4 lead #3 — the dominant off-truth-axis shift component (not a truth axis; causal probe)
    if "resid_pc1_unit" in md:
        directions.append({"name": "mag_resid_pc1", "unit_dir": md["resid_pc1_unit"],
                           "layer": layer, "a_prefix_norm": apn})
    # supervised directions at the same layer, calibrated with the same A_prefix_norm
    if os.path.exists(f"truth_dir_{ds}.npz"):
        td = np.load(f"truth_dir_{ds}.npz")
        for nm in ("mean_diff", "grad"):
            directions.append({"name": f"sup_{nm}", "unit_dir": unit(td[nm]),
                               "layer": int(td["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])})
    if a.only:
        subs = [s.strip() for s in a.only.split(",") if s.strip()]
        directions = [d for d in directions if any(s in d["name"] for s in subs)]
        if not directions:
            raise SystemExit(f"--only {a.only!r} matched no directions")
    print(f"[mag/steer] {ds}: {len(directions)} directions -> " +
          ", ".join(d["name"] for d in directions), flush=True)
    run_e4(ds, directions, a.device, limit=a.limit)


if __name__ == "__main__":
    main()
