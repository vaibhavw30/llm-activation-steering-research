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


def run_e4(ds, directions, device):
    """directions: list of dicts {name, unit_dir (d,), layer, a_prefix_norm}."""
    tok, model, dev = su.load_model(device)
    steer_rows = [("direction", "scale", "prompt", "completion")]
    flip_rows = [("direction", "tau", "flip_rate")]

    for d in directions:
        uvec = unit(np.asarray(d["unit_dir"], np.float64))
        with su.Steerer(model, int(d["layer"])) as st:
            # baseline (tau=0) yes/no answers for the flip comparison
            base = {}
            st.set(None)
            for s in YESNO_STATEMENTS:
                base[s] = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
            for tau in TAUS:
                vec = torch.tensor(injected_vector(tau, uvec, d["a_prefix_norm"]), dtype=torch.float32)
                st.set(None if tau == 0 else vec)
                # free-form factual (judged)
                for p in FACTUAL_PROMPTS:
                    c = su.generate(model, tok, p, 8)
                    steer_rows.append((d["name"], tau, p, c))
                # matched-format yes/no flips vs baseline
                flips = 0
                for s in YESNO_STATEMENTS:
                    a = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX, 3))
                    flips += int(a != base[s] and a != "?")
                flip_rows.append((d["name"], tau, flips / len(YESNO_STATEMENTS)))
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
    a = ap.parse_args()
    ds = a.dataset
    md = np.load(f"mag_dir_{ds}.npz")
    directions = [
        {"name": "mag_u_gold", "unit_dir": md["u_Q_gold_unit"],
         "layer": int(md["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])},
        {"name": "mag_u_yM", "unit_dir": md["u_Q_yM_unit"],
         "layer": int(md["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])},
    ]
    # supervised directions at the same layer, calibrated with the same A_prefix_norm
    if os.path.exists(f"truth_dir_{ds}.npz"):
        td = np.load(f"truth_dir_{ds}.npz")
        for nm in ("mean_diff", "grad"):
            directions.append({"name": f"sup_{nm}", "unit_dir": unit(td[nm]),
                               "layer": int(td["layer"]), "a_prefix_norm": float(md["A_prefix_norm"])})
    run_e4(ds, directions, a.device)


if __name__ == "__main__":
    main()
