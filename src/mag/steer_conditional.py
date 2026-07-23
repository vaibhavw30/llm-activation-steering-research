"""Arm A2 — composed steering: inject the question-mode vector v_Q (gate) together with
tau * mean_diff (content) on free-form factual stems, for OLMo judging. Hypothesis: the truth
axis is causally potent only when the model is in 'being asked about truth' mode.

    python -m mag.steer_conditional --dataset cities --device mps
    python -m mag.steer_conditional --dataset cities --device cpu --limit 2   # smoke
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
from steer_supervised import FACTUAL_PROMPTS

TAUS7 = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
GATES = (0, 1)
MAX_NEW_TOKENS = 8          # matches dct_warm_steer.py so judged runs are comparable


def direction_name(gate):
    return f"meandiff_gate{gate}"


def composed_vec(gate, tau, vq_unit, md_unit, apn):
    """gate*apn*v_Q + tau*apn*mean_diff; None for the untouched (gate=0, tau=0) baseline."""
    if gate == 0 and float(tau) == 0.0:
        return None
    vq = unit(np.asarray(vq_unit, np.float64))
    md = unit(np.asarray(md_unit, np.float64))
    return float(gate) * float(apn) * vq + float(tau) * float(apn) * md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke)")
    a = ap.parse_args()
    ds = a.dataset
    md_npz = np.load(f"mag_dir_{ds}.npz")
    layer = int(md_npz["layer"]); apn = float(md_npz["A_prefix_norm"])
    vq = md_npz["v_Q_unit"]
    td = np.load(f"truth_dir_{ds}.npz")
    assert int(td["layer"]) == layer, (
        f"truth_dir layer {int(td['layer'])} != mag layer {layer}; both vectors must live "
        f"in the same layer space to be co-injected")
    md = td["mean_diff"]
    prompts = FACTUAL_PROMPTS[:a.limit] if a.limit else FACTUAL_PROMPTS
    print(f"[a2] {ds}: layer={layer} apn={apn:.3f} gates={GATES} "
          f"{len(TAUS7)} taus x {len(prompts)} prompts", flush=True)
    tok, model, dev = su.load_model(a.device)
    rows = [("direction", "scale", "prompt", "completion")]
    with su.Steerer(model, layer) as st:
        for gate in GATES:
            for tau in TAUS7:
                vec = composed_vec(gate, tau, vq, md, apn)
                st.set(None if vec is None else torch.tensor(vec, dtype=torch.float32))
                for p in prompts:
                    c = su.generate(model, tok, p, MAX_NEW_TOKENS)
                    rows.append((direction_name(gate), tau, p, c))
                print(f"  gate={gate} tau={tau:+.1f} done", flush=True)
    out = f"mag_conditional_{ds}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[a2] wrote {out}")


if __name__ == "__main__":
    main()
