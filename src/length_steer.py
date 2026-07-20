# src/length_steer.py
"""Length-steering generation: one long greedy completion per (prompt, direction, tau), logging
per-token intrinsic signals, plus cutoff-prefix rows for the OLMo judge.

    python length_steer.py --dataset cities --device cuda
    python length_steer.py --dataset cities --device cpu --limit 4   # smoke

Writes:
    length_steer_<ds>.csv     one row per (direction, tau, prompt): completion + per-position signals
    length_prefixes_<ds>.csv  one row per (direction, tau, prompt, cutoff): prefix text for the judge
                              (schema direction,scale,prompt,completion — direction="<dir>_tau<tau>",
                               scale=<cutoff> — so judge_results.py --mode steer reads it unchanged)
"""
import argparse
import csv
import json
import numpy as np
import torch

import dct_steer_utils as su
from funnel_utils import unit
from length_prompts import get_prompt_set
from length_intrinsics import intrinsic_signals

TAUS = [-1.0, -0.3, 0.0, 0.3, 1.0]   # two-sided: -tau pushes mean_diff toward FALSE, +tau toward TRUE
CUTOFFS = [8, 16, 32, 64, 96]
MAX_NEW_TOKENS = 96


def injected_vector(tau, unit_dir, a_prefix_norm):
    return float(tau) * float(a_prefix_norm) * unit(np.asarray(unit_dir, np.float64))


def load_directions(ds):
    md = np.load(f"mag_dir_{ds}.npz")
    td = np.load(f"truth_dir_{ds}.npz")
    layer = int(md["layer"])
    assert layer == int(td["layer"]), (
        f"mag_dir.layer={layer} != truth_dir.layer={int(td['layer'])}; "
        "mean_diff and resid_pc1 must live at the same layer to inject with a shared norm")
    apn = float(md["A_prefix_norm"])
    dirs = [
        {"name": "mean_diff", "unit_dir": unit(np.asarray(td["mean_diff"], np.float64))},
        {"name": "resid_pc1", "unit_dir": unit(np.asarray(md["resid_pc1_unit"], np.float64))},
    ]
    return layer, apn, dirs


def prefixes_for_cutoffs(token_ids, cutoffs, tokenizer):
    out = {}
    for k in cutoffs:
        if k <= len(token_ids):
            out[k] = tokenizer.decode(token_ids[:k], skip_special_tokens=True).replace("\n", " ").strip()
    return out


def generate_with_logging(model, tok, prompt, max_new_tokens):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3,
            pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    gen_ids = out.sequences[0][inp["input_ids"].shape[1]:].tolist()
    scores = [s[0].float().cpu().numpy() for s in out.scores]   # list of (vocab,)
    # scores and gen_ids are aligned and equal-length (one score per generated token)
    n = min(len(scores), len(gen_ids))
    text = tok.decode(gen_ids[:n], skip_special_tokens=True).replace("\n", " ").strip()
    return text, scores[:n], gen_ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke test)")
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = dataset default (cities 300 / cc 100)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="override generation length (default 96; use small value for CPU smoke tests)")
    a = ap.parse_args()
    ds = a.dataset

    layer, apn, dirs = load_directions(ds)
    prompts = get_prompt_set(ds, n=a.n_prompts or None)
    if a.limit:
        prompts = prompts[:a.limit]
    print(f"[length] {ds}: layer={layer} A_prefix_norm={apn:.3f} "
          f"{len(prompts)} prompts x {len(dirs)} dirs x {len(TAUS)} taus", flush=True)

    tok, model, dev = su.load_model(a.device)
    steer_rows = [("direction", "tau", "prompt", "completion", "max_prob", "entropy", "rep3")]
    prefix_rows = [("direction", "scale", "prompt", "completion")]   # judge schema

    for d in dirs:
        with su.Steerer(model, layer) as st:
            for tau in TAUS:
                vec = None if tau == 0.0 else torch.tensor(
                    injected_vector(tau, d["unit_dir"], apn), dtype=torch.float32)
                st.set(vec)
                for stem, _answer in prompts:
                    text, scores, gen_ids = generate_with_logging(model, tok, stem, a.max_new_tokens)
                    sig = intrinsic_signals(scores, gen_ids)
                    steer_rows.append((d["name"], tau, stem, text,
                                       json.dumps([round(x, 4) for x in sig["max_prob"]]),
                                       json.dumps([round(x, 4) for x in sig["entropy"]]),
                                       json.dumps(sig["rep3"])))
                    for k, ptext in prefixes_for_cutoffs(gen_ids, CUTOFFS, tok).items():
                        prefix_rows.append((f"{d['name']}_tau{tau}", k, stem, ptext))
                print(f"  {d['name']} tau={tau:+.1f} done ({len(prompts)} prompts)", flush=True)

    with open(f"length_steer_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(steer_rows)
    with open(f"length_prefixes_{ds}.csv", "w", newline="") as f:
        csv.writer(f).writerows(prefix_rows)
    print(f"[length] wrote length_steer_{ds}.csv and length_prefixes_{ds}.csv")


if __name__ == "__main__":
    main()
