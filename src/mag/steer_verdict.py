"""Arm A1 — verdict-mode steering with logit readout. Steer each direction at its layer with
alpha(tau) = tau * A_prefix_norm, and read the model's first-token p_yes/p_no on
Q_TRUTH + statement + Q_SUFFIX over balanced test-split statements. No sampling, no judge.

    python -m mag.steer_verdict --dataset cities --device mps
    python -m mag.steer_verdict --dataset cities --device cpu --limit 2 --only sup_mean_diff
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dct_steer_utils as su
import funnel_utils as fu
from funnel_utils import unit
from mag.config import Q_TRUTH, Q_SUFFIX, YES_VARIANTS, NO_VARIANTS
from mag.steer import injected_vector
from mag.verdict import first_token_ids, verdict_from_logits

TAUS7 = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
N_PER_CLASS = 64
SPLIT_SEED = 42


def sample_statements(ds, n_per_class=N_PER_CLASS, seed=SPLIT_SEED):
    """Balanced (statement, label) pairs drawn only from the standard 80/20 test split,
    so none were used to fit mean_diff/grad. Capped by the smaller class pool."""
    rows = list(csv.DictReader(open(f"got_datasets/{ds}.csv")))
    labels = [int(r["label"]) for r in rows]
    idx = np.arange(len(rows))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    out = []
    for lab in (1, 0):
        pool = sorted(i for i in test_idx if labels[i] == lab)
        if len(pool) > n_per_class:
            pool = sorted(rng.choice(pool, n_per_class, replace=False))
        out += [(rows[i]["statement"], lab) for i in pool]
    return out


def build_directions(ds):
    """The 5 A1 directions as (name, unit_vec, layer); plus the shared A_prefix_norm."""
    md_npz = np.load(f"mag_dir_{ds}.npz")
    layer = int(md_npz["layer"]); apn = float(md_npz["A_prefix_norm"])
    td = np.load(f"truth_dir_{ds}.npz")
    dirs = [
        ("sup_mean_diff", unit(np.asarray(td["mean_diff"], np.float64)), int(td["layer"])),
        ("sup_grad", unit(np.asarray(td["grad"], np.float64)), int(td["layer"])),
        ("mag_resid_pc1", unit(np.asarray(md_npz["resid_pc1_unit"], np.float64)), layer),
    ]
    V, U, _ = fu.load_dct(ds)
    top = fu.top_k_by_potency(V, U, 1)[0]
    dirs.append(("cold_top", unit(V[:, top].astype(np.float64)), layer))
    rng = np.random.default_rng(SPLIT_SEED)
    dirs.append(("random_unit", unit(rng.standard_normal(V.shape[0])), layer))
    return dirs, apn


def verdict_rows(model, tok, prompts, device, batch_size=16):
    """Per-prompt first-token verdict dicts (p_yes/p_no/margin), batched with right padding.
    Mirrors mag.verdict.compute_verdicts but keeps every row's full readout."""
    yes_ids = first_token_ids(tok, YES_VARIANTS)
    no_ids = first_token_ids(tok, NO_VARIANTS)
    tok.padding_side = "right"
    out = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(device)
            logits = model(**enc).logits
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                row = logits[b, int(last[b].item()), :].float().cpu().numpy()
                out.append(verdict_from_logits(row, yes_ids, no_ids))
    return out


def run_a1(ds, device, n_per_class=N_PER_CLASS, only=None, taus=TAUS7):
    pairs = sample_statements(ds, n_per_class)
    dirs, apn = build_directions(ds)
    if only:
        subs = [s.strip() for s in only.split(",") if s.strip()]
        dirs = [d for d in dirs if any(s in d[0] for s in subs)]
        if not dirs:
            raise SystemExit(f"--only {only!r} matched no directions")
    prompts = [Q_TRUTH + s + Q_SUFFIX for s, _ in pairs]
    print(f"[a1] {ds}: {len(dirs)} dirs x {len(taus)} taus x {len(prompts)} statements "
          f"(apn={apn:.3f})", flush=True)
    tok, model, dev = su.load_model(device)
    rows = [("direction", "tau", "statement", "label", "p_yes", "p_no", "margin")]
    for name, uvec, layer in dirs:
        with su.Steerer(model, layer) as st:
            for tau in taus:
                st.set(None if tau == 0.0 else torch.tensor(
                    injected_vector(tau, uvec, apn), dtype=torch.float32))
                res = verdict_rows(model, tok, prompts, dev)
                for (s, lab), r in zip(pairs, res):
                    rows.append((name, tau, s, lab, r["p_yes"], r["p_no"], r["margin"]))
                print(f"  {name} tau={tau:+.1f} done", flush=True)
    out = f"mag_verdict_logits_{ds}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[a1] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap statements per class (smoke)")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep only matching direction names")
    a = ap.parse_args()
    run_a1(a.dataset, a.device, n_per_class=a.limit or N_PER_CLASS, only=a.only)


if __name__ == "__main__":
    main()
