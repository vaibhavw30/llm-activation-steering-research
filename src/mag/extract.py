"""extract.py — MAG forward passes -> mag_acts_<ds>.npz (all-layer last-token readouts for the
8 operators' ingredients + y^M). Run once per dataset. gemma-2-2b base, fp32.

    python -m mag.extract cities.csv --device mps
    python -m mag.extract cities.csv --limit 20 --device cpu   # smoke
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ on path
import dct_steer_utils as su
from mag.config import Q_TRUTH, Q_SUFFIX, E_FEWSHOT
from mag.verdict import compute_verdicts

DATASET_DIR = "got_datasets"
BATCH = 16
MAX_LENGTH = 96


def _readouts(model, tokenizer, texts, device, max_length=MAX_LENGTH):
    """Last non-pad-token hidden state at every layer for each text. Returns (L+1, n, d)."""
    tokenizer.padding_side = "right"
    per = []
    with torch.no_grad():
        for s in range(0, len(texts), BATCH):
            batch = texts[s:s + BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple (L+1) of (B,seq,d)
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                idx = int(last[b].item())
                per.append(torch.stack([h[b, idx, :] for h in hs]).float().cpu().numpy())
            print(f"    {min(s + BATCH, len(texts))}/{len(texts)}", flush=True)
    arr = np.stack(per, axis=0)               # (n, L+1, d)
    return np.transpose(arr, (1, 0, 2)).astype(np.float32)  # (L+1, n, d)


def _const_readout(model, tokenizer, text, device):
    """Single-text readout -> (L+1, d)."""
    return _readouts(model, tokenizer, [text], device)[:, 0, :]


def extract_mag(dataset_file, device, limit=None, q_prefix=Q_TRUTH, q_suffix=Q_SUFFIX,
                flip_labels=False, out=None, loaded=None, max_length=MAX_LENGTH):
    """Defaults reproduce every existing mag_acts_<ds>.npz exactly.

    q_prefix / q_suffix: the verdict question wrapped around each statement.
    flip_labels: store 1 - label. TruthfulQA's label 1 is UNTRUTHFUL, while y^M = 1 means
        the model said "yes, true", so without the flip agree(y^M, gold) and u_Q_gold
        would both come out sign-inverted against every other dataset.
    out: output path, default mag_acts_<ds>.npz in the cwd.
    loaded: an already-loaded (tok, model, dev), so a caller that has the model up
        does not load it twice.
    max_length: token cap for every pass. Truncation is on the right, so a cap below the
        longest prompt silently drops the verdict suffix. 96 fits the GoT statements;
        TruthfulQA's Q/A rows under a verdict question need more."""
    ds = dataset_file.replace(".csv", "")
    df = pd.read_csv(f"{DATASET_DIR}/{dataset_file}")
    statements = df["statement"].astype(str).tolist()
    labels = df["label"].to_numpy().astype(int)
    if flip_labels:
        labels = 1 - labels
    if limit is not None:
        statements, labels = statements[:limit], labels[:limit]
    n = len(statements)
    print(f"[mag/extract] {ds}: {n} statements on {device}", flush=True)

    tok, model, dev = loaded or su.load_model(device)

    prefixed = [q_prefix + p + q_suffix for p in statements]
    print("  y^M ...", flush=True)
    verd = compute_verdicts(model, tok, prefixed, dev, max_length=max_length)
    yesno = ["yes" if v else "no" for v in verd["ymL"]]

    print("  A_p ...", flush=True);        A_p = _readouts(model, tok, statements, dev, max_length)
    print("  A_Qp ...", flush=True);       A_Qp = _readouts(model, tok, prefixed, dev, max_length)
    print("  A_Qpv ...", flush=True)
    A_Qpv = _readouts(model, tok, [pf + " " + a for pf, a in zip(prefixed, yesno)], dev,
                      max_length)
    print("  A_verdict ...", flush=True);  A_verd = _readouts(model, tok, yesno, dev, max_length)
    print("  A_EQp ...", flush=True)
    A_EQp = _readouts(model, tok, [E_FEWSHOT + "Statement: " + p + q_suffix for p in statements], dev,
                      max_length)

    print("  constants ...", flush=True)
    A_Q = _const_readout(model, tok, q_prefix.rstrip(), dev)
    A_empty = _const_readout(model, tok, "", dev)   # single-token empty input (BOS only)

    out = out or f"mag_acts_{ds}.npz"
    np.savez_compressed(out, A_p=A_p, A_Qp=A_Qp, A_Qpv=A_Qpv, A_verdict=A_verd, A_EQp=A_EQp,
                        A_Q=A_Q, A_empty=A_empty,
                        ymL=verd["ymL"], margin=verd["margin"], conf=verd["conf"],
                        labels=labels, statements=np.array(statements, dtype=object),
                        model=su.MODEL_NAME, q_prefix=q_prefix, q_suffix=q_suffix,
                        labels_flipped=bool(flip_labels))
    agree = float((verd["ymL"] == labels).mean())
    print(f"[mag/extract] wrote {out}  shape A_Qp={A_Qp.shape}  agree(y^M,gold)={agree:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    extract_mag(a.dataset, a.device, a.limit)


if __name__ == "__main__":
    main()
