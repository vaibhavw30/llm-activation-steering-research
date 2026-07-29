"""calibrate_scale.py — Horizon-1 1.1 Task A4: input_scale without fitting DCT factors.

run_dct_data.py derives input_scale from dct.SteeringCalibrator(target_ratio=0.5)
BEFORE ExponentialDCT.fit (src/run_dct_data.py:136-149). The minimal refusal control
needs the calibrated budget yardstick but not the factors, so this reproduces exactly
that prefix — same tokenizer settings (left padding, left truncation), same X/Y
extraction, same calibrator, same token_idxs, same seed — and writes the result into
dct_meta_<ds>.json in place. Everything else in the meta file is left untouched.

    PYTHONPATH=src python src/calibrate_scale.py --dataset refusal --device cuda
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dct
from run_dct_data import load_statements, parse_token_idxs

CALIBRATION_SAMPLE_SIZE = 30
FACTOR_BATCH_SIZE = 128
FORWARD_BATCH_SIZE = 8
MAX_LENGTH = 64
SEED = 325                 # run_dct_data.py default
DEFAULT_MODEL = "google/gemma-2-2b"

# load_statements is imported from run_dct_data rather than re-implemented here:
# the calibration population is part of the eps* yardstick this task exists to
# preserve, so it must track the truth sampler exactly, not drift silently if that
# sampler is ever edited. Signature: load_statements(dataset, num_samples, balanced,
# seed) -> (statements, labels) (src/run_dct_data.py:66).


def run(ds, device):
    with open(f"dct_meta_{ds}.json") as f:
        meta = json.load(f)
    model_name = meta.get("model", DEFAULT_MODEL)
    src, tgt = int(meta["source_layer"]), int(meta["target_layer"])
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left",
                                        truncation_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="eager")
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    d_model = model.config.hidden_size
    sliced = dct.SlicedModel(model, start_layer=src, end_layer=tgt,
                             layers_name="model.layers")
    stmts, _ = load_statements(ds, int(meta["num_samples"]),
                               bool(meta.get("balanced", True)), SEED)
    n = len(stmts)
    enc = tok(stmts, return_tensors="pt", padding="longest", truncation=True,
              max_length=MAX_LENGTH)
    T = enc["input_ids"].shape[1]
    attn = enc["attention_mask"].to(torch.float)
    X = torch.zeros(n, T, d_model, device="cpu")
    Y = torch.zeros(n, T, d_model, device="cpu")
    for t in range(0, n, FORWARD_BATCH_SIZE):
        with torch.no_grad():
            ids = enc["input_ids"][t:t + FORWARD_BATCH_SIZE].to(dev)
            msk = enc["attention_mask"][t:t + FORWARD_BATCH_SIZE].to(dev)
            h_src = model(ids, attention_mask=msk,
                          output_hidden_states=True).hidden_states[src]
            X[t:t + FORWARD_BATCH_SIZE] = h_src.cpu()
            Y[t:t + FORWARD_BATCH_SIZE] = sliced(h_src).cpu()
        print(f"[calib] X/Y {min(t + FORWARD_BATCH_SIZE, n)}/{n}", flush=True)
    da = dct.DeltaActivations(
        sliced, target_position_indices=parse_token_idxs(meta["token_idxs"]))
    input_scale = float(dct.SteeringCalibrator(target_ratio=0.5).calibrate(
        da, X.to(da.device), Y.to(da.device),
        factor_batch_size=FACTOR_BATCH_SIZE,
        calibration_sample_size=CALIBRATION_SAMPLE_SIZE,
        attention_mask=attn.to(da.device)))
    meta["input_scale"] = input_scale
    if n != int(meta["num_samples"]):
        # record the realized sample count, not the request, matching
        # run_dct_data.py:170 (`"num_samples": n`) — the meta must not overstate
        # the population the calibration actually ran over.
        meta["num_samples"] = n
    with open(f"dct_meta_{ds}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[calib] input_scale = {input_scale:.4f} -> dct_meta_{ds}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.dataset, a.device)


if __name__ == "__main__":
    main()
