"""
run_dct_data.py — Stage 3: run DCT on true/false statement datasets.

Mirrors dct_train.py::train_combined_observations but loads statements from our
got_datasets/<dataset>.csv (columns: statement,label) instead of a HF dataset, and
is device-aware (GPU by default). Reuses the paper's SlicedModel / DeltaActivations /
SteeringCalibrator / ExponentialDCT — the method itself is unchanged.

Saves:
    dct_V_<dataset>.pt   unit-norm steering vectors (d_model x num_factors)
    dct_U_<dataset>.pt   output directions
    dct_meta_<dataset>.json   input_scale, layers, sizes (needed to steer/compare later)

Usage (GPU):
    python run_dct_data.py --dataset cities --num-factors 512 --num-iters 30 --num-samples 64
    python run_dct_data.py --dataset common_claim_true_false --device cuda
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import vmap

import dct
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
DATASET_DIR = "got_datasets"


def parse_args():
    p = argparse.ArgumentParser(description="Run DCT on true/false statement data")
    p.add_argument("--dataset", required=True, help="name without .csv, e.g. cities")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-layer", type=int, default=6)
    p.add_argument("--target-layer", type=int, default=18)
    p.add_argument("--num-factors", type=int, default=512)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--num-samples", type=int, default=64, help="# statements to use")
    p.add_argument("--balanced", action="store_true",
                   help="take an equal number of true/false statements")
    p.add_argument("--factor-batch-size", type=int, default=128)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--calibration-sample-size", type=int, default=30)
    p.add_argument("--scale", type=float, default=None, help="skip calibration; use this scale")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--token-idxs", default="-3:")
    p.add_argument("--orthogonalize", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=325)
    return p.parse_args()


def parse_token_idxs(s):
    parts = s.split(":")
    if len(parts) == 1:
        return int(parts[0])
    start = None if parts[0] == "" else int(parts[0])
    end = None if parts[1] == "" else int(parts[1])
    return slice(start, end)


def load_statements(dataset, num_samples, balanced, seed):
    df = pd.read_csv(f"{DATASET_DIR}/{dataset}.csv")
    if balanced:
        per = num_samples // 2
        pos = df[df["label"] == 1].sample(min(per, (df["label"] == 1).sum()), random_state=seed)
        neg = df[df["label"] == 0].sample(min(per, (df["label"] == 0).sum()), random_state=seed)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    df = df.head(num_samples)
    return df["statement"].astype(str).tolist(), df["label"].to_numpy().astype(int)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"  (CUDA not available — falling back to {device})")

    print(f"Loading {args.model_name} ({device}, fp32, eager attention)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # eager attention: the calibrator's forward-mode autodiff (jvp) is unsupported by
    # the fused SDPA/flash kernels (CPU and CUDA alike).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.float32, attn_implementation="eager")
    model.to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False
    d_model = model.config.hidden_size

    sliced_model = dct.SlicedModel(
        model, start_layer=args.source_layer, end_layer=args.target_layer,
        layers_name="model.layers")
    print(f"  slice source={args.source_layer} -> target={args.target_layer}, d_model={d_model}")

    statements, labels = load_statements(args.dataset, args.num_samples, args.balanced, args.seed)
    n = len(statements)
    print(f"  {n} statements ({int(labels.sum())} true / {n-int(labels.sum())} false)")

    # ---- Build X (source) and Y (clean target) over all statements ---------
    enc = tokenizer(statements, return_tensors="pt", padding="longest",
                    truncation=True, max_length=args.max_length)
    max_seq_len = enc["input_ids"].shape[1]
    attention_mask = enc["attention_mask"].to(torch.float)

    X = torch.zeros(n, max_seq_len, d_model, device="cpu")
    Y = torch.zeros(n, max_seq_len, d_model, device="cpu")
    print("Extracting X/Y activations...", flush=True)
    for t in range(0, n, args.forward_batch_size):
        with torch.no_grad():
            ids = enc["input_ids"][t:t+args.forward_batch_size].to(device)
            msk = enc["attention_mask"][t:t+args.forward_batch_size].to(device)
            hs = model(ids, attention_mask=msk, output_hidden_states=True).hidden_states
            h_source = hs[args.source_layer]
            X[t:t+args.forward_batch_size] = h_source.cpu()
            Y[t:t+args.forward_batch_size] = sliced_model(h_source).cpu()

    token_idxs = parse_token_idxs(args.token_idxs)
    delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=token_idxs)
    cpu_attention_mask = attention_mask.cpu()

    # ---- Calibrate ---------------------------------------------------------
    if args.scale is None:
        print(f"Calibrating input_scale (sample_size={args.calibration_sample_size})...", flush=True)
        calibrator = dct.SteeringCalibrator(target_ratio=0.5)
        input_scale = calibrator.calibrate(
            delta_acts_single,
            X.to(delta_acts_single.device),
            Y.to(delta_acts_single.device),
            factor_batch_size=args.factor_batch_size,
            calibration_sample_size=args.calibration_sample_size,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device))
    else:
        input_scale = args.scale
    print(f"  input_scale = {input_scale}")

    # ---- Fit DCT -----------------------------------------------------------
    print(f"Fitting ExponentialDCT (num_factors={args.num_factors}, max_iters={args.num_iters})...",
          flush=True)
    exp_dct = dct.ExponentialDCT(num_factors=args.num_factors)
    U, V = exp_dct.fit(
        delta_acts_single, X, Y,
        batch_size=1, factor_batch_size=args.factor_batch_size,
        init="rand_backward", input_scale=input_scale, max_iters=args.num_iters,
        beta=1.0, orthogonalize=args.orthogonalize, deflation=False,
        soft_ortho_temp=1.0, soft_ortho_iterations=10,
        attention_mask=cpu_attention_mask.to(delta_acts_single.device), separate_u=False)

    # ---- Save --------------------------------------------------------------
    torch.save(V.detach().cpu(), f"dct_V_{args.dataset}.pt")
    torch.save(U.detach().cpu(), f"dct_U_{args.dataset}.pt")
    meta = {
        "dataset": args.dataset, "model": args.model_name,
        "source_layer": args.source_layer, "target_layer": args.target_layer,
        "num_factors": args.num_factors, "num_iters": args.num_iters,
        "num_samples": n, "input_scale": float(input_scale),
        "token_idxs": args.token_idxs, "balanced": args.balanced,
    }
    with open(f"dct_meta_{args.dataset}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== DONE ===")
    print(f"V {tuple(V.shape)} -> dct_V_{args.dataset}.pt   "
          f"U {tuple(U.shape)} -> dct_U_{args.dataset}.pt")
    print(f"meta -> dct_meta_{args.dataset}.json (input_scale={input_scale:.3f})")


if __name__ == "__main__":
    main()
