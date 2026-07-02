"""
run_dct_minimal.py — Stage 1 smoke test for DCT (Deep Causal Transcoding).

Runs the paper's OWN implementation (dct.py) end-to-end on google/gemma-2-2b,
CPU/float32, from a single neutral prompt. Reuses their SlicedModel,
DeltaActivations, SteeringCalibrator, and ExponentialDCT — nothing is
reimplemented. Mirrors the call pattern in dct_train.py::process_single_observation.

Output: dct_V.pt (steering vectors, d_model x num_factors) and dct_U.pt
        (output directions). Expected V shape for gemma-2-2b: 2304 x num_factors.

Usage:
    .venv/bin/python run_dct_minimal.py
    .venv/bin/python run_dct_minimal.py --source-layer 4 --target-layer 10   # faster fallback
"""

import argparse
import torch

import dct  # the paper's implementation (dct.py, repo root)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"


def parse_args():
    p = argparse.ArgumentParser(description="DCT Stage-1 smoke test on gemma-2-2b (CPU)")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--device", default="cuda", help="cuda or cpu (auto-falls back to cpu)")
    p.add_argument("--prompt", default="Tell me about your day.")
    # Guide defaults: source 6 / target 18 (gemma-2-2b has 26 layers). Fallback: 4 / 10.
    p.add_argument("--source-layer", type=int, default=6)
    p.add_argument("--target-layer", type=int, default=18)
    p.add_argument("--num-factors", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=10)
    # CPU-friendly: smaller than the paper's 16/30 so the smoke test finishes.
    p.add_argument("--factor-batch-size", type=int, default=8)
    p.add_argument("--calibration-sample-size", type=int, default=8)
    p.add_argument("--token-idxs", default="-3:", help="target token positions, e.g. '-3:'")
    p.add_argument("--scale", type=float, default=None,
                   help="skip calibration and use this input_scale directly")
    return p.parse_args()


def parse_token_idxs(s):
    parts = s.split(":")
    if len(parts) == 1:
        return int(parts[0])
    start = None if parts[0] == "" else int(parts[0])
    end = None if parts[1] == "" else int(parts[1])
    return slice(start, end)


def main():
    args = parse_args()
    torch.manual_seed(325)  # same seed the paper's driver uses

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"  (CUDA not available — falling back to {device})")

    # ---- Load tokenizer + model (fp32, frozen) -----------------------------
    print(f"Loading {args.model_name} ({device}, fp32)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
        padding_side="left", truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # attn_implementation="eager": DCT's SteeringCalibrator uses forward-mode
    # autodiff (jvp), and the fused SDPA CPU kernel does NOT support forward AD
    # (NotImplementedError). Eager attention (plain matmul/softmax) does.
    try:
        # transformers <5 uses torch_dtype; >=5 renamed it to dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager",
        )
    model.to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False
    d_model = model.config.hidden_size
    print(f"  d_model={d_model}, num_hidden_layers={model.config.num_hidden_layers}")

    # ---- Verify SlicedModel reproduces the real forward (their own check) ---
    test_in = tokenizer(["colourless green sheep sleep furiously"],
                        return_tensors="pt").to(model.device)
    with torch.no_grad():
        hs = model(test_in["input_ids"], output_hidden_states=True).hidden_states
    test_slice = dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
    with torch.no_grad():
        recon = test_slice(hs[3]).float()
        target = hs[5].float()
    max_d = (recon - target).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(recon.flatten(), target.flatten(), dim=0).item()
    # Use a realistic atol: default atol=1e-8 is far too strict on near-zero activation
    # elements and gives false negatives even when the slice is faithful.
    ok = torch.allclose(recon, target, rtol=1e-2, atol=1e-2)
    print(f"  SlicedModel sanity check (layers 3->5): {'PASS' if ok else 'MISMATCH'}  "
          f"cos={cos:.6f}  max|d|={max_d:.2e}")
    if not ok:
        print("  WARNING: slice reconstruction not faithful (low cosine) — check transformers")
        print("           version (paper pins 4.51.3). Continuing to inspect V/U anyway.")

    # ---- Build the actual slice we steer through ---------------------------
    sliced_model = dct.SlicedModel(
        model, start_layer=args.source_layer, end_layer=args.target_layer,
        layers_name="model.layers",
    )
    print(f"  Slice: source_layer={args.source_layer} -> target_layer={args.target_layer}")

    # ---- Encode the single prompt (chat template if available, else raw) ----
    try:
        chat = [{"content": args.prompt, "role": "user"}]
        example = tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
    except Exception:
        print("  (no chat template on this base model — using raw prompt text)")
        example = args.prompt

    model_input = tokenizer([example], return_tensors="pt").to(model.device)
    seq_len = model_input["input_ids"].shape[1]
    attention_mask = model_input["attention_mask"].to(torch.float)
    print(f"  Prompt encoded: seq_len={seq_len}")

    # ---- Extract X (source-layer states) and Y (clean target-layer states) -
    X = torch.zeros(1, seq_len, d_model)
    Y = torch.zeros(1, seq_len, d_model)
    with torch.no_grad():
        hidden_states = model(model_input["input_ids"],
                              attention_mask=model_input["attention_mask"],
                              output_hidden_states=True).hidden_states
        h_source = hidden_states[args.source_layer]
        unsteered_target = sliced_model(h_source)
        X[0] = h_source.cpu()
        Y[0] = unsteered_target.cpu()

    token_idxs = parse_token_idxs(args.token_idxs)
    delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=token_idxs)
    cpu_attention_mask = attention_mask.cpu()

    # ---- Calibrate input scale (or use supplied --scale) -------------------
    if args.scale is None:
        print(f"Calibrating input_scale (sample_size={args.calibration_sample_size})...", flush=True)
        calibrator = dct.SteeringCalibrator(target_ratio=0.5)
        input_scale = calibrator.calibrate(
            delta_acts_single,
            X.to(delta_acts_single.device),
            Y.to(delta_acts_single.device),
            factor_batch_size=args.factor_batch_size,
            calibration_sample_size=args.calibration_sample_size,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device),
        )
    else:
        input_scale = args.scale
    print(f"  input_scale = {input_scale}")

    # ---- Fit DCT ------------------------------------------------------------
    print(f"Fitting ExponentialDCT (num_factors={args.num_factors}, max_iters={args.max_iters})...",
          flush=True)
    exp_dct = dct.ExponentialDCT(num_factors=args.num_factors)
    U, V = exp_dct.fit(
        delta_acts_single, X, Y,
        batch_size=1,
        factor_batch_size=args.factor_batch_size,
        init="rand_backward",
        input_scale=input_scale,
        max_iters=args.max_iters,
        beta=1.0,
        orthogonalize=True,
        deflation=False,
        soft_ortho_temp=1.0,
        soft_ortho_iterations=10,
        attention_mask=cpu_attention_mask.to(delta_acts_single.device),
        separate_u=False,
    )

    # ---- Save + report ------------------------------------------------------
    torch.save(V.detach().cpu(), "dct_V.pt")
    torch.save(U.detach().cpu(), "dct_U.pt")
    print("\n=== DONE ===")
    print(f"V (steering vectors) shape: {tuple(V.shape)}  -> dct_V.pt   "
          f"(expect ({d_model}, {args.num_factors}))")
    print(f"U (output directions) shape: {tuple(U.shape)}  -> dct_U.pt")
    print(f"V column norms: mean={V.norm(dim=0).mean():.4f}  "
          f"min={V.norm(dim=0).min():.4f}  max={V.norm(dim=0).max():.4f}")


if __name__ == "__main__":
    main()
