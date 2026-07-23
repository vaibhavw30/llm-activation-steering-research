"""Warm-started DCT: seed factor 0 at a supervised truth axis + soft anchor, matching the
existing cold run's config (layers, input_scale, num_iters) from dct_meta_<ds>.json.

Mirrors run_dct_data.py's extraction/calibration; changes only the DCT init.

    python dct_warm.py --dataset cities --seed-name mean_diff --anchor-lambda 0.1 --device cuda
    python dct_warm.py --dataset cities --seed-name grad --anchor-lambda 0.0 --num-factors 8 \
        --num-iters 3 --num-samples 8 --device cpu     # smoke

Writes: dct_warm_V_<ds>_<seed>_<lamtag>.pt  and  ..._U.pt
"""
import argparse
import json
import numpy as np
import torch

import dct
from funnel_utils import unit
from run_dct_data import load_statements, parse_token_idxs, MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer


def lam_tag(lam):
    return "lam" + str(lam).replace(".", "p").rstrip("p0") if lam else "lam0"


def load_seed(ds, seed_name, source_layer):
    td = np.load(f"truth_dir_{ds}.npz")
    assert int(td["layer"]) == int(source_layer), (
        f"truth_dir_{ds}.layer={int(td['layer'])} != dct source_layer={source_layer}; "
        f"seed must live in source-layer space")
    return unit(np.asarray(td[seed_name], np.float64))


def load_seed_tgt(ds, seed_name, target_layer):
    td = np.load(f"truth_dir_tgt_{ds}.npz")
    assert int(td["layer"]) == int(target_layer), (
        f"truth_dir_tgt_{ds}.layer={int(td['layer'])} != dct target_layer={target_layer}; "
        f"U-anchor seed must live in target-layer space")
    return unit(np.asarray(td[seed_name], np.float64))


def out_stem(anchor_space):
    return "dct_uwarm" if anchor_space == "u" else "dct_warm"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed-name", choices=["mean_diff", "grad"], required=True)
    p.add_argument("--anchor-lambda", type=float, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-factors", type=int, default=64)
    # overrides (default: read from dct_meta_<ds>.json)
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--factor-batch-size", type=int, default=64)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=325)
    p.add_argument("--anchor-space", choices=["v", "u"], default="v",
                   help="v: anchor the input direction (original warm-DCT); "
                        "u: anchor the factor-0 EFFECT toward the target-layer truth axis")
    return p.parse_args()


def main():
    a = parse_args()
    ds = a.dataset
    meta = json.load(open(f"dct_meta_{ds}.json"))
    src, tgt = int(meta["source_layer"]), int(meta["target_layer"])
    input_scale = float(meta["input_scale"])
    num_iters = a.num_iters or int(meta["num_iters"])
    num_samples = a.num_samples or int(meta["num_samples"])
    token_idxs = parse_token_idxs(meta.get("token_idxs", "-3:"))
    if a.anchor_space == "v":
        seed_vec = load_seed(ds, a.seed_name, src)
    else:
        seed_vec = load_seed_tgt(ds, a.seed_name, tgt)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[warm] {ds} seed={a.seed_name} lam={a.anchor_lambda} src={src}->tgt={tgt} "
          f"scale={input_scale:.3f} factors={a.num_factors} iters={num_iters} "
          f"space={a.anchor_space}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", truncation_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad = False
    d_model = model.config.hidden_size

    sliced = dct.SlicedModel(model, start_layer=src, end_layer=tgt, layers_name="model.layers")
    statements, _labels = load_statements(ds, num_samples, True, a.seed)
    n = len(statements)
    enc = tok(statements, return_tensors="pt", padding="longest", truncation=True,
              max_length=a.max_length)
    attn = enc["attention_mask"].to(torch.float)
    X = torch.zeros(n, enc["input_ids"].shape[1], d_model, device="cpu")
    Y = torch.zeros(n, enc["input_ids"].shape[1], d_model, device="cpu")
    for t in range(0, n, a.forward_batch_size):
        with torch.no_grad():
            ids = enc["input_ids"][t:t+a.forward_batch_size].to(device)
            msk = enc["attention_mask"][t:t+a.forward_batch_size].to(device)
            hs = model(ids, attention_mask=msk, output_hidden_states=True).hidden_states
            X[t:t+a.forward_batch_size] = hs[src].cpu()
            Y[t:t+a.forward_batch_size] = sliced(hs[src]).cpu()

    delta_acts_single = dct.DeltaActivations(sliced, target_position_indices=token_idxs)
    cpu_mask = attn.cpu()
    m = dct.ExponentialDCT(num_factors=a.num_factors)
    if a.anchor_space == "v":
        U, V = m.fit(
            delta_acts_single, X, Y, batch_size=1, factor_batch_size=a.factor_batch_size,
            init="warm", warm_seed=torch.as_tensor(seed_vec, dtype=torch.float32),
            anchor_lambda=a.anchor_lambda, input_scale=input_scale, max_iters=num_iters,
            beta=1.0, orthogonalize=True, deflation=False,
            attention_mask=cpu_mask.to(delta_acts_single.device), separate_u=False)
    else:
        U, V = m.fit(
            delta_acts_single, X, Y, batch_size=1, factor_batch_size=a.factor_batch_size,
            init="random", u_anchor=torch.as_tensor(seed_vec, dtype=torch.float32),
            u_anchor_lambda=a.anchor_lambda, input_scale=input_scale, max_iters=num_iters,
            beta=1.0, orthogonalize=True, deflation=False,
            attention_mask=cpu_mask.to(delta_acts_single.device), separate_u=False)

    stem = out_stem(a.anchor_space)
    tag = lam_tag(a.anchor_lambda)
    torch.save(V.detach().cpu(), f"{stem}_V_{ds}_{a.seed_name}_{tag}.pt")
    torch.save(U.detach().cpu(), f"{stem}_U_{ds}_{a.seed_name}_{tag}.pt")
    print(f"[warm] wrote {stem}_V_{ds}_{a.seed_name}_{tag}.pt  V{tuple(V.shape)}")


if __name__ == "__main__":
    main()
