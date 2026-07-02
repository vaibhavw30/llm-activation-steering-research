"""
apply_dct_vector.py — Stage 2: sanity-check that DCT steering vectors change behavior.

Loads dct_V.pt (steering vectors, d_model x num_factors), and for the top-K vectors
adds `input_scale * v` to the residual stream at the INPUT of the source layer during
generation (a forward_pre_hook on model.model.layers[source_layer]). Prints unsteered
vs steered continuations side by side for a few probe prompts.

Why pre-hook on source_layer: DCT defines X = hidden_states[source_layer], i.e. the
residual stream feeding INTO decoder layer[source_layer]. Steering must be injected at
the same point the vectors were learned for.

Usage:
    .venv-dct/bin/python apply_dct_vector.py
    .venv-dct/bin/python apply_dct_vector.py --vectors 0,1,2 --scale 25.72 --max-new-tokens 32
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
# Calibrated input_scale from the Stage-1 run (run_dct_minimal.py). The DCT vectors
# are unit-norm; the steering magnitude they were tuned for is input_scale * v.
DEFAULT_SCALE = 25.720566749572527


def parse_args():
    p = argparse.ArgumentParser(description="Apply DCT steering vectors during generation")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--v-path", default="dct_V.pt")
    p.add_argument("--source-layer", type=int, default=6, help="must match Stage-1 training")
    p.add_argument("--vectors", default="0,1,2,3", help="comma-separated V column indices to test")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompts", default=None, help="'||'-separated probe prompts (else defaults)")
    return p.parse_args()


DEFAULT_PROMPTS = [
    "I think that",
    "The best way to spend a weekend is",
    "Here is some advice:",
]


def main():
    args = parse_args()
    torch.manual_seed(0)

    print(f"Loading {args.model_name} (CPU, fp32)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Default (SDPA) attention is fine here — generation needs no forward-mode autodiff.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    model.eval()

    V = torch.load(args.v_path).float()  # (d_model, num_factors)
    d_model, num_factors = V.shape
    print(f"Loaded {args.v_path}: {V.shape}  | source_layer={args.source_layer}  "
          f"scale={args.scale:.3f}")

    vec_idxs = [int(x) for x in args.vectors.split(",")]
    prompts = DEFAULT_PROMPTS if args.prompts is None else args.prompts.split("||")

    layer = model.model.layers[args.source_layer]

    # Steering state: the hook reads `current_steer` (a d_model vector or None).
    state = {"steer": None}

    def pre_hook(module, hook_args, hook_kwargs):
        if state["steer"] is None:
            return None
        s = state["steer"].to(dtype=hook_args[0].dtype if hook_args else
                              hook_kwargs["hidden_states"].dtype)
        if hook_args:  # hidden_states passed positionally
            new0 = hook_args[0] + s
            return (new0,) + hook_args[1:], hook_kwargs
        else:          # passed as kwarg
            hook_kwargs = dict(hook_kwargs)
            hook_kwargs["hidden_states"] = hook_kwargs["hidden_states"] + s
            return hook_args, hook_kwargs

    handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, repetition_penalty=1.3,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text.replace("\n", " ").strip()

    try:
        for prompt in prompts:
            print("\n" + "=" * 78)
            print(f"PROMPT: {prompt!r}")
            print("=" * 78)
            state["steer"] = None
            print(f"  [unsteered]      {generate(prompt)}")
            for vi in vec_idxs:
                state["steer"] = args.scale * V[:, vi]
                print(f"  [steer V[{vi:>2}]]    {generate(prompt)}")
    finally:
        handle.remove()

    print("\nDone. If steered continuations differ markedly from unsteered, the DCT "
          "vectors elicit distinct behaviors.")


if __name__ == "__main__":
    main()
