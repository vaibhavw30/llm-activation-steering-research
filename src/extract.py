"""
extract.py — Step 1: extract LLM residual-stream activations from true/false statements.

CPU-only, float32. Run once per dataset. For every statement it records the
activation at the LAST non-pad token, at EVERY layer (embeddings + all blocks).

Usage:
    python extract.py cities.csv
    python extract.py cities.csv --limit 20      # smoke test on first 20 rows
    python extract.py common_claim_true_false.csv

Output: acts_<dataset>.npz  (one file per dataset) containing:
    activations : float32 (num_layers+1, num_statements, hidden_dim)
    labels      : int     (num_statements,)
    statements  : object  (num_statements,)
    model       : str
"""

import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# CONFIG
#   Primary:  google/gemma-2-2b  (matches Julian's A-LQR paper; gated on HF)
#   Fallback: Qwen/Qwen2.5-1.5B  (ungated, same code path)
# ----------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"  # matches A-LQR paper; fall back to "Qwen/Qwen2.5-1.5B" if gated
BATCH = 16
MAX_LENGTH = 64
SEED = 42

DATASET_DIR = "got_datasets"


def load_model_or_explain():
    """Load tokenizer + model on CPU/fp32. On HF license gating, print a clear
    message telling the user how to fix it, then exit cleanly (code 2)."""
    print(f"Loading model {MODEL_NAME} (CPU, fp32)...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # transformers 5.x renamed torch_dtype -> dtype; support both.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, dtype=torch.float32, output_hidden_states=True
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float32, output_hidden_states=True
            )
    except Exception as e:  # noqa: BLE001 - we want to classify any load failure
        msg = str(e).lower()
        gated = any(
            k in msg
            for k in ("gated", "401", "403", "awaiting", "access", "authenticate",
                      "is not a valid", "restricted", "login")
        )
        print("\n" + "=" * 70)
        if gated:
            print("ERROR: could not load a gated model:", MODEL_NAME)
            print("This model requires accepting its license on Hugging Face.")
            print("\nFIX — choose ONE:")
            print("  1. Accept the license at https://huggingface.co/google/gemma-2-2b")
            print("     then run:  huggingface-cli login   (paste an HF token)")
            print('  2. Edit extract.py and set MODEL_NAME = "Qwen/Qwen2.5-1.5B"')
            print("     (ungated, no other change needed).")
        else:
            print("ERROR: failed to load model", MODEL_NAME)
            print("Underlying error:", repr(e))
            print('\nIf this is a license/auth issue, set MODEL_NAME = "Qwen/Qwen2.5-1.5B".')
        print("=" * 70)
        sys.exit(2)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # IMPORTANT: Gemma (and many causal LMs) default to LEFT padding, which breaks
    # the "attention_mask.sum()-1" last-token index. Force right padding so the
    # last real token sits at sum(mask)-1. (Right padding is safe for hidden-state
    # extraction: causal attention means trailing pad tokens don't affect it.)
    tokenizer.padding_side = "right"
    model.eval()
    return tokenizer, model


def main(dataset_file, limit=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_name = dataset_file.replace(".csv", "")
    out_path = f"acts_{dataset_name}.npz"

    tokenizer, model = load_model_or_explain()

    csv_path = f"{DATASET_DIR}/{dataset_file}"
    print(f"Loading dataset {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)
    statements = df["statement"].astype(str).tolist()
    labels = df["label"].to_numpy().astype(int)
    if limit is not None:
        statements = statements[:limit]
        labels = labels[:limit]
        print(f"  [smoke test: limited to first {limit} statements]")
    n = len(statements)
    print(f"  {n} statements, {int(labels.sum())} true / {n - int(labels.sum())} false",
          flush=True)

    per_statement = []  # list of (num_layers+1, hidden_dim) arrays
    with torch.no_grad():
        for start in range(0, n, BATCH):
            batch = statements[start:start + BATCH]
            inputs = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_LENGTH,
            )
            out = model(**inputs)
            # hidden_states: tuple of (num_layers+1) tensors, each (B, seq, d)
            # Last real-token index per row = position of the LAST 1 in the mask.
            # This is correct for BOTH left- and right-padding (robust guard on top
            # of the forced right-padding above).
            am = inputs["attention_mask"]
            last_idx = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                idx = int(last_idx[b].item())
                layers = torch.stack([h[b, idx, :] for h in out.hidden_states])
                per_statement.append(layers.to(torch.float32).numpy())
            done = min(start + BATCH, n)
            print(f"  batch {start // BATCH + 1}: {done}/{n} statements", flush=True)

    arr = np.stack(per_statement, axis=0)        # (n, L+1, d)
    acts = np.transpose(arr, (1, 0, 2)).astype(np.float32)  # (L+1, n, d)
    print("Activation tensor shape (num_layers+1, num_statements, hidden_dim):",
          acts.shape, flush=True)

    np.savez_compressed(
        out_path,
        activations=acts,
        labels=labels,
        statements=np.array(statements, dtype=object),
        model=MODEL_NAME,
    )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract.py <dataset.csv> [--limit N]")
        print("e.g.:  python extract.py cities.csv")
        print("       python extract.py cities.csv --limit 20   (smoke test)")
        sys.exit(1)
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    main(sys.argv[1], limit=lim)
