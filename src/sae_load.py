"""sae_load.py — Horizon-1 1.2: GemmaScope SAE parameters for gemma-2-2b.

Downloads the raw params.npz from google/gemma-scope-2b-pt-res via huggingface_hub
rather than going through sae_lens: sae_lens is not installed, pins its own
transformers, and we need nothing from it but the decoder matrix.

Width 16k is the default — the canonical GemmaScope width, the one the Gemma Scope
paper and AxBench report feature interpretations for, and small enough for CPU.

    PYTHONPATH=src .venv/bin/python src/sae_load.py --layer 11
"""
import argparse
import re

import numpy as np

REPO = "google/gemma-scope-2b-pt-res"
TARGET_L0 = 70          # GemmaScope's canonical sparsity band
WIDTH = "16k"


def pick_l0_path(files, layer, width=WIDTH, target_l0=TARGET_L0):
    """From a repo file listing, the params.npz whose average_l0 is closest to
    target_l0 for the requested layer and width; ties go to the smaller l0."""
    pat = re.compile(rf"^layer_{int(layer)}/width_{width}/average_l0_(\d+)/params\.npz$")
    cands = [(int(m.group(1)), f) for f in files for m in [pat.match(str(f))] if m]
    if not cands:
        raise SystemExit(f"[sae] no width_{width} SAE for layer {layer} in {REPO}")
    return min(cands, key=lambda t: (abs(t[0] - target_l0), t[0]))[1]


def decoder_unit(sae):
    """Row-normalized decoder atoms (F, d). Matching pursuit needs unit atoms so the
    correlation with the residual is an inner product, not a scaled one."""
    W = np.asarray(sae["W_dec"], np.float64)
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def load_sae(layer, width=WIDTH, cache_dir=None, target_l0=TARGET_L0):
    """Download (cached) and load one GemmaScope SAE's parameters."""
    from huggingface_hub import hf_hub_download, list_repo_files
    path = pick_l0_path(list_repo_files(REPO), layer, width, target_l0)
    local = hf_hub_download(REPO, path, cache_dir=cache_dir)
    z = np.load(local)
    out = {k: np.asarray(z[k]) for k in z.files}
    out["path"] = path
    print(f"[sae] layer {layer} width {width}: {path}  W_dec {out['W_dec'].shape}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", default=WIDTH)
    a = ap.parse_args()
    load_sae(a.layer, a.width)


if __name__ == "__main__":
    main()
