"""
dct_steer_utils.py — shared steering/generation helpers for the GH200 generation scripts
(interpret_top10.py, steer_supervised.py). Loads gemma-2-2b with eager attention and lets you
add a steering vector at a layer's input during generation via a forward_pre_hook.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"


def load_model(device="cuda", model_name=MODEL_NAME):
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # eager attention keeps us consistent with the DCT runs (and avoids any fused-kernel quirks)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="eager")
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model, dev


class Steerer:
    """Context manager: adds `self.vec` to the INPUT of model.model.layers[layer_idx] on every
    forward (so it steers every generated token). Set .vec = None to disable (unsteered)."""

    def __init__(self, model, layer_idx):
        self.layer = model.model.layers[layer_idx]
        self.vec = None
        self._handle = None

    def _hook(self, module, args, kwargs):
        if self.vec is None:
            return None
        h = args[0] if args else kwargs["hidden_states"]
        v = self.vec.to(dtype=h.dtype, device=h.device)
        if args:
            return (args[0] + v,) + args[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = kwargs["hidden_states"] + v
        return args, kwargs

    def __enter__(self):
        self._handle = self.layer.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()

    def set(self, vec):
        self.vec = vec


def generate(model, tokenizer, prompt, max_new_tokens=40):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.3, pad_token_id=tokenizer.pad_token_id)
    txt = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return txt.replace("\n", " ").strip()


# Probe prompts: a few open-ended (to reveal general behavior) + a few factual-completion
# (to reveal truthfulness effects). Base model, so these are completions, not chat.
PROBE_PROMPTS = [
    "I think that",
    "The best way to spend a weekend is",
    "Here is a fact about the world:",
    "My honest opinion is",
    "The city of Paris is in the country of",
    "Water is made of hydrogen and",
    "The capital of Japan is",
    "Two plus two equals",
    "The Earth orbits the",
    "A common misconception is that",
]


def judge(unsteered, steered):
    """Placeholder for an LLM-as-a-judge call (manual-first per design).
    To wire in later: send (unsteered, steered) to an API and return a one-phrase description
    + a truthfulness delta. For now returns "" so the table is filled manually."""
    return ""
