"""y^M — the base model's own yes/no verdict (MAG Eq. 1):
    y^M(p) = 1[ Pr(yes | Q||p) > Pr(no | Q||p) ]
computed from the first-generated-token logits over Q_TRUTH + p + Q_SUFFIX.
"""
import numpy as np
import torch

from .config import YES_VARIANTS, NO_VARIANTS


def first_token_ids(tokenizer, variants):
    ids = []
    for s in variants:
        enc = tokenizer(s, add_special_tokens=False)["input_ids"]
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


def verdict_from_logits(logits_row, yes_ids, no_ids):
    x = np.asarray(logits_row, dtype=np.float64)
    x = x - x.max()                    # numerical stability
    probs = np.exp(x); probs /= probs.sum()
    p_yes = float(probs[yes_ids].sum())
    p_no = float(probs[no_ids].sum())
    return {"ymL": int(p_yes > p_no), "margin": p_yes - p_no,
            "conf": max(p_yes, p_no), "p_yes": p_yes, "p_no": p_no}


def compute_verdicts(model, tokenizer, prompts, device, batch_size=16):
    """prompts are full strings (Q_TRUTH + statement + Q_SUFFIX). Returns arrays of y^M/margin/conf.
    Uses right padding + attention mask so the final real-token logits are read per row."""
    yes_ids = first_token_ids(tokenizer, YES_VARIANTS)
    no_ids = first_token_ids(tokenizer, NO_VARIANTS)
    tokenizer.padding_side = "right"
    ym, marg, conf = [], [], []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=96).to(device)
            logits = model(**enc).logits                     # (B, seq, vocab)
            am = enc["attention_mask"]
            last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
            for b in range(len(batch)):
                row = logits[b, int(last[b].item()), :].float().cpu().numpy()
                r = verdict_from_logits(row, yes_ids, no_ids)
                ym.append(r["ymL"]); marg.append(r["margin"]); conf.append(r["conf"])
    return {"ymL": np.array(ym, dtype=int),
            "margin": np.array(marg, dtype=np.float64),
            "conf": np.array(conf, dtype=np.float64)}
