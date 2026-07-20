"""Cheap per-token coherence signals computed from generation logits.

max_prob (confidence), entropy (uncertainty), and a 3-gram repetition flag (looping).
Pure numpy so it is unit-testable without loading the model."""
import numpy as np


def _softmax(logits):
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def token_max_prob(logits):
    return float(_softmax(logits).max())


def token_entropy(logits):
    p = _softmax(logits)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def rep3_flags(token_ids):
    ids = list(token_ids)
    seen = set()
    flags = [0] * len(ids)
    for i in range(len(ids)):
        if i >= 2:
            tri = (ids[i - 2], ids[i - 1], ids[i])
            if tri in seen:
                flags[i] = 1
            seen.add(tri)
    return flags


def intrinsic_signals(scores, token_ids):
    return {
        "max_prob": [token_max_prob(s) for s in scores],
        "entropy": [token_entropy(s) for s in scores],
        "rep3": rep3_flags(token_ids),
    }
