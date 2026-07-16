"""The 8 MAG operators (Table 1) as pure functions over cached last-token readouts.

Each returns an (n, d) feature matrix at a chosen layer L. Per-statement caches are
(L+1, n, d); constants A_Q, A_empty are (L+1, d) and broadcast over n.
"""
import numpy as np

from .config import OPERATOR_NAMES


def operator_features(name, cache, layer):
    A_p = np.asarray(cache["A_p"][layer], dtype=np.float64)      # (n, d)
    A_Qp = np.asarray(cache["A_Qp"][layer], dtype=np.float64)    # (n, d)
    A_Qpv = np.asarray(cache["A_Qpv"][layer], dtype=np.float64)
    A_verd = np.asarray(cache["A_verdict"][layer], dtype=np.float64)
    A_EQp = np.asarray(cache["A_EQp"][layer], dtype=np.float64)
    A_Q = np.asarray(cache["A_Q"][layer], dtype=np.float64)      # (d,)
    A_empty = np.asarray(cache["A_empty"][layer], dtype=np.float64)  # (d,)

    if name == "Direct":
        return A_p
    if name == "Prefixed":
        return A_Qp
    if name == "Answered":
        return A_Qpv
    if name == "Verdict":
        return A_verd
    if name == "InputDelta":
        return A_Qp - A_p
    if name == "QuestionDelta":
        return A_Qp - A_Q            # (n,d) - (d,)
    if name == "Interaction":
        return A_Qp - A_p + A_empty  # (n,d) - (n,d) + (d,)
    if name == "FewShot":
        return A_EQp
    raise ValueError(f"unknown operator {name!r}")


def all_operator_features(cache, layer):
    return {name: operator_features(name, cache, layer) for name in OPERATOR_NAMES}
