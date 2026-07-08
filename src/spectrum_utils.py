"""Aggregate judged steering sweeps into a single salience number (spectrum x-axis)."""


def _rate(rows, verdict):
    n = len(rows) or 1
    return sum(r["verdict"] == verdict for r in rows) / n


def concept_salience(rows, present_verdict="FALSE", incoherent_verdict="INCOHERENT",
                     max_incoherent=0.5, direction="grad"):
    """Max swing in the concept-present rate vs. unsteered, over acceptably-coherent scales.

    Uses only the given `direction` (default 'grad'). Returns x_salience and the winning scale.
    """
    rows = [r for r in rows if r["direction"] == direction]
    by_scale = {}
    for r in rows:
        by_scale.setdefault(float(r["scale"]), []).append(r)

    baseline = _rate(by_scale.get(0.0, []), present_verdict)
    best_x, best_scale, best_rate = 0.0, 0.0, baseline
    for scale, srows in by_scale.items():
        if _rate(srows, incoherent_verdict) > max_incoherent:
            continue
        present = _rate(srows, present_verdict)
        swing = abs(present - baseline)
        if swing > best_x:
            best_x, best_scale, best_rate = swing, scale, present
    return {"x_salience": best_x, "best_scale": best_scale,
            "present_rate_at_best": best_rate, "baseline_rate": baseline}
