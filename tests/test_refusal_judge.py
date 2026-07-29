import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refusal_judge import score_rows, rates_by_scale, agreement


def _row(direction, scale, completion):
    return {"direction": direction, "scale": scale, "prompt": "p",
            "completion": completion}


def test_score_rows_adds_a_refused_flag_and_preserves_order():
    src = [_row("d", "0.0", "Sure, here is how"),
           _row("d", "1.0", "I cannot help with that")]
    out = score_rows(src)
    assert [r["refused"] for r in out] == [0, 1]
    assert [r["completion"] for r in out] == [r["completion"] for r in src]
    assert "refused" not in src[0]          # inputs are not mutated


def test_rates_by_scale_groups_and_averages():
    rows = [_row("d", "0.0", "sure"), _row("d", "0.0", "I'm sorry"),
            _row("d", "2.0", "I cannot"), _row("d", "2.0", "I apologize")]
    r = rates_by_scale(score_rows(rows))
    assert r[("d", 0.0)] == (2, 0.5)
    assert r[("d", 2.0)] == (2, 1.0)


def test_agreement_reports_kappa_for_perfect_agreement():
    a = agreement([1, 0, 1, 0], [1, 0, 1, 0])
    assert a["n"] == 4 and a["agree"] == 1.0 and abs(a["cohen_kappa"] - 1.0) < 1e-9


def test_agreement_kappa_is_zero_at_chance():
    # both raters call half of them refusals, but they disagree exactly at chance
    a = agreement([1, 1, 0, 0], [1, 0, 1, 0])
    assert a["agree"] == 0.5 and abs(a["cohen_kappa"]) < 1e-9
