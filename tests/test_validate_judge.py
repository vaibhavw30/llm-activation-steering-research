import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validate_judge import (row_to_judge_input, balanced_sample, score_agreement,
                            passes_gate)


def test_row_to_judge_input_uses_city_country_for_parity():
    # cities.csv has structured columns -> build the exact (stem, completion) steer mode sees.
    row = {"statement": "The city of Krasnodar is in South Africa.", "label": "0",
           "city": "Krasnodar", "country": "South Africa", "correct_country": "Russia"}
    stem, completion, gold = row_to_judge_input(row)
    assert stem == "The city of Krasnodar is in"
    assert completion == "South Africa"
    assert gold == 0


def test_row_to_judge_input_falls_back_to_statement():
    # datasets without city/country columns -> judge the whole statement (empty stem).
    row = {"statement": "The Great Wall of China is visible from the Moon.", "label": "0"}
    stem, completion, gold = row_to_judge_input(row)
    assert stem == ""
    assert completion == "The Great Wall of China is visible from the Moon."
    assert gold == 0


def test_balanced_sample_is_balanced_and_deterministic():
    rows = [{"label": "1", "i": i} for i in range(10)] + [{"label": "0", "i": i} for i in range(10)]
    picked = balanced_sample(rows, 6)
    assert len(picked) == 6
    assert sum(r["label"] == "1" for r in picked) == 3
    assert sum(r["label"] == "0" for r in picked) == 3
    # deterministic: takes the first k of each class
    assert [r["i"] for r in picked if r["label"] == "1"] == [0, 1, 2]


def test_score_agreement_counts_and_incoherent_is_never_agreement():
    pairs = [(1, "TRUE"),        # agree
             (0, "FALSE"),       # agree
             (1, "FALSE"),       # wrong
             (0, "TRUE"),        # wrong
             (1, "INCOHERENT")]  # never agrees
    res = score_agreement(pairs)
    assert res["n"] == 5
    assert res["agree"] == 2
    assert abs(res["accuracy"] - 0.4) < 1e-9
    assert res["confusion"][(1, "TRUE")] == 1
    assert res["confusion"][(1, "INCOHERENT")] == 1


def test_score_agreement_empty_is_zero_not_crash():
    res = score_agreement([])
    assert res["n"] == 0 and res["accuracy"] == 0.0


def test_passes_gate_threshold_boundary():
    assert passes_gate(0.85, 0.85) is True     # >= is a pass
    assert passes_gate(0.849, 0.85) is False
    assert passes_gate(0.90) is True            # default threshold 0.85
