import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import extract


def test_model_flag_defaults_to_gemma():
    a = extract.build_parser().parse_args(["cities.csv"])
    assert a.model == "google/gemma-2-2b"
    assert a.dataset == "cities.csv"
    assert a.limit is None


def test_model_flag_overrides():
    a = extract.build_parser().parse_args(["refusal.csv", "--model", "google/gemma-2-2b-it"])
    assert a.model == "google/gemma-2-2b-it"
    assert a.dataset == "refusal.csv"


def test_limit_still_parses():
    a = extract.build_parser().parse_args(["cities.csv", "--limit", "20"])
    assert a.limit == 20
