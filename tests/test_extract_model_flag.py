import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

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


def test_no_args_exits_2_with_usage_on_stderr(capsys):
    """Pins the CLI misuse contract: argparse (not the old hand-rolled
    sys.argv parsing) now owns the zero-argument case. It exits 2 and
    writes its usage/error message to stderr, mentioning the missing
    required `dataset` argument. This intentionally differs from the
    pre-argparse behavior (stdout, exit 1) — no caller in the repo
    invokes extract.py with zero arguments or inspects its exit code."""
    with pytest.raises(SystemExit) as exc_info:
        extract.build_parser().parse_args([])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "dataset" in captured.err
