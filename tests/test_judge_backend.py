import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge_results import ask, build_parser


class _FakeChat:
    """Stands in for OlmoJudge: exposes .chat, records the call. No model download."""
    def __init__(self):
        self.calls = []

    def chat(self, system, user, max_tokens=200):
        self.calls.append((system, user, max_tokens))
        return '{"verdict": "TRUE", "reason": "ok"}'


def test_ask_routes_chat_client_to_chat():
    fake = _FakeChat()
    out = ask(fake, "ignored-model", "SYS", "USER", max_tokens=80)
    assert out == '{"verdict": "TRUE", "reason": "ok"}'
    assert fake.calls == [("SYS", "USER", 80)]


def test_parser_accepts_olmo_backend_and_default_model():
    args = build_parser().parse_args(
        ["--mode", "steer", "--dataset", "cities", "--backend", "olmo"])
    assert args.backend == "olmo"
    assert args.olmo_model == "allenai/Olmo-3-7B-Instruct"


def test_parser_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--mode", "steer", "--dataset", "cities", "--backend", "nope"])
