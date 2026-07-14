import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.olmo_judge import OlmoJudge, DEFAULT_OLMO


# _build_messages is pure (no model load), so it is the unit-testable surface of the judge.
# Constructing a real OlmoJudge downloads a 7B model, so that path is only exercised by the
# env-gated smoke test below (run manually on the M5 Pro).
def test_build_messages_has_system_then_user():
    msgs = OlmoJudge._build_messages("SYS", "USER")
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_default_model_is_olmo_instruct():
    assert DEFAULT_OLMO == "allenai/Olmo-3-7B-Instruct"


@pytest.mark.skipif(not os.environ.get("RUN_OLMO_SMOKE"),
                    reason="downloads and runs the 7B OLMo model; set RUN_OLMO_SMOKE=1 to enable")
def test_olmo_smoke_true_false():
    from judge_results import STEER_SYS, extract_json
    j = OlmoJudge(device=os.environ.get("OLMO_DEVICE", "mps"))
    tokyo = extract_json(j.chat(STEER_SYS, 'Stem: "The capital of Japan is"\nCompletion: "Tokyo"', 80))
    three = extract_json(j.chat(STEER_SYS, 'Stem: "Two plus two equals"\nCompletion: "three"', 80))
    assert tokyo and tokyo.get("verdict") == "TRUE"
    assert three and three.get("verdict") == "FALSE"
