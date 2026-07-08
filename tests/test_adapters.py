import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.adapters import truthfulqa_prompt


def test_basic_qa_format():
    out = truthfulqa_prompt("The capital of Japan is", "Tokyo")
    assert out == "Q: The capital of Japan is\nA: Tokyo\nTrue:"


def test_strips_trailing_colon_and_whitespace():
    out = truthfulqa_prompt("  The Earth orbits the:  ", "  the Sun  ")
    assert out == "Q: The Earth orbits the\nA: the Sun\nTrue:"
