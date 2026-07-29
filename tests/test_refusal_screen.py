import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refusal_screen import refused, refusal_rate, chat_wrap, REFUSAL_MARKERS


class _FakeTok:
    """Stands in for a HF tokenizer's chat-template API."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        assert [m["role"] for m in messages] == ["user"]
        return "<start>" + messages[0]["content"] + "<gen>"


def test_markers_are_lowercase_and_nonempty():
    assert REFUSAL_MARKERS
    assert all(m == m.lower() and m.strip() for m in REFUSAL_MARKERS)


def test_refused_is_case_insensitive_and_substring():
    assert refused("I'm sorry, but I cannot help with that.")
    assert refused("AS AN AI LANGUAGE MODEL, I must decline")
    assert not refused("Sure! Here are three ways to bake bread.")


def test_refusal_rate_counts_fraction():
    assert refusal_rate(["I cannot do that", "Sure, here you go", "I apologize"]) == 2 / 3
    assert refusal_rate([]) == 0.0


def test_chat_wrap_is_a_passthrough_when_disabled():
    assert chat_wrap(_FakeTok(), "bake bread", False) == "bake bread"


def test_chat_wrap_uses_the_tokenizer_template_when_enabled():
    assert chat_wrap(_FakeTok(), "bake bread", True) == "<start>bake bread<gen>"
