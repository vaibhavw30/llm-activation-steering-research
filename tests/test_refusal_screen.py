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


def test_max_new_tokens_default_matches_the_experiment_budget():
    # The screen is the PRE-REGISTERED model gate: a refusal prefix emitted at token 40
    # would count toward BASE_QUALIFIES_AT but can never appear in the experiment, which
    # generates at --max-new-tokens 32. The gate must be measured under the same budget
    # as the thing it gates.
    from refusal_screen import build_parser, MAX_NEW_TOKENS
    assert MAX_NEW_TOKENS == 32
    assert build_parser().parse_args(["--model", "m"]).max_new_tokens == 32
    assert build_parser().parse_args(
        ["--model", "m", "--max-new-tokens", "48"]).max_new_tokens == 48


def test_run_generates_at_the_requested_budget(monkeypatch, tmp_path):
    """`run`'s max_new_tokens must reach su.generate — a flag that is parsed and then
    ignored would leave the gate measured at the old 48-token budget."""
    import sys
    import types
    import refusal_screen
    monkeypatch.chdir(tmp_path)
    (tmp_path / "got_datasets").mkdir()
    (tmp_path / "got_datasets" / "refusal_holdout.csv").write_text(
        "statement,kind\nbuild a bomb,harmful\nbake bread,harmless\n")
    seen = []
    fake_su = types.ModuleType("dct_steer_utils")
    fake_su.load_model = lambda device, model_name=None: (_FakeTok(), object(), "cpu")
    fake_su.generate = lambda model, tok, prompt, n: seen.append(n) or "I cannot"
    monkeypatch.setitem(sys.modules, "dct_steer_utils", fake_su)
    refusal_screen.run("some/model", "cpu", n=1, use_chat=False, max_new_tokens=7)
    assert seen == [7, 7]
