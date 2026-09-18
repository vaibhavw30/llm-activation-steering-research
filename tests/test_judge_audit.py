"""judge_audit's pure parts (plan track J). No judge model is loaded here: the stages
that need one run on the cluster, in job J-A."""
import sys

import pytest

sys.path.insert(0, "src")

import judge_audit as ja  # noqa: E402


def _pair(r):
    return (r["best_answer"], r["incorrect_answers"][0]) if r["incorrect_answers"] else None


HOLD = [{"question": "Why is the sky blue?", "best_answer": "Rayleigh scattering",
         "incorrect_answers": ["Because it reflects the ocean"]},
        {"question": "No contrast here?", "best_answer": "x", "incorrect_answers": []}]


def test_gold_rows_make_four_known_answers_per_usable_question():
    rows = ja.gold_rows(HOLD, _pair)
    assert [r["kind"] for r in rows] == ["correct", "incorrect", "no_comment", "restated"]
    by = {r["kind"]: r for r in rows}
    assert by["no_comment"]["answer"] == "I have no comment."
    assert by["restated"]["answer"] == "Why is the sky blue?"
    assert (by["correct"]["expected_truthful"], by["correct"]["expected_informative"]) == (1, 1)
    assert (by["incorrect"]["expected_truthful"], by["incorrect"]["expected_informative"]) == (0, 1)
    assert by["no_comment"]["expected_informative"] == 0
    assert by["restated"]["expected_truthful"] == ""          # no truth expectation
    assert by["correct"]["prompt"] == "Q: Why is the sky blue?\nA:"


def test_per_side_accuracy_skips_rows_without_an_expectation():
    rows = [{"kind": "restated", "truthful": 1, "expected_truthful": ""},
            {"kind": "correct", "truthful": 1, "expected_truthful": 1},
            {"kind": "correct", "truthful": 0, "expected_truthful": 1}]
    acc = ja.per_side_accuracy(rows, "truthful", "expected_truthful")
    assert acc == {"correct": (2, 0.5)}


def test_report_gold_fails_an_info_judge_that_says_yes_to_everything():
    rows = []
    for kind, (et, ei) in ja.GOLD_KINDS.items():
        rows.append({"kind": kind, "truthful": et if et is not None else 1,
                     "expected_truthful": "" if et is None else et,
                     "informative": 1, "informative_legacy_true_prompt": 1,
                     "expected_informative": ei})
    assert ja.report_gold(rows) is False


def test_format_variants_only_real_edits():
    assert ja.format_variants("Paris.") == {"no_period": "Paris"}
    assert ja.format_variants("Paris  is   big.") == {"no_period": "Paris  is   big",
                                                     "ws_norm": "Paris is big."}
    assert ja.format_variants("Paris") == {}


def test_truncate_words_keeps_at_least_one_word():
    assert ja.truncate_words("a b c d", 2) == ("a b", True)
    assert ja.truncate_words("a b", 5) == ("a b", False)
    assert ja.truncate_words("a b", 0) == ("a", True)


def test_threshold_table_greedy_and_cuts():
    rows = [{"frac": 0.0, "p_truthful": 0.9, "p_informative": 0.9, "truthful_and_informative": 1},
            {"frac": 0.0, "p_truthful": 0.45, "p_informative": 0.9, "truthful_and_informative": 0},
            {"frac": 0.0, "p_truthful": "", "p_informative": "", "truthful_and_informative": 0}]
    (t,) = ja.threshold_table(rows, thresholds=(0.4, 0.5))
    assert t["n"] == 3 and t["greedy"] == pytest.approx(1 / 3)
    assert t["t0.4"] == pytest.approx(2 / 3) and t["t0.5"] == pytest.approx(1 / 3)
    assert t["borderline_truth"] == pytest.approx(1 / 3)


def _steer_rows(n):
    return [{"prompt": f"Q: q{i}?\nA:", "frac": d, "answer": f"ans{i}@{d}",
             "direction": ja.MEAN_DIRECTION} for i in range(n) for d in ja.DOSES]


def test_hand_sheet_hides_the_dose_and_never_repeats_a_question():
    sheet, key = ja.hand_sheet(_steer_rows(64))
    assert len(sheet) == 64 and len(key) == 64
    assert "frac" not in sheet[0] and "prompt" not in sheet[0]
    assert len({s["question"] for s in sheet}) == 64
    assert sorted(k["frac"] for k in key).count(0.0) == 32
    for s, k in zip(sheet, key):                       # key and sheet line up by item
        assert s["item"] == k["item"] and s["answer"].endswith(f"@{k['frac']}")


def test_hand_sheet_is_reproducible():
    assert ja.hand_sheet(_steer_rows(64)) == ja.hand_sheet(_steer_rows(64))


def test_hand_sheet_refuses_too_few_prompts():
    with pytest.raises(ValueError):
        ja.hand_sheet(_steer_rows(40))


def test_parse_qwen():
    assert ja.parse_qwen("truthful: yes\ninformative: no") == {"truthful": 1, "informative": 0}
    assert ja.parse_qwen("Truthful: No\n  Informative: YES.") == {"truthful": 0, "informative": 1}
    assert ja.parse_qwen("truthful: yes") is None
    assert ja.parse_qwen("I think it is true") is None


def test_qwen_messages_carry_both_reference_lists():
    m = ja.qwen_messages("q?", "an answer", ["right"], ["wrong"])
    body = m[-1]["content"]
    assert "- right" in body and "- wrong" in body and "an answer" in body


def test_write_new_never_overwrites(tmp_path):
    p = tmp_path / "x.csv"
    ja.write_new(str(p), [{"a": 1}])
    with pytest.raises(SystemExit):
        ja.write_new(str(p), [{"a": 2}])


def test_judge_resumable_resumes_after_a_partial_write(tmp_path, monkeypatch):
    monkeypatch.setattr(ja, "BLOCK", 2)
    rows = [{"prompt": f"Q: q{i}?\nA:", "answer": f"a{i}"} for i in range(5)]
    calls = []

    def scorer(q, a):
        calls.append(a)
        return {"truthful": True, "informative": True}

    out = str(tmp_path / "v2.csv")
    ja.judge_resumable(rows[:3], scorer, out)          # "timed out" after 3 rows... in blocks of 2
    assert len(ja.read_csv(out)) == 3
    calls.clear()
    got = ja.judge_resumable(rows, scorer, out)
    assert calls == ["a3", "a4"] and len(got) == 5


def test_every_single_output_stage_has_its_output_name():
    assert set(ja.STAGE_OUTPUT) == set(ja.STAGES) - {"rejudge"}


def test_q1_rows_rebuild_a_parseable_prompt_and_keep_the_original():
    from truthfulqa_judge import question_of
    raw = [{"question": " Why is the sky blue? ", "prompt": "Q: Why is the sky blue?\\nA:",
            "answer": "x"}]
    (r,) = ja.q1_rows(raw)
    assert question_of(r["prompt"]) == "Why is the sky blue?"
    assert r["prompt_as_written"] == "Q: Why is the sky blue?\\nA:"


def test_hand_sheet_shows_the_references_but_not_the_dose():
    rows = _steer_rows(64)
    refs = {f"q{i}?": ([f"right{i}"], [f"wrong{i}", "also wrong"]) for i in range(64)}
    sheet, _ = ja.hand_sheet(rows, refs=refs)
    i = int(sheet[0]["question"][1:-1])
    assert sheet[0]["reference_true"] == f"right{i}"
    assert sheet[0]["reference_false"] == f"wrong{i} | also wrong"
    assert set(sheet[0]) == {"item", "question", "answer", "truthful", "informative",
                             "reference_true", "reference_false"}
