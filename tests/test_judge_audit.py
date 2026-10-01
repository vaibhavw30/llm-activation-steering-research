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


# ------------------------------------------------------------------ handscore
def test_cohen_kappa_perfect_chance_and_undefined():
    assert ja.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # observed 0.5, expected 0.5 -> kappa 0
    assert ja.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    # 3/4 agree; p_e = 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa 0.5
    assert ja.cohen_kappa([1, 1, 0, 0], [1, 1, 1, 0]) == pytest.approx(0.5)
    # both raters constant and equal: p_e = 1, kappa undefined
    assert ja.cohen_kappa([1, 1], [1, 1]) != ja.cohen_kappa([1, 1], [1, 1])  # nan


def test_label_map_refuses_a_blank_or_non_binary_row():
    ok = [{"item": "3", "truthful": "1", "informative": "0"}]
    assert ja.label_map(ok, "h") == {3: (1, 0)}
    with pytest.raises(SystemExit, match="item 4"):
        ja.label_map(ok + [{"item": "4", "truthful": "", "informative": "1"}], "h")
    with pytest.raises(SystemExit, match="item 5"):
        ja.label_map([{"item": "5", "truthful": "yes", "informative": "1"}], "h")


def test_judge_by_item_joins_through_the_key_on_prompt_dose_direction():
    key = [{"item": "1", "prompt": "Q: a\nA:", "frac": "-2.0", "direction": "d"},
           {"item": "2", "prompt": "Q: b\nA:", "frac": "0.0", "direction": "d"}]
    judged = [{"prompt": "Q: a\nA:", "frac": -2.0, "direction": "d",
               "truthful": "1", "informative": "1"},
              {"prompt": "Q: a\nA:", "frac": 0.0, "direction": "d",      # other dose
               "truthful": "0", "informative": "0"},
              {"prompt": "Q: b\nA:", "frac": 0.0, "direction": "d",
               "truthful": "0", "informative": "1"}]
    got = ja.judge_by_item(key, judged, "truthful", "informative")
    assert got == {1: (1, 1), 2: (0, 1)}


def test_judge_by_item_refuses_a_key_row_it_cannot_find():
    key = [{"item": "1", "prompt": "Q: a\nA:", "frac": "-2.0", "direction": "d"}]
    with pytest.raises(SystemExit, match="item 1"):
        ja.judge_by_item(key, [], "truthful", "informative")


def test_agreement_rows_only_score_items_the_reference_has():
    ref = {1: (1, 1), 2: (0, 1)}
    judges = {"j": {1: (1, 0), 2: (0, 1), 3: (1, 1)}}
    rows = ja.agreement_rows("human16", ref, judges)
    t = next(r for r in rows if r["axis"] == "truthful")
    i = next(r for r in rows if r["axis"] == "informative")
    assert (t["n"], t["agree"], t["accuracy"]) == (2, 2, 1.0)
    assert (i["n"], i["agree"], i["accuracy"]) == (2, 1, 0.5)
    assert t["reference"] == "human16" and t["judge"] == "j"


def test_dose_rows_report_rate_per_dose_and_an_unpaired_test():
    labels = {1: (1, 1), 2: (1, 1), 3: (0, 1), 4: (0, 1)}
    frac = {1: -2.0, 2: -2.0, 3: 0.0, 4: 0.0}
    (row,) = [r for r in ja.dose_rows("x", labels, frac) if r["axis"] == "truthful"]
    assert (row["n_frac0"], row["rate_frac0"], row["n_frac-2"], row["rate_frac-2"]) == \
        (2, 0.0, 2, 1.0)
    assert 0 < row["fisher_p"] <= 1


def test_stage_handscore_end_to_end_on_fakes(tmp_path, monkeypatch, capsys):
    import csv as _csv
    monkeypatch.chdir(tmp_path)

    def w(path, rows):
        with open(path, "w", newline="") as f:
            d = _csv.DictWriter(f, fieldnames=list(rows[0]))
            d.writeheader()
            d.writerows(rows)

    items = range(1, 9)
    frac = {i: (-2.0 if i % 2 else 0.0) for i in items}
    prompt = {i: f"Q: q{i}\nA:" for i in items}
    w(ja.HAND_KEY, [{"item": i, "prompt": prompt[i], "frac": frac[i],
                     "direction": ja.MEAN_DIRECTION} for i in items])
    w(ja.HAND_CLAUDE, [{"item": i, "truthful": i % 2, "informative": 1, "note": "",
                        "labeller": "c"} for i in items])
    w(ja.HAND_HUMAN, [{"item": i, "question": "q", "answer": "a", "truthful": i % 2,
                       "informative": 1, "note": ""} for i in (1, 2, 3, 4)])
    w(ja.V2_PATTERN.format(ds=ja.DS, arm="mean"),
      [{"direction": ja.MEAN_DIRECTION, "scale": frac[i], "prompt": prompt[i],
        "truthful": i % 2, "informative": 1} for i in items])
    w(ja.STAGE_OUTPUT["qwen"],
      [{"source": "mean", "direction": ja.MEAN_DIRECTION, "frac": frac[i],
        "prompt": prompt[i], "qwen_truthful": 1, "qwen_informative": 1} for i in items])
    monkeypatch.setattr(ja, "with_frac",
                        lambda rows: [dict(r, frac=float(r["scale"])) for r in rows])
    ja.stage_handscore()
    agree = ja.read_csv(ja.HANDSCORE_OUT)
    h_claude = next(r for r in agree if r["reference"] == "human16"
                    and r["judge"] == "claude64" and r["axis"] == "truthful")
    assert (h_claude["n"], h_claude["accuracy"]) == ("4", "1.0")
    assert {r["reference"] for r in agree} == {"human16", "claude64"}
    dose = ja.read_csv(ja.HANDSCORE_DOSE_OUT)
    assert {r["labeller"] for r in dose} == {"human16", "claude64", "allenai", "qwen"}
    assert "human16 vs claude64" in capsys.readouterr().out
