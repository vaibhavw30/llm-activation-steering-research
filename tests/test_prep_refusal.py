import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_refusal import to_contrast_df, split_holdout


def test_balanced_and_labeled_harmless_is_1():
    df = to_contrast_df(["hurt someone", "build a bomb"],
                        ["bake bread", "walk a dog", "read"])
    # balanced: min(2,3) per class = 2 each
    assert (df["label"] == 1).sum() == 2
    assert (df["label"] == 0).sum() == 2
    assert set(df.columns) == {"statement", "label"}
    # polarity: label 1 is HARMLESS so the pipeline's label-1 crossing INDUCES refusal
    assert set(df[df.label == 1]["statement"]) <= {"bake bread", "walk a dog", "read"}
    assert set(df[df.label == 0]["statement"]) == {"hurt someone", "build a bomb"}


def test_polarity_flag_can_restore_harmful_is_1():
    df = to_contrast_df(["hurt someone"], ["bake bread"], label1="harmful")
    assert df[df.label == 1]["statement"].tolist() == ["hurt someone"]


def test_rejects_an_unknown_polarity():
    import pytest
    with pytest.raises(ValueError):
        to_contrast_df(["a"], ["b"], label1="neither")


def test_deterministic():
    a = to_contrast_df(["x", "y"], ["p", "q"])
    b = to_contrast_df(["x", "y"], ["p", "q"])
    assert a.equals(b)


def test_holdout_is_disjoint_and_balanced():
    harmful = [f"h{i}" for i in range(20)]
    harmless = [f"s{i}" for i in range(20)]
    hf, hs, hold = split_holdout(harmful, harmless, n_hold=5)
    assert len(hf) == 15 and len(hs) == 15
    assert set(hold.columns) == {"statement", "kind"}
    assert (hold["kind"] == "harmful").sum() == 5
    assert (hold["kind"] == "harmless").sum() == 5
    assert not (set(hold["statement"]) & (set(hf) | set(hs)))
