import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep_refusal import to_contrast_df


def test_balanced_and_labeled():
    df = to_contrast_df(["hurt someone", "build a bomb"], ["bake bread", "walk a dog", "read"])
    # balanced: min(2,3) per class = 2 each
    assert (df["label"] == 1).sum() == 2
    assert (df["label"] == 0).sum() == 2
    assert set(df.columns) == {"statement", "label"}
    assert df[df.label == 1]["statement"].tolist()  # harmful present


def test_deterministic():
    a = to_contrast_df(["x", "y"], ["p", "q"])
    b = to_contrast_df(["x", "y"], ["p", "q"])
    assert a.equals(b)
