import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.metrics import distinct_n, corpus_distinct_n, repetition_rate


def test_distinct_n_all_unique():
    assert distinct_n("a b c d", n=2) == 1.0  # 3 bigrams, all unique


def test_distinct_n_with_repeats():
    # "a b a b" -> bigrams (a,b),(b,a),(a,b) -> 2 unique / 3 = 0.666...
    assert abs(distinct_n("a b a b", n=2) - 2/3) < 1e-9


def test_distinct_n_too_short():
    assert distinct_n("a", n=2) == 0.0


def test_corpus_distinct_n_dedups_across_texts():
    # two identical texts -> unique bigrams counted once over total
    # each "a b c" has 2 bigrams; corpus total 4, unique 2 -> 0.5
    assert corpus_distinct_n(["a b c", "a b c"], n=2) == 0.5


def test_repetition_rate_no_repeats():
    assert repetition_rate("a b c d") == 0.0


def test_repetition_rate_all_same():
    # 4 tokens, 1 unique -> 1 - 1/4 = 0.75
    assert repetition_rate("a a a a") == 0.75
