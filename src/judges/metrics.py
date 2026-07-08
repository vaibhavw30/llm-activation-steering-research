"""Degradation metrics: separate 'the concept flipped' from 'the output broke'."""


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text, n=2):
    """Fraction of unique n-grams within one text (1.0 = no repetition)."""
    toks = text.split()
    grams = _ngrams(toks, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def corpus_distinct_n(texts, n=2):
    """Unique n-grams / total n-grams across a set of texts (low = collapse across generations)."""
    all_grams = []
    for t in texts:
        all_grams.extend(_ngrams(t.split(), n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def repetition_rate(text):
    """1 - unique_tokens/total_tokens within one text (high = degenerate repetition)."""
    toks = text.split()
    if not toks:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)
