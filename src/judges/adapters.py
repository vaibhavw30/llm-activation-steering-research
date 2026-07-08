"""Format adapters mapping our (stem, completion) rows into judge-specific prompts."""


def truthfulqa_prompt(stem, completion):
    """Map a completion-style (stem, completion) into the allenai TruthfulQA judge format.

    The allenai/truthfulqa-*-judge-llama2-7B models expect `Q: <q>\\nA: <a>\\nTrue:` and
    generate ` yes`/` no`.
    """
    q = stem.strip().rstrip(":").strip()
    a = completion.strip()
    return f"Q: {q}\nA: {a}\nTrue:"
