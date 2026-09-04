"""Build got_datasets/truthfulqa.csv (+ truthfulqa_holdout.csv) for track Q.

Public data: truthfulqa/truthful_qa, the `generation` config, 817 questions.

POLARITY (important): label 1 = UNTRUTHFUL, label 0 = TRUTHFUL. This is INVERTED
relative to cities.csv and common_claim_true_false.csv, deliberately. reach_analyze
computes eps* for label-1 statements crossing into the label-0 halfspace, so this
convention makes the certificate describe steering an untruthful reading TOWARD
truthful. That is the direction the advisor's first note asks for and it is the
direction with headroom: TruthfulQA is adversarially constructed so the model answers
falsely often, whereas on cities it completes truthfully 94% of the time and there is
almost nothing to move (docs/D2_PREFIX_TRANSFER.md).

THE CONTRASTIVE PAIR: each kept question yields exactly two rows, (prompt + correct
answer) and (prompt + incorrect answer). The two share a byte-identical prompt prefix,
so what separates the classes is the answer's truthfulness in the generation format,
not a prompt-format offset. That is the fix for what D2 measured, where a direction fit
on declarative statements and read on generation prefixes came out at chance, AUC 0.510
with the readout never changing sign.

FIT POINT vs STEER POINT: extract.py reads the last non-pad token, which on these rows
is the last token of the ANSWER, while generation happens at the last token of the
PROMPT. Those are still two different tokens, so the transfer D2 measured has to be
measured again here rather than assumed. The difference is that it is now measurable:
the holdout supplies real generation prompts on a population where the model is wrong
often enough for both classes to exist. Q1 measures that base rate before any steering.

"I HAVE NO COMMENT": 56 questions carry it as best_answer and 110 carry it somewhere in
correct_answers. It is truthful and uninformative, so putting it on the truthful side
would build a "decline to answer" component into the truth direction. Our refusal
control PASSED, so a truth direction contaminated with refusal is the one confound that
would invalidate the whole 2x2. Those answers are skipped, and a question with no
informative correct answer is dropped.

HOLDOUT: held-out QUESTIONS never enter direction fitting in either polarity. They
supply the mean-arm generation prompts for Q1 and Q2, so behavioral evaluation is
leakage-free. The default is larger than prep_refusal's 32 because Q1 reports a RATE
off this set, and 64 questions is +/- 6 points rather than +/- 9.

CHAT TEMPLATE: --chat-template <model-id> renders every question through that model's
chat template ONCE, here, exactly as prep_refusal.py does, so extract.py,
reach_margins.py and reach_steer.py all tokenize the identical string. Leave it off for
base models, which have no template; the fallback is the plain Q/A format the allenai
TruthfulQA judges were trained on.

    .venv/bin/python src/prep_truthfulqa.py
    .venv/bin/python src/prep_truthfulqa.py --chat-template google/gemma-2-2b-it
"""
import argparse
import json

import numpy as np
import pandas as pd

N_HOLDOUT = 64
HF_DATASET = "truthfulqa/truthful_qa"
PLAIN_PROMPT = "Q: {q}\nA:"
NO_COMMENT = "no comment"
LIST_COLS = ("correct_answers", "incorrect_answers")
LABEL1_CHOICES = ("untruthful", "truthful")


def prompt_of(question, tok=None):
    """The generation prompt for one question: everything the model sees before it has
    to produce an answer. Its last token is the point generation starts from, and is
    the point the certificate must be linearized about."""
    q = str(question).strip()
    if tok is None:
        return PLAIN_PROMPT.format(q=q)
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)


def statement_of(question, answer, tok=None):
    """prompt + answer, joined the way the model itself would continue the prompt.

    A chat template ends in a newline and takes the answer directly. The plain format
    ends in "A:" and takes a leading space, which is how the tokenizer sees a real
    completion; without it the answer's first word becomes a different token than the
    one generation would produce."""
    p = prompt_of(question, tok)
    sep = "" if p.endswith(("\n", " ")) else " "
    return p + sep + str(answer).strip()


def _informative(answer):
    """False for empty answers and for the "I have no comment" family."""
    return bool(str(answer).strip()) and NO_COMMENT not in str(answer).lower()


def answer_pair(row):
    """(correct, incorrect) for one TruthfulQA row, or None if it cannot make an
    informative contrast.

    correct   = best_answer, unless it is a no-comment non-answer, in which case the
                first informative entry of correct_answers.
    incorrect = the first incorrect answer, which is the canonical misconception the
                question was written to elicit."""
    correct = str(row["best_answer"]).strip()
    if not _informative(correct):
        correct = next((str(a).strip() for a in row["correct_answers"]
                        if _informative(a)), None)
    incorrect = next((str(a).strip() for a in row["incorrect_answers"]
                      if _informative(a)), None)
    if correct is None or incorrect is None:
        return None
    return correct, incorrect


def to_contrast_df(rows, tok=None, seed=42, label1="untruthful"):
    """Balanced, deterministic (statement, label) frame, two rows per kept question.
    label1 selects which class gets label 1; default "untruthful", see the module
    docstring on polarity."""
    if label1 not in LABEL1_CHOICES:
        raise ValueError(f"label1 must be one of {LABEL1_CHOICES}, got {label1!r}")
    lab_wrong = 1 if label1 == "untruthful" else 0
    out = []
    for r in rows:
        pair = answer_pair(r)
        if pair is None:
            continue
        correct, incorrect = pair
        for answer, lab in ((correct, 1 - lab_wrong), (incorrect, lab_wrong)):
            out.append({"statement": statement_of(r["question"], answer, tok),
                        "label": lab,
                        "question": str(r["question"]).strip(),
                        "answer": answer,
                        "category": r.get("category", ""),
                        "type": r.get("type", "")})
    df = pd.DataFrame(out, columns=["statement", "label", "question", "answer",
                                    "category", "type"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_holdout(rows, n_hold=N_HOLDOUT, seed=42):
    """Deterministically peel n_hold whole QUESTIONS off for behavioral evaluation.

    Whole questions, not rows: holding out one polarity of a question would leak its
    prompt into the direction fit through the other one, and the prompt is the part
    both rows share. Returns (fit_rows, holdout_rows)."""
    rows = list(rows)
    k = min(max(n_hold, 0), len(rows))
    idx = np.random.default_rng(seed).permutation(len(rows))
    hold, fit = sorted(idx[:k]), sorted(idx[k:])
    return [rows[i] for i in fit], [rows[i] for i in hold]


def holdout_df(rows, tok=None):
    """The behavioral-evaluation frame. `statement` is the generation PROMPT with no
    answer attached, under the column name reach_steer.load_prompt_set already reads
    for refusal_holdout. The reference answers ride along as JSON so the TruthfulQA
    judges can be scored against them without reloading the source dataset."""
    return pd.DataFrame(
        [{"statement": prompt_of(r["question"], tok),
          "question": str(r["question"]).strip(),
          "best_answer": str(r["best_answer"]).strip(),
          "correct_answers": json.dumps([str(a) for a in r["correct_answers"]]),
          "incorrect_answers": json.dumps([str(a) for a in r["incorrect_answers"]]),
          "category": r.get("category", ""),
          "type": r.get("type", "")} for r in rows],
        columns=["statement", "question", "best_answer", "correct_answers",
                 "incorrect_answers", "category", "type"])


def load_holdout(path="got_datasets/truthfulqa_holdout.csv"):
    """Read the holdout back with the JSON list columns parsed into lists."""
    df = pd.read_csv(path)
    for c in LIST_COLS:
        df[c] = df[c].apply(json.loads)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="generation")
    p.add_argument("--split", default="validation")
    p.add_argument("--out", default="got_datasets/truthfulqa.csv")
    p.add_argument("--holdout-out", default="got_datasets/truthfulqa_holdout.csv")
    p.add_argument("--n-holdout", type=int, default=N_HOLDOUT)
    p.add_argument("--label1", default="untruthful", choices=list(LABEL1_CHOICES))
    p.add_argument("--chat-template", default=None,
                   help="model id whose chat template to apply to every question")
    args = p.parse_args()

    from datasets import load_dataset
    rows = [dict(r) for r in load_dataset(HF_DATASET, args.config)[args.split]]
    tok = None
    if args.chat_template:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.chat_template)
        print(f"applied the {args.chat_template} chat template to every question")

    fit_rows, hold_rows = split_holdout(rows, args.n_holdout)
    df = to_contrast_df(fit_rows, tok, label1=args.label1)
    hold = holdout_df(hold_rows, tok)
    df.to_csv(args.out, index=False)
    hold.to_csv(args.holdout_out, index=False)

    dropped = len(fit_rows) - len(df) // 2
    print(f"wrote {args.out}: {len(df)} rows from {len(df) // 2} questions, "
          f"label1={args.label1}, {int(df.label.sum())} label-1 / "
          f"{int((df.label == 0).sum())} label-0")
    if dropped:
        print(f"  dropped {dropped} questions with no informative correct answer")
    print(f"wrote {args.holdout_out}: {len(hold)} held-out generation prompts "
          f"({int((hold['type'] == 'Adversarial').sum())} adversarial / "
          f"{int((hold['type'] != 'Adversarial').sum())} non-adversarial)")


if __name__ == "__main__":
    main()
