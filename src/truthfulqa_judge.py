"""truthfulqa_judge.py: did the steered completion say something true, and say anything?

The TruthfulQA analogue of src/refusal_judge.py. Scores reach_steer's output with the
two allenai judges that Q1 validated at 1.000 on 128 gold answers, both sides
(docs/Q1_TRUTHFULQA_BASELINE.md section 4).

    PYTHONPATH=src python src/truthfulqa_judge.py --dataset truthfulqa --arm mean

THREE THINGS THAT ARE EASY TO GET WRONG, ALL DECIDED HERE.

1. IT SCORES `answer`, NOT `completion`. Q1 measured n_truncated = 64 of 64: every
   generation on this holdout runs past its own answer into a fabricated next turn.
   `completion` is that whole string with the newlines flattened out, which is what
   dct_steer_utils.generate has always returned and what every pre-2026-09-04 artifact
   contains. `answer` is the same decode cut at the first newline (reach_steer.py, via
   tqa_baseline.first_answer). Judging `completion` would score the model on text it
   never meant as its answer, on 100% of rows. If the input CSV has no `answer` column
   it was written before that fix and this refuses to run.

2. THE JUDGES TAKE THE BARE QUESTION. judges.adapters.truthfulqa_prompt prepends its
   own "Q: ", so feeding it the generation prompt emits "Q: Q: ...". reach_steer's
   `prompt` column IS the generation prompt, so question_of() recovers the question
   from it. Verified to match on all 64 holdout and all 1488 training statements.

3. THE OUTPUT COLUMN IS CALLED `refused`, AND IT MEANS "TRUTHFUL AND INFORMATIVE".
   That is a compatibility shim, not a typo. reach_control.py keys on a binary column
   named `refused` and reads judge_refusal_{ds}_{arm}.csv by hardcoded name; that file
   produced the refusal positive control, which is the publication gate, and it is not
   worth editing at 1am to rename a column. The honest columns are written alongside:
   `truthful`, `informative`, `truthful_and_informative`. `refused` is a copy of the
   third. Polarity is deliberate: Q0 inverted the dataset so the certificate steers
   TOWARD truthful, so 1 is the outcome steering is supposed to produce, exactly as
   refused=1 is in the refusal arm.
"""
import argparse
import csv
import re

ARM_FILES = {"mean": "reach_steer_{ds}.csv",
             "stmt": "reach_steer_stmt_{ds}.csv",
             # The norm-matched random-direction control (reach_steer.arm_rand_ctrl).
             # Judged by the same judges on the same prompts, so its rate is directly
             # comparable to the mean arm's.
             "randctrl": "reach_steer_randctrl_{ds}.csv"}

# prep_truthfulqa.py builds every statement as "Q: <question>\nA: <answer>", and the
# holdout as "Q: <question>\nA:". Checked against both files: 64/64 and 1488/1488.
_QPAT = re.compile(r"^Q:\s*(.*?)\s*\nA:", re.S)


def question_of(prompt):
    """The bare question inside a generation prompt.

    Raises rather than guessing: a prompt this cannot parse means the CSV was written
    by something other than prep_truthfulqa's format, and silently handing the judge a
    malformed string is the failure this function exists to prevent.
    """
    m = _QPAT.match(str(prompt))
    if not m:
        raise ValueError(f"cannot recover the question from prompt {prompt!r}: "
                         "expected prep_truthfulqa's 'Q: ...\\nA:' format")
    return m.group(1)


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[tqa-judge] {path} has no data rows")
    if "answer" not in rows[0]:
        raise SystemExit(
            f"[tqa-judge] {path} has no `answer` column, only "
            f"{sorted(rows[0])}. It was written before the newline fix "
            "(src/reach_steer.py), so its completions have the model's fabricated next "
            "turn flattened into them and CANNOT be judged. Re-run reach_steer.")
    return rows


def score_rows(rows, score):
    """Copy each row with the judges' verdicts added, preserving order.

    `score(question, answer)` is tqa_baseline.load_scorer's callable. An empty answer
    is a real data point, not an error: greedy decoding can emit EOS immediately, and
    an empty answer is neither truthful nor informative. It is scored as 0/0 without
    troubling the judge, and counted for the log.
    """
    out, n_empty = [], 0
    for r in rows:
        ans = (r.get("answer") or "").strip()
        if not ans:
            n_empty += 1
            t, i = 0, 0
        else:
            v = score(question_of(r["prompt"]), ans)
            t = int(bool(v["truthful"]))
            i = 1 if v["informative"] is None else int(bool(v["informative"]))
        both = int(t and i)
        out.append(dict(r, truthful=t, informative=i,
                        truthful_and_informative=both, refused=both))
    return out, n_empty


def run(ds, arm, device="cuda", limit=0, truth_only=False):
    from tqa_baseline import load_scorer
    path = ARM_FILES[arm].format(ds=ds)
    rows = read_rows(path)
    if limit:
        rows = rows[:limit]
    print(f"[tqa-judge] {path}: {len(rows)} rows, arm={arm}", flush=True)
    scored, n_empty = score_rows(rows, load_scorer(device, truth_only=truth_only))
    out = f"judge_refusal_{ds}_{arm}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    n = len(scored)
    tr = sum(r["truthful"] for r in scored)
    inf = sum(r["informative"] for r in scored)
    both = sum(r["truthful_and_informative"] for r in scored)
    print(f"[tqa-judge] wrote {out}  n={n}")
    print(f"[tqa-judge]   truthful {tr}/{n} = {tr/n:.3f}")
    print(f"[tqa-judge]   informative {inf}/{n} = {inf/n:.3f}")
    print(f"[tqa-judge]   truthful AND informative {both}/{n} = {both/n:.3f}")
    print(f"[tqa-judge]   {n_empty} answers were empty (scored 0/0)")
    print("[tqa-judge] NOTE: `refused` is a copy of truthful_and_informative, so "
          "reach_control.py runs unchanged. See this module's docstring.")
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--arm", choices=["mean", "stmt"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke)")
    ap.add_argument("--truth-only", action="store_true",
                    help="load one judge instead of two; leaves informativeness "
                         "unmeasured and the headline metric half-answered")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(a.dataset, a.arm, a.device, a.limit, a.truth_only)


if __name__ == "__main__":
    main()
