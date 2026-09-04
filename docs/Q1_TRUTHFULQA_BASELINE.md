# Q1. gemma-2-2b's unsteered TruthfulQA baseline

*Measured 2026-09-04 on CLUSTER, SLURM job 3081925, 6m40s on one GH200, 37.6 GB peak RSS.
Script: [`src/tqa_baseline.py`](../src/tqa_baseline.py). Job:
[`deltaai/run_tqa_baseline.slurm`](../deltaai/run_tqa_baseline.slurm). This is step Q1 of
[`PLAN_ADVISOR_NOTES_2026-09.md`](PLAN_ADVISOR_NOTES_2026-09.md) section 3, the gate in front of
the whole Q track.*

## 1. The question this had to answer

The plan states the gate:

> **Q1. Baseline, before any steering.** Measure gemma-2-2b's unsteered truthful rate and
> informative rate on TQA. This is the number that decides whether the track is viable at all, and
> it must be measured rather than assumed. **If the base rate is near ceiling as it is on cities,
> stop and reconsider.**

The reason it is a gate and not a footnote: `cities` is **94.3% truthful**, so its spontaneous lie
rate is 5.7% and there is essentially nothing for a steering intervention to move. Every truth
null in this project was measured against that ceiling. `CEILING = 0.90` operationalises "near
ceiling" in the script.

## 2. What was run

The 64 held-out questions from Q0, none of which were seen by the direction fit. Three stages in
one job:

1. **Generate.** Greedy, `max_new_tokens=48`, `repetition_penalty=1.3`, mirroring
   `dct_steer_utils.generate` because that is the decoder `reach_steer.py` calls in Q2. The answer
   is cut at the first newline (`first_answer`), for the reason in section 6.
2. **Judge.** `allenai/truthfulqa-truth-judge-llama2-7B` and
   `allenai/truthfulqa-info-judge-llama2-7B`, the standard TruthfulQA judges, run locally on the
   compute node.
3. **Gold check.** The same judges re-scored TruthfulQA's own `best_answer` (truthful by
   construction) and first informative `incorrect_answers` entry (untruthful by construction), for
   all 64 questions, 128 answers.

## 3. Result

| metric | rate | Wilson 95% | count |
|---|---:|---|---:|
| truthful | **0.281** | [0.186, 0.401] | 18/64 |
| informative | **0.984** | [0.917, 0.997] | 63/64 |
| **truthful AND informative** | **0.266** | [0.173, 0.385] | 17/64 |

The 2x2 those three imply is degenerate in a useful way:

| | informative | not informative |
|---|---:|---:|
| **truthful** | 17 | 1 |
| **not truthful** | 46 | 0 |

**Every uninformative answer was also truthful, and there was exactly one of them.** That is the
"I have no comment" cell, and it is nearly empty. `n_empty_answers = 0`: no generation came back
blank.

## 4. The judge is validated on this holdout

| gold side | n | accuracy |
|---|---:|---:|
| correct answers, expected TRUE | 64 | **1.000** |
| incorrect answers, expected FALSE | 64 | **1.000** |
| both | 128 | **1.000** |

Reported per side deliberately. A judge that answers TRUE to everything scores 1.000 on the
correct side and 0.000 on the incorrect side, and a single pooled accuracy of 0.500 would hide
which failure it was. Neither side degenerates here. This is the same discipline that made the
refusal control readable, where the substring judge validated at 0.969.

**The honest caveat.** Gold answers are short, clean reference strings; model answers are messy
48-token generations. Perfect accuracy on the gold distribution is a **necessary** condition for
trusting the 0.281, not a proof of accuracy on the distribution actually being scored. What it
rules out is a judge that is broken, mis-prompted, or stuck on one verdict, which is exactly the
failure that would have made this number meaningless.

## 5. Verdict: PROCEED, and the margin is not close

`headline_rate = 0.266` against `CEILING = 0.90`. The upper end of the Wilson interval is 0.385.
The gate does not trigger, and it does not come near triggering.

| dataset | truthful rate | headroom |
|---|---:|---|
| `cities` | 0.943 | 0.057. This is why every truth null there was unreadable |
| **`truthfulqa` holdout** | **0.281** | **0.719** |

**The two facts that matter, together.** The model is untruthful on 72% of these questions **and**
informative on 98% of them. Either alone would be useless. A low truthful rate with a low
informative rate would mean the model was dodging, and a steering result would be measuring
whether we made it dodge more. A high informative rate with a high truthful rate is the cities
ceiling again. This is the one combination that leaves room for a behavioral effect to be visible.

## 6. Three things Q2 inherits from this run, measured rather than assumed

**`n_truncated = 64`, i.e. 100%.** Every one of the 64 generations continued past its own answer
into a fabricated next turn, and `first_answer` cut it at the newline. This settles a wiring
question that was previously a judgement call: `dct_steer_utils.generate` calls
`.replace("\n", " ")`, so on a base model in `Q:`/`A:` format it flattens the real answer together
with a hallucinated follow-up Q and A into one string. **Q2 must take the raw decode and apply
`tqa_baseline.first_answer`, not the flattened string.** Without that fix the judge would score
the wrong text on 100% of rows, not on some of them.

**The stem column.** `judges/adapters.truthfulqa_prompt` prepends its own `Q: `, so it must be fed
the holdout's bare `question` column. Feeding it `statement` emits `Q: Q: ...`. Q2 must make the
same choice.

**Q2 has to be analysed paired, not as two independent rates.** At n=64 the Wilson half-width on
the headline is +-0.106, and an unpaired two-proportion test at 80% power would need the steered
rate to reach **0.51**, a +0.24 absolute move, before it cleared significance. That is a large
effect to demand. The same 64 questions steered and unsteered is a paired design, and McNemar on
the discordant pairs needs only **6 flips all in one direction** to reach p = 0.031, 8 to reach
p = 0.008. This is exactly how the refusal positive control was read: 14 flips against 0. Q2
should register the paired analysis before it runs.

## 7. What this does not say

- **The adversarial split is underpowered and should not be leaned on.** Adversarial questions
  scored 0.250 (9/36) against 0.321 (9/28) for non-adversarial. The direction is what TruthfulQA's
  design predicts, since adversarial questions are the ones written to elicit imitative falsehoods,
  but Fisher exact gives p = 0.583. Report it as consistent, never as evidence.
- **This is one model, one decoder setting, 64 questions.** The rate is a property of
  gemma-2-2b at greedy decoding with `repetition_penalty=1.3` and a 48-token budget, not of the
  model in general. Q2 must steer at the same budget or this is not its control.
- **A low truthful rate is not by itself evidence that truth is steerable here.** It establishes
  that there is headroom, which is the only thing the gate asked. Whether the certificate can move
  anything into that headroom is Q2, and `context_factor` (see
  [`DERIVATION_SHORT.md`](DERIVATION_SHORT.md) section 4) is the reason it might not.

## 8. Artifacts

| file | rows | what |
|---|---:|---|
| `tqa_baseline_completions.csv` | 64 | question, prompt, answer, raw decode, truncated, empty, category, type |
| `tqa_baseline_judged.csv` | 64 | the above plus the two judge verdicts |
| `tqa_baseline_gold.csv` | 128 | gold answers with expected and observed verdicts |
| `tqa_baseline_summary.csv` | 1 | every rate, interval, count, and the gate verdict |
| `tqa_baseline_3081925.out` | | the run log, kept at the repo root like the other SLURM logs |

Reproduce the summary from the CSVs on LAPTOP with
`PYTHONPATH=src ./.venv/bin/python src/tqa_baseline.py --summarize`, which imports no torch.
