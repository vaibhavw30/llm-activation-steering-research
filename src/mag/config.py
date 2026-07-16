"""Fixed strings and constants for the MAG battery (arXiv:2607.04222, adapted to truth).
Kept in one file so a reviewer can see and change the exact instruction Q and few-shot E."""

# Base gemma-2-2b: plain completion, not chat. Readout / y^M taken at the last token of
#   Q_TRUTH + statement + Q_SUFFIX  (the next token is the yes/no verdict).
Q_TRUTH = "Answer in yes or no only. Is the following statement true?\nStatement: "
Q_SUFFIX = "\nAnswer:"

E_FEWSHOT = (                      # fixed 2-shot preamble for phi_FewShot (one true, one false)
    "Answer in yes or no only. Is the following statement true?\n"
    "Statement: The sky is blue.\nAnswer: yes\n"
    "Statement: Fish can fly.\nAnswer: no\n"
)

# First-token variants summed for p_yes / p_no (Eq. 1). Space-prefixed forms matter for gemma's BPE.
YES_VARIANTS = [" yes", "yes", " Yes", "Yes"]
NO_VARIANTS = [" no", "no", " No", "No"]

# MAG sweeps tau in {0, 0.3, 1.0}; we add the negatives so the lie-asymmetry test still applies.
TAUS = [-1.0, -0.3, 0.0, 0.3, 1.0]

OPERATOR_NAMES = ["Direct", "Prefixed", "Answered", "Verdict",
                  "InputDelta", "QuestionDelta", "Interaction", "FewShot"]

DATASETS = ["cities", "sp_en_trans", "companies_true_false", "common_claim_true_false"]
