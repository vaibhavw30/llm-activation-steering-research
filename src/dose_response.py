"""D1: the dose-response kill test (design spec section 7).

Assumption under test: the behavioral null is a fact about the model, not an artifact of
sampling only oversteered magnitudes.

S3 showed the two committed arms bracket an unmeasured band. The naive arm's smallest
non-trivial dose is already at baseline (relative magnitude 0.23 to 0.29) and the
reachability arm never rose above 0.13. Nothing has ever been generated between 0.29 and
0.78, which is exactly where a clean low-magnitude effect would live. D1 sweeps that band.

    PYTHONPATH=src python src/dose_response.py --dataset cities --device cuda
    PYTHONPATH=src python src/dose_response.py --dataset cities --device cpu --limit 2 \
        --grid 0.3,1.0 --out-prefix dose_smoke                          # local smoke

WHY THE MAGNITUDE AXIS IS RELATIVE, AND NOT tau OR eps*
-------------------------------------------------------
The two committed arms parameterise dose differently and neither normaliser is the
activation norm:

    naive:  alpha = tau  * A_prefix_norm      (mag_dir_<ds>.npz)
    reach:  alpha = frac * eps_star           (reach_summary_<ds>.json)

That is the whole reason their ranges were never comparable. D1 uses the S3 canonical axis
directly: alpha = rel * median ||h_src|| at the injection layer, so every direction here and
every point in both committed arms live on one ruler by construction. On cities the two
happen to be close (A_prefix_norm 113.80 against median ||h_src|| 118.06), so the committed
tau = 0.3 and tau = 1.0 land at rel 0.289 and 0.964 and can be read straight off the D1 plot.

TWO OUTCOMES, NEVER COLLAPSED
-----------------------------
"The model lied" and "the model broke" are different events and the spec forbids merging
them. This script emits two independent measurements per (direction, dose):

  * free-form:  32 FACTUAL_PROMPTS -> dose_<ds>.csv, scored later by judge_results.py into
                TRUE / FALSE / INCOHERENT. Schema matches run_steer so no judge change.
  * yes/no:     24 YESNO_STATEMENTS -> dose_yesno_<ds>.csv, per statement, paired against
                that statement's own unsteered answer.

WHY THE YES/NO PROBE LOGS A MARGIN AND NOT JUST A FLIP
------------------------------------------------------
S1 measured the detection floor of the committed flip rate: 24 binary trials cannot resolve
an asymmetry smaller than 9/24 = 0.375. Eight of nine directions reading exactly 0/24 on
common_claim therefore means "no effect larger than a third", not "no effect". So alongside
the discrete answer this logs the continuous first-token margin p_yes - p_no (MAG Eq. 1, via
mag.verdict), paired to the same statement's unsteered margin. A dose effect far too small
to move an argmax is visible in that margin, and the pairing makes it a signed within-item
difference rather than a difference of two rates.

ARTIFACT SAFETY
---------------
Writes only dose_<ds>.csv, dose_yesno_<ds>.csv, dose_meta_<ds>.json. Refuses to start if an
output already exists unless --resume or --overwrite is given, so a partially finished
cluster run is never silently clobbered by a restart. --limit (a smoke run) is refused unless
--out-prefix is changed, so a two-prompt test can never land on the real artifact names.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dct_steer_utils as su
from funnel_utils import unit
from mag.config import Q_TRUTH, Q_SUFFIX, YES_VARIANTS, NO_VARIANTS
from mag.steer import YESNO_STATEMENTS, _yes_no_answer
from mag.verdict import first_token_ids, verdict_from_logits
from steer_supervised import FACTUAL_PROMPTS

# Revised grid. The spec's original put 8 of 12 points below 0.20, which S3 showed is
# territory where the naive directions are already at baseline. Rationale: docs/S3_COMMON_AXIS.md
# section 7. Six of these twelve points fall inside the previously unmeasured 0.29 to 0.78 band.
REL_GRID = [0.01, 0.02, 0.04, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52, 0.66, 0.82, 1.00]

RAND_SEED = 1234
MAX_NEW_FACTUAL = 8      # matches mag/steer.py, so completions are judged on the same length
MAX_NEW_YESNO = 3        # matches mag/steer.py
LIVENESS_REL = 5.0       # a perturbation this large MUST change the text; see check_hook_live

FACTUAL_FIELDS = ["direction", "scale", "prompt", "completion"]
YESNO_FIELDS = ["direction", "scale", "statement", "answer", "base_answer", "flipped",
                "margin", "base_margin", "d_margin", "p_yes", "p_no"]


def signed_grid(rels):
    """Zero first, then ascending magnitude, both signs. 12 magnitudes -> 25 points."""
    return [0.0] + [s * r for r in sorted(rels) for s in (+1.0, -1.0)]


# ------------------------------------------------------------------ geometry and directions
def load_geometry(ds):
    """Injection layer and the canonical denominator, cross-checked across all three files
    that claim to know them. The three directions families were fit independently; if they
    disagree about the layer then a shared magnitude axis is meaningless and we stop."""
    md = np.load(f"mag_dir_{ds}.npz")
    td = np.load(f"truth_dir_{ds}.npz")
    meta = json.load(open(f"dct_meta_{ds}.json"))
    layers = {"mag_dir": int(md["layer"]), "truth_dir": int(td["layer"]),
              "dct_meta.source_layer": int(meta["source_layer"])}
    if len(set(layers.values())) != 1:
        raise SystemExit(f"[D1] refusing to run: injection layer disagrees across files: {layers}")

    z = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    h = np.linalg.norm(np.asarray(z["h_src"], np.float64), axis=1)
    summ = json.load(open(f"reach_summary_{ds}.json"))
    return {
        "layer": int(md["layer"]),
        "h_src_median": float(np.median(h)),
        "A_prefix_norm": float(md["A_prefix_norm"]),
        "input_scale": float(meta["input_scale"]),
        # The key is "model", not "model_name". reach_hop.load_meta reads it this way and
        # reach_steer.py:102 documents why it matters: falling through to the base gemma-2-2b
        # literal on an -it run would steer the wrong checkpoint silently. `or` not a get()
        # default, because make_reach_meta.py writes the key and it can be present as null.
        "model_name": str(meta.get("model") or su.MODEL_NAME),
        "eps_star_mean_diff_tgt": float(summ["directions"]["mean_diff_tgt"]["median_eps_star"]),
    }


def jtw_mean_direction(ds):
    """The reachability arm's direction: the label-1 mean of the per-statement unit J^T w for
    w = mean_diff_tgt, exactly as src/reach_steer.py:arm_mean builds it."""
    mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
    acts = np.load(f"reach_acts_{ds}.npz", allow_pickle=True)
    store_names = [str(x) for x in mz["store_names"]]
    ks = store_names.index("mean_diff_tgt")
    jtw = np.asarray(mz["jtw"], np.float64)
    y = np.asarray(acts["labels"]).astype(int)[:jtw.shape[0]]
    return unit(jtw[y == 1, ks, :].mean(axis=0))


def load_directions(ds, geo):
    """The six directions Gate 1 selected, all unit, all native to the injection layer.

    sup_grad          : the only direction S1 found cleanly directional (monotone in both signs).
    sup_mean_diff     : the contrastive axis; the reference truth direction throughout.
    mag_u_gold        : the MAG InputDelta direction, the naive arm's headline.
    mag_resid_pc1     : NOT a truth axis. It is the dominant off-truth-axis shift component and
                        the only direction that flipped anything on common_claim, so it is the
                        norm-effect control: whatever it does at a given dose is what a large
                        perturbation does regardless of truth content.
    jtw_mean_diff_tgt : the reachability arm, so both arms land on the same grid.
    rand_ctrl         : seeded Gaussian unit vector. Norm matching is automatic here since every
                        direction is unit and every dose is the same alpha.
    """
    md = np.load(f"mag_dir_{ds}.npz")
    td = np.load(f"truth_dir_{ds}.npz")
    dim = int(md["u_Q_gold_unit"].shape[0])
    rng = np.random.default_rng(RAND_SEED)
    out = [
        ("sup_grad", unit(np.asarray(td["grad"], np.float64))),
        ("sup_mean_diff", unit(np.asarray(td["mean_diff"], np.float64))),
        ("mag_u_gold", unit(np.asarray(md["u_Q_gold_unit"], np.float64))),
        ("mag_resid_pc1", unit(np.asarray(md["resid_pc1_unit"], np.float64))),
        ("jtw_mean_diff_tgt", jtw_mean_direction(ds)),
        ("rand_ctrl", unit(rng.standard_normal(dim))),
    ]
    for name, v in out:
        if v.shape[0] != dim:
            raise SystemExit(f"[D1] direction {name} has dim {v.shape[0]}, expected {dim}")
    return out


def direction_cosines(dirs):
    """Pairwise cosines, recorded in the meta file. If two directions in the panel are nearly
    parallel then their curves are not independent evidence and the writeup must say so."""
    names = [n for n, _ in dirs]
    M = np.stack([v for _, v in dirs])
    C = M @ M.T
    return {f"{names[i]}|{names[j]}": float(C[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))}


# ------------------------------------------------------------------ measurement
def yesno_readout(model, tok, statement, yes_ids, no_ids, dev):
    """One forward over Q_TRUTH + statement + Q_SUFFIX with whatever hook is currently set.
    Returns the MAG first-token verdict dict. Single row, so no padding is involved and the
    last position is unambiguous."""
    enc = tok(Q_TRUTH + statement + Q_SUFFIX, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**enc).logits
    return verdict_from_logits(logits[0, -1, :].float().cpu().numpy(), yes_ids, no_ids)


def check_hook_live(model, tok, steerer, direction, h_med, baseline_text):
    """A perturbation of five times the median activation norm cannot leave the text
    unchanged. If it does, the hook is not firing and every null in this run is an artifact
    of a dead hook rather than a fact about the model. Same spirit as the oracle line in
    run_token_steer.slurm: fail loudly before producing a plausible-looking wrong answer."""
    steerer.set(torch.tensor(LIVENESS_REL * h_med * direction, dtype=torch.float32))
    txt = su.generate(model, tok, FACTUAL_PROMPTS[0], MAX_NEW_FACTUAL)
    steerer.set(None)
    if txt == baseline_text:
        raise SystemExit(
            f"[D1][assert] HOOK IS DEAD. At rel={LIVENESS_REL} the completion is byte-identical "
            f"to the unsteered one ({txt!r}). Nothing in this run would be meaningful. "
            f"Check dct_steer_utils.Steerer against the model's layer signature.")
    print(f"[D1][assert] hook live: rel={LIVENESS_REL} changes the text "
          f"({baseline_text!r} -> {txt!r})", flush=True)


# ------------------------------------------------------------------ incremental IO
class Appender:
    """Append-and-flush CSV writer. A run of this size will sometimes hit the SLURM wall, and a
    truncated file that is still valid CSV is worth far more than a complete file that was
    never written."""

    def __init__(self, path, fields, resume):
        self.path = path
        self.fields = fields
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self.f = open(path, "a" if (resume and exists) else "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if not (resume and exists):
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        self.w.writerow(row)

    def flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def done_pairs(path):
    """(direction, rounded scale) cells already present, so --resume can skip them.

    Keyed on the PARSED float, not the formatted string. A partial file that has been through
    pandas, a spreadsheet, or any other round trip writes 1.0 where this script wrote +1.0000,
    and a string key would silently treat every such cell as unfinished and regenerate the
    whole run on top of itself."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["direction"], round(float(r["scale"]), 6)) for r in csv.DictReader(f)}


def fmt_scale(s):
    return f"{s:+.4f}" if s != 0.0 else "0.0000"


def align_resume(fac_path, yn_path, fac_fields, yn_fields):
    """Make the two dose files agree on which cells are finished, and return that set.

    A cell writes all of its free-form rows, then all of its yes/no rows, then flushes both.
    A kill in between leaves a cell present in one file and absent from the other. Trusting
    either file alone then loses data silently: skipping on the factual file would leave that
    cell's 24 yes/no rows missing forever, and not skipping would duplicate its 32 factual
    rows. So only cells complete in BOTH files count as done, and any half-written cell is
    stripped from both before appending resumes."""
    done = done_pairs(fac_path) & done_pairs(yn_path)
    for path, fields in ((fac_path, fac_fields), (yn_path, yn_fields)):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if (r["direction"], round(float(r["scale"]), 6)) in done]
        if len(keep) != len(rows):
            print(f"[D1] resume: dropping {len(rows) - len(keep)} rows from {path} belonging "
                  f"to cells that were only half written", flush=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(keep)
    return done


# ------------------------------------------------------------------ main sweep
def run(ds, device, grid, limit=0, only=None, resume=False, prefix="dose"):
    geo = load_geometry(ds)
    dirs = load_directions(ds, geo)
    if only:
        subs = [s.strip() for s in only.split(",") if s.strip()]
        dirs = [(n, v) for n, v in dirs if any(s in n for s in subs)]
        if not dirs:
            raise SystemExit(f"[D1] --only {only!r} matched no directions")

    h_med = geo["h_src_median"]
    scales = signed_grid(grid)
    factual = FACTUAL_PROMPTS[:limit] if limit else FACTUAL_PROMPTS
    yesno = YESNO_STATEMENTS[:limit] if limit else YESNO_STATEMENTS
    n_gen = (len(dirs) * (len(scales) - 1) + 1) * (len(factual) + len(yesno))

    print(f"[D1] {ds}: layer {geo['layer']}, median ||h_src|| = {h_med:.3f}", flush=True)
    print(f"[D1] certified budget eps* = {geo['eps_star_mean_diff_tgt']:.3f} "
          f"= rel {geo['eps_star_mean_diff_tgt'] / h_med:.4f}", flush=True)
    print(f"[D1] {len(dirs)} directions x {len(scales)} doses x "
          f"({len(factual)} factual + {len(yesno)} yes/no) = {n_gen} generations", flush=True)

    fac_path, yn_path = f"{prefix}_{ds}.csv", f"{prefix}_yesno_{ds}.csv"
    for p in (fac_path, yn_path):
        if os.path.exists(p) and not resume:
            raise SystemExit(f"[D1] {p} already exists. Pass --resume to continue it, or "
                             f"--overwrite to discard it. Refusing to clobber a finished run.")

    tok, model, dev = su.load_model(device, model_name=geo["model_name"])
    yes_ids = first_token_ids(tok, YES_VARIANTS)
    no_ids = first_token_ids(tok, NO_VARIANTS)

    skip = align_resume(fac_path, yn_path, FACTUAL_FIELDS, YESNO_FIELDS) if resume else set()
    fac = Appender(fac_path, FACTUAL_FIELDS, resume)
    yn = Appender(yn_path, YESNO_FIELDS, resume)
    if skip:
        print(f"[D1] resuming: {len(skip)} (direction, dose) cells already done", flush=True)

    # ---- baseline. The hook is off, so the unsteered result is identical for every direction.
    # Generated once and stored under direction "baseline"; the analysis joins it to each curve.
    steerer = su.Steerer(model, geo["layer"])
    base_ans, base_marg = {}, {}
    with steerer:
        steerer.set(None)
        base_text0 = su.generate(model, tok, factual[0], MAX_NEW_FACTUAL)
        if ("baseline", 0.0) not in skip:
            for p in factual:
                c = su.generate(model, tok, p, MAX_NEW_FACTUAL)
                fac.write({"direction": "baseline", "scale": fmt_scale(0.0),
                           "prompt": p, "completion": c})
        for s in yesno:
            base_ans[s] = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX,
                                                     MAX_NEW_YESNO))
            r0 = yesno_readout(model, tok, s, yes_ids, no_ids, dev)
            base_marg[s] = r0["margin"]
            if ("baseline", 0.0) not in skip:
                yn.write({"direction": "baseline", "scale": fmt_scale(0.0), "statement": s,
                          "answer": base_ans[s], "base_answer": base_ans[s], "flipped": 0,
                          "margin": f"{r0['margin']:.6g}", "base_margin": f"{r0['margin']:.6g}",
                          "d_margin": "0", "p_yes": f"{r0['p_yes']:.6g}",
                          "p_no": f"{r0['p_no']:.6g}"})
        fac.flush(); yn.flush()
        n_unparsed = sum(1 for v in base_ans.values() if v == "?")
        parse_rate = 1.0 - n_unparsed / max(len(yesno), 1)
        print(f"[D1] baseline done; {sum(1 for v in base_ans.values() if v == 'yes')} yes, "
              f"{sum(1 for v in base_ans.values() if v == 'no')} no, {n_unparsed} unparsed, "
              f"mean margin {np.mean(list(base_marg.values())):+.4f}", flush=True)
        # A "?" baseline is not a neutral starting point. flipped is defined as "the steered
        # answer parses AND differs from baseline", so for a statement whose unsteered answer
        # does not parse, ANY parseable steered answer counts as a flip, including "yes" on a
        # true statement. Where the baseline parse rate is low the flip rate is partly a
        # measure of whether steering made the model answer at all, which is not the same
        # claim as making it lie. The verdict margin is unaffected: p_yes - p_no is defined
        # whether or not the sampled token happens to be a yes/no word.
        if parse_rate < 0.75:
            print(f"[D1][warn] only {parse_rate:.0%} of unsteered yes/no prompts produced a "
                  f"parseable yes or no. Read the flip rate as parseability-confounded and "
                  f"lead with the verdict margin. This applies equally to the committed "
                  f"mag_verdict_flips_{ds}.csv, which uses the same definition.", flush=True)

        check_hook_live(model, tok, steerer, dirs[0][1], h_med, base_text0)

        # ---- the sweep
        for name, vec in dirs:
            for sc in scales:
                if sc == 0.0:
                    continue                      # the baseline block above covers the zero dose
                if (name, round(sc, 6)) in skip:
                    continue
                steerer.set(torch.tensor(sc * h_med * vec, dtype=torch.float32))
                for p in factual:
                    c = su.generate(model, tok, p, MAX_NEW_FACTUAL)
                    fac.write({"direction": name, "scale": fmt_scale(sc),
                               "prompt": p, "completion": c})
                flips = 0
                for s in yesno:
                    a = _yes_no_answer(su.generate(model, tok, Q_TRUTH + s + Q_SUFFIX,
                                                   MAX_NEW_YESNO))
                    r = yesno_readout(model, tok, s, yes_ids, no_ids, dev)
                    # Same flip definition as src/mag/steer.py:88, kept byte-identical so D1 and
                    # the committed arm's flip rates are the same measurement.
                    f = int(a != base_ans[s] and a != "?")
                    flips += f
                    yn.write({"direction": name, "scale": fmt_scale(sc), "statement": s,
                              "answer": a, "base_answer": base_ans[s], "flipped": f,
                              "margin": f"{r['margin']:.6g}",
                              "base_margin": f"{base_marg[s]:.6g}",
                              "d_margin": f"{r['margin'] - base_marg[s]:.6g}",
                              "p_yes": f"{r['p_yes']:.6g}", "p_no": f"{r['p_no']:.6g}"})
                fac.flush(); yn.flush()
                print(f"  {name} rel={sc:+.4f} alpha={sc * h_med:+.2f} "
                      f"flips={flips}/{len(yesno)}", flush=True)
        steerer.set(None)

    fac.close(); yn.close()
    meta = {
        "dataset": ds, **geo,
        "rel_grid": list(grid), "n_doses": len(scales),
        "directions": [n for n, _ in dirs],
        "direction_cosines": direction_cosines(dirs),
        "n_factual_prompts": len(factual), "n_yesno_statements": len(yesno),
        "max_new_tokens": {"factual": MAX_NEW_FACTUAL, "yesno": MAX_NEW_YESNO},
        # su.generate hardcodes greedy decoding with repetition_penalty=1.3. Both committed
        # arms generated under exactly that setting, so D1 keeps it for comparability even
        # though the token-space work flagged rp != 1.0 as breaking the argmax cone geometry.
        # That geometry is not what D1 measures; comparability with the arms it is auditing is.
        "decoding": {"do_sample": False, "repetition_penalty": 1.3},
        "baseline_parse_rate": parse_rate,
        "baseline_yes": sum(1 for v in base_ans.values() if v == "yes"),
        "baseline_no": sum(1 for v in base_ans.values() if v == "no"),
        "baseline_mean_margin": float(np.mean(list(base_marg.values()))),
        "rand_seed": RAND_SEED, "smoke": bool(limit or prefix != "dose"),
    }
    with open(f"{prefix}_meta_{ds}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[D1] wrote {fac_path}, {yn_path}, {prefix}_meta_{ds}.json", flush=True)
    return fac_path, yn_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap both prompt lists (smoke test)")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep only matching direction names")
    ap.add_argument("--grid", default=None,
                    help="comma-separated relative magnitudes, overriding REL_GRID")
    ap.add_argument("--resume", action="store_true",
                    help="append to existing dose CSVs, skipping (direction, dose) cells already done")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard existing dose CSVs and start over")
    ap.add_argument("--out-prefix", default="dose",
                    help="output filename prefix. A smoke run MUST pass something other than "
                         "the default so it cannot land on the real artifact names.")
    a = ap.parse_args()
    if a.resume and a.overwrite:
        raise SystemExit("[D1] --resume and --overwrite are mutually exclusive")
    if a.limit and a.out_prefix == "dose":
        raise SystemExit("[D1] --limit is a smoke run; pass --out-prefix dose_smoke so it "
                         "cannot overwrite the real dose_<ds>.csv artifacts.")
    if a.overwrite:
        for p in (f"{a.out_prefix}_{a.dataset}.csv", f"{a.out_prefix}_yesno_{a.dataset}.csv"):
            if os.path.exists(p):
                os.remove(p)
                print(f"[D1] removed {p}")
    grid = [float(x) for x in a.grid.split(",")] if a.grid else REL_GRID
    run(a.dataset, a.device, grid, limit=a.limit, only=a.only,
        resume=a.resume, prefix=a.out_prefix)


if __name__ == "__main__":
    main()
