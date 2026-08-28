"""token_steer.py — E1 (naive last-layer steer) and E3 (certified per-layer steer).

The PI's triage. Steer at the LAST layer, where the map to tokens is exactly linear
(logits = W z, softcap is monotone), and see what happens:

  incoherent  -> the linear-feature-in-the-final-layer assumption is wrong
  clean lie   -> the final layer is fine; the hop / linearization / transfer broke it
  nothing     -> the direction is wrong, or the budget never bought a flip

That third branch is the one the existing negative result could not distinguish,
because it reported a norm budget and never reported what fraction of the LOGIT
MARGIN the budget bought. Every row here logs `frac_margin` so the branch is decided
by the data instead of by argument.

Sites (`--site`):
  postnorm   add to the output of model.model.norm — the purest test. Nothing
             nonlinear remains between the perturbation and the logits, so a failure
             here indicts the direction and nothing else.
  prenorm    add to the input of model.model.norm — the honest "steer layer 26"
             experiment, in which RMSNorm discards the radial component.
  layer:L    add to the input of decoder layer L (the convention every previous run
             used), for the E3 certified sweep.

Directions (`--dirs`):
  oracle       the per-statement least-norm displacement from token_geom. MUST flip
               the token at scale 1.0. This is a harness assertion, not a result: if
               oracle fails, stop and fix the harness before reading anything else.
  md_full      final-layer mean-difference truth direction, fitted at the
               FULL-statement position (the label carries no signal at the stem)
  jtw_token    unit(J_L^T a_i) from token_jac — the certified actuator for layer:L
  jtw_legacy   the direction the existing reach_steer runs used, for continuity

Ablations the deep-research pass asked for, all first-class flags:
  --positions all|last     ActAdd injects at one aligned position; the existing
                           Steerer broadcasts. `all` also perturbs BOS.
  --rep-penalty            DEFAULT 1.0. The existing runs used 1.3, which makes the
                           decoder not-argmax and voids the cone geometry.

    PYTHONPATH=src python src/token_steer.py --dataset cities --device cuda \
        --site postnorm --dirs oracle,md_full --gen
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch

import dct_steer_utils as su

FRACS = [0.25, 0.5, 1.0, 2.0]
MAX_NEW = 8
MAX_LEN = 64

STEER_COLUMNS = ("direction", "stmt", "stem", "scale", "frac", "argmax_id",
                 "argmax_tok", "hit_target", "margin", "frac_margin",
                 "tgt_minus_top_delta", "completion")

# `readout_delta` is what the last column was called in every token_steer_*.csv written
# before 2026-08-04. It never held a probe readout. It is
#
#     Delta(logit[j_tgt] - logit[j_top]) = m0 - m,
#
# i.e. `frac_margin` times the per-statement constant |m0| — one measurement wearing two
# column names, not two measurements. The old name read as the legacy `mean_diff_tgt`
# readout (w . h) from reach_margins, and on that reading an analysis was designed around
# a readout-versus-behavior 2x2 cross-tab that these files cannot support. The name is
# fixed here; the twelve CSVs from the finished cluster runs are NOT regenerated, so both
# spellings are live and readers must take either.
LEGACY_COLUMN_ALIASES = {"readout_delta": "tgt_minus_top_delta"}


def load_steer_csv(path):
    """Read a token_steer_*.csv under either header generation.

    Use this rather than a bare `pd.read_csv` anywhere the delta column is touched: the
    twelve files on disk from the finished cluster runs carry the old header, are the
    inputs to in-flight analysis, and will not be rewritten."""
    return pd.read_csv(path).rename(columns=LEGACY_COLUMN_ALIASES)


class SiteSteerer:
    """Adds `vec` at one of three sites. Batch size 1 throughout: per-statement
    directions and per-statement scales make batching a false economy, and it keeps
    the position bookkeeping exact."""

    def __init__(self, model, site, positions="all"):
        self.site, self.positions, self.vec, self._h = site, positions, None, None
        if site == "postnorm":
            self.mod, self.kind = model.model.norm, "post"
        elif site == "prenorm":
            self.mod, self.kind = model.model.norm, "pre"
        elif site.startswith("layer:"):
            self.mod, self.kind = model.model.layers[int(site.split(":")[1])], "pre"
        else:
            raise ValueError(f"unknown site {site!r}")

    def _add(self, h):
        if self.vec is None:
            return h
        v = self.vec.to(dtype=h.dtype, device=h.device)
        if self.positions == "all" or h.shape[1] == 1:
            return h + v
        out = h.clone()
        out[:, -1, :] = out[:, -1, :] + v          # last position only (ActAdd style)
        return out

    def _pre(self, module, args, kwargs):
        if self.vec is None:
            return None
        if args:
            return (self._add(args[0]),) + args[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = self._add(kwargs["hidden_states"])
        return args, kwargs

    def _post(self, module, args, output):
        return self._add(output) if self.vec is not None else None

    def __enter__(self):
        self._h = (self.mod.register_forward_pre_hook(self._pre, with_kwargs=True)
                   if self.kind == "pre"
                   else self.mod.register_forward_hook(self._post))
        return self

    def __exit__(self, *e):
        if self._h:
            self._h.remove()

    def set(self, v):
        self.vec = v


def probe(model, tok, prompt, W_rows, j_top, j_tgt, dev):
    """One forward; returns (argmax token id, margin, -margin).

    SIGN CONVENTION. `W_rows` is `Weff[[j_top, j_tgt]]`, so `lg[0]` is the incumbent's
    logit and `lg[1]` the target's, and the two floats are ONE number with opposite
    signs:

        [1] margin           = logit[j_top] - logit[j_tgt]   (> 0 while the top wins)
        [2] tgt_minus_top    = logit[j_tgt] - logit[j_top]   = -margin

    Return [2] exists only so the caller can log a delta (`tgt_minus_top_delta` =
    r - r0 = m0 - m) that is POSITIVE when steering moves probability toward the target.
    It is NOT `w . h` for any probe direction — in particular not the legacy
    `mean_diff_tgt` readout from reach_margins — and the delta it feeds is `frac_margin`
    times |m0|, carrying no information `frac_margin` lacks. See `STEER_COLUMNS`.

    Margins are recomputed from z against the *uncapped* plain unembedding rows, so
    they are the linear quantity the geometry is stated in. The argmax is taken from
    the model's own (softcapped) head; softcap is monotone, so the two agree."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    z = out.hidden_states[-1][0, -1].float()
    lg = (W_rows @ z).cpu().numpy()
    return int(out.logits[0, -1].argmax()), float(lg[0] - lg[1]), float(lg[1] - lg[0])


def load_dirs(ds, which, site, geom, jac, layer, positions="all"):
    """Per-statement unit directions, expressed in the coordinates of `site`."""
    D = {}
    n = len(geom["delta_cone"])
    if "oracle" in which:
        if site != "postnorm":
            print("[dirs] oracle is defined in post-norm coordinates; on a "
                  f"{site} site it is used as a raw direction and its scale "
                  "calibration no longer certifies a flip")
        o = np.asarray(geom["delta_opt"], np.float64)
        D["oracle"] = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
    if "md_full" in which:
        # Fitted at the FULL-statement position, where the label is informative. A fit
        # at the stem would be noise: the same stem serves both labels. See
        # token_geom.load_directions.
        A = np.load(f"token_acts_{ds}.npz", allow_pickle=True)
        y = np.asarray(A["labels"], int)
        key = "z_full" if site == "postnorm" else "h_full"
        if key not in A.files or not ((y == 1).any() and (y == 0).any()):
            print(f"[dirs] md_full needs {key} and both labels; skipped")
        else:
            Z = np.asarray(A[key], np.float64)
            v = Z[y == 1].mean(0) - Z[y == 0].mean(0)
            D["md_full"] = np.repeat((v / np.linalg.norm(v))[None, :], n, 0)
    if "jtw_token" in which:
        # The actuator must match the INJECTION CONVENTION, not just the layer.
        # token_jac computes two pullbacks: `jtw` from the gradient summed over all
        # positions (what a broadcast injection actuates) and `jtw_last` from the
        # gradient at the last position alone. Using `jtw` while injecting at one
        # position certifies a budget the run never spends, and the two differ by the
        # broadcast gain, which is up to 3.3x at the early layers.
        key = "jtw" if positions == "all" else "jtw_last"
        if jac is None or key not in jac.files:
            raise SystemExit(f"jtw_token at --positions {positions} needs '{key}' — "
                             "run token_jac.py --save-dirs")
        if layer is None:
            raise SystemExit("jtw_token is a per-layer actuator; use --site layer:L")
        v = np.asarray(jac[key][:, layer, :], np.float64)
        D["jtw_token"] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    if "jtw_legacy" in which:
        # reach_margins rows follow reach_margins.load_statements, which applies the
        # stratified DATASET_SAMPLE cap. `cities` has no cap, so position k IS csv row
        # k and a direct join works. `common_claim` keeps 2000 of 4450, so position k is
        # csv row idx[k] and v[row_index] is WRONG even where it lands in bounds. Use
        # the mapping in reach_acts_<ds>.npz rather than assuming identity.
        mz = np.load(f"reach_margins_{ds}.npz", allow_pickle=True)
        sn = [str(x) for x in mz["store_names"]]
        v = np.asarray(mz["jtw"], np.float64)[:, sn.index("mean_diff_tgt"), :]
        ri = np.asarray(geom["row_index"])
        pos = legacy_positions(ds, ri, len(v))
        hit = pos >= 0
        if not hit.any():
            print(f"[dirs] jtw_legacy: none of the {len(ri)} statements appear in "
                  f"reach_margins_{ds}.npz; arm omitted")
        else:
            # Statements outside the reach subsample get NaN and are skipped per
            # statement in the sweep. Dropping them from the frame instead would
            # silently misalign this direction against every other one.
            L = np.full((len(ri), v.shape[1]), np.nan)
            L[hit] = unit_rows(v[pos[hit]])
            if not hit.all():
                print(f"[dirs] jtw_legacy covers {int(hit.sum())}/{len(ri)} statements; "
                      f"the rest fall outside the reach_margins subsample and are "
                      f"skipped for this arm only")
            D["jtw_legacy"] = L
    return D


def legacy_positions(ds, row_index, n_rows):
    """csv row positions -> reach_margins row positions, -1 where the row is absent.

    reach_acts_<ds>.npz carries the `row_index` the reach pipeline actually kept. When
    it is missing we fall back to the identity, which is right exactly when no
    DATASET_SAMPLE cap applied to that dataset."""
    src = f"reach_acts_{ds}.npz"
    ridx = (np.asarray(np.load(src, allow_pickle=True)["row_index"])
            if os.path.exists(src) else np.arange(n_rows))
    lut = {int(r): k for k, r in enumerate(ridx) if k < n_rows}
    return np.array([lut.get(int(r), -1) for r in row_index])


def unit_rows(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def required_scale(name, i, delta_cone, eps, layer):
    """The budget that, on the certificate's own terms, should buy exactly one flip.

    `eps` must be the array matching the injection convention in use: eps_all for a
    broadcast, eps_last for a single position. They differ by the broadcast gain, so
    crossing them sweeps a budget the run does not actually spend.

    Takes plain arrays, never the NpzFile: indexing an NpzFile member re-inflates the
    whole array on every access, and this is called once per statement per direction."""
    if name == "jtw_token" and eps is not None and layer is not None:
        return float(eps[i, layer])
    return float(delta_cone[i])        # oracle exactly; others rescaled by alpha below


def oracle_line(site, hr):
    """The oracle verdict, which is a HARNESS GATE at exactly one site and a
    MEASUREMENT everywhere else.

    At post-norm the map to the logits is the identity, so `delta_cone` provably lands
    inside the target argmax cone and a hit rate below ~0.9 means the injection is not
    landing where the code says. Nothing else in that file is then readable.

    At any other site the same vector is a post-norm quantity being injected upstream
    of RMSNorm, which annihilates its radial component and rescales the survivor by
    sqrt(d)/||h||. It no longer certifies anything, so a zero is the number the arm was
    run to obtain (how much of a certified displacement survives the intervening
    layers), not a fault. Labelling that BROKEN sends the reader off to debug arithmetic
    that is working correctly."""
    if site == "postnorm":
        ok = "HARNESS OK" if hr > 0.9 else "HARNESS BROKEN — stop and debug"
        return f"[assert] oracle @ +1.00 hit rate {hr:.3f} ({ok})"
    return (f"[measure] oracle @ +1.00 hit rate {hr:.3f} at site={site}; the oracle is "
            f"calibrated post-norm, so this is the survival rate of a certified "
            f"displacement through the intervening layers, NOT a harness check "
            f"(the harness gate is the postnorm arm)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--site", default="postnorm")
    p.add_argument("--positions", default="all", choices=["all", "last"])
    p.add_argument("--dirs", default="oracle,md_full")
    p.add_argument("--rep-penalty", type=float, default=1.0)
    p.add_argument("--gen", action="store_true", help="also emit 8-token completions")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    ds = a.dataset
    geom = np.load(f"token_geom_{ds}.npz", allow_pickle=True)
    acts = np.load(f"token_acts_{ds}.npz", allow_pickle=True)
    gcsv = pd.read_csv(f"token_geom_{ds}.csv")
    jac = (np.load(f"token_jac_{ds}.npz", allow_pickle=True)
           if os.path.exists(f"token_jac_{ds}.npz") else None)
    layer = int(a.site.split(":")[1]) if a.site.startswith("layer:") else None

    which = [x.strip() for x in a.dirs.split(",") if x.strip()]
    D = load_dirs(ds, which, a.site, geom, jac, layer, a.positions)
    # Bind every npz member ONCE. Indexing an NpzFile re-decompresses the whole array.
    stems = np.asarray(acts["stems"], object)
    j_top = np.asarray(geom["j_top"])
    j_tgt = np.asarray(geom["j_tgt"])
    delta_cone = np.asarray(geom["delta_cone"], np.float64)
    row_index = np.asarray(geom["row_index"])
    eps_key = "eps_all" if a.positions == "all" else "eps_last"
    eps = np.asarray(jac[eps_key], np.float64) if jac is not None else None
    alpha_of = {nm: gcsv[f"alpha_{nm}"].to_numpy() for nm in D
                if f"alpha_{nm}" in gcsv.columns}
    n = len(stems) if not a.limit else min(a.limit, len(stems))

    tok, model, dev = su.load_model(a.device, model_name=a.model)
    # PLAIN E, not E*(1+gamma): hidden_states[-1] is already post-RMSNorm and carries
    # the gain, so logits = E z. Folding gamma in here would double-count it and every
    # margin below would be wrong. Same convention as token_geom.load_unembed.
    Weff = model.get_output_embeddings().weight.detach().float()

    tag = a.tag or f"{a.site.replace(':', '')}_{a.positions}_rp{a.rep_penalty:g}"
    rows = [STEER_COLUMNS]

    with SiteSteerer(model, a.site, a.positions) as st:
        for i in range(n):
            Wr = Weff[[int(j_top[i]), int(j_tgt[i])]]
            st.set(None)
            am0, m0, r0 = probe(model, tok, str(stems[i]), Wr, j_top[i], j_tgt[i], dev)
            for name, U in D.items():
                if not np.isfinite(U[i]).all():
                    continue          # this statement has no actuator for this arm
                base = required_scale(name, i, delta_cone, eps, layer)
                if name not in ("oracle", "jtw_token") and name in alpha_of:
                    # eps(u) = eps* / alpha: the budget this actuator needs to buy the
                    # flip the oracle buys optimally. Without this rescale every
                    # non-oracle arm is swept at a budget two orders of magnitude short.
                    base = base / max(float(alpha_of[name][i]), 1e-12)
                for f in [0.0] + [s * f for f in FRACS for s in (+1.0, -1.0)]:
                    sc = f * base
                    st.set(None if sc == 0.0 else
                           torch.tensor(sc * U[i], dtype=torch.float32))
                    am, m, r = probe(model, tok, str(stems[i]), Wr,
                                     j_top[i], j_tgt[i], dev)
                    comp = ""
                    if a.gen:
                        inp = tok(str(stems[i]), return_tensors="pt").to(dev)
                        with torch.no_grad():
                            o = model.generate(
                                **inp, max_new_tokens=MAX_NEW, do_sample=False,
                                repetition_penalty=a.rep_penalty,
                                pad_token_id=tok.pad_token_id)
                        comp = tok.decode(o[0][inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True
                                          ).replace("\n", " ").strip()
                    rows.append((name, int(row_index[i]), str(stems[i]),
                                 f"{sc:.6g}", f"{f:+.2f}", am,
                                 tok.decode([am]).replace("\n", "\\n"),
                                 int(am == int(j_tgt[i])), f"{m:.6g}",
                                 f"{(m0 - m) / abs(m0) if m0 else 0.0:.6g}",
                                 # == frac_margin * |m0|; kept only as the absolute-units
                                 # view of the very same quantity. Report frac_margin.
                                 f"{r - r0:.6g}", comp))
            if i % 10 == 0:
                print(f"  stmt {i}/{n} done", flush=True)
            st.set(None)

    out = f"token_steer_{ds}_{tag}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    d = pd.DataFrame(rows[1:], columns=rows[0])
    d["frac_margin"] = d.frac_margin.astype(float)
    d["hit_target"] = d.hit_target.astype(int)
    print(f"\n=== token_steer {ds} site={a.site} positions={a.positions} "
          f"rep_penalty={a.rep_penalty} ===")
    print(d.groupby(["direction", "frac"]).agg(
        hit_rate=("hit_target", "mean"),
        frac_margin=("frac_margin", "median")).to_string())
    orc = d[(d.direction == "oracle") & (d.frac == "+1.00")]
    if len(orc):
        print("\n" + oracle_line(a.site, orc.hit_target.mean()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
