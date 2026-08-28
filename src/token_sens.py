"""token_sens.py — E6: the directional sensitivity spectrum (the PI's "chaos" claim).

Claim 4 was: *small perturbations in different directions can have vastly different
chains, so the latent space is chaotic.* If that is true, a scalar norm budget is the
wrong currency for a certificate, because the same epsilon buys wildly different
amounts of movement depending on which way you push. This measures that directly.

For each statement i, each unit direction u, and each budget eps, inject eps*u at the
site and record what the model actually did:

    gain(u, eps)   = || z(eps u) - z(0) ||  /  eps           realized amplification
    dM(u, eps)     = a_i . ( z(eps u) - z(0) )               realized readout move
    slope(u, eps)  = dM / eps                                movement PER UNIT BUDGET

Two distinct questions come out of the same table, and conflating them is the easy
mistake:

  **Heterogeneity across directions** (the actual claim 4). At fixed eps, look at the
  spread of `gain` over u. `spread = p90/p10`. A spread near 1 means the layer is
  close to isotropic and eps is a fair currency; a spread of orders of magnitude means
  it is not, and the honest budget is gain-normalized.

  **Departure from linearity** (chaos proper). Compare `gain` at the smallest eps
  against `gain` at the largest. A linear map has the SAME gain at every eps by
  definition, so `scale_dep = gain(eps_max)/gain(eps_min)` is a pure nonlinearity
  reading, per direction. This is what distinguishes "anisotropic but linear" (fine,
  just rescale the budget) from "chaotic" (no budget works).

The budget ladder is denominated in each statement's own `delta_cone` from token_geom,
not in absolute norm, so `frac = 1.0` is exactly the displacement that provably buys
the token flip at the post-norm site. Absolute norms are not comparable across
statements; fractions of the certified budget are.

**Built-in assertion.** At `--site postnorm` the map from the injection point to z is
the identity, so `gain` must be 1.0 and `slope` must equal `a_i . u` for every
direction and every eps, to numerical precision. The script checks this and says so.
If it fails there, the harness is wrong and the deeper-site numbers mean nothing. Same
discipline as the oracle arm in token_steer.

    PYTHONPATH=src python src/token_sens.py --dataset cities --site postnorm --n-stmt 20
    PYTHONPATH=src python src/token_sens.py --dataset cities --site layer:13 --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import dct_steer_utils as su
from token_steer import SiteSteerer, unit_rows

MAX_LEN = 64
SEED = 42
FRACS = (0.01, 0.1, 1.0, 4.0)


def families(ds, geom, acts, site, layer, n_random, rng):
    """Named families of unit directions, as (name, array) with array (n, d) or (d,).

    Everything except `random` is a direction somebody has actually proposed steering
    with, so the spectrum is read against real candidates rather than against noise
    alone. `random` is the reference: it is what a direction carrying no information
    about this decision looks like.

    All of these live in POST-NORM coordinates. Injected at a deeper site they are raw
    directions, not actuators for that site; that is the point of comparing them
    against `jtw`, which is the actuator, and it is why the caller prints the caveat."""
    out = []
    n, d = np.asarray(geom["delta_opt"]).shape

    R = rng.normal(size=(n_random, d))
    for k, r in enumerate(R / np.linalg.norm(R, axis=1, keepdims=True)):
        out.append((f"random{k}", r))

    out.append(("a_unit", np.asarray(geom["a_unit"], np.float64)))
    out.append(("oracle", unit_rows(np.asarray(geom["delta_opt"], np.float64))))

    lab = np.asarray(acts["labels"], int)
    key = "z_full" if site == "postnorm" else "h_full"
    if key in acts.files and (lab == 1).any() and (lab == 0).any():
        Zf = np.asarray(acts[key], np.float64)
        v = Zf[lab == 1].mean(0) - Zf[lab == 0].mean(0)
        out.append(("md_full", v / np.linalg.norm(v)))

    src = f"reach_dirs_{ds}.npz"
    if os.path.exists(src):
        Dz = np.load(src, allow_pickle=True)
        names = [str(x) for x in Dz["names"]]
        for want in ("mean_diff_tgt", "probe_grad_tgt"):
            if want in names:
                v = np.asarray(Dz["W"][names.index(want)], np.float64)
                out.append((f"{want}_asis", v / np.linalg.norm(v)))

    jp = f"token_jac_{ds}.npz"
    if layer is not None and os.path.exists(jp):
        J = np.load(jp, allow_pickle=True)
        if "jtw" in J.files:
            out.append(("jtw", unit_rows(np.asarray(J["jtw"][:, layer, :], np.float64))))
    return out


def read_z(model, tok, prompt, dev):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float64)


def summarize(d, site):
    """The two readings, kept apart on purpose."""
    print(f"\n=== token_sens site={site} ===")
    print(f"rows {len(d)}  statements {d.stmt.nunique()}  "
          f"directions {d.direction.nunique()}  budgets {sorted(d.frac.unique())}")

    print("\n-- gain = ||dz|| / eps, by direction (median over statements) --")
    per = d.groupby(["direction", "frac"]).gain.median().unstack()
    print(per.to_string(float_format=lambda x: f"{x:10.4g}"))

    print("\n-- claim 4: spread of gain ACROSS directions, at fixed budget --")
    for f in sorted(d.frac.unique()):
        g = d[d.frac == f].groupby("direction").gain.median()
        lo, hi = g.quantile(.1), g.quantile(.9)
        print(f"  frac {f:<6g} p10 {lo:10.4g}  p90 {hi:10.4g}  "
              f"spread p90/p10 {hi / max(lo, 1e-12):8.2f}")

    print("\n-- chaos: same direction, gain at the largest budget / at the smallest --")
    f_lo, f_hi = min(d.frac), max(d.frac)
    a = d[d.frac == f_lo].groupby("direction").gain.median()
    b = d[d.frac == f_hi].groupby("direction").gain.median()
    sc = (b / a.replace(0, np.nan)).dropna()
    for nm, v in sc.sort_values().items():
        print(f"  {nm:24s} {v:8.4f}")
    if len(sc):
        print(f"  median scale-dependence {sc.median():.4f}  "
              f"(1.0 => exactly linear over a {f_hi / f_lo:g}x budget range)")

    print("\n-- slope = dM / eps: readout movement bought per unit budget --")
    print(d.groupby(["direction", "frac"]).slope.median().unstack().to_string(
        float_format=lambda x: f"{x:10.4g}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--site", default="postnorm")
    p.add_argument("--positions", default="all", choices=["all", "last"])
    p.add_argument("--n-stmt", type=int, default=40)
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--fracs", default=",".join(str(f) for f in FRACS))
    p.add_argument("--tag", default="")
    a = p.parse_args()

    ds = a.dataset
    geom = np.load(f"token_geom_{ds}.npz", allow_pickle=True)
    acts = np.load(f"token_acts_{ds}.npz", allow_pickle=True)
    layer = int(a.site.split(":")[1]) if a.site.startswith("layer:") else None
    fracs = [float(x) for x in a.fracs.split(",") if x.strip()]

    # Bind every npz member once: indexing an NpzFile re-inflates the whole array.
    stems = np.asarray(acts["stems"], object)
    a_unit = np.asarray(geom["a_unit"], np.float64)
    delta_cone = np.asarray(geom["delta_cone"], np.float64)
    row_index = np.asarray(geom["row_index"])

    rng = np.random.default_rng(SEED)
    fam = families(ds, geom, acts, a.site, layer, a.n_random, rng)
    n = min(a.n_stmt, len(stems))
    print(f"[sens] {n} statements x {len(fam)} directions x {len(fracs)} budgets "
          f"= {n * len(fam) * len(fracs)} forwards")
    if a.site != "postnorm":
        print(f"[sens] post-norm directions injected at a {a.site} site are raw "
              f"directions, not actuators for that site; `jtw` is the actuator")

    tok, model, dev = su.load_model(a.device, model_name=a.model)

    rows = []
    with SiteSteerer(model, a.site, a.positions) as st:
        for i in range(n):
            st.set(None)
            z0 = read_z(model, tok, str(stems[i]), dev)
            base = float(delta_cone[i])
            ai = a_unit[i]
            for name, U in fam:
                u = U[i] if np.ndim(U) == 2 else U
                for f in fracs:
                    eps = f * base
                    if eps <= 0:
                        continue
                    st.set(torch.tensor(eps * u, dtype=torch.float32))
                    dz = read_z(model, tok, str(stems[i]), dev) - z0
                    rows.append(dict(
                        stmt=int(row_index[i]), direction=name, frac=f, eps=eps,
                        gain=float(np.linalg.norm(dz)) / eps,
                        dM=float(ai @ dz), slope=float(ai @ dz) / eps,
                        cos_first=float(ai @ u),
                        z_norm=float(np.linalg.norm(z0))))
            st.set(None)
            if i % 5 == 0:
                print(f"  stmt {i}/{n} done", flush=True)

    d = pd.DataFrame(rows)
    tag = a.tag or f"{a.site.replace(':', '')}_{a.positions}"
    out = f"token_sens_{ds}_{tag}.csv"
    d.to_csv(out, index=False)
    summarize(d, a.site)

    if a.site == "postnorm":
        # A = I here, so this is arithmetic, not an experiment: gain must be 1 and the
        # realized slope must equal a . u. If it does not, the injection is not landing
        # where the code says and nothing else here is readable.
        #
        # The tolerance CANNOT be a fixed absolute number across a 400x budget range.
        # gain = ||dz||/eps, and dz is a difference of two fp32 activations of norm
        # ||z||, so its absolute error is about ||z|| * eps_mach * sqrt(d). Divided by
        # eps that inflates without limit as eps shrinks. On common_claim, whose p10
        # delta_cone is 0.151, frac=0.01 injects eps ~ 0.0015 into a vector of norm 168
        # and the floor alone is ~1%. A flat 1e-2 threshold flags float noise as a
        # broken harness, which is a false alarm on the one arm that is pure arithmetic.
        prec = 6e-6                      # fp32 eps * sqrt(2304), rounded up
        tol = np.maximum(1e-3, prec * d.z_norm / d.eps)
        eg, es = (d.gain - 1.0).abs(), (d.slope - d.cos_first).abs()
        bad = int(((eg > tol) | (es > tol)).sum())
        ok = bad == 0
        print(f"\n[assert] postnorm identity: max |gain-1| {eg.max():.2e}, "
              f"max |slope - a.u| {es.max():.2e}, "
              f"{bad}/{len(d)} rows outside the precision-scaled tolerance "
              f"({'HARNESS OK' if ok else 'HARNESS BROKEN — stop and debug'})")
        if not ok:
            worst = d.assign(err=eg).sort_values("err", ascending=False).head(3)
            print(worst[["direction", "frac", "eps", "gain", "z_norm"]].to_string(
                index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
