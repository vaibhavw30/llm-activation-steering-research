"""token_geom.py — E0: the token-space geometry of the last layer.

Every downstream number in the token-space program is denominated in the quantities
this script produces. Nothing here needs a Jacobian and nothing here needs steering:
it is the *denominator* the existing negative result was never divided by.

At temperature 0 the decoder is the argmax over logits, and (ignoring the strictly
increasing softcap, which preserves order) the logits are an exactly LINEAR function
of the post-final-norm activation z:

    logits = E z ,      z = (h / rms(h)) * (1 + gamma)

E is the plain tied embedding matrix. The gain lives in z, NOT in the unembedding —
see load_unembed for why the other convention gives the same argmax but different
distances, and why that matters when every budget here is a distance.

So the set of activations that emit token j is the polyhedral cone

    C_j = { z : (E[j] - E[k]) . z >= 0  for all k }.

Three quantities, per statement i:

  M_i      logit margin from the current argmax to the target token  (pre-softcap)
  delta_i  the LEAST-NORM displacement that lands inside C_{j_target}, measured in
           the units of the injection SITE. Computed by active set: solve the tiny
           dual QP on the currently violated faces, re-check the full 256k argmax,
           add the violator, repeat. At the post-norm site the check is exact and a
           `solved` row is a proof, not an optimizer's opinion.
  alpha_i  |a_i . u| / ||a_i||, the alignment between a candidate steering direction u
           and the token-difference direction a_i = E[j_target] - E[j_argmax].

and the number the whole program turns on,

    eps_required(u, i) = delta_i / alpha_i(u)

the budget you must spend ALONG u to buy the flip that delta_i buys optimally. If a
steering run spent eps and alpha was 0.03, it bought 3% of a flip, and the resulting
behavioral null says nothing whatsoever about truth.

Pre-norm injection is handled as a second site rather than a correction factor. The
RMSNorm Jacobian is c * diag(1+gamma) * P_perp with c = sqrt(d)/||h||, so the cone
constraints are simply re-expressed through its transpose (see rmsnorm_site and
cone_budget). Two columns report the consequences: `rmsnorm_penalty` is the ratio of
the pre-norm to the post-norm budget, and `radial_<dir>` is the fraction of a
candidate direction that P_perp discards outright.

    PYTHONPATH=src python src/token_geom.py --dataset cities --device cuda
    PYTHONPATH=src python src/token_geom.py --dataset cities --stage geom   # no model
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SOFTCAP = 30.0          # gemma-2 final_logit_softcapping; monotone, argmax-irrelevant
MAX_FACES = 48          # active-set cap; typical solves use 1-3
DUAL_ITERS = 20000
INTERIOR = 1e-3         # solve for a STRICTLY interior point, not the boundary
N_DEFAULT = 200
SEED = 42
BATCH = 16
MAX_LEN = 64


# --------------------------------------------------------------------------- W

def resolve_snapshot(model_name="google/gemma-2-2b", snapshot=None):
    if snapshot:
        return snapshot
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_name, local_files_only=True)
    except Exception:                                        # noqa: BLE001
        pat = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
            "/snapshots/*")
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"cannot locate a local snapshot for {model_name}; "
                             f"pass --snapshot")
        return hits[-1]


def load_unembed(snap):
    """(E, gamma), kept SEPARATE on purpose.

    HF's Gemma2RMSNorm returns normalized * (1 + weight), so `hidden_states[-1]` —
    the z this script works in — already carries the gain. The unembedding that pairs
    with z is therefore the plain tied matrix E:

        logits = E z ,      z = (h / rms(h)) * (1 + gamma)

    Folding gamma into the unembedding (W = E * (1 + gamma)) is the OTHER valid
    convention, and it pairs with the pre-gain h / rms(h). The two give identical
    logits and identical argmax, but different DISTANCES, and every budget in this
    program is a distance. Read straight from safetensors so the geometry stage needs
    neither torch nor a GPU."""
    from safetensors import safe_open
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]

    def get(key):
        with safe_open(f"{snap}/{idx[key]}", framework="np") as f:
            return f.get_tensor(key)

    E = np.asarray(get("model.embed_tokens.weight"), np.float32)
    gamma = np.asarray(get("model.norm.weight"), np.float32)
    print(f"[W] E {E.shape}  gamma range [{gamma.min():.3f}, {gamma.max():.3f}]")
    return E, gamma


# ------------------------------------------------------------------- cone solve

def _min_norm_ineq(A, b, iters=DUAL_ITERS):
    """argmin ||D||^2 s.t. A D >= b, by projected gradient on the dual.

    Lagrangian stationarity gives D = A^T lam with lam >= 0, and the dual is
    min_{lam>=0} 0.5 lam^T (A A^T) lam - b^T lam. A is tiny (<= MAX_FACES rows),
    so this is microseconds."""
    G = A @ A.T
    L = float(np.linalg.eigvalsh(G).max())
    if not np.isfinite(L) or L <= 0:
        return np.zeros(A.shape[1], np.float64)
    lam = np.zeros(len(b), np.float64)
    for _ in range(iters):
        lam = np.maximum(lam - (G @ lam - b) / L, 0.0)
    return A.T @ lam


def rmsnorm_site(h, gamma):
    """The pre-norm injection site as an explicit linear pair (push, pull).

    z = (h / rms(h)) * (1 + gamma) with rms(h) = ||h|| / sqrt(d), so

        dz = A dh ,   A   = (sqrt(d)/||h||) diag(1 + gamma) P_perp
                      A^T = (sqrt(d)/||h||) P_perp diag(1 + gamma)

    with P_perp = I - h_hat h_hat^T. Two facts fall straight out of A and are the
    whole reason this site is treated separately: the radial component of any
    perturbation is annihilated (P_perp kills it), and what survives is rescaled by
    sqrt(d)/||h||, which for these activations is a contraction of roughly 5x."""
    d = len(h)
    hn = float(np.linalg.norm(h))
    hh = h / max(hn, 1e-12)
    c = np.sqrt(d) / max(hn, 1e-12)
    g1 = 1.0 + gamma

    def push(v):                                    # site -> z
        return c * g1 * (v - hh * (hh @ v))

    def pull(v):                                    # z -> site (A^T)
        w = g1 * v
        return c * (w - hh * (hh @ w))

    return push, pull


def cone_budget(E, z, j, site=None, max_faces=MAX_FACES):
    """Least-norm perturbation, AT THE INJECTION SITE, making token j the argmax.

    The cone lives in z-space: (E[j] - E[k]) . z >= 0 for all k. A site S reaches z
    through a linear map A_S, so a perturbation D at the site moves z by A_S D and the
    constraint reads

        (A_S^T (E[j] - E[k])) . D  >=  (E[k] - E[j]) . z .

    Only the NORMALS change with the site; the offsets are the logit gaps either way.
    Pass `site` as the (push, pull) pair from rmsnorm_site, or None for the post-norm
    site where A_S = I.

    Active set: solve on the faces seen so far, ask the full vocabulary who wins, add
    that violator, repeat. `solved` means an explicit full-vocabulary argmax check
    confirmed j wins. For the post-norm site that check is exact and the row is a
    proof; for the pre-norm site the map is only a first-order model of RMSNorm, so
    `solved` there means "first-order certified" and token_steer verifies it against
    the real model."""
    push, pull = (None, None) if site is None else site
    D = np.zeros(E.shape[1], np.float64)
    act = []
    for _ in range(max_faces):
        dz = D if push is None else push(D)
        k = int(np.argmax(E @ (z + dz)))
        if k == j:
            return D, True, len(act)
        if k in act:                       # QP satisfied it yet argmax disagrees:
            return D, False, len(act)      # numerical tie, report honestly
        act.append(k)
        A = E[j][None, :] - E[act]                              # (n, d)
        if pull is not None:
            A = np.stack([pull(np.asarray(r, np.float64)) for r in A])
        b = (E[act] - E[j][None, :]) @ z                        # (n,)
        # The exact least-norm point sits ON the cone boundary, where the target ties
        # with the binding competitor and np.argmax breaks the tie by index. Aim a
        # hair inside instead: costs INTERIOR in relative budget, buys a clean verdict.
        b = np.asarray(b, np.float64) + INTERIOR * np.abs(b).max()
        D = _min_norm_ineq(np.asarray(A, np.float64), b)
    return D, False, len(act)


# ---------------------------------------------------------------- extract stage

def stem_of(statement, min_words=4):
    """Statement minus its final word. Identical rule to reach_steer.stem_of, kept
    local so the geometry stage imports no torch."""
    words = str(statement).rstrip(" .").split()
    return " ".join(words[:-1]) if len(words) >= min_words else None


def stem_for(statement, city=None, template=None):
    """The decision-position text for one row. With a template (cities only) the stem is
    template.format(city=city), which fixes two faults of stem_of on cities: after "is in"
    gemma's next token is " the" (174/200 rows in the Sep 4 run), not a country; and on a
    multi-word country stem_of cuts mid-name ("... is in South" for South Africa)."""
    return template.format(city=city) if template else stem_of(statement)


def clean_rows(cities, correct, pool):
    """Positions to keep under --target countries_clean: the first row of each distinct
    city whose correct country is in the pool. The templated stem depends only on the city,
    so a city's true and false rows would otherwise be the same geometry twice."""
    seen, keep = set(), []
    for i, (c, k) in enumerate(zip(cities, correct)):
        if c in seen:
            continue
        seen.add(c)
        if k in pool:
            keep.append(i)
    return keep


def _read_positions(model, tok, texts, dev, pre_store):
    """Post-norm z and pre-norm h at the last real token of each text."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_LEN).to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    am = enc["attention_mask"]
    # gemma's tokenizer LEFT-pads by default, so `mask.sum(1) - 1` silently reads a
    # pad row. This is reach_hop.last_nonpad_index: the last index whose mask is 1,
    # correct for either padding side.
    last = am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)
    ar = torch.arange(len(last), device=am.device)
    return (out.hidden_states[-1][ar, last].float().cpu().numpy(),
            pre_store["x"][ar, last].float().cpu().numpy())


def extract(ds, device, n, model_name, max_new=0, template=None, acts_suffix=""):
    """Post-final-norm z and pre-norm h at TWO positions per statement.

    stem  the last token of the statement minus its final word, i.e. the position
          whose logits choose the next word. This is where the token-space decision
          happens, and it is NOT where reach_acts_<ds>.npz sampled.
    full  the last token of the whole statement (its period), which IS where
          reach_acts sampled and where the existing directions were fitted.

    Both are needed, and for a reason worth stating. At the STEM position the label
    carries no information: for cities the same stem "The city of X is in" serves both
    the true and the false variant of the row, so a mean-difference direction fitted
    there is pure noise. Truth is only determined by what comes after. So a final-layer
    truth direction has to be fitted at `full`, and the interesting quantity is its
    alignment with the decision direction at `stem`. That gap IS the context-transfer
    problem (D1), now measured directly instead of inferred.

    row_index is the index into the ORIGINAL csv, unfiltered, so every downstream join
    against got_datasets/<ds>.csv and reach_* artifacts is valid. Rows are NOT filtered
    to label 1: the stem asks for the country either way, and keeping both labels is
    what makes the `full`-position direction fittable."""
    import dct_steer_utils as su

    df = pd.read_csv(f"got_datasets/{ds}.csv")
    rng = np.random.default_rng(SEED)
    pick = rng.permutation(len(df))[: n if n else len(df)]
    if template and "city" not in df.columns:
        raise SystemExit(f"--template needs a city column; {ds} has {list(df.columns)}")
    rows = [(int(i), str(df["statement"][i]),
             stem_for(df["statement"][i], df["city"][i] if template else None, template))
            for i in pick]
    rows = [(i, s, st) for i, s, st in rows if st]
    lab = df["label"].values[[r[0] for r in rows]]
    print(f"[extract] {len(rows)} stems from {ds} "
          f"({int((lab == 1).sum())} label-1, {int((lab == 0).sum())} label-0)")

    tok, model, dev = su.load_model(device, model_name=model_name)
    pre = {}
    h_pre = model.model.norm.register_forward_pre_hook(
        lambda m, a: pre.__setitem__("x", a[0].detach()))

    Zs, Hs, Zf, Hf = [], [], [], []
    try:
        for b0 in range(0, len(rows), BATCH):
            chunk = rows[b0:b0 + BATCH]
            zs, hs = _read_positions(model, tok, [c[2] for c in chunk], dev, pre)
            zf, hf = _read_positions(model, tok, [c[1] for c in chunk], dev, pre)
            Zs.append(zs); Hs.append(hs); Zf.append(zf); Hf.append(hf)
            print(f"  batch {b0}-{b0 + len(chunk)} done", flush=True)
    finally:
        h_pre.remove()

    cat = np.concatenate
    np.savez_compressed(
        f"token_acts_{ds}{acts_suffix}.npz",
        z=cat(Zs), h=cat(Hs), z_full=cat(Zf), h_full=cat(Hf),
        labels=np.asarray(lab, int),
        row_index=np.asarray([r[0] for r in rows]),
        statements=np.asarray([r[1] for r in rows], object),
        stems=np.asarray([r[2] for r in rows], object),
        model=model_name, template=template or "")
    print(f"[extract] wrote token_acts_{ds}{acts_suffix}.npz  z_stem{cat(Zs).shape}")


# ------------------------------------------------------------------- geom stage

def first_token(tok, word):
    return int(tok(" " + str(word), add_special_tokens=False)["input_ids"][0])


def geom(ds, snapshot, target_mode, dirs_file, tag="", acts_suffix=""):
    suffix = f"_{tag}" if tag else ""
    from transformers import AutoTokenizer
    snap = resolve_snapshot(snapshot=snapshot)
    W, gamma = load_unembed(snap)          # W is the PLAIN tied E; see load_unembed
    tokz = AutoTokenizer.from_pretrained(snap)
    acts = f"token_acts_{ds}{acts_suffix}.npz"
    A = np.load(acts, allow_pickle=True)
    Z = np.asarray(A["z"], np.float64)
    H = np.asarray(A["h"], np.float64)
    stmts, stems, ridx = A["statements"], A["stems"], A["row_index"]
    d = Z.shape[1]

    df = pd.read_csv(f"got_datasets/{ds}.csv")
    # candidate false first-tokens (cities only has a semantic false set)
    false_pool = None
    if target_mode == "countries":
        if "correct_country" not in df.columns:
            raise SystemExit(f"--target countries needs a correct_country column; "
                             f"{ds} has {list(df.columns)}")
        countries = sorted(df["correct_country"].astype(str).unique())
        false_pool = {c: first_token(tokz, c) for c in countries}
        print(f"[geom] country target pool: {len(set(false_pool.values()))} tokens")
    keep = list(range(len(Z)))
    if target_mode == "countries_clean":
        # The xfer_cities pool, so this run and J-D1 aim at the same targets: bare names
        # (no leading "the"), no shared first token, no generic first word (North Korea).
        import xfer_cities as xcit
        first = {xcit.bare(c): first_token(tokz, xcit.bare(c))
                 for c in df["correct_country"].astype(str).unique()}
        false_pool = xcit.clean_pool(first)
        true_of = [xcit.bare(df["correct_country"].iloc[int(r)]) for r in ridx]
        keep = clean_rows([str(df["city"].iloc[int(r)]) for r in ridx], true_of,
                          false_pool)
        print(f"[geom] clean country pool: {len(false_pool)} countries; "
              f"{len(keep)} distinct cities of {len(Z)} rows kept")

    U = load_directions(ds, dirs_file, acts)

    rows, deltas, dirs_out = [], [], []
    for i in keep:
        z = Z[i]
        logit = W @ z
        order = np.argsort(-logit)
        j_top = int(order[0])
        if target_mode == "countries":
            # ridx is a POSITION in the unfiltered csv (see extract), so .iloc is the
            # correct accessor and stays correct if the frame ever gains a real index.
            true_c = str(df["correct_country"].iloc[int(ridx[i])])
            bad = [t for c, t in false_pool.items() if c != true_c]
            bad = sorted(set(bad) - {j_top})
            j_tgt = int(bad[int(np.argmax(logit[bad]))])      # cheapest false country
        elif target_mode == "countries_clean":
            true_c = true_of[i]
            bad = sorted({t for c, t in false_pool.items() if c != true_c} - {j_top})
            j_tgt = int(bad[int(np.argmax(logit[bad]))])
        else:
            j_tgt = int(order[1])                             # cheapest token of any kind

        a = W[j_tgt] - W[j_top]
        M = float(logit[j_top] - logit[j_tgt])                # > 0 by construction
        na = float(np.linalg.norm(a))                         # m_opt at the post-norm site
        delta_face = M / na                                   # single-face lower bound
        D, solved, nf = cone_budget(W, z, j_tgt)
        delta = float(np.linalg.norm(D))

        # pre-norm site: same cone, normals pulled back through the RMSNorm Jacobian
        push, pull = rmsnorm_site(H[i], gamma)
        a_pre = pull(a)
        Dp, solved_p, nfp = cone_budget(W, z, j_tgt, site=(push, pull))
        hn = float(np.linalg.norm(H[i]))
        hh = H[i] / hn

        r = dict(idx=int(ridx[i]), stem=str(stems[i]), statement=str(stmts[i]),
                 j_top=j_top, tok_top=tokz.decode([j_top]),
                 j_tgt=j_tgt, tok_tgt=tokz.decode([j_tgt]),
                 margin=M, margin_capped=float(
                     SOFTCAP * (np.tanh(logit[j_top] / SOFTCAP)
                                - np.tanh(logit[j_tgt] / SOFTCAP))),
                 m_post=na, m_pre=float(np.linalg.norm(a_pre)),
                 delta_face=delta_face, delta_cone=delta,
                 cone_solved=bool(solved), cone_faces=int(nf),
                 delta_cone_pre=float(np.linalg.norm(Dp)),
                 cone_solved_pre=bool(solved_p), cone_faces_pre=int(nfp),
                 rmsnorm_penalty=float(np.linalg.norm(Dp)) / max(delta, 1e-12),
                 z_norm=float(np.linalg.norm(z)), h_norm=hn,
                 delta_rel_z=delta / float(np.linalg.norm(z)))
        if target_mode == "countries_clean":
            r["true_country"] = true_c
            r["top_is_correct"] = bool(j_top == false_pool[true_c])
        for name, u in U.items():
            ui = u[i] if u.ndim == 2 else u
            r[f"alpha_{name}"] = float(abs(a @ ui) / na)
            r[f"radial_{name}"] = float(abs(hh @ ui))
            al = r[f"alpha_{name}"]
            r[f"eps_req_{name}"] = delta / al if al > 1e-12 else np.inf
        rows.append(r)
        deltas.append(D)
        dirs_out.append(a / na)
        if len(rows) % 25 == 1:
            print(f"  [geom] {i}/{len(Z)} margin={M:.3f} delta={delta:.4f} "
                  f"faces={nf} solved={solved} pre/post={r['rmsnorm_penalty']:.2f}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"token_geom_{ds}{suffix}.csv", index=False)
    np.savez_compressed(f"token_geom_{ds}{suffix}.npz",
                        delta_opt=np.asarray(deltas, np.float32),
                        a_unit=np.asarray(dirs_out, np.float32),
                        j_top=out.j_top.values, j_tgt=out.j_tgt.values,
                        margin=out.margin.values, delta_cone=out.delta_cone.values,
                        row_index=out.idx.values, target_mode=target_mode)
    summarize(out, ds, target_mode, suffix)


def load_directions(ds, dirs_file, acts):
    """Candidate steering directions, all unit, all in post-norm coordinates.

    md_full   final-layer mean-difference truth direction, fitted at the FULL-statement
              position where the label is actually informative. Deliberately NOT fitted
              at the stem: the same stem serves both labels, so a stem fit is noise.
    *_asis    the directions the existing runs used, taken verbatim from
              reach_dirs_<ds>.npz. Those were fitted at the TARGET LAYER, not the final
              layer, so their alpha here is a cross-layer carryover and is labelled as
              such. It answers "how aligned is the direction we actually steered with
              to the thing that moves the next token", which is the question we need,
              but it is not a claim that the two live in the same basis."""
    U = {}
    A = np.load(acts, allow_pickle=True)
    lab = np.asarray(A["labels"], int)
    if "z_full" in A.files and (lab == 1).any() and (lab == 0).any():
        Zf = np.asarray(A["z_full"], np.float64)
        md = Zf[lab == 1].mean(0) - Zf[lab == 0].mean(0)
        U["md_full"] = md / np.linalg.norm(md)
    else:
        print("[geom] md_full not fittable (need both labels and z_full); skipped")
    src = dirs_file or f"reach_dirs_{ds}.npz"
    if os.path.exists(src):
        Dz = np.load(src, allow_pickle=True)
        names = [str(x) for x in Dz["names"]]
        for want in ("mean_diff_tgt", "probe_grad_tgt"):
            if want in names:
                v = np.asarray(Dz["W"][names.index(want)], np.float64)
                U[f"{want}_asis"] = v / np.linalg.norm(v)
    print(f"[geom] candidate directions: {list(U)}")
    return U


def summarize(out, ds, target_mode, suffix=""):
    ok = out.cone_solved
    print(f"\n=== token_geom {ds}{suffix} (target={target_mode}) ===")
    print(f"statements                     {len(out)}")
    print(f"post-norm cone proved          {int(ok.sum())}/{len(out)}")
    print(f"pre-norm cone first-order ok   {int(out.cone_solved_pre.sum())}/{len(out)}")
    print(f"top tokens     {out.tok_top.value_counts().head(5).to_dict()}")
    print(f"target tokens  {out.tok_tgt.value_counts().head(5).to_dict()}")
    if "top_is_correct" in out:
        print(f"argmax is the correct country  {int(out.top_is_correct.sum())}/{len(out)}")
    for c in ("margin", "delta_cone", "delta_rel_z", "delta_cone_pre",
              "rmsnorm_penalty", "z_norm", "h_norm"):
        q = out.loc[ok, c]
        print(f"  {c:18s} median {q.median():10.4f}  p10 {q.quantile(.1):10.4f}"
              f"  p90 {q.quantile(.9):10.4f}")
    print(f"  delta_cone / delta_face  median "
          f"{(out.loc[ok, 'delta_cone'] / out.loc[ok, 'delta_face']).median():.4f}"
          f"   (1.0 => the single binding face IS the whole constraint)")
    base = out.loc[ok, "delta_cone"].median()
    for c in [c for c in out.columns if c.startswith("alpha_")]:
        nm = c[len("alpha_"):]
        e, rr = f"eps_req_{nm}", f"radial_{nm}"
        print(f"  {nm:22s} alpha {out.loc[ok, c].median():.4f}"
              f"  radial {out.loc[ok, rr].median():.4f}"
              f"  eps_req {out.loc[ok, e].median():10.3f}"
              f"  = {out.loc[ok, e].median() / base:8.1f}x optimal")
    print(f"\nwrote token_geom_{ds}{suffix}.csv and token_geom_{ds}{suffix}.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage", default="all", choices=["extract", "geom", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--target", default="runnerup",
                   choices=["runnerup", "countries", "countries_clean"])
    p.add_argument("--template", default=None,
                   help='cities: stem = template.format(city=...), e.g. '
                        '"The city of {city} is in the country of"')
    p.add_argument("--acts-tag", default="",
                   help="suffix for token_acts_<ds>.npz, so a templated extract does "
                        "not clobber the one xfer_cities.targets reads")
    p.add_argument("--dirs", default=None)
    p.add_argument("--tag", default="",
                   help="suffix for the geom outputs, so a second target "
                        "mode does not clobber the first")
    a = p.parse_args()
    acts_suffix = f"_{a.acts_tag}" if a.acts_tag else ""
    if a.stage in ("extract", "all"):
        extract(a.dataset, a.device, a.n, a.model, template=a.template,
                acts_suffix=acts_suffix)
    if a.stage in ("geom", "all"):
        geom(a.dataset, a.snapshot, a.target, a.dirs, a.tag, acts_suffix)


if __name__ == "__main__":
    main()
