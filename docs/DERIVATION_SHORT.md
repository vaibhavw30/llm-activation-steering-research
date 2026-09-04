# The short derivation: input layer to output layer, in one thread

`docs/math_map.tex` is the reference. It is correct and complete, and it is 494 lines of
paper-style LaTeX with twelve sections. This note is the front-end: one pass from token input to
token output, the certificate stated once and instantiated twice, and a table saying where every
measured number in the project plugs into the algebra. Where a claim needs the full statement,
this note points into the .tex by section.

Model throughout: `gemma-2-2b`. Width `d = 2304`, `L = 26` decoder blocks, vocabulary
`V = 256,000`, tied embedding `E` (the same matrix embeds and unembeds).

---

## 1. The forward map, with the linear parts marked

A prompt goes through five stages. Only two of them are linear, and it matters which two.

| stage | map | linear? |
|---|---|---|
| embed | `h0[t] = sqrt(d) * E[x_t]` | **yes**, a lookup and a scale |
| 26 decoder blocks | `h(l+1) = h(l) + Attn_l(RN(h(l))) + MLP_l(RN(h(l)))` | **no**, and it mixes across positions |
| final RMSNorm | `z = (h(L) / rms(h(L))) * (1 + gamma)` | **no**, but degree-zero homogeneous, so its Jacobian is closed form |
| unembedding | `u = E z` | **yes, exactly** |
| softcap then softmax | `u~ = 30 tanh(u/30)`, then `softmax(u~/T)` | monotone coordinatewise, so **argmax-irrelevant** |

Three consequences carry the rest of the note.

**Softcapping never changes which token wins.** `s -> c tanh(s/c)` is strictly increasing and
applied per coordinate, so every statement about *which token is emitted* can be made about the
raw logits `u = E z`. Statements about *probabilities* cannot.

**At temperature 0 the decoder is an argmax**, so the map from `z` to the emitted token is
piecewise constant, and the pieces are convex polyhedral cones

```
C_j = { z : (E[j] - E[k]) . z >= 0  for all k != j }
```

with `V - 1 = 255,999` faces. This is a target set defined in token space. It is not learned, not
a heuristic, and no readout can dissociate from it, because membership in `C_j` **is** the
behavior. (math_map `eq:cone`, §2.)

**RMSNorm is degree-zero homogeneous**, so its Jacobian is

```
A_pre = (sqrt(d) / ||h(L)||) * diag(1 + gamma) * (I - h_hat h_hat^T)
```

The projector annihilates the radial component of anything injected before the norm, and what
survives is scaled by `sqrt(d)/||h(L)||`, measured at `48/726 ~ 0.066` on our activations. Cost of
injecting before the norm rather than after: **4.4x**. (math_map `eq:rmsjac`, §2 item 3.)

**One convention warning, because it was a real bug.** Folding the gain into `z` and pairing it
with plain `E` gives the same logits and the same argmax as folding the gain into `W = E * (1+gamma)`
and pairing it with the ungained `z'`. It does **not** give the same *distances*, and every budget
in this project is a distance. Mixing the two inflates every reported `eps` by a per-coordinate
factor. Fix the convention once, state it, and stay in it.

---

## 2. The certificate, stated once

Fix an injection **site** `S` (a layer index, or the input or output of the final norm) and a
**readout**: a vector `w` and a threshold `t`, such that the property of interest is `w . z >= t`.
Let `A_S` be the Jacobian of the map from the site to `z`, evaluated at the unperturbed activation.
A perturbation `delta` injected at `S` moves the readout by `w . A_S delta = (A_S^T w) . delta` to
first order. So:

```
g    = t - w . z            the gap, how far there is to travel
m    = || A_S^T w ||        the controllability margin, the gain of this site on this readout
eps* = g / m                the least ||delta|| that reaches the halfspace
```

with minimiser `delta* = eps* * A_S^T w / m`. Two structural facts:

1. **`m` carries all the dynamics.** Nothing else about the network enters the formula.
2. **Steering along some other unit direction `u` costs `eps(u) = eps* / alpha`**, where
   `alpha = |(A_S^T w) . u| / ||A_S^T w||` is the cosine between the actuator you used and the
   optimal one.

Point 2 is the whole diagnosis of this project. **A steering experiment that fixes a budget without
reporting `alpha` cannot distinguish "the property is not causally actuable" from "we pushed almost
perpendicular to the thing that moves it."** (math_map `eq:cert` and `eq:alpha`, §3.1.)

---

## 3. The same formula, twice

### 3a. The probe halfspace (what we ran)

- site: input of decoder layer `l_src`
- readout: `w = unit(mean h_tgt[y=1] - mean h_tgt[y=0])`, the contrastive mean difference at `l_tgt`
- threshold: `t02`, from a one-dimensional logistic fit of the label on the score, solved at
  `P(y=1) = 0.2`
- `A_S = J`, the layer-to-layer Jacobian, obtained by **one** vector-Jacobian product, never `d`
  forward passes

Layer pairs actually used: `cities` 11 to 20, `common_claim_true_false` 13 to 22, `truthfulqa` 11
to 20.

**Its failure mode is structural, not numerical.** `w` is fitted, so `w . z >= t` is a claim about
a *readout*, and nothing forces a readout to be what the rest of the network consumes. We measured
exactly that: readout displacement tracked the certificate at `R^2 = 0.999` while behavior did not
move at all.

### 3b. The argmax cone (what to run)

Let `j_top` be the current argmax at the position that chooses the next word and `j_tgt` the token
we want instead. Put

```
a = E[j_tgt] - E[j_top]
M = -a . z = u[j_top] - u[j_tgt] > 0        the logit margin
```

Then "the model emits `j_tgt` rather than `j_top`" is exactly `a . z >= 0`, and section 2 applies
verbatim with `w := a`, `t := 0`, `g = M`, and

```
eps*_token = M / || A_S^T a ||
```

**Same formula, same code, same single VJP.** What changed is that `a` is not fitted. It is read
off the unembedding, and `a . z >= 0` is not a proxy for the behavior, it is a restatement of it.
The readout-versus-behavior dissociation becomes impossible by construction.

Two refinements the .tex works out and this note only records. One face is not the whole cone:
solving for all of `C_j` is a small QP whose active set is measured at 1 to 3 faces, and the true
`||delta*||` exceeds the single-face bound by about **8.8%**. And the cones are non-empty: by
Demeter et al. (2020) plus a minimax argument, all 182 tested tokens are achievable, so
unargmaxability is not the obstruction at `d = 2304`. Non-emptiness is a necessary condition only.
(math_map `eq:coneqp` and `eq:hull`, §3.3.)

### 3c. The broadcast operator, which is easy to forget

Every certificate we computed added `delta` at *every* position of the prompt, not one. That is a
different linear map, `A_S . B` with `B delta = 1_T (x) delta`, and its adjoint **sums** the
per-position gradients. The ratio of the two is the broadcast gain `rho`, measured at **1.355**
(cities, layer 16) and **2.016** (common_claim, layer 8). The broadcast site has the larger margin,
so a budget quoted under broadcast is easier to meet than the same number quoted per position, and
an ActAdd-style single-position `eps` is not comparable to ours without dividing by `rho`.
(math_map §4.1.)

---

## 4. Where each measured number plugs in

| symbol | reads | measured | plugs into |
|---|---|---|---|
| `eps*` | least budget that reaches the halfspace | cities **2.80**, common_claim **10.69** | `g/m`, section 2 |
| trust radius | where the first-order model still holds | cities **2.99**, common_claim **4.87** | validity of `eps*`: cities `0.94x` (**inside**), common_claim `2.20x` (**outside**) |
| `calibration` | realized slope over predicted slope, **at the linearization point** | cities **0.9997**, common_claim **0.7025** | says whether `A_S^T` is right. On cities the pullback is essentially exact |
| `context_factor` | the same ratio **at the generation stem** | cities **0.0729**, common_claim **0.1944** | says whether the Jacobian transfers to where we steer. Shortfall **13.7x** and **5.1x** |
| `alpha` | cosine of the actuator against the optimal direction | **0.0133** on cities, against a chance floor `sqrt(2/pi d) = 0.0166` | `eps(u) = eps*/alpha`. The direction we steered with cost **79x** the optimum |
| `phi` | fraction of the logit margin consumed | **never reported in any results table** | the honest currency, `eps` times `abs((A_S^T a) . u)` over `M` |
| `rho` | broadcast gain | 1.355, 2.016 | whether "budget eps" meant `eps` or `eps sqrt(T)` |
| `norm(delta*) / norm(z)` | size of the cheapest certified token flip | **0.034** | a 3.4% nudge flips the country |

Read the first four rows together and the two datasets fail for different reasons, which is why
they are never pooled. **`cities` is a genuine readout-versus-behavior dissociation**: `eps*` is
inside the trust radius and calibration is 0.9997, so the linear algebra is right and controlling
the readout simply buys no behavioral change. **`common_claim` is extrapolation**: `eps*` is 2.2x
the trust radius and calibration is already 0.70 at its own point, so the certificate is being
evaluated where the first-order model is not valid.

The `context_factor` row is separate from both and applies to each: the Jacobian was linearized
about the *statement*, and we steered at the *generation stem*, which is a 13.7x loss of gain on
cities before any of the above is considered. This is the defect the TruthfulQA track (Q0 to Q3)
removes by construction, because on a generation task the fit population and the steer population
are the same object.

---

## 5. What the halfspace target set actually was

In one line:

> The certified target set was `{ h at layer l_tgt : w . h >= t02 }`, where `w` is the contrastive
> mean-difference direction fitted on statement activations and `t02` is the score at which a
> one-dimensional logistic fit assigns `P(true) = 0.2`. That is **the set of activations on which
> one fitted linear readout would report FALSE at 80% confidence.** It is a statement about the
> readout's opinion, not about anything the model says out loud.

Two distinct emptiness results follow, and they should not be run together.

**The readout is uninformative on the population we steer.** `reach_stemprobe` activations,
recomputed read-only: balanced accuracy is exactly **0.500** on both datasets, AUC 0.510 and 0.568,
and the readout never changes sign (every cities stem scores between -83.5 and -43.1). So on the
generation stems the halfspace boundary carries no information about whether the completion will be
truthful. The target set is not semantically empty here so much as *unlocated*.

**The token-space target set was semantically empty, which S4 measured.** Moving to the argmax cone
of section 3b fixes the readout problem by construction but does not fix the *choice of target
token*, and minimum norm makes that choice badly. On `cities`, 200 of 200 certified argmax flips
succeeded, `hit_target` = 1.000, and yet only **16 distinct target tokens** appear across 200
statements: " North" was chosen 87 times and " South" 83, so **85% of certified targets are
directional modifiers that cannot carry the truth value of the claim.** Of the 110 statements whose
unsteered completion named the true country, 94 still named it after the certified flip. Vacuity
rate **0.855** [0.775, 0.915] as a lower bound, and **at most 5.5%** of certified flips
demonstrably made the claim false.

On `common_claim` the failure is different again and worse: `target_mode` was `runnerup`, so the
objective aimed at the model's own second-most-likely token. It never aimed at falsity at all, and
that target set is vacuous by definition rather than by accident.

**So the correct summary is that we certified, and hit, a target set that was the wrong set twice
over**: a fitted readout that is at chance where we steer, and a minimum-norm token target that is
semantically inert. Neither result says truth is un-actuable. Both say the target set has to be
specified before the certificate means anything, which is what the token-space program and the
TruthfulQA track are for.

---

## 6. Pointers into the reference

| for | read |
|---|---|
| the full forward map and the convention warning | `math_map.tex` §1 |
| softcap, argmax cones, the RMSNorm Jacobian, temperature | §2 |
| the certificate, `alpha`, both instantiations, the cone QP, non-emptiness | §3 |
| the three injection sites and the harness assertion | §4 |
| the broadcast operator and the `_asis` cross-layer caveat | §4.1, §4.2 |
| whether a norm budget is the right currency at all (anisotropy vs nonlinearity) | §5 |
| the full diagnosis table with all measured numbers | §6 |
| what each planned measurement pins down, E0 to E8 | §7 |
