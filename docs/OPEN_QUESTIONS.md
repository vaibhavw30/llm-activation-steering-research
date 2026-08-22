# Open Questions, Against the Literature

*The three questions at the end of `onboarding_sri.md`, each answered in three parts: what the
literature settles, what is still genuinely unanswered, and the cheapest experiment that would
resolve it. Citations resolve in [`LITERATURE.md`](LITERATURE.md).*

**Rule used for "cheapest."** An experiment counts as cheap if it runs on artifacts already on
disk, or costs under about 4 GPU-hours on DeltaAI. Anything larger is labelled as such.

---

## Q1. Is the null real, or an artifact?

The candidate confounds named in the onboarding doc were: aggregate cancellation from
anti-steerability, radial components annihilated by the final RMSNorm, a poorly chosen actuation
layer, broadcast injection across all positions including BOS, and a stray
`repetition_penalty = 1.3`.

### What the literature says

**On anti-steerability.** Tan et al. (arXiv:2407.12404) established that per-input steerability
is highly variable and that a non-trivial fraction of inputs move the wrong way, which makes
aggregate cancellation a real and standard concern. Li et al. (arXiv:2602.01654) give the
geometric reason: a static global vector assumes the concept-improving direction is constant
across representation space, and when the locally effective direction varies, a single vector
"can become misaligned, which yields weak or reversed effects."

**But our own test came back the other way, and this is worth saying loudly.** Running the Tan
protocol on our arms gives, on cities, 103 of 199 statements **inert** against 24 movers and 4
anti-movers (p = 0.00018); on common_claim, 8 of 197 inert with 35 against 21 (p = 0.081).
**Inertness dominates, not cancellation.** So the field's leading candidate confound is measured
and rejected on our data.

**On the RMSNorm confound.** No published work I could verify writes down that the final norm's
Jacobian exactly annihilates the radial component of a perturbation. The nearest neighbours are
empirical: Aparin & Gaintseva (arXiv:2606.06735) decompose steering into angular and radial parts
across seven models and find concepts live primarily in angular structure while norm still governs
stability; Vu & Nguyen (arXiv:2510.26243) build a rotation-based method on the same intuition.
**Neither states the null-space fact.** We have it exactly:
`A_pre = (sqrt(d)/||h||)·diag(1+gamma)·P_perp` with `P_perp h = 0`, which is why our pre-norm arm
is uninformative rather than negative, and why the measured RMSNorm penalty is 4.380 on cities and
5.177 on common_claim.

**On the layer choice.** Settled against us, and by more than LEACE §5.3 now. Labiosa et al.
(arXiv:2608.03842) report sensitivity and causality **anti-correlated at rho = −0.72 to −0.88**
across layers. Billa (arXiv:2604.15557) predicts layer selection from an unembedding-based
diagnostic at rho = +0.63 to +0.92. Góral et al. (arXiv:2512.07667) show that a Gaussian
allocation across depth beats single-layer intervention in six of seven models under equal budget.
**Our single layer chosen at the decodability peak is the weakest option in a family that has been
benchmarked.** Our own sweep agrees: controllability is front-loaded at L0 to L8, with 65 to 87%
lost by the layers we used.

**On the position set.** Nobody has published the ablation. What exists is the mechanism around
it: Barbero et al. (arXiv:2504.02732) argue the first-token sink exists to prevent over-mixing;
Queipo-de-Llano et al. (arXiv:2510.06477) trace sinks and representational compression to massive
residual-stream activations. **Our own 2.5x measurement (0.340 broadcast against 0.135
last-position at matched budget on cities) appears to be the best evidence anywhere on this
question**, against a first-order prediction of only 1.355.

**On the decoder confound.** Kang et al. (arXiv:2605.10664) name a mechanism we had not
considered: **KV-cache contamination**, where steered token states are cached and reused so a
local perturbation becomes cumulative coherence degradation. They report coherence drift improving
from −18.6 to −1.9 once interventions follow prompt-mediated pathways instead. **This is a live
alternative explanation for our degeneracy and empty-completion results, which we attributed to
over-steering.**

**And the framing risk.** LiSeCo **Lemma C.1** (Appendix C, "Identifying the allowable region in
latent space") states that for activation-space control to translate into attribute-space control,
the probe must apply to every reachable point in latent space, and the paper concedes this rarely
holds. **Our null is the measured failure case of that lemma.** Meanwhile Nadaf
(arXiv:2604.02608) reports the converse dissociation and finds decodable-without-steerable "nearly
empty (3 of 72)," which will be read as contradicting us unless we draw the distinction first:
their decodability is the model's own logit lens, ours is an externally fitted probe.

### What remains unanswered

1. **Whether the instrument can detect actuation at all.** Every result in this project is a null.
   Until the refusal positive control runs, "truth is not actuatable" and "our harness cannot
   detect actuation" are not separated. The literature cannot settle this for us. Arditi et al.
   (arXiv:2406.11717) establishes that refusal *is* a single-direction, causally effective concept,
   which is exactly why it is the right control, but it does not certify our pipeline.
2. **Whether KV-cache contamination or over-steering explains our incoherence.** Untested.
3. **Whether a better layer buys actuation or just buys margin.** Labiosa et al.'s cascade
   disruption predicts that moving to L5 to L9 raises controllability and damages downstream
   coherence. That prediction is testable and has not been tested by us or by them on our setup.

### Cheapest resolving experiment

**Run the refusal positive control.** It is fully implemented, 976 + 64 rows of prompts prepared,
four gated SLURM jobs written, roughly 4 GPU-hours, and never launched. It is the only experiment
that separates the two readings of every null in the project, and no amount of further reading
substitutes for it. **Nothing else on this page should be scheduled ahead of it.**

Three cheap things to run alongside, all on existing artifacts:

| Check | Cost | Decides |
|---|---|---|
| Compute Billa's `A_lin` at every layer on our cached activations and correlate against our measured flip rates | CPU, no new forward passes | Whether our layer choice is refutable by a published, better-powered diagnostic, and gives a citable baseline for our alignment statistic |
| Extend the pre-norm sweep past `frac = 5` | under 1 GPU-hr | Converts the uninformative pre-norm arm into either a confirmation of the RMSNorm penalty or a real anomaly. The prediction is registered: `oracle` should start flipping somewhere near the measured penalty of 4.4 to 5.2 |
| Re-run one cities arm with steering applied to newly generated positions only, leaving the cached prompt states untouched | 1 GPU-hr | Separates KV-cache contamination from over-steering as the source of our degeneracy |

---

## Q2. Does the token-space reformulation close the gap, or just move it?

### What the literature says

**The reformulation is well founded and the pieces are individually settled.** At temperature 0
the decoder is an argmax, so the emit-`j` region is a convex polyhedral cone; Brett
(arXiv:2604.06767) studies the same object as a Voronoi tessellation and supplies the margin
vocabulary. Emptiness is a real failure mode with a known bound: Demeter et al.
(arXiv:2005.02433) show interior-of-hull tokens are probability-bounded, and Grivas et al.
(arXiv:2203.06462) give exact detection and find the effect rare in practice (13 of 150 models,
negligible frequency). **We ran that check and got 182/182 achievable**, by the minimax identity
rather than a search, so the target set is non-empty by proof rather than by hope.

**The gap it closes is real, and it is the one LiSeCo Lemma C.1 names.** A probe halfspace is a
proxy for behavior and can fail to transfer. An argmax cone at temperature 0 **is** the behavior
at that position, so Lemma C.1's condition is satisfied by construction rather than assumed.

**The gap it moves is equally real, and the literature already names it.** Three items:

- **The horizon.** Our certificate is a first-token claim, and we measured its limit: `oracle`
  flips the first token on 200 of 200 cities statements while only 0.085 of those completions name
  the target country and 0.700 still name the correct one. Karnik & Bansal (arXiv:2509.21528) is
  precisely the formalism for extending a target set over a horizon, via a backward reachable tube
  learned by a Bellman-style recursion along the **token** axis. **Our construction runs along the
  depth axis. Composing the two is the obvious next problem and nobody has done it.**
- **Determinism.** He et al., Thinking Machines (Sept 2025) show that temperature 0 is not
  bit-deterministic under dynamic batching, because kernels are not batch-invariant and server
  load varies. Our cone geometry assumes a clean argmax. On single-stream local inference this is
  a non-issue; it becomes one the moment results are reproduced through a batched serving stack,
  and it should be stated as an assumption rather than discovered by a referee.
- **Off-manifold-ness.** Mishra et al. (arXiv:2604.09839) prove that steered activations almost
  surely have no prompt preimage. Reaching the cone is therefore a statement about the white-box
  system, not about anything a user could elicit. **This bounds what the result licenses and the
  paper explicitly cautions against the stronger reading.**

### What remains unanswered

1. **Whether reaching the cone at position `t` can be made to persist.** Measured limit: it does
   not, on its own. Unmeasured: whether re-planning per token (recomputing the cone and the
   pullback at each step) holds the trajectory. That is the MPC variant and it is expensive.
2. **Whether the token pullback beats a published unembedding diagnostic at picking directions.**
   Billa's `A_lin` is the incumbent and we have never compared against it.
3. **Whether the construction survives contact with a target set larger than one token.** A
   semantically false *statement* is a set of token sequences, not a single argmax cone.

### Cheapest resolving experiment

**A two-token horizon test, on the arm we have already run.** Take the cities `oracle` arm, where
the first token flips 200 of 200. At position `t+1`, recompute the cone for the continuation of
the target country name and re-solve the least-norm displacement, then measure how many
completions still name the target at the end. Cost: roughly 1 to 2 GPU-hours, no new
infrastructure, because the geometry code already solves per-statement cones and the steering hook
already exists.

**Why this is the right test.** It cleanly separates the two hypotheses. If one extra re-plan step
lifts completion-level target rate substantially above the current 0.085, the gap is a **horizon**
problem and BRT-Align style recursion is the fix, which is a strong, fundable direction. If it
does not move, the gap is **semantic** rather than positional, and no amount of per-token control
will produce a coherent lie. **Either outcome is a result, and it costs one afternoon.**

Cheap companion: state the batch-invariance assumption explicitly and verify determinism on our
own stack by re-running 32 statements twice at batch size 1 and at batch size 16, checking for
byte-identical output. Under an hour, and it closes a referee objection permanently.

---

## Q3. Is DCT-style discovery the right primitive, or should we select by controllability margin?

### What the literature says

**On DCT's validation gap.** The DCT writeup (Mack & Turner, Dec 2024) reports strong positive
results and **no systematic failure analysis**: no denominator for how many of its 512 directions
are interpretable or behaviorally meaningful. The authors record only a subjective impression that
projected linear DCTs miss interesting vectors. **Our 0 of 10 judged interpretation rate on the
top vectors by potency, behind a judge validated at 0.970, appears to be the first quantitative
negative denominator attached to DCT output anywhere.** That is a contribution independent of the
reachability work and should be reported as one.

**On whether magnitude is the right ranking.** AxBench (arXiv:2501.17148), run on gemma-2-2b,
found difference-in-means best for **detection** and prompting best for **steering**, with SAEs
uncompetitive on both. Their conclusion that better classification does not lead to better
steering is the general form of the question being asked here. **Ranking by downstream-change
magnitude, which is what DCT's `||U||` does, optimizes for the thing that moves the model most,
not the thing that moves the model toward a target.** Our own measurement of the same distinction
is sharp: the mean-difference truth direction consumes a median 1.0877 of the deciding logit
margin (more than the optimal displacement's 1.0010) and flips 1 of 101 statements, because
spending that much norm off-axis moves the other 255,999 logits too.

**On the alternative criterion.** Billa (arXiv:2604.15557) is the strongest evidence that a
target-aware, training-free scalar predicts steering success: peak `A_lin` at rho = +0.86 to +0.91
for effectiveness and +0.63 to +0.92 for layer selection, over 24 concept families and five
models. **This is the same idea as selecting by controllability margin, arrived at without the
Jacobian.** A-LQR (arXiv:2604.19018) supplies the other half: layer-wise Jacobians of LLMs are
cheap enough to use as a control model, with error bounds, and the code is public.

**On what a good discovery primitive would need.** Two 2026 methods show where the field is
heading, and both still reintroduce supervision at evaluation time. Rosser's Gradient Atoms
(arXiv:2603.14665) decomposes training gradients rather than activations and recovers refusal,
arithmetic and QA behaviors label-free, with atoms that double as steering vectors (bulleted-list
33% to 94%, refusal 50% to 0%), but reports no failure rate across its 500 atoms. Cho et al.
(arXiv:2606.08236, ICML 2026 Spotlight) cluster continuations by joint semantic and mechanistic
attribution, and validate by steering. **Nobody publishes the fraction of discovered directions
that do nothing.**

### What remains unanswered

1. **Whether the controllability margin `||J^T w||` outranks `A_lin` at predicting actuation.**
   They are different objects: the margin transports a target readout back through the inter-layer
   Jacobian; `A_lin` reads the unembedding at the layer with no transport. No comparison exists.
2. **Whether either criterion beats DCT's `||U||` when the goal is a *specified* concept.** The
   comparison is not apples-to-apples (DCT is unsupervised and target-free), which is exactly the
   framing the answer needs: **magnitude ranking is right for discovery and wrong for targeting,
   and the two should not share a pipeline.**
3. **Whether an unsupervised method can be validated without a labelled behavior at the end.** Open
   across the whole field, per §6.5 of the literature compilation.

### Cheapest resolving experiment

**A three-way ranking bake-off on artifacts already on disk. No new forward passes.**

Take our existing direction battery and, for each direction, compute three scores:

| Score | What it is | Cost |
|---|---|---|
| `&#124;&#124;U&#124;&#124;` | DCT's own potency ranking | already computed |
| `A_lin` | Billa's unembedding diagnostic at the layer | CPU minutes |
| `&#124;&#124;J^T a&#124;&#124;` and `alpha` | our controllability margin and alignment against the token target | already computed |

Then correlate each against the realized flip rates we already measured in `token_steer_*.csv`,
and report three Spearman coefficients with confidence intervals. **The whole thing is a CPU
script over committed files.**

**Why it settles the question.** If `||J^T a||` and `alpha` beat both `||U||` and `A_lin` at
predicting realized flips, the answer to Q3 is "select by controllability margin," with our own
data as the evidence and Billa as the strong baseline we cleared. If `A_lin` wins, the honest
answer is that the Jacobian is not earning its cost, and the project should pivot to the cheaper
diagnostic and spend the saved compute on the horizon problem in Q2. If `||U||` wins, the
reachability framing is in trouble and we would want to know that before writing it up.

**One caution on interpretation.** cities draws its targets from only 16 distinct tokens, so
alignment there takes only 35 distinct values and quartile splits collapse. Our existing
alignment-versus-behavior test read rho = 0.316 (p = 5.3e-06) on common_claim and rho = 0.129
(p = 0.068) on cities, and the cities number is evidence for nothing in either direction. **Run
the bake-off on common_claim as the powered dataset and report cities as underpowered rather than
as a second data point.**

---

## Sequencing

If all three are to be answered, the order is forced by what gates what.

1. **Refusal positive control** (Q1). Roughly 4 GPU-hours. Gates every other claim in the project.
2. **The `A_lin` and margin bake-off** (Q3). CPU only, no new runs, and it can proceed in parallel
   with (1) since it touches no GPU.
3. **The two-token horizon test** (Q2). 1 to 2 GPU-hours, but only worth running after (1) tells
   us the instrument works.
4. **Pre-norm budget extension and the KV-cache ablation** (Q1). Under 2 GPU-hours combined, and
   both close named objections rather than opening new lines.

**Separately and immediately, at zero compute cost: clone the A-LQR repository.** It is public, it
covers truthfulness and refusal, and its example scripts reference gemma-2b. The behavioral
evaluation this project has been treating as blocked on an access request is available now.
