# Deep-Research Prompt: Backward Reachability for Truth-Steering in LLM Activation Space

*Paste everything below the line into Claude (deep research mode). It is self-contained.*

---

I am an LLM-interpretability researcher. I need a deep, technical literature review and research
plan for applying **control-theoretic backward reachability / set propagation** to a concrete
problem in transformer activation steering. Below: (1) my project state and the exact null result
I'm stuck on, (2) my PI's proposed reframe, (3) what I already know from a first literature pass,
(4) the specific questions I need answered with depth and citations.

## 1. Project state — where I'm stuck

Model: **gemma-2-2b base** (26 layers, residual stream d=2304), fp32. Datasets: true/false factual
statements (curated "cities" — clean; "common_claim" — messy). Established results:

- **Truth is linearly decodable but not causally steerable.** A logistic probe reads true-vs-false
  from residual activations at ~99% (clean data). But steering along the supervised truth axis
  (difference-of-class-means "mean_diff", or the probe gradient) with calibrated strengths
  (τ ∈ [−1, +1] × a per-dataset norm scale) **never makes the model assert falsehoods** — judged
  by a validated open-LLM judge (TRUE/FALSE/INCOHERENT) over free-form generations. Pushing hard
  only degrades coherence. Slogan: **decodable ≠ causal**.
- **DCT (Deep Causal Transcoding)** — an unsupervised method that learns input directions V at a
  source layer whose injection maximally changes target-layer activations through the multi-block
  map (cities: layers 11→20; common_claim: 13→22; the objective works with a linearized/Jacobian
  view of that map; U = the effect directions at the target layer, V = input directions at the
  source layer, with a calibrated input_scale per dataset ≈ 48 and 87 respectively). DCT's most
  causally potent factors are **orthogonal to the truth axis** (|cos| ≈ 0), and its top-k causal
  subspace carries no more linearly-decodable truth than random-k directions.
- **Warm-started DCT:** anchoring DCT's input-direction search toward the supervised truth axis
  with a soft anchor λ ∈ {0, 0.3, 1, 3} interpolates cleanly in geometry (cos to seed: 0.03 → 0.95)
  but **no point on the path is a truth lever** — behavior runs from "mild coherence degrader"
  (low λ) to "inert" (high λ), never through "makes the model lie."
- **MAG (a second, geometry-based miner):** prepend "Is the following statement true?" and measure
  the activation shift Δ. The shift is highly one-dimensional (cos of its principal direction to
  the full shift: 0.84–0.98) and truth is linearly readable *from* the shift — but the shift
  direction is **orthogonal to the truth axis** (|cos| < 0.05), and the base model's verbalized
  verdict is a stuck "yes" (essentially 0 "no" answers over thousands of statements) whose
  answer-position activations read truth at chance. So "being asked about truth" is a real, crisp,
  linear direction — and it is not the truth-content direction.
- **In flight (built, not yet run):** (a) verdict-mode steering at scale with a p_yes−p_no logit
  margin readout, per gold label — testing whether the truth axis is causally potent only
  *conditional on question-mode*; (b) composed steering (question-gate vector + truth-content
  vector injected together); (c) **U-space-anchored DCT** — anchor DCT's *effect* (target-layer)
  direction toward the truth readout and leave the input free: "find any input whose causal effect
  moves the truth readout."

**The stuck point:** every experiment is a statement that some **single direction** fails. I cannot
distinguish between (i) "no coherent-lie state is reachable at all from bounded source-layer
perturbations" (a strong, almost theorem-shaped negative) and (ii) "the reachable lie-region
exists but is thin/curved/off-axis, and every 1-D probe missed it" (a constructive positive that
would immediately give better steering). I need set-level, not direction-level, machinery.

## 2. The PI's reframe (robotics/control-theory background)

Define a **target set in the output space** — the target-layer activation region where the truth
readout says FALSE (probe halfspace {h : w·h ≤ c − δ}), intersected with a coherence/on-distribution
constraint — and compute its **backward-reachable set** at the source layer under a perturbation
budget ‖Δh‖ ≤ input_scale. "The formalization is control theory; the actual tools are just linear
algebra": preimages of polytopes/ellipsoids/halfspaces under (locally) linear maps, Jacobian
pseudoinverses, SVD. Empty/off-distribution preimage ⇒ the null becomes an unreachability
statement. Non-empty ⇒ steer *within the set* instead of along an axis. The PI also flagged
Anthropic's global-workspace paper ("J-lens"/"J-space") — look at how they define their output
matrices (J_ℓ = E[∂h_final/∂h_ℓ]) and define acceptable-output sets in that Jacobian space.

## 3. What I already know (first pass — verify, deepen, extend)

- **arXiv 2509.21528, BRT-Align** (Karnik & Bansal, Stanford): backward reachable tubes in LLM
  latent space across *token steps*; failure set = sublevel set of a toxicity margin; value
  function learned with a Bellman min-recursion (DeepReach-style); least-restrictive filter
  steering. Approximate, no certificates, no code released yet. My case differs: my "dynamics" are
  *depth-wise* (one 9-block hop, linearizable), not a recurrent token rollout.
- **arXiv 2603.00140** (same lab): reachability-constrained RL for diffusion memorization; safety
  critic ≈ BRT; constrained-MDP trade-off (task reward vs staying out of the tube); compresses the
  control space to 64-dim before doing reachability.
- **Anthropic "A Global Workspace in Language Models"** (transformer-circuits, July 2026): J-lens
  J_ℓ = E[∂h_final/∂h_ℓ]; J-space = sparse nonnegative cone over J-lens vectors; "selective
  engagement" — the same info can be linearly present yet causally inert per task mode. This
  predicts my null via workspace gating.
- **NN preimage tools:** INVPROP (NeurIPS 2023, in α,β-CROWN), PREMAP (JMLR 2025) — provable
  preimage bounds for linearly-constrained output sets; validated to ~167k neurons, never on
  multi-block transformers with attention. BReach-LP / DRIP / hybrid-zonotope backward reachability
  (2310.06921, 2303.10513) for NN feedback loops, low-dimensional demos only.
- **LiSeCo** (arXiv 2405.15454): linear-probe halfspace constraint + closed-form minimal-norm
  projection, demonstrated on gemma-2-2b — forward safe-set control, not backward reachability.
- **Classical:** preimage of a polytope under a linear map is a polytope (closed-form for
  halfspaces: pull w back through Jᵀ); zonotopes/support functions scale to high d; grid-based
  Hamilton–Jacobi dies at ~6 dims (only enters via learned value functions).
- Note: arXiv 1910.13272 turned out to be feedback linearization via RL (Westenbroek et al.,
  Tomlin/Sastry) — inverse-output-map algebra (u = A(x)⁻¹(v−b)) but no set machinery.

## 4. What I need from you (be technical; cite precisely; flag uncertainty)

**A. The right formalization for a depth-wise, single-hop reachability problem.**
1. For h_tgt = F(h_src + Δ) with F = 9 transformer blocks, what are the failure modes of treating
   F via a single empirical Jacobian J (averaged vs per-sample; token-position handling; layernorm
   and attention nonlinearity within the hop)? What does the literature say about the validity
   radius of such linearizations in transformers (cite work measuring local linearity of residual
   stream maps, e.g. Jacobian-based lenses, tuned-lens-adjacent work, DCT's own linearization
   assumptions)?
2. Given a halfspace target {h : w·h ≤ c} and budget ‖Δ‖₂ ≤ ε, the linear answer is: reachable iff
   w·F(h_src) − c ≤ ε·‖Jᵀw‖. So everything hinges on ‖Jᵀw‖ and the geometry of Jᵀw. Is there prior
   art analyzing ‖Jᵀw‖ (the "controllability margin" of a probe direction) as an interpretability
   quantity? Anything relating probe directions to the row space of inter-layer Jacobians?
3. How should the **coherence constraint** be formalized as a set? Options I see: Mahalanobis ball
   around the activation distribution at the target layer; convex hull / percentile box of observed
   activations; a perplexity-proxy probe halfspace. What does the safe-set / control-barrier
   literature suggest for "stay on the data manifold" constraints in high-d latent spaces?

**B. Multi-hop and nonlinear upgrades.**
4. Realistically, can INVPROP or PREMAP be run through even ONE gemma-2 block (attention + MLP +
   RMSNorm, d=2304) today? What would have to be relaxed/approximated (e.g., freezing attention
   patterns to make the block linear-in-activations — is that a recognized technique, and who has
   done it for verification purposes)? Cite any attempt to verify/propagate sets through
   attention layers (GenBaB's general-nonlinearity support, any transformer-verification papers
   2023–2026, e.g. robustness certification for ViTs/BERT).
5. Is there work composing per-block linearizations into interval/zonotope products across many
   blocks with error accumulation bounds (rather than one end-to-end Jacobian)?

**C. Controllability theory for LLMs.**
6. Beyond Soatto et al. 2305.18449 and LiSeCo: map the 2024–2026 "control theory of LLMs"
   literature — steering as control, reachable output sets under prompt or activation control,
   empirical controllability audits ("what can we actually steer?"), adversarial-reachability
   framings. Which of these define reachable SETS in activation space (vs output/token space)?
7. Is there any published attempt at exactly my construction — backward-reachable set of a probe
   halfspace through inter-layer transformer dynamics? If yes, what did they find; if no, what's
   the nearest miss?

**D. The global-workspace connection.**
8. Reconcile the two frames: if truth sits outside J-space (the verbalizable workspace) during
   declarative processing, does that PREDICT a small ‖Jᵀw‖ for the block-hop Jacobian into the
   verdict-relevant region? Can J-lens matrices themselves serve as the "output matrices" defining
   my target set (target set = cone of J-lens vectors around the FALSE-verdict vocabulary
   direction), and has anyone combined J-lens-style objects with reachability/steering?

**E. Experiment design critique.**
9. Critique this v0 plan: (i) empirical Jacobian J of the 9-block map at k sample statements
   (autograd, batched); (ii) compute ‖Jᵀw‖ and cos(Jᵀw, mean_diff@source) per statement; (iii)
   decide reachability of the FALSE-margin halfspace within ε = input_scale; (iv) if reachable,
   steer along Jᵀw (the maximally-efficient direction for moving the readout) and behaviorally
   judge the completions; (v) compare against the already-built U-space-anchored DCT run. What are
   the statistical and conceptual traps (Jacobian averaging washing out per-input structure;
   probe-direction non-identifiability; the readout moving without the *behavior* moving —
   activation-space vs output-space gap that LiSeCo flags; coherence set mis-specification)?
10. Propose the strongest version of the "unreachability theorem" claim that could survive review:
    what exactly can be claimed from bounded-budget linear-hop unreachability plus empirical
    nonlinear spot-checks, and what hedges are mandatory?

**Deliverable:** a structured report with sections A–E, each with: findings, key citations
(author, year, arXiv/venue, one-line method), confidence level, and a final prioritized
reading list (top 10) + a recommended 2-week experiment sequence.
