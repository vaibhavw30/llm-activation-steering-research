# Literature Compilation: Automated Feature Discovery

*Compiled 2026-08-22 for the shared research base. Every entry below was checked against a
primary source (the arXiv abstract page, the publisher page, or the paper's own HTML) during
compilation. Where verification failed or a claim in `onboarding_sri.md` did not survive
checking, the entry says so rather than smoothing it over.*

**Companion:** [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) turns the three open questions at the end
of the onboarding doc into literature summaries plus the cheapest resolving experiment for each.

**Note on the onboarding file.** `onboarding_sri.md` is not committed to this repo. This
compilation was built from the version pasted into the compiling session. If the file is later
committed, re-check the four citation corrections in §7.

### Confidence key

| Tag | Meaning |
|---|---|
| **[primary]** | Title, authors, date and abstract read off the arXiv/publisher page during compilation |
| **[primary+]** | As above, plus a specific in-paper claim (appendix, lemma, section) verified in the paper body |
| **[citing]** | Verified through a citing source or search index only; the primary page was not readable |
| **[unverified]** | Could not confirm. Flagged, not summarized |

---

## 0. Executive summary

### What the literature settles

1. **Probing accuracy does not imply causal importance.** This is settled and has been since
   Elazar et al. 2021. Our result is a sharper instrument for a known phenomenon, not a discovery
   of it. Any framing that treats decodable-versus-causal as novel will be rejected.
2. **Steering reliability is highly variable and largely a property of the concept and dataset,
   not the model.** Tan et al. 2024 established per-input variance and anti-steerability; Fan et
   al. 2026 predict steerability from early hidden states across 150 concepts and 1.4M
   generations. "Steering works" is not a claim anyone still makes unconditionally.
3. **Difference-in-means is a strong detector and a weak actuator.** AxBench (on gemma-2-2b
   specifically) reports difference-in-means best for concept detection while prompting beats
   every representation method for steering. The onboarding's framing of mean-difference as "a
   heuristic" is charitable; the benchmark evidence is stronger than that.
4. **The layer where a concept is most decodable is systematically not the layer where it is
   most actuable.** LEACE §5.3 was one data point. There are now several, including a reported
   anti-correlation of rho = −0.72 to −0.88 between sensitivity and causality across layers.
5. **Steering interventions move the residual stream off the manifold reachable by any prompt.**
   Mishra et al. 2026 prove this. It does not invalidate steering research, but it does mean a
   steered state is not evidence about what a prompt could do.
6. **Unargmaxable tokens are a real but rare phenomenon at our scale.** Demeter et al. 2020
   established the convex-hull bound; Grivas et al. 2022 gave an exact detection algorithm and
   found 13 of 150 public models affected, with negligible frequency. At d = 2304 this is a
   check to run, not a threat. Our own run confirms it: 182/182 targets achievable.

### What is genuinely open

1. **Whether an internal readout crossing a threshold implies anything about the emitted text.**
   LiSeCo states the sufficient condition as a lemma and concedes it rarely holds. Nobody has
   measured how badly it fails, or given the mechanism. That gap is where our result sits.
2. **Certified reachability against an unembedding-defined target region.** No work found pulls
   an argmax decision cone back through inter-layer dynamics. See §3.4 for the novelty verdict
   and its limits.
3. **Position sets as part of a steering specification.** The literature reports layer and
   coefficient as the tuned hyperparameters. The set of sequence positions the perturbation is
   added at is treated as an implementation detail almost everywhere, and we measured it as worth
   2.5x on realized flip rate.
4. **How to validate an unsupervised direction without a labelled behavior.** Every discovery
   method surveyed in §6 validates with a supervised or human-read behavioral check at the end.
   The label reappears at evaluation time in all of them.

### The three findings most likely to change how we run the next experiment

**1. The A-LQR code is public, and it already covers truthfulness and refusal on gemma-2b.**
Repository: `github.com/trustworthyrobotics/lqr-activation-steering`, linked from the paper
itself, 94 commits, public. Its four task areas are toxicity, truthfulness (TQA), concept
steering, and refusal. **Our standing ask "can we get A-LQR access for behavioral evaluation" is
stale**, and the plan built around waiting for it should be rewritten. This is the single most
actionable item in this document.

**2. Someone has already built a predictor of steering success out of the unembedding, and it is
far better powered than ours.** Billa 2026 (arXiv:2604.15557) defines `A_lin` by applying the
unembedding to intermediate hidden states, requires no training, and reports peak `A_lin`
predicting steering effectiveness at rho = **+0.86 to +0.91** and layer selection at rho =
**+0.63 to +0.92**, across 24 concept families and five models. Our own alignment-versus-behavior
check reached Spearman **0.316** on one dataset and was underpowered on the other. `A_lin` is a
one-afternoon computation on our existing activations and gives us both a much stronger baseline
and an independent layer-selection criterion. **It is also the nearest prior art to the intuition
behind our alignment statistic**, and it needs to be cited as such.

**3. Our null is the measured failure case of a published lemma, not an unexplained anomaly.**
LiSeCo Appendix C, Lemma C.1: for control in activation space to translate to control in
attribute space, the probe must apply to every reachable point in the latent space, and the paper
states this rarely holds in practice. That reframes our contribution. We are not reporting a
surprise; we are quantifying how badly a known-fragile condition breaks and naming the mechanism.
That is a better paper and a harder one to dismiss.

---

## 1. Negative and reliability results in steering

### 1.1 Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso, Kirk (2024). *Analyzing the Generalization and Reliability of Steering Vectors.* arXiv:2407.12404. NeurIPS 2024. **[primary]**

**Method.** Takes contrastive activation addition steering vectors for a set of behavioral
concepts and evaluates them per-input rather than in aggregate. Defines a per-input steerability
score, then decomposes variance in that score across inputs, concepts, and datasets. Tests
out-of-distribution robustness by perturbing the prompt format rather than the content.

**Load-bearing claim.** From the abstract: steering vectors have "substantial limitations both
in- and out-of-distribution," and "in-distribution, steerability is highly variable across
different inputs." The anti-steerability finding (a non-trivial fraction of inputs move the
*wrong* way, so aggregate means understate per-input effects) is developed in the results
section, not the abstract.

**Relevance.** Open question 1 directly. Anti-steerability is our leading candidate for
aggregate cancellation: a mean over statements can read null while individual statements move in
both directions. We tested this and found the opposite of the naive prediction, which is worth
reporting: on cities, 103 of 199 statements were inert with 24 movers against 4 anti-movers
(p = 0.00018); on common_claim, 8 of 197 inert with 35 against 21 (p = 0.081). Inertness, not
cancellation, is our failure mode.

**Code / gemma-2-2b.** Code released. Their models are Llama-family; the method transfers.

---

### 1.2 Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* arXiv:2501.17148. ICML 2025. **[primary]**

**Method.** A benchmark with two tracks, steering and concept detection, run on **Gemma-2-2B and
9B**. Compares prompting, finetuning, SAEs, difference-in-means, linear probes, LAT, and
representation finetuning on a common set of concepts with a judge-scored steering rubric.
Introduces ReFT-r1, a rank-1 weakly-supervised representational method, and releases SAE-scale
dictionaries for ReFT-r1 and DiffMean.

**Load-bearing claim.** Abstract: "For steering, we find that prompting outperforms all existing
methods, followed by finetuning. For concept detection, representation-based methods such as
difference-in-means, perform the best. On both evaluations, SAEs are not competitive."

**Relevance.** Open question 3, and it is the benchmark our result has to be legible against.
The detection-versus-steering split is our decodable-versus-causal claim measured at scale on our
exact model. It also warns against any plan that routes steering through SAEs.

**Code / gemma-2-2b.** Code and dictionaries released. **Runs on gemma-2-2b, and was run on it.**

---

### 1.3 Nadaf (2026). *Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens.* arXiv:2604.02608. **[primary]**

**Method.** Extracts function vectors as mean differences across in-context-learning
demonstrations, then for each of 12 tasks and 6 models across 3 families tests two things
independently: whether the FV steers the model to the correct answer, and whether the logit lens
can decode that answer from any intermediate layer. 4,032 directed cross-template pairs.

**Load-bearing claim.** Abstract: "FV steering routinely succeeds where the logit lens cannot
decode the correct answer at any intermediate layer, while the converse — decodable without
steerable — is nearly empty (3 of 72)."

**Relevance. This is the paper most likely to be used against us, and it must be addressed in
any writeup.** Read carelessly, "decodable without steerable is nearly empty" is a direct
contradiction of our headline. The resolution is that the two papers mean different things by
decodable: Nadaf's decodability is the model's **own unembedding** applied to intermediate
states, ours is an **externally fitted probe**. A statement whose truth an external probe reads
at 0.99 while the logit lens reads nothing is precisely our low-alignment case, and Nadaf's axis
would score it as *not* decodable. So the results are compatible, but only after that distinction
is drawn explicitly. **Do not let a reviewer draw it first.**

**Code / gemma-2-2b.** Not stated on the abstract page. Model families listed do not obviously
include Gemma.

---

### 1.4 Li, Li, Huang (2026). *Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models.* arXiv:2602.01654. **[primary]**

**Method.** Diagnoses static steering vectors geometrically: a single global vector assumes the
concept-improving direction is constant across representation space. Replaces it with a learned
differentiable concept scoring function whose local gradient defines the steering direction at
each activation, giving an explicitly context-dependent intervention, coordinated across layers
in a shared concept space.

**Load-bearing claim.** Abstract: "When the locally effective direction varies with the current
activation, a single global vector can become misaligned, which yields weak or reversed effects."

**Relevance.** Open questions 1 and 3. This is the constructive counterpart to our context-shift
finding. We measured that removing one word of context replaces 29 of 32 SAE atoms in the
actuating direction (Jaccard 0.049 against a 0.684 same-quantity control); SVF is what you build
if you take that seriously. **It also implies our single fixed direction was the wrong object
regardless of which direction we picked.**

**Code / gemma-2-2b.** Not stated on the abstract page.

---

### 1.5 Ye, Ran, Yao, Wang, Jiang, Hou, Li, Pan (2026). *Where Steering Signals Come From: Activation Source Selection in Activation Steering.* arXiv:2607.25270. **[primary]**

**Method.** Holds the downstream intervention fixed and varies only the *source* of the
activations the steering vector is built from: which context, and which readout policy over
positions. Three instruction-tuned models, four task families. Introduces "tail subtraction,"
which removes shared prompt and continuation semantics from boundary states.

**Load-bearing claim.** Abstract: "strong signals come from execution-boundary states, where the
model is about to produce or continue the target behavior," and "steering depends on
representations of what the model is about to do, not merely on what has already appeared."

**Relevance. This is the strongest convergent external evidence for our D2 mechanism and I would
put it in the related-work paragraph.** Our probe is fitted on complete statements and read on
generation prefixes, where its balanced accuracy is exactly 0.500. In their vocabulary, we built
the actuator from a post-realization state and applied it at an execution boundary. Their tail
subtraction is a cheap candidate fix that we have not tried.

**Code / gemma-2-2b.** Not stated on the abstract page.

---

### 1.6 Fan, Cheng, Li, Feizi, Zhou (2026). *When is Your LLM Steerable?* arXiv:2606.11599. **[primary]**

**Method.** Builds a dataset of 1.4M steered generations across 150 concepts labelled
under-steer / success / over-steer, then trains a predictor that reads early hidden states (after
the first few generated tokens) to forecast steering outcome before running the full rollout.

**Load-bearing claim.** Roughly 0.7 macro-F1 on unseen concepts, from early decoding dynamics
alone.

**Relevance.** Open question 1. Gives an external reference point for how much of steering
success is predictable in advance, and a labelling scheme (under/success/over) that is better
than our binary flip metric.

**Code / gemma-2-2b.** Not stated on the abstract page.

---

### 1.7 Billa (2026). *Predicting Where Steering Vectors Succeed.* arXiv:2604.15557. **[primary]**

**Method.** Defines the Linear Accessibility Profile (LAP): apply the model's unembedding matrix
to intermediate hidden states, layer by layer, and read off a scalar `A_lin` measuring how
linearly accessible the target concept is at that depth. No training required. Evaluated on 24
controlled binary concept families across five models from Pythia-2.8B to Llama-8B.

**Load-bearing claim.** Abstract: "peak `A_lin` predicts steering effectiveness at rho = +0.86 to
+0.91 and layer selection at rho = +0.63 to +0.92."

**Relevance. Read this one before doing anything else.** It bears on all three open questions.
It is the nearest prior art to our alignment statistic and to the token-space reformulation's
core intuition: use the unembedding, not a fitted probe, to decide whether a direction can
actuate. Two things follow. First, we must cite it and position against it. Second, its
correlations are an order of magnitude better established than our Spearman 0.316, and `A_lin` is
computable from activations we already have on disk. **What it does not do is compute a preimage:
`A_lin` is a diagnostic read at a layer, with no Jacobian, no target set, and no budget.** That
is the space our construction occupies.

**Code / gemma-2-2b.** Not stated on the abstract page. The method needs only the unembedding and
hidden states, so it runs on gemma-2-2b unchanged.

---

### 1.8 On BOS, attention sinks, and all-position injection

The specific question asked, does injecting at all sequence positions rather than one aligned
position cause degeneration in Gemma-family models, **is not directly answered by any paper I
could verify.** What exists is the surrounding mechanism, in three parts.

- **Barbero, Arroyo, Gu, Perivolaropoulos, Bronstein, Veličković, Pascanu (2025). *Why do LLMs
  attend to the first token?* arXiv:2504.02732.** **[primary]** Argues theoretically and
  empirically that the attention sink is a mechanism to avoid over-mixing of information across
  positions. Gemma is not mentioned in the abstract. Relevance: gives a *reason* corrupting BOS
  should be harmful rather than neutral, which is what our broadcast convention risks.
- **Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv (2025). *Attention
  Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin.* arXiv:2510.06477.**
  **[primary]** Traces both phenomena to massive activations in the residual stream, proves that
  massive activations necessarily produce representational compression, and bounds the entropy
  reduction. Models from 410M to 120B. Gemma not mentioned in the abstract.
- **Community reports** that GemmaScope SAEs were not trained on the BOS token and that BOS
  activations sit far outside the training distribution, and that BOS-dominant features introduce
  generation artifacts while contributing little steering effect. **[citing]** These are
  LessWrong and Medium posts, not peer-reviewed, and are flagged as such. They are consistent
  with our own measurement but should not be cited as evidence.

**Our own number is currently the best evidence on this question**: broadcasting to all positions
versus last-position-only changed the realized flip rate by 2.5x on cities (0.340 against 0.135
at matched budget) while the first-order broadcast gain predicts only 1.355. **That gap is
publishable on its own and appears to be unclaimed.**

---

### 1.9 Compact entries

| Work | Claim | Bearing | Conf |
|---|---|---|---|
| Yap (2026), *Behavioral Steering in a 35B MoE LM via SAE-Decoded Probe Vectors*, arXiv:2603.16335 | Five trait vectors all modulate one agency axis; the risk-calibration vector "produces only suppression"; steering during decoding alone has zero effect (p > 0.35) | An explicit epiphenomenal-direction case study, and a warning that our per-concept directions may not be independent | [primary] |
| Wenkmann & Garreau (2025), *On The Variability of Concept Activation Vectors*, arXiv:2509.24058 | CAV variance decreases as 1/N in the number of sampled examples | Sets the sampling floor for our n = 200 direction estimates | [primary] |
| Abdullaev et al. (2026), *Concept Heterogeneity-aware Representation Steering*, arXiv:2603.02237, ICML 2026 | Models source and target representations as Gaussian mixtures and steers by optimal-transport barycentric projection, input-dependent | The clustering counterpart to SVF; relevant if our inert statements form a cluster | [citing] |
| Kang et al. (2026), *Prompt-Activation Duality*, arXiv:2605.10664 | Identifies KV-cache contamination: steered states are cached and reused, turning a local perturbation into cumulative coherence degradation; coherence drift improves from −18.6 to −1.9 | **A named mechanism for our degeneracy and empty-completion results that we had attributed to over-steering** | [primary] |
| Cheng & Kriegeskorte (2026), *Decomposing how prompting steers behavior*, arXiv:2606.03093 | Prompt-induced representational change is largely shape-preserving (translation, rigid + uniform scaling); affine is the first tier that nearly recovers target-prompt task geometry | Directly bears on MAG's question-shift direction `v_Q` and on why `eps_Q` exceeded 1.0 | [citing] |

---

## 2. Layer selection for actuation

### 2.1 What the literature settles

**The decodability peak and the actuation peak are different layers, and the effect is now
reported by several independent groups.** LEACE §5.3 was the data point in the onboarding doc.
Two more, both verified:

- **Labiosa, Buff, Nayak, Donno (2026). *Sensitivity, Causality, and Repair Dissociate: A
  Layer-Wise Analysis of Perturbation Robustness and Its Scaling.* arXiv:2608.03842.**
  **[primary]** On the two models meeting an 80% identity-patch gate, **sensitivity and causality
  are anti-correlated at rho = −0.72 to −0.88.** The authors name the mechanism "cascade
  disruption": intervening at causally implicated early layers damages downstream computation, so
  diagnostic-flagged sensitive sites are counterproductive for repair. Relevance: this is the
  strongest external form of our layer finding, and it complicates it. Our own sweep found
  controllability front-loaded at L0 to L8 with 65 to 87% lost by the layers we used, and a best
  window at L5 to L9. **Cascade disruption predicts that moving to L5-L9 buys margin and costs
  coherence.** That is a testable prediction against our own next run.
- **Billa (2026), arXiv:2604.15557**, §1.7 above: peak `A_lin` predicts *layer selection* at
  rho = +0.63 to +0.92. This is a published, cheap layer-selection protocol and the closest thing
  to a copyable procedure in this area.

**On protocols to copy.** ActAdd (Turner et al., arXiv:2308.10248, **[primary]**) tunes layer and
coefficient as hyperparameters over a sweep, which is the field's default and is what the
onboarding's note 8 refers to. Góral, Winkels, Basart (2025), *Depth-Wise Activation Steering for
Honest Language Models*, arXiv:2512.07667 **[primary]**, is the most relevant alternative: rather
than choosing one layer, it weights steering strength across depth with a Gaussian schedule, and
reports that on the MASK benchmark this beats no-steering and single-layer baselines in six of
seven models across LLaMA, Qwen and Mistral, with equal-budget ablations showing the Gaussian
schedule beats random, uniform and box-filter allocations. **The honest reading for us: our
single-layer choice is not just possibly-wrong, it is the weakest member of a family of
allocations that have been benchmarked against each other.**

### 2.2 What is still open

Whether a *certified* layer choice (maximize the controllability margin `||J^T w||`) beats a
*predicted* one (maximize `A_lin`) has not been compared by anyone, including us. They are
different quantities: the margin is a pullback through the inter-layer Jacobian, `A_lin` is a
direct unembedding read at the layer with no transport. **We are in a position to run that
comparison on artifacts already on disk, and it would be a clean, small, publishable result.**

---

## 3. Token-space and unembedding geometry

### 3.1 The settled base

- **Demeter, Kimmel, Downey (2020). *Stolen Probability: A Structural Weakness of Neural Language
  Models.* arXiv:2005.02433. ACL 2020.** **[primary]** Numerical, theoretical and empirical
  analysis showing that words on the **interior of the convex hull** of the embedding space have
  their probability bounded by the probabilities of words on the hull. Relevance: a token-space
  target set can be *empty by construction*, which would silently invalidate a reachability claim.
  This is a precondition check, not a result. **We ran it: 182/182 targets achievable**, proved by
  the minimax identity `max_{||z||<=1} min_k (E[j]-E[k])·z = dist(E[j], conv{E[k]})`, which is
  exact in both directions.
- **Grivas, Bogoychev, Lopez (2022). *Low-Rank Softmax Can Have Unargmaxable Classes in Theory but
  Rarely in Practice.* arXiv:2203.06462. ACL 2022.** **[primary]** Gives an exact detection
  algorithm and finds **13 of 150 public models** have unargmaxable tokens, but that they are
  "very infrequent and unlikely to impact model quality." Relevance: reassuring at d = 2304, and
  the correct citation for why the check is cheap rather than the reason to worry.
- **Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, Steinhardt (2023). *Eliciting
  Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.** **[primary]**
  Trains an affine probe per block to decode every hidden state into a vocabulary distribution,
  and shows with causal experiments that the tuned lens uses similar features to the model itself.
  Relevance: prior art for the general move of reading intermediate states in the final basis,
  and the honest comparison for our `_asis` convention. **The tuned lens *learns* the transport;
  we currently apply none.**

### 3.2 New, and relevant

- **Brett (2026). *Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds
  of Large Language Models.* arXiv:2604.06767.** **[primary]** Studies the tessellation the
  unembedding induces over the representation manifold on Qwen3.5-4B-Base, validates scaling
  laws, and introduces margin-refinement procedures comparing direct margin maximization against
  Fisher-information distance. **Analyzes the decision regions as Voronoi cells, not as cones, and
  performs no reachability or preimage computation through inter-layer dynamics.** Relevance:
  closest published treatment of the same geometric object, arrived at from a different direction,
  and a source for the margin vocabulary.
- **Mishra, Khashabi, Liu (2026). *Steered LLM Activations are Non-Surjective.* arXiv:2604.09839.
  ICLR 2026 Workshops (Sci4DL, Re-Align).** **[primary]** Casts prompt-reachability as a
  surjectivity question and proves, under stated practical assumptions, that activation steering
  pushes the residual stream off the manifold of states reachable from discrete prompts: "Almost
  surely, no prompt can reproduce the same internal behavior induced by steering." Illustrated
  empirically on three LLMs. **This is about the surjectivity of the prompt-to-activation map, not
  about computing reachable sets in activation space.** Relevance: a real caveat on our
  certificate. The states we certify arrival at are, with probability one, states no prompt
  produces. That does not make the certificate wrong, but it does bound what it licenses, and the
  paper explicitly cautions against reading white-box steerability as evidence about black-box
  behavior.

### 3.3 The verification tools, if we harden the claim

- **Kotha, Brix, Kolter, Dvijotham, Zhang (2023). *Provably Bounding Neural Network Preimages.*
  arXiv:2302.01404. NeurIPS 2023 (Spotlight).** **[primary]** The INVPROP algorithm verifies
  properties over the preimage of a **linearly constrained output set**, combinable with
  branch-and-bound, GPU-accelerated, no LP solver required. Reports over-approximations up to
  2500x tighter than prior work at 2.5x the speed, and is incorporated into α,β-CROWN. Their own
  demo application is backward reachability for a dynamical system. Relevance: **an argmax cone is
  exactly a linearly constrained output set**, so INVPROP is the tool that would upgrade our
  first-order certificate to a sound one. The obstacle is scale: their largest reported model is
  167k neurons, and a 26-layer 2.3k-wide transformer is far outside that.
- **Zhang, Wang, Kwiatkowska, Zhang (2024/2025). *PREMAP: A Unifying PREiMage APproximation
  Framework for Neural Networks.* arXiv:2408.09262, JMLR v26 (2025).** **[primary]** Produces
  under- and over-approximations of the preimage of **any polyhedral output set** using cheap
  parameterised linear relaxations plus anytime refinement that splits on input features and
  neurons. Same relevance and same scale obstacle as INVPROP, with a sound-and-complete algorithm
  for quantitative verification.

### 3.4 Novelty verdict on the token-space reformulation

**Question asked: has anyone certified reachability against an unembedding-defined target region,
or pulled an argmax cone back through inter-layer dynamics?**

**Verdict: I found no such work, and I searched for it specifically. The claim appears open. Four
caveats on that verdict, in descending order of how much they should worry us.**

1. **Billa 2026 (arXiv:2604.15557) already uses the unembedding to predict steerability and select
   layers**, with much better statistics than ours. The *diagnostic* use of the unembedding is
   taken. What is not taken is the *constructive* use: a target set, a preimage, a budget, and a
   least-norm displacement verified against the full vocabulary. Our novelty claim must be scoped
   to the construction, not to the idea of consulting the unembedding.
2. The pieces all exist separately and are individually unsurprising: preimages of polyhedral sets
   (INVPROP, PREMAP), Voronoi/cone geometry of the unembedding (Brett, Demeter, Grivas), backward
   reachability in latent space along the token axis (BRT-Align), and layer-wise Jacobians of LLMs
   (A-LQR, Tuned Lens). **A reviewer can reasonably call the combination straightforward.** The
   defense is the measurement it enabled, not the construction.
3. A negative search result is weak evidence. Searches in this area return heavily polluted
   results, and I could not read OpenReview during compilation (browser verification wall).
4. This literature moves fast enough that the verdict has a short shelf life. Re-run the §3
   searches before submission.

---

## 4. Normalization geometry

### 4.1 What exists

- **Aparin & Gaintseva (2026). *A Geometric Account of Activation Steering through Angle-Norm
  Decomposition.* arXiv:2606.06735.** **[primary]** Disentangles the angular and radial components
  of a steering intervention in a controlled study across **seven language models**. Finds that
  "concepts are represented primarily in angular structure," which supports spherical methods,
  "but that norm remains important for the stability and downstream effects of steering."
  Concludes that steering should be parameterized by interpretable angular and radial components
  rather than a single additive coefficient that entangles them. **The abstract page does not
  discuss RMSNorm annihilating the radial component**; I could not verify that specific claim in
  the body.
- **Vu & Nguyen (2025). *Angular Steering: Behavior Control via Rotation in Activation Space.*
  arXiv:2510.26243. NeurIPS 2025 (Spotlight).** **[primary]** Rotates activations within a fixed
  two-dimensional subspace, unifying additive and orthogonalization techniques under a rotation
  framework, with an adaptive variant that rotates only activations already aligned with the
  target feature.

### 4.2 What is not settled, and where we are ahead

**Nobody I could verify writes down the final-norm Jacobian and observes that it exactly
annihilates the radial component.** The angle-norm literature above is empirical and is about the
*intervention's* decomposition, not about the *normalization operator's* null space. The exact
statement, which we do have, is that RMSNorm is degree-zero homogeneous, so

```
A_pre = (sqrt(d)/||h||) · diag(1 + gamma) · P_perp ,   P_perp = I − h_hat h_hat^T
```

and `P_perp h = 0` exactly. Two consequences follow with no experiment: the radial part of any
pre-norm perturbation is destroyed, and what survives is contracted by `sqrt(d)/||h||`, which on
our activations is roughly 5x. **Our measured RMSNorm penalty of 4.380 on cities and 5.177 on
common_claim is the empirical form of that constant.**

This looks like a genuine, small, defensible contribution. It is also the explanation for our
pre-norm null arm, which is uninformative rather than negative: a displacement certified after the
norm was never given enough budget to survive it.

---

## 5. Control theory and reachability for LLMs

### 5.1 The map, by axis and by object

The most useful way to organize this literature is by two binary questions: does the work define
reachable **sets** in **activation** space, and does it operate along the **token** axis (across
generation steps) or the **depth** axis (across layers)?

| Work | Reachable sets? | Space | Axis | Guarantees |
|---|---|---|---|---|
| Karnik & Bansal 2025, BRT-Align, arXiv:2509.21528 | **Yes**, a backward reachable tube | activation (latent) | **token** | learned value function, no certificates |
| Cheng & Amo Alonso 2024/2026, LiSeCo, arXiv:2405.15454 | **Yes**, a probe halfspace safe set | activation | token (per generated token) | closed-form optimal projection; Lemma C.1 caveat |
| Skifstad, Yang, Chou 2026, A-LQR, arXiv:2604.19018 | No (setpoint tracking, not sets) | activation | **depth** (layer-wise LTV) | theoretical bounds on setpoint tracking error |
| Mishra et al. 2026, arXiv:2604.09839 | No (surjectivity of prompt→activation) | activation | n/a | proof under stated assumptions |
| Nguyen et al. 2025, PID Steering, arXiv:2510.04309 | No | activation | token | classical stability connections |
| Soatto, Tabuada, Chaudhari, Liu 2023, arXiv:2305.18449 | **Yes**, in a quotient space of "meanings" | **meaning** space, not activations | token (prompts) | see §5.3 |
| Kotha et al. 2023 / Zhang et al. 2025 | **Yes**, preimages of polyhedral sets | generic NN input space | n/a | sound over- and under-approximation |
| Karnik, Kim, Koyejo, Lee, Bansal 2026, RADS, arXiv:2603.00140 | **Yes**, a backward reachable tube | diffusion latent | denoising steps | constrained RL, approximate |
| **Ours** | **Yes**, halfspace and argmax cone | activation | **depth** | exact first-order, verified against full vocabulary |

**The gap this table shows is real and is worth stating in a paper: every reachable-set
construction in the LLM literature runs along the token axis. The one work along the depth axis
(A-LQR) does setpoint tracking, not sets.**

### 5.2 Entries

**Karnik & Bansal (2025). *Preemptive Detection and Steering of LLM Misalignment via Latent
Reachability.* arXiv:2509.21528.** **[primary]** Method: models autoregressive generation as a
dynamical system in latent space, learns a safety value function via backward reachability
(DeepReach-style, an MLP fit by a Bellman-type recursion), and uses it two ways: a runtime monitor
that forecasts unsafe completions several tokens ahead, and a least-restrictive steering filter
that perturbs latent states minimally. Load-bearing claim, from the abstract: BRT-Align "provides
more accurate and earlier detection of unsafe continuations than baselines" and "substantially
reduces unsafe generations while preserving sentence diversity and coherence." **Correction to the
onboarding doc: "BRT-Align" is the method name and it does appear in the paper, but the paper's
title is *Preemptive Detection and Steering of LLM Misalignment via Latent Reachability*.** Code:
none found. Relevance: nearest prior art, and the natural formalism for our first-token limit,
since extending our target set over a horizon is exactly their recursion.

**Cheng & Amo Alonso (2024, rev. 2026). *LiSeCo: Linear Semantic Control for Language Generation.*
arXiv:2405.15454. TMLR 2026; earlier version NeurIPS MINT Workshop 2024.** **[primary+]** Method:
takes concepts as linearly represented, treats generation as a trajectory in latent space, and
intervenes online on the activations of the token being generated, computing in closed form the
minimal-norm projection that brings activations into a pre-defined allowed region defined by a
probe. Gradient-free, minimal generation-time overhead. Demonstrated on toxicity, sentiment, and
English/Spanish steering, **on gemma-2-2b among others**.

> **The load-bearing item for us is Appendix C, and it is verified.** Appendix C is titled
> "Identifying the allowable region in latent space," and **Lemma C.1** states that for control in
> activation space to translate into control in attribute space, the probe must apply to **every
> reachable point** in the latent space. The paper acknowledges that this condition rarely holds
> in practice.

Relevance: this is the exact gap our result inhabits, stated as a lemma by someone else, on our
model, two years before we measured it. **It changes our framing from discovery to quantification,
and that is an improvement.** Note also the correction in §7: the author list is two people, not
three. Code: released.

**Skifstad, Yang, Chou (2026). *Local Linearity of LLMs Enables Activation Steering via
Model-Based Linear Optimal Control.* arXiv:2604.19018.** **[primary]** Method: shows empirically
that layer-wise transformer dynamics are well approximated by locally linear models across
architectures and scales, models inference as a linear time-varying system, and adapts the LQR to
compute feedback controllers from layer-wise Jacobians, steering activations toward semantic
setpoints in closed loop with no offline training. Derives bounds on setpoint tracking error.
Introduces an adaptive semantic feature setpoint signal. Load-bearing claim, from the abstract:
state-of-the-art modulation of "toxicity, truthfulness, refusal, and arbitrary concepts,
surpassing baseline steering methods," with formal guarantees on steering performance.

> **Code: PUBLIC, and this is the headline of this compilation.**
> `github.com/trustworthyrobotics/lqr-activation-steering`, linked from the paper, 94 commits,
> implements A-LQR and S-PID, with task areas covering toxicity, **truthfulness (TQA)**, concept
> steering, and **refusal**. The data-collection scripts reference `gemma2b` as the example model.
> Contact listed as Julian Skifstad, jskifstad3@gatech.edu.

Relevance: the linearization justification our whole approach inherits, and now also the
behavioral evaluation we have been treating as blocked. **Our project docs list "get access to the
A-LQR code" as an open ask; that ask is answered.**

**Soatto, Tabuada, Chaudhari, Liu (2023). *Taming AI Bots: Controllability of Neural States in
Large Language Models.* arXiv:2305.18449.** **[primary, with a caveat]** The onboarding doc asked
for its actual controllability claim to be verified, because it had not been confirmed. Verified
form: the paper introduces a formal definition of "meaning," characterizes "meaningful data" and
"well-trained LLMs" by conditions the authors argue today's models largely meet, and shows that
while the embedding space of meanings is Euclidean, **meanings themselves form a quotient space
rather than a linear subspace**. It then characterizes the subset of meanings reachable by the
LLM state for some input prompt and shows a well-trained bot **can reach any meaning, but with
small probability**. It then introduces a stronger notion, **almost-certain reachability**, and
shows that **restricted to the space of meanings, an AI bot is controllable**.

> **Three things to be careful about before citing this.** The controllability is over *prompts*
> and over an abstract *meaning* quotient space, **not over residual-stream activations**, so it
> is not a statement about activation steering at all. The result rests on definitional
> conditions the authors themselves characterize rather than verify empirically. And I could not
> confirm from the abstract page whether the claim carries a numbered theorem. **Treat it as
> conceptual framing, not as a controllability result for our setting.**

**Nguyen, Vu, Pham, Zhang, Nguyen (2025/2026). *Activation Steering with a Feedback Controller.*
arXiv:2510.04309. ICLR 2026 Poster.** **[primary]** Shows that popular steering methods correspond
to **proportional controllers** with the steering vector as the feedback signal, then proposes PID
Steering using the full controller. Relevance: the cheapest available upgrade to open-loop
steering, and it makes explicit what our single additive push is in control terms.

**Karnik, Kim, Koyejo, Lee, Bansal (2026). *Steering Away from Memorization:
Reachability-Constrained Reinforcement Learning for Text-to-Image Diffusion.* arXiv:2603.00140.**
**[primary]** Models diffusion denoising as a dynamical system, approximates the backward
reachable tube of memorized samples, and formulates mitigation as constrained RL that steers the
trajectory away via minimal perturbations in caption-embedding space. Relevance: the same group's
transfer of the BRT idea to a second modality, and the source of the compress-before-reachability
design lesson.

### 5.3 What the control literature settles and what is open

**Settled:** local linearity of layer-wise LLM dynamics is well supported and now has a
best-in-class method built on it (A-LQR) with error bounds and public code. Reachability in latent
space is a working tool along the token axis (BRT-Align, RADS). Closed-form minimal projections
into probe-defined safe regions are cheap and deployable (LiSeCo).

**Open:** no work computes a backward-reachable set along the **depth** axis with a **behaviorally
defined** target. LiSeCo's Lemma C.1 names the condition under which activation-space control
transfers to text, and nobody has measured how badly it fails. Formal verification tools
(INVPROP, PREMAP) are sound but three orders of magnitude short of transformer scale.

---

## 6. Unsupervised feature discovery

### 6.1 Mack & Turner (2024). *Deep Causal Transcoding: A Framework for Mechanistically Eliciting Latent Behaviors in Language Models.* LessWrong/AlignmentForum, 3 Dec 2024; OpenReview `gvboE2A04D` under the title *Mechanistically Eliciting Latent Behaviors in Language Models*. **[citing]**

**Method.** Models the effect of causally intervening on the residual stream of a deep (roughly
10+ layer) transformer slice with a shallow MLP, fit by a heuristic generalization of tensor
decomposition. The objective is a functional loss over the derivative tensors of the sliced
transformer, `L = sum_k (1/k!)||R^k T^(k) − That^(k)||^2`, learning matrices `V` (input steering
directions) and `U` (their downstream effects) with `m` factors, typically **m = 512**, under a
choice of activation `sigma` from linear, quadratic, exponential:
`Deltahat(theta) = sum_l alpha_l · sigma(<vhat_l, theta>) · uhat_l`. Exponential DCTs generalize
best. Models used in the original work: Qwen-1.5-7B-Chat, Qwen-1.5-32B-Chat, Mistral-7B-Instruct-v2,
and a password-locked Deepseek-Math-7B.

**Load-bearing claim.** Extreme data efficiency: a large number of interpretable features from a
single prompt, with input directions serving as steering vectors and output directions inducing
predictable changes via directional ablation. The authors report more than 200 linearly
independent jailbreak-inducing directions.

**How it validates, and the gap.** **The post reports no systematic failure analysis.** There is
no quantitative assessment of how many of the 512 discovered directions lack interpretability or
behavioral validity; the authors note only a subjective impression that projected linear DCTs miss
qualitatively interesting vectors. **Relevance: our 0/10 judged interpretation rate on the top
vectors by potency is, as far as this compilation can tell, the first quantitative negative
denominator anyone has attached to DCT output.** That is worth stating plainly, and it is a
contribution independent of the reachability work.

**Confidence caveat.** Marked **[citing]** because OpenReview was behind a browser-verification
wall during compilation and the LessWrong mirror was the readable source. **Re-verify the loss
formula against the primary before it appears in a paper.**

### 6.2 Lieberum, Rajamanoharan, Conmy, Smith, Sonnerat, Varma, Kramár, Dragan, Shah, Nanda (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* arXiv:2408.05147. **[primary]**

**Method.** An open suite of JumpReLU SAEs trained on all layers and sublayers of Gemma 2 2B and
9B and select layers of 27B, evaluated on standard metrics, with weights and tutorial released.

**Relevance.** The dictionary we use. **The critical operational caveat is separate: GemmaScope
trains a separate dictionary per layer, so a feature id at layer 11 and the same id at layer 20
are unrelated atoms.** Our current SAE files decompose the readout and the steering vectors at
different layers, which is why the SAE target-set question is not answerable from them.

### 6.3 Mayne, Yang, Mahdi (2024). *Can sparse autoencoders be used to decompose and interpret steering vectors?* arXiv:2411.08790. **[primary]**

**Method.** Investigates why applying SAEs directly to steering vectors yields misleading
decompositions, identifying two causes.

**Load-bearing claim.** Abstract: "(1) steering vectors fall outside the input distribution for
which SAEs are designed, and (2) steering vectors can have meaningful negative projections in
feature directions, which SAEs are not designed to accommodate."

**Relevance.** The justification for our methodological choice to run **orthogonal matching
pursuit against the decoder** rather than the SAE encoder. Anyone reviewing our SAE forensics will
ask why we did not just encode; this is the answer, and it should be cited at the point of use.

### 6.4 New unsupervised methods, 2026

| Work | Method | Validation | Conf |
|---|---|---|---|
| Rosser (2026), *Gradient Atoms*, arXiv:2603.14665 | Decomposes per-document **training gradients** into sparse atoms by dictionary learning in a preconditioned eigenspace; 500 atoms | Highest-coherence atoms recover refusal, arithmetic, yes/no, trivia QA with no labels; atoms double as weight-space steering vectors (bulleted-list 33%→94%, refusal 50%→0%). **Failure rate over the 500 atoms is not reported in the abstract** | [primary] |
| Cho, Roh, Kim (2026), *Shared Semantics, Divergent Mechanisms*, arXiv:2606.08236, ICML 2026 Spotlight | Clusters sampled continuations using both semantic embeddings and prefix-to-continuation attribution signatures, optimizing a rate-distortion objective | Clustering analyses plus steering interventions providing "interventional evidence that cluster signatures correspond to actionable mechanistic factors" | [primary] |

### 6.5 What §6 settles and what is open

**Settled:** unsupervised discovery produces large numbers of directions that change behavior, and
does so cheaply. SAEs are not competitive for steering (AxBench), and naive SAE encoding of
steering vectors is the wrong operation (Mayne et al.).

**Open, and this is the field-level gap our project sits in: none of these methods validates a
discovered direction without reintroducing supervision or a human read at the end.** DCT reports
no failure denominator. Gradient Atoms reports headline shifts for its best atoms. The 2026
clustering work validates by steering, judged externally. **Nobody publishes the fraction of
discovered directions that do nothing.** Answering open question 3 well means proposing a
selection criterion computable *before* the behavioral test, which is what a controllability
margin or `A_lin` offers and what downstream-change magnitude does not.

---

## 7. Where the literature contradicts or corrects our documents

Stated explicitly rather than smoothed over, as requested.

| # | Our documents say | The primary source says | Severity |
|---|---|---|---|
| 1 | "Get access to the A-LQR code" is a standing open ask (`PI_QUESTIONS_ANSWERED.md` F9, `RESEARCH_ROADMAP.md` Horizon 2) | The code is **public** at `github.com/trustworthyrobotics/lqr-activation-steering`, linked from arXiv:2604.19018, and covers truthfulness and refusal on gemma-2b | **High.** Rewrite the ask and the roadmap item |
| 2 | LiSeCo authors are "Cheng, Baroni & Amo Alonso" (onboarding Tier 3) | arXiv:2405.15454 lists **Emily Cheng, Carmen Amo Alonso**. Venue is **TMLR 2026**, with an earlier NeurIPS MINT Workshop 2024 version | Low, but fix the citation |
| 3 | Karnik & Bansal's paper is "BRT-Align" (onboarding Tier 3) | BRT-Align is the **method** name; the **title** is *Preemptive Detection and Steering of LLM Misalignment via Latent Reachability* | Low |
| 4 | INVPROP is "Kotha et al., NeurIPS 2023" (onboarding Tier 3) | Correct. Full list: Kotha, Brix, Kolter, Dvijotham, Zhang. Title *Provably Bounding Neural Network Preimages*, **Spotlight** | None; recorded for completeness |
| 5 | arXiv:1910.13272 is a reachability reference (project docs, flagged as unconfirmed) | **Confirmed mismatch.** That ID is Westenbroek, Fridovich-Keil, Mazumdar, Arora, Prabhu, Sastry, Tomlin, *Feedback Linearization for Unknown Systems via Reinforcement Learning*, math.OC. **Feedback linearization, not reachability** | Medium. The PI still needs to confirm what was meant |
| 6 | Our headline is a decodable-but-not-steerable dissociation | Nadaf 2026 (arXiv:2604.02608) reports "decodable without steerable" is **nearly empty (3 of 72)** | **High as a framing risk.** The two use different definitions of decodable; see §1.3. Address it preemptively |
| 7 | The readout-versus-behavior dissociation "appears to be unclaimed in the literature" (`PROJECT_PROGRESS_TO_DATE.md` §0) | LiSeCo **Lemma C.1** names the sufficient condition and says it rarely holds in practice | **Medium.** The *observation* is anticipated; the *measurement and mechanism* still appear unclaimed. Narrow the claim |
| 8 | Incoherence under steering attributed to over-steering / budget | Kang et al. 2026 (arXiv:2605.10664) identify **KV-cache contamination**: steered states are cached and reused, turning a local perturbation into cumulative degradation | Medium. A named alternative mechanism we have not ruled out |

---

## 8. References

Alphabetized by first author. Confidence tag as defined in the key.

1. Abdullaev, L. U., Wong, N. Y. L., Lee, R. T. Z., Jiang, S., Nguyen, K. N. M., Nguyen, T. M. (2026). *Concept Heterogeneity-aware Representation Steering.* arXiv:2603.02237. ICML 2026 (PMLR 306). **[citing]**
2. Aparin, G., Gaintseva, T. (2026). *A Geometric Account of Activation Steering through Angle-Norm Decomposition.* arXiv:2606.06735. **[primary]**
3. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717. **[primary]**
4. Barbero, F., Arroyo, Á., Gu, X., Perivolaropoulos, C., Bronstein, M., Veličković, P., Pascanu, R. (2025). *Why do LLMs attend to the first token?* arXiv:2504.02732. **[primary]**
5. Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., Steinhardt, J. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112. **[primary]**
6. Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., Biderman, S. (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form.* arXiv:2306.03819. **[primary]**
7. Billa, J. (2026). *Predicting Where Steering Vectors Succeed.* arXiv:2604.15557. **[primary]**
8. Brett, M. (2026). *Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models.* arXiv:2604.06767. **[primary]**
9. Cheng, E., Amo Alonso, C. (2024, rev. 2026). *LiSeCo: Linear Semantic Control for Language Generation.* arXiv:2405.15454. TMLR 2026. **[primary+]**
10. Cheng, F. L., Kriegeskorte, N. (2026). *Decomposing how prompting steers behavior.* arXiv:2606.03093. **[citing]**
11. Cho, H., Roh, Y., Kim, J. (2026). *Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms.* arXiv:2606.08236. ICML 2026 Spotlight. **[primary]**
12. Demeter, D., Kimmel, G., Downey, D. (2020). *Stolen Probability: A Structural Weakness of Neural Language Models.* arXiv:2005.02433. ACL 2020. **[primary]**
13. Elazar, Y., Ravfogel, S., Jacovi, A., Goldberg, Y. (2020/2021). *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals.* arXiv:2006.00995. TACL. **[primary]**
14. Fan, C., Cheng, Y., Li, M., Feizi, S., Zhou, T. (2026). *When is Your LLM Steerable?* arXiv:2606.11599. **[primary]**
15. Góral, G., Winkels, M., Basart, S. (2025). *Depth-Wise Activation Steering for Honest Language Models.* arXiv:2512.07667. **[primary]**
16. Grivas, A., Bogoychev, N., Lopez, A. (2022). *Low-Rank Softmax Can Have Unargmaxable Classes in Theory but Rarely in Practice.* arXiv:2203.06462. ACL 2022. **[primary]**
17. He, H., et al., Thinking Machines Lab (2025). *Defeating Nondeterminism in LLM Inference.* Blog, 10 Sept 2025. **[primary]**
18. Kang, D., Liu, Z., Ma, N., Huang, Y., Tan, Z., Jiang, M. (2026). *Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions.* arXiv:2605.10664. **[primary]**
19. Karnik, S., Bansal, S. (2025). *Preemptive Detection and Steering of LLM Misalignment via Latent Reachability* (method: BRT-Align). arXiv:2509.21528. **[primary]**
20. Karnik, S., Kim, J., Koyejo, S., Lee, J.-S., Bansal, S. (2026). *Steering Away from Memorization: Reachability-Constrained Reinforcement Learning for Text-to-Image Diffusion* (method: RADS). arXiv:2603.00140. **[primary]**
21. Kotha, S., Brix, C., Kolter, Z., Dvijotham, K., Zhang, H. (2023). *Provably Bounding Neural Network Preimages* (algorithm: INVPROP). arXiv:2302.01404. NeurIPS 2023 Spotlight. **[primary]**
22. Labiosa, N., Buff, D., Nayak, E., Donno, E. (2026). *Sensitivity, Causality, and Repair Dissociate: A Layer-Wise Analysis of Perturbation Robustness and Its Scaling.* arXiv:2608.03842. **[primary]**
23. Li, J., Li, Y., Huang, K.-H. (2026). *Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models.* arXiv:2602.01654. **[primary]**
24. Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* arXiv:2306.03341. NeurIPS 2023 Spotlight. **[primary]**
25. Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., Nanda, N. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* arXiv:2408.05147. **[primary]**
26. Mack, A., Turner, A. M. (2024). *Deep Causal Transcoding: A Framework for Mechanistically Eliciting Latent Behaviors in Language Models.* LessWrong/AlignmentForum, 3 Dec 2024. OpenReview `gvboE2A04D` (as *Mechanistically Eliciting Latent Behaviors in Language Models*). **[citing]**
27. Marks, S., Tegmark, M. (2023). *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* arXiv:2310.06824. COLM 2024. **[primary]**
28. Mayne, H., Yang, Y., Mahdi, A. (2024). *Can sparse autoencoders be used to decompose and interpret steering vectors?* arXiv:2411.08790. **[primary]**
29. Mishra, A., Khashabi, D., Liu, A. (2026). *Steered LLM Activations are Non-Surjective.* arXiv:2604.09839. ICLR 2026 Workshops. **[primary]**
30. Nadaf, M. S. B. (2026). *Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens.* arXiv:2604.02608. **[primary]**
31. Nguyen, D. V., Vu, H. M., Pham, N. Y., Zhang, L., Nguyen, T. M. (2025/2026). *Activation Steering with a Feedback Controller.* arXiv:2510.04309. ICLR 2026 Poster. **[primary]**
32. Queipo-de-Llano, E., Arroyo, Á., Barbero, F., Dong, X., Bronstein, M., LeCun, Y., Shwartz-Ziv, R. (2025/2026). *Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin.* arXiv:2510.06477. **[primary]**
33. Rosser, J. (2026). *Gradient Atoms: Unsupervised Discovery, Attribution and Steering of Model Behaviors via Sparse Decomposition of Training Gradients.* arXiv:2603.14665. **[primary]**
34. Skifstad, J., Yang, X. A., Chou, G. (2026). *Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control* (method: A-LQR). arXiv:2604.19018. Code: `github.com/trustworthyrobotics/lqr-activation-steering`. **[primary]**
35. Soatto, S., Tabuada, P., Chaudhari, P., Liu, T. Y. (2023). *Taming AI Bots: Controllability of Neural States in Large Language Models.* arXiv:2305.18449. **[primary, scope caveat in §5.2]**
36. Tan, D., Chanin, D., Lynch, A., Kanoulas, D., Paige, B., Garriga-Alonso, A., Kirk, R. (2024). *Analyzing the Generalization and Reliability of Steering Vectors.* arXiv:2407.12404. NeurIPS 2024. **[primary]**
37. Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. (2023). *Steering Language Models With Activation Engineering* (method: ActAdd). arXiv:2308.10248. **[primary]**
38. Vu, H. M., Nguyen, T. M. (2025). *Angular Steering: Behavior Control via Rotation in Activation Space.* arXiv:2510.26243. NeurIPS 2025 Spotlight. **[primary]**
39. Wenkmann, J., Garreau, D. (2025). *On The Variability of Concept Activation Vectors.* arXiv:2509.24058. Submitted to AAAI-26. **[primary]**
40. Westenbroek, T., Fridovich-Keil, D., Mazumdar, E., Arora, S., Prabhu, V., Sastry, S. S., Tomlin, C. J. (2019/2020). *Feedback Linearization for Unknown Systems via Reinforcement Learning.* arXiv:1910.13272. **[primary]** *Listed only to record that this ID does not match the description it was given in our notes.*
41. Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., Potts, C. (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* arXiv:2501.17148. ICML 2025. **[primary]**
42. Yap, J. Q. (2026). *Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors: One Agency Axis, Not Five Traits.* arXiv:2603.16335. **[primary]**
43. Ye, J., Ran, L., Yao, Z., Wang, C., Jiang, Y., Hou, L., Li, J., Pan, L. (2026). *Where Steering Signals Come From: Activation Source Selection in Activation Steering.* arXiv:2607.25270. **[primary]**
44. Zhang, X., Wang, B., Kwiatkowska, M., Zhang, H. (2024/2025). *PREMAP: A Unifying PREiMage APproximation Framework for Neural Networks.* arXiv:2408.09262. JMLR v26 (2025). **[primary]**
45. Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., Hendrycks, D. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405. **[primary]**

### Not verified, and therefore not summarized

- **A "Verify Before You Conclude" intervention-validity item** surfaced in search on a
  non-archival site (`iamhumanityfirst.com`, July 2026) reporting a truthfulness ablation that
  converted 0 of 180 instructed lies and moved 3 of 720 answers. The numbers are strikingly close
  to ours, but the source has no discoverable peer review or archival identifier and shows signs
  of automated generation. **Flagged, not cited. Do not use it as corroboration.**
- **Several arXiv identifiers surfaced in search snippets** for angular steering, crystallization
  gaps, and unsupervised jailbreak simulation whose abstract pages I did not open. They are
  deliberately absent from the reference list rather than listed on the strength of a snippet.
