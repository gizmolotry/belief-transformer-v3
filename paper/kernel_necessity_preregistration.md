# Kernel Necessity Preregistration

Date frozen: 2026-08-21

Status: pre-run protocol. No real kernel-necessity experiment may be run, interpreted, or used in paper text until the provenance tests named here pass and this preregistration is committed in git.

## 1. Narrow Claim Boundary

This protocol tests one narrow claim:

> On a fresh, held-out, human-labeled real-article framing task, this protocol can support or fail to support kernel necessity under the frozen tested arms and endpoints. It cannot establish metaphysical or global necessity.

This is not a test of whether the whole Belief Transformer thesis is true, whether real-world ideology has been solved, or whether any earlier synthetic Matern result should be promoted. Existing repository evidence motivates the test but does not count as evidence for this preregistered claim.

Local rationale:

- [METHODS.md](../METHODS.md), Claim Tiering, states that real unlabeled corpus runs can support geometry/control/provenance claims, but not label-recovery claims without an independent validation target.
- [RESULTS.md](../RESULTS.md), Table 0B, records that simple PCA baselines are serious reviewer baselines and that the original local recompute path is not universally superior.
- [paper/observer_recentering_robustness_brief_20260528.md](observer_recentering_robustness_brief_20260528.md), Kernel-Level Finding, reports a Matern-specific controlled synthetic pattern, but frames it as conditional and not universal.
- [notes/REAL_CORPUS_CLAIM_BOUNDARY_20260528.md](../notes/REAL_CORPUS_CLAIM_BOUNDARY_20260528.md) states that real-corpus ideological label accuracy requires an independent label target.
- [results/exp_B_manifest.json](../results/exp_B_manifest.json), labels block, shows a prior label evaluation was skipped because the expected 60 rows had only 30 observed rows. Existing human label coverage is therefore not sufficient for this test.

Label acquisition is an unmet prerequisite. This protocol freezes how the test will be run after fresh labels exist; it does not assert that those labels already exist.

## 2. Data, Labels, and Eligibility

Primary data must be a fresh held-out real-article set not used to choose this protocol, tune the kernel arms, write the label codebook, debug kernel provenance, or select thresholds. The unit of analysis is the article.

Primary labels must be fresh held-out human article-level framing labels:

- Minimum label-complete sample size: `N >= 120` articles.
- Raters: at least 3 independent blinded human raters per article.
- Codebook: fixed before rating begins, including frame categories, inclusion/exclusion examples, and tie rules.
- Rater blinding: raters must not see model outputs, kernel arm names, source/outlet labels, MBFC labels, LLM labels, or previous experiment results.
- Consensus policy: majority label when at least 2 of 3 raters agree. If more than 3 raters are used, use plurality only when the winning category has at least 50% support; otherwise adjudicate by a blinded fourth rater using the frozen codebook.
- Reliability statistic and gate: nominal Krippendorff's alpha, computed on the raw pre-adjudication rater-by-article label matrix, must be `>= 0.67`. No equivalent or substitute statistic may satisfy this gate.
- Missing-rating handling is frozen as follows: an absent, withdrawn, illegible, or otherwise unusable rating is encoded as missing, never as a label category; missing values are not imputed; nominal Krippendorff's alpha uses its standard available-pair treatment; an article with fewer than two nonmissing ratings contributes no observed-disagreement pair; and counts of missing ratings by rater and article are reported before adjudication. This rule may not be changed after rating begins.
- Any other inter-rater statistic (including Fleiss' kappa, Cohen's kappa, or an ICC) is descriptive only and cannot satisfy, replace, or rescue the nominal Krippendorff's-alpha gate.
- Usable consensus gate: at least 80% of rated articles must have usable consensus labels.

If `N < 120`, reliability `< 0.67`, or usable consensus `< 80%`, the inferential run is `invalid`. The data may be reported only as a failed feasibility attempt.

LLM labels, MBFC/source labels, publisher labels, and other proxy labels are secondary only. They may be used for audit, sensitivity analysis, or comparison with human consensus. They cannot rescue a failed or null primary human-label result.

## 3. Matched Arms

All arms use the identical articles, article IDs, text preprocessing, CLS extraction, centering rule, split definitions, seeds, bandwidth source, clustering algorithm, and metric code. The insertion point is after article-level CLS embeddings have been computed and centered.

The preregistered arms are:

1. `raw_centered_cls`: centered CLS geometry with no learned, random, or kernel transform.
2. `fixed_rank_pca`: fixed-rank PCA fit only on the split's included articles.
3. `matched_random_linear`: random linear projection with the same internal feature budget and seeds as the RKS arms.
4. `matched_random_cosine_nonkernel`: random cosine features with the same internal feature budget and seeds as RKS, but not sampled from or claimed to approximate any stationary kernel.
5. `exact_rbf`: exact RBF affinity matrix.
6. `rks_rbf`: Random Kitchen Sinks approximation to the RBF kernel.
7. `rks_matern_1p5`: Random Kitchen Sinks approximation to the Matern kernel with `nu = 1.5`.
8. `rks_imq`: Random Kitchen Sinks approximation to the inverse multiquadratic kernel.

For inference, every arm must output the same object type: one article-by-article affinity matrix of shape `N_split x N_split` for each fixed cell. Coordinate-producing arms are converted to an affinity matrix by the same frozen evaluator. RKS arms may use an internal random feature dimension, but the inferential output is the affinity matrix, not a higher-dimensional feature advantage.

Default fixed values, unless the committed run config explicitly records stricter values before execution:

- Seeds: `42`, `420`, `4200`, matching the canonical seed pattern in [METHODS.md](../METHODS.md), Canonical Configuration.
- Splits: 5 deterministic article-level evaluation splits generated from sorted article IDs and saved before arm execution.
- Internal random feature budget for random-linear, random-cosine, and RKS arms: `2048`, matching existing RKS configuration scale in [results/exp_B_manifest.json](../results/exp_B_manifest.json), config block.
- PCA rank for the PCA arm: `min(64, N_split - 1)`, fixed before labels are evaluated.
- Bandwidth selection for kernel arms: median pairwise Euclidean distance on centered CLS embeddings inside the same split. The selected bandwidth is shared across `exact_rbf`, `rks_rbf`, `rks_matern_1p5`, and `rks_imq` for that split and seed. No bandwidth is tuned to NMI, ARI, MBFC, source labels, LLM labels, or human labels.

The strongest non-kernel comparator is selected mechanically after all non-kernel arms are evaluated on the primary human labels: choose the non-kernel arm among `raw_centered_cls`, `fixed_rank_pca`, `matched_random_linear`, and `matched_random_cosine_nonkernel` with the highest mean NMI across all fixed cells. Ties are broken in this order: `raw_centered_cls`, `fixed_rank_pca`, `matched_random_linear`, `matched_random_cosine_nonkernel`.

## 4. Hard Pre-Run Provenance Gate

The provenance gate must pass before the inferential run starts. Failure invalidates the entire kernel-necessity run; it cannot be repaired by excluding an arm after seeing results.

Required provenance tests:

- Analytic toy tests: for fixed toy points and parameters, the implemented analytic kernels must match closed-form values for RBF, Matern `nu=1.5`, and IMQ within a committed tolerance. Example required checks at `sigma = 1`, `||x-y|| = 1`: RBF `exp(-1/2)`, Matern `nu=1.5` `(1 + sqrt(3)) * exp(-sqrt(3))`, and IMQ `1 / sqrt(2)`.
- Requested family equals actual family: every arm manifest must record requested family, actual family, sampler name, sampler parameterization, seed, bandwidth, internal dimension, basis hash, and code path. Requested `rks_matern_1p5` cannot be backed by an RBF, generic Gaussian, fallback, or mislabeled Student-t sampler.
- RKS-to-exact approximation smoke tests: each RKS family must be tested against its own exact analytic kernel on both a fixed toy grid and a fixed deterministic random grid. These unit-test grids detect gross implementation errors but cannot satisfy or substitute for the runtime fidelity gate below.
- Runtime RKS-to-exact implementation-fidelity gate: for every RKS family and fixed inferential cell, compare the RKS affinity matrix with that family's exact analytic affinity matrix on the actual frozen centered-CLS representation, article split, seed, bandwidth, sampler parameters, and internal feature dimension `D = 2048`. On that cell, the Pearson correlation of finite upper-triangle affinity entries must be `>= 0.95`; MAE and RMSE must also be recorded. A cell below `0.95`, or with undefined/nonfinite diagnostics, invalidates the run before human-label endpoints are inspected. This correlation is an operational implementation-fidelity screen, not an inferential endpoint, an effect-size criterion, or a theoretical consequence of random-feature theory.
- Family-correct diagnostics: a diagnostic must never compare Matern, IMQ, or any other requested family against RBF as a fallback truth. Every diagnostic reference kernel must be the exact analytic kernel for the requested family.
- Validated path provenance: every active direct or injected basis path must carry and validate requested family, actual family, sampler name and version, sampler parameterization, seed, bandwidth, internal dimension, basis hash, and code path. Missing, inconsistent, inferred, or silently substituted provenance invalidates the run.
- Canonical agreement: all active construction, injection, diagnostic, persistence, and restore paths must use the canonical kernel and sampler definitions for the requested family and must agree on family identity, parameterization, and generated-basis identity.
- Persisted basis integrity and strict restore provenance: the tensor hash of every persisted basis, covering all basis tensors together with their shapes, dtypes, and byte order, must be stored in the manifest. Restore must recompute that hash from the loaded tensors and require an exact match, and must validate the persisted requested family, actual family, sampler name and version, sampler parameterization, seed, bandwidth, internal dimension, and code path against the current run request. Regeneration, substitution, fallback, missing restore metadata, or any mismatch invalidates the run.
- Nonfinite guard: any NaN, Inf, nonpositive bandwidth, zero-variance non-control feature collapse, shape mismatch, missing metadata field, missing basis hash, missing article ID, duplicate article ID, or implicit fallback makes the run `invalid`.

The gate is pass/fail. If it fails, allowed language is: "The kernel-necessity experiment was not run because kernel provenance failed the preregistered pre-run gate."

Random Fourier feature theory motivates checking approximation quality at the frozen feature budget because approximation error generally decreases on the order of `D^(-1/2)` under the relevant assumptions; it does not derive or guarantee the operational `0.95` correlation cutoff used here. See Rahimi and Recht (2007), [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html), and Sutherland and Schneider (2015), [On the Error of Random Fourier Features](https://proceedings.mlr.press/v37/sutherland15.html). The cutoff is therefore frozen as a conservative project-level quality-control rule and must be evaluated at the actual `D = 2048` runtime configuration rather than justified retrospectively from observed performance.

## 5. Primary Endpoint

The primary endpoint is mean paired delta NMI:

`delta_nmi(kernel_arm) = NMI(kernel_arm) - NMI(strongest_non_kernel_arm)`

Pairing is by fixed cell: article split, model seed, bandwidth selection, and clustering seed. A kernel arm can support necessity only if all of the following are true:

- Holm-adjusted one-sided permutation result passes for the relevant directional contrast in the single preregistered family.
- Mean paired `delta_nmi >= +0.05`.
- The lower bound of the paired bootstrap 95% confidence interval for `delta_nmi` is `> 0`.
- At least 80% of fixed cells have `delta_nmi > 0`.
- A shuffled-label null, run through the same pipeline including strongest-non-kernel selection, cannot pass the same decision rule.

Delta ARI is secondary. It will be reported descriptively and with the same pairing, but it is not co-primary because adding it as a co-primary endpoint would inflate multiplicity and change the meaning of the test.

## 6. Controlled Synthetic Recovery Diagnostic

Component-level recovery on a controlled synthetic corpus is retained as a secondary, non-inferential diagnostic. It is run and reported separately from the held-out human-label analysis, with distinct run IDs, manifests, output tables, and result headings. For every arm and fixed synthetic cell, report both NMI and ARI against the known generating-component labels, together with the generator configuration, component parameters, seeds, and sample size.

Synthetic recovery cannot enter the primary endpoint, strongest-non-kernel selection, Holm family, bootstrap interval, TOST decision, label-reliability gate, or final decision tree. It cannot rescue, alter, qualify upward, or reverse an `invalid`, `fail`, or `inconclusive` primary human-label result. A positive synthetic diagnostic supports only the statement that a method recovered known components under the frozen controlled generator; it is not evidence that kernelization is necessary for real-article framing recovery.

## 7. Multiplicity and Permutation Testing

There is exactly one Holm-Bonferroni family at `FWER = 0.05`.

The five directional permutation contrasts are:

1. `exact_rbf > strongest_non_kernel`
2. `rks_matern_1p5 > strongest_non_kernel`
3. `rks_imq > strongest_non_kernel`
4. `rks_matern_1p5 > exact_rbf`
5. `rks_matern_1p5 > rks_imq`

The `rks_rbf` arm is included to verify that the RKS approximation behaves like exact RBF and to diagnose approximation error. It is not an additional primary superiority contrast; adding it would create a sixth family member without changing the scientific question.

Permutation procedure:

- Use the fixed human consensus labels as the primary labels.
- For each permutation, shuffle labels at the article level within the same split.
- Recompute NMI for all arms under the permuted labels.
- Recompute the strongest non-kernel arm inside that permutation using the same mechanical rule.
- Compute the paired deltas and directional contrast statistics.
- Use at least 10,000 permutations or all exact permutations if fewer are possible.

Holm decision procedure:

- Sort the five raw permutation p-values from smallest to largest.
- Compare the smallest p-value to `0.05 / 5 = 0.010`.
- If it passes, compare the next p-value to `0.05 / 4 = 0.0125`.
- Continue with thresholds `0.0167`, `0.025`, and `0.05`.
- Once a p-value fails, that contrast and all remaining larger p-values are not rejected.

Practical thresholds, confidence intervals, fixed-cell positivity, and shuffled-null failure are required in addition to Holm rejection.

## 8. Equivalence Test If Superiority Fails

TOST equivalence is evaluated only if superiority fails for all kernel-vs-strongest-non-kernel contrasts.

Equivalence bounds are `delta_nmi in [-0.02, +0.02]`. This interval is chosen because the repository already treats effects around `+0.05` NMI as meaningful enough to discuss in synthetic robustness summaries, while smaller differences near two NMI points are not practically strong evidence that kernelization is needed. For example, [RESULTS.md](../RESULTS.md), Table 0B, reports mean NMI gains that distinguish serious baseline behavior from stronger candidates, and [final_thesis_metrics.md](../final_thesis_metrics.md), Synthetic Recoverability, reports per-kernel mean NMI differences on the order of a few hundredths. This protocol therefore treats `+0.05` as the minimum practical superiority threshold and `[-0.02, +0.02]` as the practical equivalence band.

Equivalence decision:

- Compute paired `delta_nmi` between the best kernel arm and the strongest non-kernel arm.
- Use a 90% paired bootstrap confidence interval.
- If the entire 90% CI lies inside `[-0.02, +0.02]`, conclude equivalence for the primary claim: kernelization was not necessary under this protocol.
- If superiority fails and equivalence also fails, conclude `inconclusive`.

## 9. Partial-Pass Decision Tree and Allowed Language

Final decision is assigned in this order:

1. `invalid`: any provenance gate failure, label eligibility failure, missing manifest field, uncommitted code state, nonfinite value, fallback, article-ID mismatch, or undeclared analysis change.
2. `pass`: at least one kernel-vs-strongest-non-kernel primary contrast passes Holm, practical threshold, CI, fixed-cell positivity, and shuffled-null checks.
3. `fail`: superiority fails and TOST equivalence passes.
4. `inconclusive`: superiority fails and TOST equivalence fails.

Allowed language:

- `invalid`: "The preregistered kernel-necessity experiment is invalid because a pre-run or manifest requirement failed. No inference about kernel necessity is made."
- `pass`: "Under the preregistered human-label protocol, kernelization was necessary for the tested framing-recovery endpoint: at least one kernel arm beat the strongest matched non-kernel arm by the required statistical and practical margins."
- `fail`: "Under the preregistered human-label protocol, kernelization was not shown to be necessary: the best kernel arm was practically equivalent to the strongest matched non-kernel arm within the preregistered NMI bounds."
- `inconclusive`: "Under the preregistered human-label protocol, the result is inconclusive: kernel superiority did not pass, but equivalence was also not established."

Disallowed language:

- "The kernel works" when only one kernel family passes.
- "Matern proves the theory" based on synthetic or proxy labels.
- "MBFC validates ideology recovery" unless MBFC was separately validated for this exact article-level claim.
- "LLM labels confirm the primary result" when human labels fail, are underpowered, or are unreliable.
- "Kernelization is necessary" if the strongest non-kernel arm is omitted, changed after seeing results, or not selected by the mechanical rule.

## 10. Fail-Closed Manifest Requirements

The run manifest must contain enough information for a reviewer to reproduce the exact inferential packet without reading lab notes.

Required fields:

- `kernel_necessity`: one of `pass`, `fail`, `inconclusive`, `invalid`.
- `preregistration_file`: `paper/kernel_necessity_preregistration.md`.
- `preregistration_commit`: clean git commit hash containing this file before any inferential run starts.
- `run_commit`: clean git commit hash for the code used to run the experiment.
- `git_clean_at_run_start`: boolean, must be `true`.
- `command`: exact command line.
- `run_started_at`, `run_finished_at`, and machine/environment summary.
- `config_hash`, full config path, and serialized config.
- `article_ids`, `article_id_hash`, text source paths, and exclusion list.
- `label_file`, `label_file_hash`, rater IDs or blinded pseudonyms, the raw pre-adjudication rater-by-article label-matrix hash, consensus policy, the frozen missing-rating policy and missingness counts by rater and article, `reliability_statistic = "nominal_krippendorff_alpha"`, nominal Krippendorff's-alpha value, any separately identified descriptive reliability statistics, usable consensus count, and label-complete `N`.
- `arm_ids`, arm configs, requested family, actual family, sampler provenance, basis hash, bandwidth, seed, split ID, internal dimension, and output affinity hash for every arm and fixed cell.
- `provenance_gate_status`, analytic toy test results, toy/random-grid RKS smoke-test diagnostics, nonfinite scan, metadata completeness scan, and fallback scan.
- Runtime RKS-to-exact fidelity records for every RKS family and fixed inferential cell: centered-CLS representation hash, article split and article-ID hash, seed, bandwidth, sampler parameters, `D = 2048`, exact-affinity hash, approximate-affinity hash, upper-triangle correlation, MAE, RMSE, and pass/fail status against the operational `0.95` threshold.
- `strongest_non_kernel_arm` and its mechanical-selection evidence.
- Raw NMI and ARI by arm and fixed cell.
- Paired deltas, bootstrap intervals, permutation p-values, Holm ordering, Holm thresholds, TOST result if evaluated, shuffled-null results, and final allowed-language string.
- Controlled synthetic diagnostic records, stored separately from primary outputs: distinct synthetic run IDs and table paths, generator/config hashes, known component labels and label hash, component parameters, seeds, sample size, raw NMI and ARI by arm and fixed cell, and an explicit `primary_decision_eligible = false` marker.

Any missing required manifest field sets `kernel_necessity = "invalid"`.

## 11. Run Permission

Before any real inferential experiment is launched:

1. The kernel provenance tests must pass.
2. This preregistration must be committed.
3. The worktree must be clean.
4. The fresh human labels must satisfy the primary-label eligibility gates.
5. The exact run command and config must be recorded in the manifest before metrics are interpreted.

Until all five conditions are true, the only permitted project language is: "Kernel necessity remains untested under the preregistered human-label protocol."
