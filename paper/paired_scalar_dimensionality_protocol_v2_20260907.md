# Paired-Scalar Dimensionality Selection: Corrected V2 Protocol

**Frozen:** 2026-09-07, before implementation or real-data execution of v2

## Status

This is a post-review corrected exploratory reanalysis of already inspected data and spectra. It
is not a preregistration, confirmation, or independent replication. Q1 paraphrases Q0's authored
dialectics, so agreement can show phrasing stability only. No component will be named and no
manuscript file will be edited by this run.

V1 is preserved as a superseded audit artifact. Its grouped held-out reconstruction criterion was
defective for model selection: ordinary PCA projection is nested, so adding components
mechanically reduces squared reconstruction error without paying a likelihood or complexity cost.
Its `supports_no_more_than_one` branch also converted non-detection into an invalid absence claim.
V2 replaces that criterion and vocabulary; it does not rewrite v1.

## Measurement and frozen inputs

The only measurement is the corrected paired scalar `c = s_A - s_B`, with
`s = P(entailment) - P(contradiction)` computed from the existing stored paragraph-aggregated
logits. Q0 and Q1 are analyzed separately. No model inference is performed.

V2 requires and hash-binds:

- `outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl`
- `data/controlled_N_dense_corpus.jsonl`
- `config/framing_queries.yaml`
- `config/prompt_banks/probe_hypotheses_paraphrase_a.json`
- `paper/paired_scalar_structure_results_20260906.json`
- `paper/paired_scalar_dimensionality_protocol_20260907.md` (v1)
- `paper/paired_scalar_dimensionality_results_20260907.json` (v1)
- this v2 protocol;
- the v2 runner;
- the imported v1 runner; and
- the paired-scalar scoring runner.

All hashes are computed before and after analysis. Any mutation aborts output. The result is
created exclusively and refuses overwrite. The expected design is exactly 320 IDs in identical
checkpoint/corpus order and balanced 16-publisher by 20-article blocks.

## Determinism

- Root seed: `20260907`.
- `numpy.random.SeedSequence.spawn` creates isolated streams, in this fixed order: Q0 primary
  publisher-block null, Q1 primary publisher-block null, Q0 article-level sensitivity null, Q1
  article-level sensitivity null, Q0 publisher bootstrap, Q1 publisher bootstrap.
- Primary and sensitivity nulls each use 10,000 draws per bank.
- Publisher-cluster bootstraps use 10,000 draws per bank.
- Leave-one-publisher-out PPCA folds are deterministic and require no random stream.

## Analysis 1: leave-one-publisher-out PPCA predictive likelihood

There are exactly 16 folds. Each fold trains on all 300 articles from 15 publishers and evaluates
the 20 articles from the one held-out publisher. Publisher is the evaluation unit. No held-out
value enters the training mean, covariance, eigendecomposition, noise estimate, or numeric floor.
Columns are centered by the training mean and are not standardized.

For each candidate `k` in `{1,2,3}`, calculate the training maximum-likelihood covariance
`S = X_centered^T X_centered / n_train`, with descending eigenvalues `lambda_i` and eigenvectors
`U`. Define

`raw_sigma_k^2 = mean(lambda_{k+1}, ..., lambda_8)`

and the standard closed-form PPCA covariance

`C_k = U_k diag(lambda_1, ..., lambda_k) U_k^T + sigma_k^2 (I - U_k U_k^T)`.

The numerical covariance-unit floor is

`f = max(lambda_1 * 1e-12, eps_float64 * max(lambda_1, 1))`.

Use `max(raw_sigma_k^2, f)` for `sigma_k^2` and `max(lambda_i, f)` for each retained
eigenvalue. Serialize the raw and floored retained eigenvalues, raw and floored noise variance,
and floor for every fold/candidate. The resulting covariance must be finite and positive definite.

Score each held-out article by multivariate-Gaussian negative log likelihood using only the
training mean and `C_k`:

`NLL(x) = 0.5 * [8 log(2*pi) + log(det(C_k)) + (x-mu)^T C_k^-1 (x-mu)]`.

Record the mean NLL for each held-out publisher and candidate. Aggregate exactly 16 publisher
means per candidate. Model selection uses the conventional one-standard-error rule: find the
candidate with the lowest mean publisher NLL; calculate its sample standard deviation across 16
publisher means divided by `sqrt(16)`; select the smallest `k` whose mean is no greater than the
best mean plus that standard error. Ties prefer smaller `k`. This is descriptive model selection,
not a hypothesis test.

Before the real run, deterministic fixed synthetic tests must establish two nontrivial properties:
isotropic Gaussian noise selects `k=1`, not mechanically `k=3`, and a fixed strong rank-2 signal
with isotropic residual noise selects `k=2`.

## Analysis 2: primary publisher-block eigenvalue null

For every draw and score column separately:

1. Treat each publisher's complete 20 values as one source block.
2. Independently permute the 16 source blocks onto the 16 target publisher positions.
3. Independently randomize the 20-value row order inside every assigned block before insertion
   into the target publisher's original row positions.

Thus each score column retains every original within-publisher block exactly as a multiset, while
cross-column publisher identity and accidental article/date/row alignment are destroyed. The
balanced 16-by-20 structure is mandatory. Refit covariance PCA to every null matrix.

PC2 and PC3 are compared with their same-ranked null eigenvalues. Report the observed eigenvalue,
null mean, conservative 95th percentile (`numpy.quantile(..., method="higher")`), count of null
values greater than or equal to observed, and plus-one p-value `(count+1)/(10000+1)`. Detection
requires observed strictly above the conservative percentile and plus-one `p <= .05`.

An independently seeded article-level column-permutation null is retained only as a clearly
labeled sensitivity analysis. It uses the same summaries and 10,000 draws, but cannot determine
the primary verdict.

Component statuses are descriptive detection statements only:

- failed primary PC2: `pc2_not_detected_under_cluster_null`;
- failed primary PC3: `pc3_not_detected`;
- passed primary PC2/PC3: `pc2_detected_under_cluster_null` / `pc3_detected_under_cluster_null`.

Non-detection is not evidence of absence, equivalence, or exact dimensionality.

## Analysis 3: publisher-cluster bootstrap

Reuse the v1 publisher-cluster bootstrap unchanged: resample 16 complete publisher clusters with
replacement and report direct absolute PC2 cosine, sign-aligned loading intervals, PC2/PC3 swap
rates, bidirectional containment in reference/replicate PC2--PC3 subspaces, principal-angle
cosines, eigen-gap diagnostics, failures, and the rotational-ambiguity flag. The frozen v1
stability thresholds remain unchanged. Bootstrap instability prohibits interpretation of a
specific PC2 loading direction; it is not proof that PC2 does not exist.

## Frozen verdicts

For each bank:

- `evidence_for_at_least_three` only when primary PC2 and PC3 are both detected, PC2 bootstrap
  stability passes, and PPCA selects `k=3`.
- `evidence_for_at_least_two` when primary PC2 is detected, PC2 bootstrap stability passes, and
  PPCA selects `k=2` or `k=3`, but the full at-least-three conjunction is not met. PC3's separate
  status remains visible.
- `pc2_not_detected_under_cluster_null` whenever primary PC2 is not detected. PPCA and bootstrap
  outputs remain descriptive and cannot rescue or strengthen that statement.
- `inconclusive_mixed_evidence` for every other combination, including bootstrap instability or
  PPCA selection of `k=1` despite primary PC2 detection.

Identical Q0/Q1 verdicts may be called paraphrase-stable only. Differing verdicts are cross-bank
inconclusive. No outcome establishes topic-independent structure, external validity, exact
dimensionality, or an interpretable axis. Thresholds and branches will not be changed after seeing
the v2 result.
