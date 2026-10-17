# Paired-Scalar Dimensionality-Selection Protocol

**Frozen:** 2026-09-07, before implementation and execution of this analysis

## Status and scope

This is a prospective analysis specification applied to an already inspected dataset and already
inspected PCA spectra. It is **not** a preregistration, a confirmatory test, or an independent
replication. Q1 is a paraphrase bank derived from the same authored dialectics as Q0. Agreement
between Q0 and Q1 can establish phrasing stability only.

The analysis uses the existing aggregated-logit checkpoint and performs no model inference. Its
sole measurement is the corrected paired scalar

`c = s_A - s_B`, where `s = P(entailment) - P(contradiction)` after softmax of the stored,
paragraph-aggregated logits. Q0 and Q1 are analyzed separately. A-side-only scores and hidden-state
displacements are out of scope.

The earlier observation that two PCs contain approximately 72--75% of paired-scalar variance does
not establish that the data are two-dimensional. This protocol compares one, two, and three
components without interpreting or naming any component.

## Frozen inputs

- Checkpoint: `outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl`
- Corpus and publisher labels: `data/controlled_N_dense_corpus.jsonl`
- Q0 prompts: `config/framing_queries.yaml`
- Q1 prompts: `config/prompt_banks/probe_hypotheses_paraphrase_a.json`
- Contextual prior result: `paper/paired_scalar_structure_results_20260906.json`
- Expected sample: 320 unique article IDs, in exact checkpoint/corpus order, from 16 publishers
  with 20 articles each.

The result must bind SHA-256 hashes for every input above, this protocol, the dimensionality
runner, and the paired-scalar scoring runner. Inputs and sources are hashed before and after
analysis; any mutation aborts the run. Output creation is exclusive and refuses overwrite.

## Determinism

- Root seed: `20260907`.
- NumPy `SeedSequence.spawn` creates fixed, isolated streams for the grouped split schedule, Q0
  eigenvalue null, Q1 eigenvalue null, Q0 bootstrap, and Q1 bootstrap, in that order.
- Grouped cross-validation: 16 repeats of four folds. In every repeat, the 16 publisher names are
  deterministically shuffled and partitioned into four folds of four publishers. Every publisher
  is held out exactly once per repeat. The same split schedule is used for Q0 and Q1.
- Eigenvalue null: 10,000 permutations per bank.
- Publisher-cluster bootstrap: 10,000 replicates per bank.

## Analysis 1: publisher-grouped held-out reconstruction

For every split and every `k` in `{1, 2, 3}`:

1. Fit covariance PCA on the training publishers after subtracting the training column means.
   Do not standardize columns.
2. Project and reconstruct the held-out publishers using only the training mean and first `k`
   training loadings.
3. Record held-out mean squared error and relative squared reconstruction error
   `SSE_k / sum((X_test - training_mean)^2)`.

Report every fold's publishers and errors, plus the paired foldwise decreases for 1-to-2 and
2-to-3 components. These decreases are descriptive only: nested PCA mechanically cannot increase
in-sample projection capacity, so a positive decrease or sign test is never evidence for an
additional dimension.

Model selection uses the predeclared one-standard-error rule on relative squared error:

1. Average the four fold errors within each repeat, yielding 16 repeat-level errors per `k`.
2. Find the candidate with the smallest grand mean error. Ties choose smaller `k`.
3. Compute that candidate's standard error across the 16 repeat means using sample standard
   deviation divided by `sqrt(16)`.
4. Select the smallest `k` whose grand mean is no greater than the best mean plus that one
   standard error.

This is descriptive model selection/generalization, not a hypothesis test. It can select only
among one, two, and three components and cannot establish that three is sufficient.

## Analysis 2: sequential column-permutation eigenvalue null

For each bank, fit covariance PCA to the complete, column-centered 320-by-8 matrix and retain raw
covariance eigenvalues. For each null draw, independently permute article rows within each of the
eight columns, preserving every marginal distribution while destroying cross-column covariance,
then refit PCA.

PC2 and PC3 are compared separately to the same-ranked null eigenvalue. For each component report:

- the observed eigenvalue;
- the null mean;
- the conservative 95th percentile using NumPy's `method="higher"`;
- exceedance count `#{null >= observed}`;
- plus-one Monte Carlo p-value `(exceedances + 1) / 10001`.

A component passes only when its observed eigenvalue is strictly above the conservative 95th
percentile and its plus-one p-value is at most 0.05. Tests are interpreted sequentially: PC3 is
eligible to support a dimensionality claim only if PC2 passes. These tests concern dependence in
the eight authored scores, not external validity or ontological dimensionality.

## Analysis 3: publisher-cluster bootstrap of PC2

For each bank, draw 16 publisher labels with replacement and include all 20 article rows from each
drawn publisher. Refit covariance PCA without column standardization. Sign is handled by absolute
cosine for directional comparisons and by aligning replicate PC2 to the sign of reference PC2 for
coordinate-wise loading intervals.

Every replicate records:

- direct `abs(cos(replicate PC2, reference PC2))`;
- containment of reference PC2 in the replicate PC2--PC3 subspace;
- containment of replicate PC2 in the reference PC2--PC3 subspace;
- both principal-angle cosines between reference and replicate PC2--PC3 subspaces;
- whether reference PC2 matches replicate PC3 more closely than replicate PC2;
- whether replicate PC2 matches reference PC3 more closely than reference PC2;
- the relative PC2--PC3 eigenvalue gap; and
- sign-aligned PC2 loadings.

A replicate fails, and is excluded from distribution summaries, if fewer than three distinct
publishers are drawn, a fit is nonfinite, rank is below three, or the PC2--PC3 eigenvalues are tied
within relative tolerance `1e-10`. Failure counts and reasons remain in the result.

PC2 loading stability passes only if all three predeclared conditions hold:

1. at least 95% of valid replicates have direct absolute cosine at least 0.90;
2. each of the two PC2/PC3 swap rates is at most 5%; and
3. at most 1% of all requested replicates fail.

Report empirical 2.5%, 50%, and 97.5% quantiles for all similarity diagnostics, percentile 95%
intervals for every sign-aligned loading, swap rates, failure rates, and eigen-gap summaries.
High 2D-subspace containment accompanied by weak direct PC2 similarity or frequent swaps is
diagnosed as PC2/PC3 rotational ambiguity, not stable PC2 recovery. The descriptive rotational-
ambiguity flag is set when both containment medians are at least 0.90 and either the direct-cosine
fraction or either swap-rate stability condition fails.

## Frozen conjunctive decisions

Each bank receives one outcome. No result may be rescued by changing thresholds after execution.

- `supports_two_over_one_not_three`: PC2 passes the eigenvalue null, PC3 fails it, PC2 loading
  stability passes, and grouped CV selects `k=2`.
- `supports_at_least_three`: PC2 and PC3 pass their eigenvalue nulls, PC2 loading stability passes,
  and grouped CV selects `k=3`. This does not establish that exactly three dimensions suffice.
- `supports_no_more_than_one`: PC2 fails its eigenvalue null, PC3 does not anomalously pass while
  PC2 fails, and grouped CV selects `k=1`. PC2 loading stability is then descriptive only.
- `inconclusive_pc2_unstable`: PC2 passes its eigenvalue null but PC2 loading stability fails.
  Component naming or interpretation is prohibited even if grouped CV selects two or three.
- `inconclusive_mixed_evidence`: every other combination, including disagreement between the
  eigenvalue null and grouped CV, anomalous PC3-without-PC2 significance, or CV selection of a
  dimension not supported by the sequential null.

Across banks, identical outcomes may be described only as **paraphrase-stable**. Differing outcomes
must be reported as cross-bank inconclusive. Neither branch is independent replication.

## Claim boundaries

The run cannot validate axis names, distinguish stance from topic salience, establish generality
outside this corpus, or justify a manuscript change by itself. The artifact will report the frozen
decision mechanically, including negative or inconclusive outcomes, and the protocol will not be
edited after the first real-data execution.
