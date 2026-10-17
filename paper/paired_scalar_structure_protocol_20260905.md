# Paired Scalar Structure Protocol

**Date:** 2026-09-05

## Status and retrospective provenance boundary

An extraction was run on 2026-09-05 before this protocol and its runner were committed or
cryptographically bound together. The draft protocol file timestamp preceded the extraction,
but the run did **not** write the promised pre-run manifest, source hashes, prompt hashes, model
snapshot hash, or corpus hash. It is therefore classified as exploratory and post hoc, not as a
preregistered or fully provenance-bound run.

- Aggregated-logit checkpoint SHA-256:
  `6d9a266aefeb88dd5a45b2ea03f160d295ba60ff3adb33fcb4d8c5a7edf5333f`
- Draft result SHA-256:
  `24c3aa39ac6f209cf448edf73eb28c1968ce401b40c2c43b63305d59f287d5b6`
- Exact draft runner SHA-256 when inspected after the run:
  `7539d52c78c9053547f6ae9d970ab1cf1b5d9d3d9fbcb7aaf57260ee50f9be53`

The checkpoint contains 320 unique canonical article IDs in exactly the current corpus order,
plus finite Q0/Q1 A and B logit arrays of shape 320-by-8-by-3. Both fresh A arrays are bitwise
identical to the corresponding historical `logits_raw` artifacts. This strongly supports the
declared deterministic extraction path, but it does not retroactively create a pre-run manifest
for B or prove that current input files were unchanged during extraction.

The runner was hardened after this extraction. Its current source is not claimed to be the exact
source that generated the checkpoint; the hash above records that boundary explicitly.

## Motivation
This analysis addresses a reviewer's concern regarding the methodology's claim of using "contrastive displacement" (A - B) versus what the original scripts actually measured ("one-sided entailment" A only). The goal is to perform a fresh extraction on the N-Dense corpus of 320 articles and calculate both the A-only and A-minus-B contrastive scalar structures strictly, keeping them separated.

## Inputs
- **Corpus**: data/controlled_N_dense_corpus.jsonl (320 unique articles)
- **Q0 Prompts**: config/framing_queries.yaml (8 pairs of A and B framing hypotheses)
- **Q1 Prompts**: config/prompt_banks/probe_hypotheses_paraphrase_a.json (16 alternating A/B paraphrase hypotheses)
- **Model**: microsoft/deberta-v2-xlarge-mnli (loaded entirely locally without network access, using float32)

## Preprocessing & Data Extraction
- Each article's text is parsed and split into up to 3 paragraphs.
- Paragraphs are weighted as [0.5, 0.3, 0.2] (normalized to sum to 1 depending on the number of available paragraphs).
- Logits for each paragraph against the hypothesis are extracted. 
- Aggregation is performed at the logit level across paragraphs.
- Using stable_softmax, probabilities for Entailment (E) and Contradiction (C) are extracted from the aggregated logits.
- The scalar score is calculated as s = P(E) - P(C).
- The A-only score is s_A. The primary paired scalar contrast is c = s_A - s_B. It is not
  mathematically equivalent to the hidden-state displacement `phi_A - phi_B`.

## Analysis
Two parallel analyses will be run and compared across Q0 and Q1:
1. **Fresh A-only baseline**: Using only s_A scores for the 8 axes.
2. **Primary paired scalar contrast**: Using s_A - s_B for the 8 axes.

For each of these channels and for each prompt bank (Q0 and Q1):
- Perform deterministic PCA on the article-by-axis matrix.
- Report the full eigen spectrum, first 2 PCA loadings, and variance explained.
- Calculate cross-bank comparisons (Q0 vs Q1) using:
  - **Variance column-null test**: 1000 permutations within each column.
  - **Procrustes row-pairing-null test**: 1000 permutations of row-correspondence, fixing the PCA space.
  - **Pairwise-distance Spearman correlation**: Descriptive comparison of inter-article distances in projected space.
  - **Principal angles** between the Q0 and Q1 PCA subspaces.

## Integrity & Exclusivity
- No historical data will be loaded.
- Outputs will be strictly isolated from historical A-only results.
- Analysis uses deterministic `np.random.default_rng(42)` with 10,000 permutations per null.
- The hardened runner fails closed on malformed banks, missing or duplicate IDs, nonfinite
  arrays, input mutation during analysis, and output overwrite.
- Future extractions must write a pre-run manifest before the first model inference. That
  manifest binds article IDs and text hashes, complete prompt pairs, model-file hashes, source
  hashes, and preprocessing settings. The analysis validates that manifest and records its hash
  in the result. That requirement was not satisfied by the completed exploratory extraction
  described above.

## Exploratory result (verified 2026-09-06)

The hardened analysis reloaded the existing checkpoint, required all 320 IDs to match the corpus
in exact order, verified all analyzed input hashes before and after computation, and wrote
`paper/paired_scalar_structure_results_20260906.json`. The analysis code and all imported helper
files are hash-bound in that result.

An initial 1,000-permutation analysis was followed by a resolution-only rerun at 10,000
permutations after review identified the original Monte Carlo floor as too coarse for manuscript
reporting. The statistic definitions, data, channels, null schemes, and seed were unchanged; the
10,000-permutation artifact supersedes the 1,000-permutation artifact.

| Quantity | Fresh A-side baseline | Paired scalar `s_A-s_B` |
|---|---:|---:|
| Q0 variance in first two PCs | 78.09% | 75.01% |
| Q1 variance in first two PCs | 78.28% | 72.29% |
| Q0/Q1 Procrustes disparity | 0.173 | 0.189 |
| Q0/Q1 pairwise-distance Spearman rho | 0.796 | 0.846 |
| Principal angles | 25.54 deg, 8.64 deg | 27.62 deg, 9.73 deg |

For both channels and both banks, zero of 10,000 column-permutation draws matched or exceeded the
observed two-PC concentration; the plus-one estimate is `1/10001`. Zero of 10,000 row-pairing
permutations produced a Procrustes disparity as low as observed, again yielding `1/10001`.
These nulls test cross-score dependence and article-row correspondence respectively. They do not
establish that exactly two dimensions are sufficient or that either space is externally valid.
`1/10001` is the Monte Carlo resolution floor for this run and does not identify how far below
that floor the underlying tail probability may lie.

The paired scalar result therefore **survives as a low-dimensional, paraphrase-stable exploratory
pattern**, but it is not numerically identical to the A-side result. Its two-PC concentration is
lower, its cross-bank distance ordering is more similar, and its component weights change
materially. Interpretation of the historical A-side loadings cannot be transferred to the paired
scalar components.

Because the extraction lacked a hash-bound pre-run manifest and Q1 is a paraphrase rather than an
independently authored bank, this result must not be described as confirmatory replication. No
new threshold was introduced after observing it, and it does not alter any earlier null, tensor
stop, or kernel decision.
