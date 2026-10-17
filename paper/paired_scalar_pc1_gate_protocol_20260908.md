# Bounded quantitative PC1 gate protocol

**Protocol date:** 2026-09-08

## Status and boundaries

This is a posthoc exploratory analysis of already inspected data. It is not preregistered, confirmatory, or an independent replication. The only measurement is the corrected paired scalar contrast \(c=s_A-s_B\). Q1 is a paraphrase bank only. This analysis performs no qualitative coding, assigns no semantic name to any axis, and authorizes no manuscript change or metadata analysis.

PC1 has **not** previously passed a publisher-aware null. Outcomes from this protocol may be described only as detected or not detected under the stated null. They must not be described as real, validated, one-dimensional, replicated, or independently confirmed. Non-detection is not evidence of absence or equivalence.

## Frozen inputs and integrity checks

The run is bound to the following files by SHA-256 before analysis and re-hashes the same files after analysis:

- `outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl`
- `data/controlled_N_dense_corpus.jsonl`
- `config/framing_queries.yaml` (Q0)
- `config/prompt_banks/probe_hypotheses_paraphrase_a.json` (Q1)
- `paper/paired_scalar_structure_protocol_20260905.md`
- `paper/paired_scalar_structure_results_20260906.json`
- `paper/paired_scalar_dimensionality_protocol_v2_20260907.md`
- `paper/paired_scalar_dimensionality_results_v2_20260907.json`
- this protocol
- the new PC1-gate runner
- every repository helper module imported by the runner

The checkpoint and corpus must contain exactly the same 320 canonical article IDs in the same order. Publisher membership must be exactly 16 publishers with 20 articles each. The paired structure and corrected dimensionality-v2 results must bind the current checkpoint, corpus, Q0, and Q1 hashes and declare the corrected paired scalar measurement. Any missing file, malformed binding, ID mismatch, publisher-count mismatch, input/source mutation during analysis, or attempted result overwrite fails closed without output.

All random operations use local `numpy.random.Generator` instances created from isolated named child streams of root seed `20260908`. The runner must not read or mutate NumPy's global RNG state.

## Score matrix and PCA

For each bank separately, convert checkpoint logits to the corrected eight-column paired scalar matrix \(c=s_A-s_B\). Fit covariance PCA to the column-centered, unstandardized matrix. Eigenvalues use the sample-covariance denominator \(n-1\). The Gate 1 statistic is the leading eigenvalue \(\lambda_1\).

## Gate 1: publisher-block PC1 detection

Run Gate 1 separately for Q0 and Q1 with 10,000 null draws per bank. Each draw uses the exact corrected dimensionality-v2 publisher-block scheme: independently for every score column, reassign the 16 complete 20-value publisher blocks among the 16 target publishers, then independently permute row order inside every assigned block. Refit covariance PCA and record \(\lambda_1\).

For each bank, report the observed \(\lambda_1\), null mean, null 95th percentile computed with `method="higher"`, the count of null values greater than or equal to observed, and the plus-one p-value \((b+1)/(10000+1)\). A bank is detected under this null if and only if observed \(\lambda_1\) is strictly greater than the null 95th percentile and the plus-one p-value is at most 0.05. Gate 1 passes only if both Q0 and Q1 are detected under this null. Otherwise the joint outcome is not detected under this null and Gate 2 is not run.

## Gate 2: Q0/Q1 phrasing stability, conditional on Gate 1

Gate 2 runs only if Gate 1 passes. Sign-align the full-data Q1 PC1 loading to the full-data Q0 PC1 loading.

Run 10,000 paired publisher-cluster bootstrap replicates. In each replicate, sample 16 publisher clusters with replacement; include all 20 article rows for every sampled cluster, including repeated clusters; use the identical sampled row indices for Q0 and Q1; refit both covariance PCAs. Sign-align the replicate Q0 PC1 loading to the full-data Q0 loading, then sign-align the replicate Q1 PC1 loading to that replicate's Q0 loading. Record their loading cosine. Report requested, valid, and failed replicate counts, categorized failure reasons, the full-data Q0-versus-Q1 loading cosine, and the bootstrap cosine distribution including its 2.5th, median, and 97.5th percentiles. The percentile interval is the 95% interval.

Using the full-data aligned loadings, compute each bank's article PC1 scores from its own globally centered score matrix. Center those article scores within publisher. Report the Spearman correlation \(\rho\) between the publisher-centered Q0 and Q1 article scores.

For the correspondence null, run 10,000 draws. Within every publisher independently permute only Q1 row correspondence, preserving the publisher-centered Q1 score multiset and publisher mean, and recompute Spearman \(\rho\) against fixed publisher-centered Q0 scores. Report the null mean, null 95th percentile with `method="higher"`, the count of null correlations greater than or equal to observed, and the one-sided plus-one p-value.

The Gate 2 outcome vocabulary is frozen:

- `strong_phrasing_stability` if bootstrap cosine q025 is at least 0.80, observed \(\rho\) is at least 0.50, and the correspondence-null p-value is at most 0.05;
- `detected_below_practical_threshold` if the p-value is at most 0.05 but either practical threshold fails;
- `not_detected` if the p-value is greater than 0.05.

Any Gate 2 result is phrasing stability only and is not independent replication.

## Output contract

The result is written exactly once with exclusive-create semantics to `paper/paired_scalar_pc1_gate_results_20260908.json`. The result records the protocol boundaries, dimensions, decision rules, named RNG spawn keys, runtime versions, all bound paths and SHA-256 hashes, Gate 1 results, and Gate 2 results only when Gate 1 passes. No metadata analysis, qualitative interpretation, semantic axis label, or manuscript language is produced.
