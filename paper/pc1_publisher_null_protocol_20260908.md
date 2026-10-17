# Gate 1 PC1 publisher-block null protocol

**Frozen:** 2026-09-08, before implementation and before the one authorized real-data run

## Status and scope

This is a post-hoc exploratory test of already inspected corrected paired-scalar data. The only
measurement is the paired scalar `c = s_A - s_B`, where
`s = P(entailment) - P(contradiction)` is reconstructed from the existing stored aggregated
logits. No model inference, article-text inspection, qualitative coding, metadata analysis,
component interpretation, or manuscript modification is authorized.

This protocol contains Gate 1 only. Gate 2 paraphrase stability is out of scope and must not run
unless the completed Gate 1 artifact is first independently reviewed.

## Frozen files and fail-closed provenance

Before computation, the runner SHA-256 binds:

- `outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl`;
- `data/controlled_N_dense_corpus.jsonl`;
- `config/framing_queries.yaml`;
- `config/prompt_banks/probe_hypotheses_paraphrase_a.json`;
- `paper/paired_scalar_structure_protocol_20260905.md`;
- `paper/paired_scalar_structure_results_20260906.json`;
- `paper/paired_scalar_dimensionality_protocol_v2_20260907.md`;
- `paper/paired_scalar_dimensionality_results_v2_20260907.json`;
- this protocol;
- the Gate 1 runner; and
- every repository helper source used by the runner, including any transitively or dynamically
  imported paired-scalar and dimensionality helpers.

The paired-scalar structure result and corrected dimensionality-v2 result must bind the current
checkpoint, corpus, Q0, and Q1 hashes and declare the corrected paired-scalar measurement. Their
recorded protocol and runner bindings must match the current files. The runner recomputes every
bound hash after computation and before output creation. Any missing or malformed input,
provenance mismatch, hash mutation, identity/order mismatch, publisher imbalance, nonfinite
value, global-RNG mutation, or output overwrite attempt fails closed.

The only output is exclusively created at
`paper/pc1_publisher_null_results_20260908.json`; an existing path is never overwritten.

## Rows, score matrices, and observed statistic

The checkpoint and corpus must contain exactly the same 320 unique canonical article IDs in the
same order. The corpus must identify exactly 16 publishers with exactly 20 rows each. Prompt-bank
files must pass the existing audited paired-scalar loader checks.

For Q0 and Q1 separately, the existing audited loaders reconstruct the 320-by-8 paired
`A-minus-B` scalar matrix. Each column is centered across all 320 rows and is not standardized.
The observed statistic is `lambda_1`, the largest eigenvalue of the sample covariance

`S = X_centered^T X_centered / (n - 1)`.

Thus the covariance convention is `ddof=1`; computing the statistic as the largest squared
singular value divided by `319` is exactly equivalent.

## Publisher-block null

Q0 uses fixed seed `20260908` and Q1 uses the distinct fixed seed `20260909`. Each bank receives
an isolated local `numpy.random.Generator`; Python's `random` module and NumPy's legacy global RNG
must be unchanged by the analysis. Each bank receives exactly 10,000 draws.

For every draw and each of the eight columns independently:

1. Treat each source publisher's complete 20 values as one block.
2. Randomly reassign the 16 source blocks without replacement among the 16 target publishers.
3. Independently shuffle the 20 values inside every assigned target block.

This is the corrected dimensionality-v2 construction. It preserves every column's complete
publisher-block value distributions while destroying common cross-column article
correspondence. Refit sample-covariance PCA and retain null `lambda_1` for every draw.

## Frozen summaries and decisions

For each bank, report the observed `lambda_1`, null mean, null standard deviation with `ddof=1`,
null minimum and maximum, and null q95. Null q95 uses
`numpy.quantile(values, 0.95, method="higher")`, equivalently the sorted order statistic at
one-based position `ceil(0.95 * (B - 1)) + 1`; for `B=10000`, this is position 9501.

An exceedance is a null value greater than or equal to the observed value, so ties count against
detection. With `b` exceedances, report the plus-one value
`p = (b + 1) / (10000 + 1)`. A bank is detected if and only if both
`observed > q95` and `p <= 0.05`.

Gate 1 passes if and only if both Q0 and Q1 are detected. The only permitted joint conclusion on
pass is:

> dominant-component dependence detected under the specified publisher-block null.

The only permitted joint conclusion on failure is:

> not detected under the specified publisher-block null.

No outcome may be called real, meaningful, ideological, validated, one-dimensional, or an axis
interpretation. A failed gate is a non-detection under this null, not evidence of absence or
equivalence.

## Required preflight and post-run verification

Before the real-data run, the protocol and deterministic synthetic tests must already exist, and
both `py_compile` and the focused pytest file must pass. Tests cover block preservation and
correspondence destruction, deterministic local streams, greater-than-or-equal tie handling,
plus-one p-values, the higher q95 order statistic, the two-bank conjunction, conclusion
vocabulary, hash-tamper failure, exclusive-create overwrite failure, and preservation of both
global RNG states.

After that preflight, the runner may be executed against real data exactly once. The same focused
tests must then pass again. No Gate 2 command is part of this protocol.
