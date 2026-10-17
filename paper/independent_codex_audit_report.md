# Independent Codex Audit of the Prospective, Ukraine, and Cross-Model Claims

Date: 2026-09-13

## Executive verdict

I independently inspected the named corpora, registry, serialized caches, result JSON, and implementation code. I reconstructed the paired contrast from cached logits, recomputed the PCA spectra and correlations in double precision, and ran targeted corpus-overlap and sensitivity checks.

| Target | Numerical reproduction | Scientific/provenance verdict |
|---|---:|---|
| Prospective Q0/Q1, N=120 | **Pass** | **Exact registry exclusion passes, but the strong independence/leakage claim fails.** Exact fingerprint/title/URL intersections are zero, yet at least 37/120 articles have substantial content-level overlap with earlier project corpora at 5-word-shingle Jaccard >= 0.50, including exact or near-exact copies. The result is numerically robust to removing those matches, but the sample cannot be described as cleanly unseen without qualification. |
| Russia-Ukraine, N=60 | **Pass** | **Topical replication invalid/inconclusive.** PC1 = 0.480166 and p = 0.341366 reproduce; the axes are nondegenerate. However, the matrix is variance-dominated by individual axes, the corpus contains substantial syndication and off-topic contamination, and the nominal PC2/PC3 findings do not establish a rank-2 structure. |
| Cross-model, N=60 | **Pass** | **Qualified pass.** All three reported Pearson correlations reproduce as correlations between separately fitted PC1 article scores, up to the arbitrary PCA sign. Cache provenance is inadequate, and a publisher-field bug invalidates collateral publisher-null output, but neither issue changes the reported Pearson values. |

These are conditional recomputations from surviving cache files. The caches do not cryptographically bind model revision, tokenizer, label mapping, prompt bank, corpus order, or article identities. Therefore this audit verifies arithmetic on the stored logits and checks consistency with the present code and corpora; it does not prove how or when those logits were generated.

## Scope and methods

The audit used CPU loading of the named PyTorch files, conversion to `float64`, a numerically stable softmax, and the declared contrast

\[
\Delta_{ik} = [P_{A,ik}(E)-P_{A,ik}(C)]-[P_{B,ik}(E)-P_{B,ik}(C)].
\]

For each resulting article-by-axis matrix, I centered columns, formed the sample covariance matrix, sorted `eigvalsh` eigenvalues in descending order, calculated explained-variance ratios, and calculated participation ratio as

\[
\mathrm{PR}=(\sum_j \lambda_j)^2 / \sum_j \lambda_j^2.
\]

PC1 article scores were obtained by projection on the fitted first component. Cross-bank and cross-model PC1 signs were aligned before correlation because PCA eigenvector sign is arbitrary. An SVD reconstruction of the prospective spectra agreed with the covariance route to less than `8.4e-17` in EVR.

For prospective leakage, I first reproduced the exact stored fingerprint formula, then tested exact and normalized title/URL intersections. I separately compared article body text against 1,100 rows in ten earlier project corpora using lowercase alphanumeric tokens, sets of consecutive five-token shingles, and maximum pairwise Jaccard overlap. That latter test is deliberately broader than the registry's exact-key check; it detects syndicated or lightly edited copies that cryptographic hashes cannot.

For Ukraine corpus quality, I inspected the selection and scoring paths and clustered the consumed text using word-plus-bigram TF-IDF cosine similarity. I also reran the dimensionality calculation after removing high-similarity duplicates and on a stricter topical subset. I did not regenerate logits from model checkpoints.

## 1. Prospective Q0/Q1 replication

### 1.1 Exact registry exclusion passes

`data/touched_fingerprints_registry.json` contains 1,290 unique entries: 486 SHA-256 values, 409 URLs, and 395 titles. `data/prospective_confirmatory_120_corpus.jsonl` has 120 rows and 120 unique stored hashes.

The stored corpus hashes reproduce exactly as SHA-256 of the lowercased, stripped title, a literal `|`, and the first 200 characters of the original content/body after the title separator. Against the registry, I found:

- stored hash intersection: **0**;
- recomputed hash intersection: **0**;
- lowercased exact-title intersection: **0**;
- exact-URL intersection: **0**;
- punctuation-insensitive normalized-title intersection: **0**;
- canonical host/path URL intersection: **0**.

This agrees with the sampler's actual exclusion logic in `scripts/exp_prospective_preregistered_replication.py:100-107`: it computes that truncated-content hash and checks the hash, lowercased title, or exact URL against the registry. Thus the narrow statement “no exact registry collision” is true.

### 1.2 Exact-key exclusion did not prevent content-level leakage

The registry result is not equivalent to an unseen-content result. The content sweep covered these earlier corpora: abortion (60), climate (60), control-adversarial (160), control-unstructured (80), controlled N=60 (60), controlled dense (320), fresh unseen (120), immigration (60), Ukraine (60), and untouched confirmatory (120), totaling 1,100 prior rows. I excluded the raw real-corpus candidate pool and the prospective corpus itself from this comparison.

Among the 120 prospective articles, the count whose best earlier-corpus match reached each five-word-shingle Jaccard threshold was:

| Threshold | Prospective rows matched |
|---:|---:|
| >= 0.50 | 37 |
| >= 0.70 | 35 |
| >= 0.80 | 33 |
| >= 0.90 | 27 |
| >= 0.95 | 20 |

Examples include prospective ID `0001` versus controlled-N60 row 4 (`J=0.9515`), `0009` versus controlled-N60 row 9 (`J=0.98995`), `0037` versus controlled-dense row 14 (`J=0.98734`), `0077` versus controlled-N60 row 38 (`J=1.0`), `0050` versus untouched-confirmatory row 103 (`J=0.9904`), and `0024` versus untouched-confirmatory row 36 (`J=0.9724`). There are also 23 within-prospective pairs at `J >= 0.50`.

These are not hash collisions; they are content-level exact or near duplicates that the exact title/URL/truncated-hash mechanism does not capture. Accordingly, the exact intersection is zero while the broader independence claim has a major leakage caveat.

### 1.3 Raw numerical reconstruction

`data/cache_prospective_120_q0_q1_logits.pt` contains `q0_la`, `q0_lb`, `q1_la`, and `q1_lb`; each is finite, shape `[120, 8, 3]`, and stored as `float16`. The cache has no corpus IDs/hashes, model identifier or revision, bank hash, label-map assertion, tokenizer version, or creation metadata, and is gitignored.

The double-precision reconstruction is:

| Quantity | Q0 | Q1 |
|---|---:|---:|
| Eigenvalues | `[0.35944754294, 0.09278380182, 0.06827677369, 0.02977554262, 0.02230991154, 0.01847246896, 0.01642000194, 0.00846460265]` | `[0.29191584366, 0.07985562168, 0.05633428996, 0.03609396256, 0.02124232185, 0.01405890013, 0.01109990308, 0.00982302546]` |
| PC1 EVR | **0.583565493725** | **0.560919399366** |
| Participation ratio | **2.624847505114** | **2.793998729336** |

After PC1 sign alignment, the Q0-Q1 article-score Pearson correlation is **`|r|=0.925279343541`**, `p=1.645627e-51`. The result JSON gives Q0 PC1 `0.583563799213`, PR `2.624858735233`; Q1 PC1 `0.560891538350`, PR `2.794231828225`; and `r=0.925295701618`. Those tiny differences are compatible with mixed-precision/intermediate implementation details. The published rounded claims—58.36%, 56.09%, and 0.9253—are reproduced.

### 1.4 Sensitivity to prior-corpus overlap

Removing every prospective row whose best earlier-corpus shingle overlap was at least 0.50 leaves `n=83`:

- Q0 PC1 EVR `0.598701`, PR `2.515006`;
- Q1 PC1 EVR `0.620348`, PR `2.394563`;
- aligned PC1 `|r|=0.953342`.

Using the stricter removal threshold `J >= 0.80` leaves `n=87`:

- Q0 PC1 EVR `0.575493`, PR `2.670120`;
- Q1 PC1 EVR `0.573683`, PR `2.695869`;
- aligned PC1 `r=0.929892`.

Thus the leakage finding damages the asserted sample provenance, not the numerical phenomenon: the leading structure and cross-bank agreement remain strong after these removals.

### 1.5 Operator audit

The implementation is the requested paired contrast. `scripts/exp_prospective_preregistered_replication.py:154-165` extracts A and B hypotheses separately. Lines 176-185 softmax both arrays, define `s_A = probs_a[...,2] - probs_a[...,0]`, define `s_B` analogously, and return `s_A - s_B`. This is exactly `(P_A(E)-P_A(C))-(P_B(E)-P_B(C))`, assuming labels 2 and 0 are entailment and contradiction. The weakness is that the indices are hardcoded; the cached-load path does not bind or reassert the checkpoint's label mapping.

**Verdict 1:** numerical result **PASS**; strict operator **PASS, conditional on the label-index convention**; exact 1,290-key registry exclusion **PASS**; broad “independent/unseen corpus” claim **FAIL / major caveat** because substantial content-level duplicates survive exact fingerprinting. The result itself is robust in the performed overlap-removal sensitivities.

## 2. Russia-Ukraine N=60 result

### 2.1 Raw reconstruction and nondegeneracy

`data/cache_paired_logits_russia_ukraine_war_N60.pt` contains finite `la` and `lb` arrays of shape `[60,8,3]`, stored as `float16`, with no article IDs/hashes or model/query/label metadata. `scripts/exp_paired_operator_reruns.py:109-132` implements the same strict paired contrast, and lines 135-174 fit PCA and invoke the null test.

Every axis varies. Axis variances and counts of distinct stored values are:

| Axis | Variance | Unique values |
|---:|---:|---:|
| 1 | 0.01238725 | 60 |
| 2 | 0.02047857 | 59 |
| 3 | 0.00844844 | 60 |
| 4 | 0.00882876 | 59 |
| 5 | 0.00319232 | 60 |
| 6 | 0.04211725 | 60 |
| 7 | 0.10514509 | 59 |
| 8 | 0.02466508 | 60 |

The seventh-to-fifth variance ratio is about 32.9. The bank is not constant or numerically collapsed, but it is strongly heteroscedastic.

The article-wise axis correlation matrix is:

```text
[[ 1.000000, -0.133972,  0.010966,  0.007668,  0.168931,  0.308709, -0.272227,  0.169881],
 [-0.133972,  1.000000,  0.093673,  0.080502, -0.002177, -0.188220, -0.160364, -0.457112],
 [ 0.010966,  0.093673,  1.000000,  0.134604,  0.280435, -0.481538,  0.106006, -0.072440],
 [ 0.007668,  0.080502,  0.134604,  1.000000, -0.046915, -0.137889,  0.283787, -0.156225],
 [ 0.168931, -0.002177,  0.280435, -0.046915,  1.000000,  0.005471, -0.179941, -0.036304],
 [ 0.308709, -0.188220, -0.481538, -0.137889,  0.005471,  1.000000, -0.059023, -0.082848],
 [-0.272227, -0.160364,  0.106006,  0.283787, -0.179941, -0.059023,  1.000000,  0.054872],
 [ 0.169881, -0.457112, -0.072440, -0.156225, -0.036304, -0.082848,  0.054872,  1.000000]]
```

This does not show a uniform positive common factor. The largest absolute pairwise relationships are axis 3 versus 6 (`-0.4815`) and axis 2 versus 8 (`-0.4571`).

The covariance eigenvalues are

```text
[0.10816355, 0.04700586, 0.03290596, 0.01159532,
 0.01115429, 0.00715709, 0.00482398, 0.00245671]
```

with EVRs

```text
[0.4801661311, 0.2086712665, 0.1460781006, 0.0514746509,
 0.0495168084, 0.0317722042, 0.0214148812, 0.0109059571]
```

and PR `3.3098270917`. These reproduce the result JSON.

### 2.2 Why PC1 is null while PC2 appears nominally significant

In 10,000 independent-column permutations, PC1 has mean eigenvalue `0.10787923`, 95th-percentile eigenvalue `0.11145957`, 3,413 exceedances, and `p=0.3413658634` (reported q95 EVR `0.494798`). PC2 has mean eigenvalue `0.04317781`, q95 eigenvalue `0.04647229`, 327 exceedances, and `p=0.0327967203` (q95 EVR `0.206303`). PC3 has `p=0.00059994`.

This pattern is possible because each ordered eigenvalue has its own null distribution. Here the null's first and second eigenvalues are negatively correlated (`r=-0.574`): permutations with unusually large lambda-1 tend to leave less variance for lambda-2. The observed lambda-1 is ordinary while lambda-2 lies just above its marginal ordered-null tail. The loadings show why this should not be read as a common rank-2 factor: PC1 loads `0.9831` on axis 7 and PC2 loads `0.9129` on axis 6. These are primarily the two highest-variance marginal axes, not broad shared directions. Standardizing axes before PCA reduces PC1 to 23.03% and raises PR to 6.34.

Moreover, the nominal PC2 result is not multiplicity-robust: Bonferroni over eight components gives `0.2624`, and Holm over only PC1 and PC2 gives `0.0656`. In `scripts/exp_systematic_dimensionality_gate.py:145-150`, `detected_rank_column_null` counts all individually significant ordered components. Therefore the reported `detected_rank_col=2` means that PC2 and PC3 crossed their separate thresholds even though PC1 failed. It does **not** establish a contiguous rank-two signal and is misleading if read that way.

### 2.3 Corpus contamination and sensitivity

The corpus builder in `scripts/exp_ukraine_cross_topic.py:62-101` admits an article when broad substrings occur anywhere, requires only two secondary strings, hashes only an initial text fragment, and takes the first 60 qualifying rows sequentially. The rerun then scores only the first 2,000 characters (`scripts/exp_paired_operator_reruns.py:64-79`). Manual inspection found numerous airliner-related or otherwise weakly relevant items.

Using word-and-bigram TF-IDF on `content[:2000]` with English stop words, I connected row pairs at cosine similarity >= 0.80 and took graph connected components. Six syndication clusters cover 21 of 60 rows (zero-based indices):

```text
[0, 6, 15, 17, 19, 27, 28, 31, 39]
[1, 5]
[4, 11, 13, 34]
[7, 53]
[8, 9]
[38, 57]
```

Keeping the lowest index from each component leaves `N=45`, with retained indices `[0, 1, 2, 3, 4, 7, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 29, 30, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 58, 59]`. The covariance eigenvalues are `[0.09756930, 0.05295253, 0.04097245, 0.01282546, 0.01221222, 0.00774597, 0.00582480, 0.00288645]`; EVRs are `[0.41877182, 0.22727463, 0.17585559, 0.05504743, 0.05241540, 0.03324605, 0.02500031, 0.01238877]`; and PR is `3.76491604`. With seed 42 and 10,000 permutations, PC1 has `p=0.16298370`, q95 eigenvalue `0.10129017`, and q95 EVR `0.43474194`; PC2 has `p=0.56084392`, q95 eigenvalue `0.05845454`, and q95 EVR `0.25088952`; PC3 has `p=0.00049995`, q95 eigenvalue `0.03566018`, and q95 EVR `0.15305508`. Thus the PC2 result disappears after deduplication, while an isolated later-component significance remains and reinforces the warning against interpreting the count as rank. A stricter weak topical subset (`N=10`) gives PC1 `0.48897`, `p=0.32737`, and PC2 `0.23632`, `p=0.54805`.

**Verdict 2:** the reported PC1 null **numerically passes** and is not an operator or constant-axis bug. But it is also not a clean test of a coherent Ukraine framing bank: syndication, topical contamination, truncation, heteroscedastic axes, and lack of cache binding make the topical result **invalid/inconclusive**. The nominal PC2 and `detected_rank_col=2` interpretation **fail** as evidence for rank two.

## 3. Cross-model correlations

### 3.1 Exact reconstruction from the three caches

The inspected files are the N=60 paired-logit caches for `microsoft/deberta-v2-xlarge-mnli`, `facebook/bart-large-mnli`, and `roberta-large-mnli`. Each contains only `la` and `lb`, shape `[60,8,3]`. BART and RoBERTa are `float32`; DeBERTa is `float16`. The arrays, dtypes, and hashes differ, so they are not literal copies.

Applying the paired contrast and independently fitting PCA to each model's 60-by-8 matrix gives these PC1 **article-score** correlations:

| Model pair | Pearson r | Pearson p |
|---|---:|---:|
| DeBERTa vs BART | **0.8057187453706554** | `8.235495250502596e-15` |
| DeBERTa vs RoBERTa | **0.5589566147945043** | `3.454642137852358e-6` |
| BART vs RoBERTa | **0.5469995599798573** | `6.1213927957952925e-6` |

These exactly reproduce `paper/paired_operator_cross_model_cross_topic_rerun.json`, including the two values singled out in the audit request: BART-RoBERTa `r=0.5470`, `p=6.12e-6`, and DeBERTa-BART `r=0.8057`.

The statistic must be described precisely. It is not a correlation of all 480 paired-contrast cells. Flattening all 60-by-8 cells gives only `0.6749648033`, `0.5477227505`, and `0.4705365018`, respectively. The headline values compare one independently fitted PC1 score per article across models.

### 3.2 Sign, ordering, leakage, and precision checks

PCA signs are arbitrary. `scripts/exp_paired_operator_reruns.py:277-287` obtains each model's PC1 scores and flips the second score/component when their correlation is negative. Therefore the stored positive correlations are correlations **up to PCA sign**, effectively absolute correlations. For the current raw decompositions, the unaligned correlations were already positive, so no flip was required.

As a row-order sanity check, the unshifted correspondence produced the largest absolute PC1 correlation among cyclic shifts for all pairs. This supports matched row order but is not a cryptographic provenance guarantee. Shared articles and shared prompts are intentional in a cross-model paired design and are not themselves leakage; they are what makes the across-model comparison paired. No evidence of copying logits between models was found.

The DeBERTa cache's lower precision has little effect. Recasting/recomputing in float64 changed individual contrasts by at most about `0.0007133` and altered correlations by at most approximately `3.3e-5`.

Two near-duplicate BBC article pairs were detected (rows 18/55, cosine `0.9976`; rows 19/56, cosine `0.9771`). Removing the duplicates gives DeBERTa-BART `0.8033674`, DeBERTa-RoBERTa `0.5651687`, and BART-RoBERTa `0.5448457`. They do not explain the reported agreement.

### 3.3 Provenance and collateral implementation defect

All three caches lack model revision, model identifier, tokenizer/config hashes, label mapping, query-bank hash, corpus hash, article IDs/order, software versions, and generation timestamp. On a cache hit, `scripts/exp_paired_operator_reruns.py:221-225` loads `la` and `lb` directly; it does not validate those bindings or the entailment/contradiction label indices. The present arithmetic is correct, but independent end-to-end regeneration cannot be established from these artifacts alone.

There is also a separate metadata bug. `data/controlled_N60_corpus.jsonl` uses the field `publication`, while `scripts/exp_paired_operator_reruns.py:201-206` reads `publisher`. All rows therefore become `"unknown"`; the code accepts this one-group list as if it were a balanced publisher design. Consequently, collateral `detected_rank_pub` output for these cross-model analyses is invalid. This does not enter the PCA PC1 score calculation or the Pearson correlations above.

**Verdict 3:** the three requested cross-model PC1 article-score correlations **PASS exactly**, including BART-RoBERTa and DeBERTa-BART. There is no detected sign error or evidence that one model's cache is a copy of another. The result remains **qualified** because sign alignment makes the measure unsigned, repeated articles slightly reduce effective sample size, the publisher-null metadata path is broken, and the caches are not bound to their purported inputs.

## Final assessment

The code computes the requested paired contrast, and all three headline numerical claims reproduce from the stored logits. That is the strongest conclusion supported by the artifacts.

It would be an overstatement, however, to treat the artifacts as three equally clean replications. The prospective result is numerically strong and sensitivity-robust, but the corpus is not cleanly independent at the content level despite zero exact registry intersections. The Ukraine PC1 null is arithmetically real, but its corpus and dimensionality interpretation are too compromised for a strong topical conclusion. The cross-model PC1-score correlations are real properties of the cached matrices, but their provenance remains unverified because the caches are unmanifested.

For future confirmatory use, cache files should fail closed unless they bind: complete ordered article IDs and content hashes; query-bank hashes; model repository and immutable revision; tokenizer/config hashes; explicit label-map assertions; truncation settings; software versions; generation time; and the producing commit. Corpus exclusion should add normalized full-text similarity/deduplication, not rely only on exact title, URL, and a first-200-character hash.
