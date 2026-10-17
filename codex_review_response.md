# Belief Transformer V3: Pre-Launch Architectural and Empirical Review

**Review snapshot:** `389e4781cdd9f13d8b5af78df1b048b2aacb0e79` (`master`, 2026-08-13)  
**Intended launch:** 58,756-article full-corpus sweep  
**Decision:** **HARD NO-GO**

## Executive judgment

The refactor has made real progress: Experiments A and B now fail closed when their required embedding tensor is absent or malformed; Experiment B actually invokes the production spectral, Dirichlet, Hadamard, and conformal modules; the original synthetic `NMI=0.6205` claim has been correctly retracted; and the product-kernel/scaled-concatenation equality has been correctly demoted to an algebraic identity check.

That progress is not enough to authorize the 58,756-article run. The current repository fails on all three launch dimensions:

1. **Operationally**, the default CLS/Dirichlet path has a deterministic configuration mismatch, Track 4 has a broken public/internal API, and a fresh isolated test run ends at **331 passed, 8 failed**.
2. **At scale**, several active paths allocate dense `N x N` arrays. At `N=58,756`, one matrix is 12.86 GiB in float32 or 25.72 GiB in float64. The T3 checkpoint path alone constructs two float64 matrices totaling about 51.44 GiB. Attention, Hadamard kernels, strict Riemannian assembly, and the walker add further dense matrices and workspaces.
3. **Scientifically**, the new `NMI=0.0862` “real publisher” result is not a verified real-data join; it positionally combines a bare 60-row embedding tensor with the first 60 labels from an 80-row synthetic-control artifact. The Fréchet control is unmatched, and the reported power law compares two transforms of the same mean-CLS Euclidean distances rather than the production Hadamard engine.

The correct disposition is therefore:

- **No** full engineering run.
- **No** full scientific run.
- **No** paper/thesis use of the current publisher, Fréchet-premium, or pathway-fidelity claims.
- A small, explicitly non-interpretive engineering canary may begin only after the deterministic startup failures are fixed. The full corpus must wait for sparse/blockwise geometry, sharded extraction, atomic recovery, and a complete provenance contract.

## Scope, method, and limitations

This review covered:

- The requested ten-commit history.
- Both external ledgers: `single_instance_tensor_trace.md` and `in_house_system_map.md`.
- All six requested diagnostics.
- The production runner, complete pipeline, NLI extractor, attention, Dirichlet, Hadamard, Track 4, checkpoint, cache, resume, and verifier paths necessary to test the ledgers' claims.
- The live binary/JSON artifacts used by the diagnostics.
- A read-only structural audit of `data/real_corpus.jsonl`.
- Fresh numerical reconstructions of the fidelity proxy, the production-module pathway, and the centered Fréchet control.
- A fresh test run using an isolated workspace temp directory: **331 passed, 8 failed** in 109.5 seconds.

The initial default pytest invocation was polluted by a pre-existing locked `.pytest_local_tmp` directory on Windows; its setup errors are not treated as code failures. The isolated rerun produced the authoritative 331/8 count. No source file was modified as part of this review. None of the six requested diagnostic scripts has a direct test reference under `tests/`, so their scientific assertions are not protected by regression tests.

## Recent commit history and what it means

The requested history is:

1. `389e4781` — `refactor: align riemannian_eval to canonical N=60 real article dataset and run true graph-level permutation D[perm,perm]`
2. `067cc275` — `refactor: walk back synthetic cross-val circularity, run graph-level distance permutations D[perm,perm], and calibrate Fréchet base anisotropy`
3. `98352f53` — `feat: purge mock code, wire production modules to exp_A/B, verify pathway fidelity, and enforce provenance stamping`
4. `5ff4f7a6` — `feat: purge synthetic fallbacks, implement zero-variance consensus normalization, and refactor fail-closed validation suite`
5. `9fbef9a9` — `Untrack large datasets and ignore them`
6. `55fab740` — `Implement actual data wiring for validation and fighting words pipelines`
7. `07a21e32` — `Comprehensive update for v3 pipeline and thesis materials`
8. `587ec40c` — `refactor: physically consolidate 119 test suites into Block A and Block B without AST mangling to preserve decorators and variants`
9. `733b9374` — `feat: formalize reflector ablation suites and fix Riemannian transport`
10. `85119b0e` — `Harden verification DAG and Dash provenance`

The four most recent commits show a rapid and commendable attempt to remove circularity and mock data. The problem is that the replacement evidence was not traced end to end. `389e4781` and `067cc275` changed the names and calculations in the diagnostics, but the replacement publisher labels still come from the old synthetic `control_unstructured` output. `98352f53` made Experiment B materially more production-like, but the separate fidelity script introduced in that commit does not call the same production pathway. `9fbef9a9` reduced repository size but also made the critical inputs untracked, which means git commit identity alone can no longer establish which corpus or pilot tensor was used.

Historical inspection of `07a21e32` also explains the stale `NMI=0.4653` claim: that script read the 80-row `outputs/non_circular_suite/exp1_unstructured/observer_42.pt`, swept four `K` values by five shear weights, then tested the stated `K=5, lambda=2` configuration without selection correction. Its empirical p-value omitted the plus-one correction and could report `p=0`. That result cannot be recast as independent real-publisher validation.

## A. Architectural integrity, mock purge, and provenance

### A1. What is genuinely improved

- Experiment A rejects missing, malformed, non-floating, wrongly shaped, and non-finite inputs (`scripts/exp_A_original_premise.py:18-43`).
- Experiment B similarly rejects malformed embeddings (`scripts/exp_B_component_necessity.py:33-47`) and only evaluates ratings when their row count exactly matches (`:50-86`).
- Experiment B executes `SpectralPolarity`, `DirichletFusion`, `HadamardFusion`, and `ConformalMetric` (`:222-275`) rather than generating replacement tensors.
- Experiments A and B stamp git commit, embedding SHA-256, timestamp, and effective configuration (`exp_A_original_premise.py:46-83`; `exp_B_component_necessity.py:350-407`).
- The current pilot tensor hash is consistently recorded as `2e0f6a37...ce4` in both `results/exp_A_manifest.json:11-13` and `results/exp_B_manifest.json:87-89`.

These are meaningful improvements. They justify saying that **the two A/B scripts no longer silently synthesize a replacement pilot tensor**.

### A2. The mock/random purge is not complete

It is not accurate to say that every production module is wired to learned or empirically calibrated production code without arbitrary fallback behavior.

1. **Default attention is a frozen random network.** `PipelineRuntimeConfig.use_attention` defaults to `True` (`core/pipeline_config.py:39-50`). `CrossArticleAttention` explicitly describes itself as a “frozen random observer,” with untrained `nn.MultiheadAttention` weights (`core/cross_article_attention.py:1-16,29-58`). The complete pipeline constructs it by default (`core/complete_pipeline.py:1700-1705`) and uses its dense weights to mix articles (`:2547-2618`). This is not a hidden data mock, but it is an arbitrary random transformation in the production path and must not be represented as learned attention.
2. **The temporal GRU is also randomly initialized and frozen** (`core/temporal_gru.py:18-23,47-86`), but the complete pipeline currently hard-disables it because the expected and actual CLS dimensions disagree (`core/complete_pipeline.py:113-119,1578-1586`). Thus a nominally enabled architectural track is actually absent.
3. **Automatic observers can emit deterministic placeholder hypotheses.** When enabled without an LLM, or when generation fails in non-strict mode, `core/auto_observers.py:196-243` substitutes generic “positive/negative semantic axis” statements. The feature is disabled by default, so this is not active in the normal launch, but the purge claim is still too broad.
4. **Several production-adjacent failures are fail-open.** Unsupported NLI constructor fields are filtered and can downgrade CLS extraction (`core/complete_pipeline.py:1529-1537`); missing Dirichlet support becomes a warning (`:1684-1685`); a requested CUDA device silently becomes CPU and `.to()` failures are swallowed (`:1900-1909`); phase-space, provenance, and checkpoint errors become warnings (`:3428-3460,3759-3760,3792-3795`). `core/metric_fusion.py:309-326` can substitute zero walker work when its artifact is absent.
5. **Legitimate stochastic numerical methods must be distinguished from mocks.** RKS frequency sampling, Dirichlet common-random-number weights, clustering seeds, bootstraps, and permutation tests are legitimate when seeded and manifested. They are not evidence of mock data. The frozen attention/GRU and placeholder observer substitutions are different because they can alter semantic output without learned or external calibration.

### A3. The live production wiring is internally inconsistent

The highest-confidence startup defect is the removed `mix_in_rkhs` option:

- `PipelineRuntimeConfig.from_mode_config` explicitly rejects the field (`core/pipeline_config.py:64-72`).
- `DirichletFusionConfig` no longer defines it (`core/dirichlet_fusion.py:44-90`).
- `initialize_full_pipeline` still accepts and passes it (`core/complete_pipeline.py:1431,1654-1669`).
- `run_experiments.py` still injects it when requested and unconditionally when `--dirichlet-fusion` is active (`run_experiments.py:1400-1418`).
- Runtime payload reconstruction still accesses a nonexistent config property (`run_full_experiment_suite.py:1190-1215`), and a synthetic suite path still passes the deleted constructor field (`:6088-6110`).

The CLS mode selected by the suite enables Dirichlet fusion (`run_experiments.py:207-222`), so this is not dead code. Fresh tests reproduce both `TypeError: DirichletFusionConfig.__init__() got an unexpected keyword argument 'mix_in_rkhs'` and `AttributeError: PipelineRuntimeConfig has no attribute mix_in_rkhs`.

Track 4 has a second deterministic mismatch. `run_stress_triggered_cyclic_walk` does not accept `adaptive_tpt_connectivity` (`core/physarum_walk.py:936-945`), but callers/tests pass it, and the return payload refers to an undefined local of that name (`:1088-1105`). The related outer API does expose the option (`:1108-1121`). Four fresh tests fail on this API/name drift.

### A4. Provenance manifests are not launch-ready

The A/B manifests are useful experiment receipts, but neither they nor the full runner form a reproducible production contract.

- The A/B manifests pin a bare tensor, not its ordered article IDs, source corpus, label join, query bank, model/tokenizer revision, weight hash, package environment, CUDA/driver environment, invocation, or dirty-tree state. Both manifests were produced at `98352f53`, not the reviewed HEAD `389e4781` (`results/exp_A_manifest.json:11-13`; `results/exp_B_manifest.json:87-89`).
- `data/real_pilot_embeddings.pt`, `outputs/non_circular_suite/exp1_unstructured/observer_42.pt`, `data/control_unstructured.jsonl`, and `data/real_corpus.jsonl` are untracked/ignored. `.gitignore:4-9,85-87` explicitly excludes outputs and JSONL corpora.
- Production observer artifacts record git state, timestamps, IDs, titles, sources, URLs, and partial configuration (`core/complete_pipeline.py:4148-4210`), but not the corpus SHA-256, ordered-ID digest, model/tokenizer revision or weights, query-bank hash, dependency lock, CUDA/driver details, or exact command.
- `run_experiments.verify_articles` treats an absent or empty hash list as success, retains records on exceptions, and filters rather than asserting an exact ordered bijection (`run_experiments.py:389-447`). Its provenance backfill hashes configuration strings rather than the actual basis/weight artifacts (`:450-499`).
- The high-level launcher does not forward `--manifest` or `--require-cuda` in its subprocess command (`run_full_experiment_suite.py:526-550`), although the lower runner supports both.
- Suite and checkpoint writes are direct, non-atomic writes; some errors are swallowed (`run_full_experiment_suite.py:401-451`; `core/waterfall_checkpoints.py:207-415,481-488`; `core/complete_pipeline.py:4239-4244`). File existence can therefore be mistaken for a valid completion.

**Architectural conclusion:** the core refactor is directionally better, but “mock purge complete,” “all modules production-wired,” and “provenance-ready” are all false at this snapshot.

## B. Non-circular validation and retraction assessment

### B1. The `NMI=0.6205` retraction is correct

The old cross-validation implementation at `07a21e32` ran `sythgen/high_quality_articles.jsonl`, clustered the generated features, and evaluated against generator-provided `perspective_tag` labels. It measured recoverability of the generator's own synthetic structure, not external generalization. The walk-back is required and should remain permanent.

`data/non_circular_results.json:6-9` still publishes `NMI=0.6205388, ARI=0.4188741`, however. That stale tracked artifact must be removed, quarantined, or marked retracted so it cannot be accidentally re-ingested.

### B2. The replacement `NMI=0.0862` must also be retracted as a real-publisher result

The number itself is reproducible; its provenance claim is not.

- `run_cross_validation.py:40-59` loads 60 rows from `data/real_pilot_embeddings.pt`, then loads publisher labels from `outputs/non_circular_suite/exp1_unstructured/observer_42.pt` and truncates the resulting label vector to 60.
- `run_riemannian_eval.py:95-111` performs the same positional splice.
- The label artifact contains 80 articles and explicitly records `meta.corpus="control_unstructured"`, `n_articles=80`, and `git_dirty=true`. Its 80 titles and publisher fields match `data/control_unstructured.jsonl` 80 out of 80. The text is templated synthetic prose; URLs are empty and provenance sources are `unknown`.
- `real_pilot_embeddings.pt` is a bare, untracked tensor. There is no `data/real_pilot_embeddings.manifest.json` and no article-ID vector with which to prove that embedding row 0 corresponds to metadata row 0.

Accordingly, `0.0862378935/-0.0087124362` in `data/cross_val_metrics.json:14-16` is only the score produced by the present positional pairing. At best it uses synthetic publisher assignments; at worst it pairs unrelated rows. It is not evidence about real publisher labels.

The method is also not cross-validation in the usual sense. `run_cross_validation.py:64-102` separately averages the same 60 articles over full, A, and B observer subsets and clusters each matrix. No model is trained on A and transferred to B, no article is held out, there are no folds or repeated splits, and there is no uncertainty or permutation test. “Single observer split association” is an accurate name; “held-out observer cross-validation” is not.

### B3. The canonical Riemannian/Hadamard comparison is likewise not established

The newer values in `data/riemannian_metrics.json` are numerically current, but they inherit the unjoined synthetic labels above. There are additional limitations:

- The “deployed Hadamard” branch is actually mean-CLS Euclidean distance followed by an ad hoc density scale (`scripts/run_riemannian_eval.py:116-127`). It never invokes `HadamardFusion`, so the label is incorrect.
- The observed statistic averages ten clustering seeds (`:40-58`), whereas every null permutation uses one fixed clustering seed (`:60-88`). The observed and null statistics are therefore not identical.
- The cluster count is taken from target-label cardinality (`:42-43,68`), making the target part of model selection.
- No Euclidean permutation p-value is computed, even though the tensor ledger says all three metrics have `p>.18`.

The tensor ledger's current numbers (`single_instance_tensor_trace.md:12`) are `NMI=0.1303` Euclidean, `0.1278` Riemannian with `p=.182`, and `0.0993` proxy-Hadamard with `p=.734`. The architecture map still asserts the obsolete `NMI=0.4653, ARI=0.3527, p<.001` (`in_house_system_map.md:14`). The map's claim that it is authoritative is therefore contradicted by the newer ledger and by the underlying artifact provenance.

Low NMI cannot “validate the core premise” that publisher identity is uninformative. It is also compatible with a bad row join, a weak proxy metric, insufficient power, or failed representation learning. The only defensible conclusion is that the current analysis does not demonstrate publisher association.

### B4. The machine-epsilon identity check is valid but scientifically neutral

Experiment B standardizes semantic and spectral features, multiplies their same-bandwidth RBF kernels, concatenates the standardized features, and computes the corresponding RBF (`scripts/exp_B_component_necessity.py:317-336`). The recorded maximum difference, `1.6653345369377348e-16` (`results/exp_B_manifest.json:190-202`), is exactly what follows from

`exp(-||x-x'||²/2) * exp(-||z-z'||²/2) = exp(-(||x-x'||²+||z-z'||²)/2)`.

This confirms an implementation identity to floating-point precision. It does not show that the product is necessary, improves prediction, or captures a new interaction. The ledger's retraction to “identity check, not empirical discovery” is correct.

More broadly, Experiment B has not completed its labeled component-necessity test. Every labeled evaluation is marked `skipped` because the ratings file contains 30 rows for 60 embeddings (`results/exp_B_manifest.json:148-209`; `gemini_blind_ratings.csv:1-31`). The production matrices were generated, but component necessity remains untested.

### B5. The Fréchet `81.43%` versus `92.79%` premium is a control artifact

The script's real residuals are measured around each article's eight-observer mean, so they sum to zero per article. Its control adds eight independent Gaussian vectors without centering those vectors per article (`scripts/exp_frechet_negative_control.py:79-89`). The mean of those eight noise vectors therefore contributes additional variance to the control consensus. The control is not orthogonal despite the comment, is not encoder-generated, and is not covariance- or spectrum-matched.

Fresh reconstruction gave:

| Statistic | Real frames | Original Gaussian control | Centered, variance-matched control |
|---|---:|---:|---:|
| Total variance | 18.152046 | 18.134008 | 18.152046 |
| Consensus variance | 1.308058 | 3.367464 | 1.308058 |
| Observer collapse | 92.793884% | 81.430117% | 92.793884% |
| Cumulative double collapse | 94.709936% | 94.521576% | 94.709936% |

Once the null is centered consistently and variance matched, it is exactly identical under these statistics. That is not surprising: the metric only decomposes total versus consensus variance; it does not inspect directional structure. The `+11.36%` “adversarial framing premium” therefore disappears.

The ledger also misnames `92.79%` as “double collapse.” In the code, `92.79%` is only observer collapse (`scripts/exp_frechet_negative_control.py:34-40`); cumulative observer-plus-PC1 collapse is `94.71%` (`:42-50,91-103`). The original control's cumulative result is already very close at `94.52%`. Finally, one random seed with no repeated controls, confidence interval, or hypothesis test cannot support the script's “significantly greater” wording (`:105-114`). `81.43%` should not be called transformer “base anisotropy,” because the null vectors are Gaussian perturbations, not transformer outputs.

`data/frechet_metrics.json:1-12` is also stale: it records a different variance scale (`197.80`) and older `92.65%/94.04%` results and is not the output of the current negative-control script.

### B6. The reported power law is proxy-versus-proxy, not production fidelity

`scripts/exp_fidelity_pathway_alignment.py` imports `HadamardFusion` and `MaternKernel` but does not use them (`:28-30`). The tested matrices are:

- `D_riemannian = compute_matern_distance_matrix(mean_cls)` (`:99-102`), which is a monotone Matérn-kernel transform of mean-CLS Euclidean distance (`core/riemannian_geometry.py:166-208`).
- `D_raw = cdist(mean_cls, mean_cls)` (`exp_fidelity_pathway_alignment.py:104-106`).
- `D_conformal`, which applies an ad hoc per-article variance scale to that same Euclidean matrix (`:108-113`).

That construction forces pre-scaling Spearman correlation to exactly 1.0. Fresh reproduction of the published proxy gives:

| Comparison | Spearman | Pearson | Exponent `b` | R² |
|---|---:|---:|---:|---:|
| Proxy, pre-scale | 1.000000 | 0.825670 | 4.02398 | 0.85394 |
| Proxy, post-scale | 0.999456 | 0.820053 | 4.15995 | 0.85049 |
| Proxy, post-scale excluding article 34 | — | — | 3.48288 | 0.86232 |

The fit is primarily the inverse shape of a Matérn transformation of the same Euclidean distance, not evidence that a production engine expands a separate geodesic manifold.

A fresh reconstruction using the actual modules and configuration exercised by Experiment B produced:

| Comparison | Spearman | Pearson | Exponent `b` | R² |
|---|---:|---:|---:|---:|
| Production-module Hadamard, pre-scale | 0.24343 | 0.27367 | 0.17345 | 0.07859 |
| Production-module Hadamard, post-scale | 0.36282 | 0.40393 | 0.26687 | 0.16786 |

Thus `b=3.48...4.16`, `R²=0.8623`, and CI `[2.34,5.21]` do **not** characterize the deployed Hadamard engine. The script is internally contradictory as well: it computes and prints pre-scaling `b≈4.024, R²≈.854` (`exp_fidelity_pathway_alignment.py:115-121`), then hard-codes `b_pre=1.0000, R²=1.0000` and says conformal scaling generates 100% of the effect (`:146-152`). It contains no Mantel test, Spearman test, or Pearson test despite the ledgers' claims. The fail-closed external suite explicitly refuses the requested fidelity comparison because a continuous geodesic engine is not available (`scripts/run_external_validation.py:309-314`).

The bootstrap (`exp_fidelity_pathway_alignment.py:47-82`) can quantify variability in this proxy calculation, but it cannot repair construct invalidity. A tight or non-unit CI for the wrong comparison is not production validation.

### B7. What survives the audit

The following narrower findings remain defensible:

- The original synthetic `NMI=0.6205` result was circular and is correctly retracted.
- The product-kernel/scaled-concatenation machine-epsilon result is a correct identity check.
- Experiment A's completed Householder/orthogonal map preserves norms to floating-point error; its manifest records maximum norm error `5.72e-6`, mean aligned held-out Frobenius error `18.55`, and shuffled error `20.75` (`results/exp_A_manifest.json:15-20`). This is a valid numerical property of the constructed map, not proof of a corpus-wide framing-space isomorphism. It uses one 48/12 split and one shuffled baseline.
- Experiment B successfully emitted finite production-module tensors and hashes for the 60-row pilot. Its label-dependent ablations were all skipped, so no component-performance conclusion survives.
- The full corpus file is structurally clean at the JSON level, as described below.

## C. Readiness for the 58,756-article corpus sweep

### C1. Corpus audit

The input itself is the strongest part of launch readiness. A full streaming audit found:

- 58,756 valid JSON records.
- No malformed records, blank lines, or empty text.
- 58,756 unique non-null `id` values; no collisions in the audited canonical identifiers.
- File size: 554,620,642 bytes.
- SHA-256: `b003a856580bf9bb3ae09f90913426728793525f9da8572b279ea8820ce3e7c0`.
- Text is uniformly available through `content`.
- Every record currently forms one paragraph under the extractor's blank-line splitting behavior.
- `publication` is absent from all records. `source` has 3,604 distinct values.

That last point matters scientifically: production label validation looks first for semantic labels and does not permit source fallback by default (`core/complete_pipeline.py:190-200,507-553`). A full run therefore will not reproduce the N=60 “publisher” analysis unless a separately documented, ID-joined labeling plan is introduced. `UNAVAILABLE` must not be promoted to validation success.

### C2. Launch blockers versus scientific-claim blockers

| Class | Blocker | Consequence |
|---|---|---|
| **Launch blocker** | Deleted `mix_in_rkhs` option still passed through CLS/Dirichlet paths | Deterministic startup failure |
| **Launch blocker** | Track 4 missing parameter/undefined local | Deterministic Track 4 failure |
| **Launch blocker** | Dense attention, pairwise distances, kernels, T3 topology | RAM/VRAM exhaustion before or during full run |
| **Launch blocker** | Sequential, unbatched NLI loop and ineffective cache injection | Prohibitive runtime and repeated extraction |
| **Launch blocker** | No shard-level checkpoint; non-atomic output and existence-based resume | Late interruption can lose work or accept corrupt output |
| **Launch blocker** | Missing resource preflight, durable log, heartbeat, and fail-closed CUDA requirement | Unobservable, unsafe long-running operation |
| **Scientific blocker** | Embeddings and publisher labels are not joined by canonical ID | Publisher metrics uninterpretable |
| **Scientific blocker** | “Hadamard” validation uses Euclidean proxy | Production pathway unvalidated |
| **Scientific blocker** | Fréchet null is uncentered/unmatched | `+11.36%` premium unsupported |
| **Scientific blocker** | Power law compares transforms of the same Euclidean matrix | Expansion and fidelity claims unsupported |
| **Scientific blocker** | Ledgers and tracked JSON artifacts contradict current results | Claim provenance unsafe |
| **Scientific blocker** | Experiment B labels mismatch and all outcome ablations skip | Component necessity unknown |

Operational fixes would not automatically clear scientific claims, and correct labels would not make the current full-corpus implementation runnable. Both sets of gates must pass independently.

### C3. Dense-matrix infeasibility

For `N=58,756`:

- `N² = 3,452,267,536` cells.
- One float32 matrix = 13,809,070,144 bytes = **12.86 GiB**.
- One float64 matrix = 27,618,140,288 bytes = **25.72 GiB**.

Concrete allocations include:

- Cross-article attention returns `[N,N]` weights (`core/cross_article_attention.py:66-103`), is enabled by default, and is consumed in the complete pipeline (`core/complete_pipeline.py:2547-2618`). The returned float32 matrix alone is 12.86 GiB; score workspaces and feature tensors add substantially more.
- The T3 checkpoint creates `np.eye(N)` and `np.zeros((N,N))` in default float64 (`core/complete_pipeline.py:3714-3758`): **51.44 GiB combined** before serialization, even though they contain only self-bonds and no cracks.
- Track 4 builds a full `torch.cdist` matrix before selecting neighbors (`core/physarum_walk.py:79-88`), defeating the memory advantage of a k-nearest-neighbor graph.
- Hadamard fusion constructs two dense distance matrices, two kernels, and their product (`core/hadamard_fusion.py:158-216`). Its default is `use_nystrom=False` (`:88-96`), and the current Nyström function accepts an already materialized full kernel (`:260-321`), so toggling it would not remove the upstream `N²` allocation.
- Strict Riemannian fusion constructs multiple dense pairwise matrices and performs classical MDS (`core/hadamard_fusion.py:581-627`).

The six N=60 diagnostics do not exercise any of these scale paths. No amount of RAM tuning turns this into a responsible 58,756-row launch; the geometry must be redesigned around sparse or blockwise representations.

### C4. Extraction and cache explosion

For one one-paragraph article, the extractor loops over eight hypothesis pairs and runs A and B separately (`core/nli_extraction.py:488-523`): 16 transformer forward calls. `extract_batch` still loops one article at a time (`:543-601`), and the compatibility wrapper disables its progress reporting (`:985-995`). It retains both logits and all eight 1536-dimensional observer vectors (`:1000-1025`).

For 58,756 articles, one extraction pass is therefore:

`58,756 * 8 * 2 = 940,096` sequential model calls.

The cache cannot currently provide the advertised reuse:

- The in-memory fingerprint omits `content`, the only text field used by this corpus (`core/complete_pipeline.py:2289-2303`).
- Disk validation checks only article count and channel (`:4018-4031`). A different corpus with the same count can be accepted.
- Loaded cache data is injected without restoring `_nli_cache_fingerprint` (`:4101-4104`), so `process_month` misses the fingerprint comparison and recomputes extraction.
- Cache saving happens only after the complete extraction/pass (`:4122-4135`), with no per-shard recovery.

With the default five kernels, two channels, and three seeds (`run_full_experiment_suite.py:7051-7117`), the broken reuse can produce up to 30 extraction passes, or **28,202,880 sequential forward calls**, before controls. Even ideal reuse would still require a measured, batched, sharded extraction phase and a disk/VRAM budget.

### C5. Resume and fault tolerance are unsafe

- Resume treats `observer_{seed}.pt` existence as completion and never opens or validates checksum, schema, count, IDs, or configuration (`run_full_experiment_suite.py:7657-7726`).
- Final observer output is a direct `torch.save` to its final name (`core/complete_pipeline.py:4239-4244`). A partial file can satisfy the existence check.
- NumPy checkpoints and their manifest are also direct writes (`core/waterfall_checkpoints.py:207-415,481-488`).
- Suite config and completion-marker errors are swallowed (`run_full_experiment_suite.py:401-451`).
- Although resume calculates missing seeds, the actual call passes the complete `args.seeds` list (`:7657-7709,7823-7842`), rerunning and potentially overwriting completed seeds in a partially complete combination.
- Resume restores only selected fields (`:6705-6761`), allowing mixed configurations.
- Failure at `:7847-7853` breaks only the inner kernel loop; later channels/corpora can continue after a failed leaf.

There is no article- or shard-level recovery. An interruption near the end of 940,096 calls can discard the whole extraction pass.

### C6. Suite defaults and observability are not production-safe

- The suite defaults to `--limit 500`, not the full corpus (`run_full_experiment_suite.py:7051-7055`). A launch command must explicitly bind an immutable count/manifest rather than rely on a manual limit.
- Defaults include real plus three controls (`:7111-7117`), but `data/control_constant.jsonl`, `data/control_shuffled.jsonl`, and `data/control_random.jsonl` are currently absent. A long real run can be followed by a predictable control failure.
- The high-level subprocess streams output without capturing a durable run log (`:565-588`).
- CUDA can silently fall back to CPU (`core/complete_pipeline.py:1900-1909`), and the high-level suite does not forward the lower-level `--require-cuda` safeguard.
- There is no startup estimate or hard gate for RAM, VRAM, disk, model throughput, output growth, NaN/Inf rate, or wall-clock timeout.

### C7. Fresh test evidence

The authoritative isolated run was:

`pytest -q tests --basetemp D:\belief-transformer\V3\.codex_arch_pytest_20260813a`

Result: **331 passed, 8 failed**.

The eight failures are:

1. Four Track 4 API/name failures in `tests/integration/dag_orchestration/test_core_stability.py`, caused by `core/physarum_walk.py:936-945,1088-1105`.
2. Two verifier control-ordering failures. `analysis/verification/verify_run.py:382-431` only consults fallback control metrics after a valid primary check returns false; when primary feature provenance is unavailable it returns `pass=None` and never applies either a supportive or non-supportive fallback. The tests require an explicit true/false decision.
3. One `mix_in_rkhs` constructor failure at `core/complete_pipeline.py:1654-1669`.
4. One stale runtime-config access at `run_full_experiment_suite.py:1190-1215`.

This is not a cosmetic red suite: all eight failures lie directly in launch or verification pathways under review.

## Required safeguards and release gates

### Gate 1: restore a coherent executable architecture

1. Remove `mix_in_rkhs` from every caller, payload migrator, mode, manifest, and test, or restore it as one deliberately specified API. Do not keep a half-removed compatibility flag.
2. Make Track 4's `adaptive_tpt_connectivity` contract consistent across public API, internal config, call sites, return payload, and tests.
3. Decide whether frozen random attention and GRU are scientific components, ablations, or legacy code. The production default should not include an untrained semantic mixer. Record their state/seed/weights if retained.
4. Convert required Dirichlet, phase-space, provenance, checkpoint, and CUDA failures from warnings to fatal production errors.
5. Require the full test suite to be green from a clean checkout and clean temp directory.

### Gate 2: split extraction from geometry

1. Implement true tokenized batching for the NLI model rather than 940,096 single-example forwards.
2. Extract logits and CLS together once, in deterministic shards, because the model already emits both.
3. Store shard ranges and ordered canonical IDs with tensor shapes, dtypes, finite-value counts, and hashes.
4. Write each shard to a temporary name, flush/sync it, atomically rename it, reopen it, validate it, and only then write a success marker.
5. Resume only missing verified shards. Injected caches must restore and verify the complete fingerprint.

### Gate 3: remove global dense geometry

1. Replace cross-article attention with local/sparse attention or disable it for the corpus run.
2. Build kNN graphs with approximate-nearest-neighbor or blockwise search; never form full `torch.cdist(N,N)`.
3. Use sparse edge lists for T3. Never serialize dense identity/zero topology matrices.
4. Compute kernels blockwise or via landmarks/features from the outset. Do not materialize a full kernel before Nyström.
5. Put an automated invariant in the production path that rejects any attempted `N x N` allocation above a configured threshold.

### Gate 4: make provenance immutable and replayable

At minimum, the run manifest must bind:

- Corpus SHA-256 (`b003a856...e7c0` for the audited file), byte size, exact row count, schema version, and ordered canonical-ID digest.
- Duplicate/collision statistics and the UID-generation contract.
- Query-bank contents/hash and observer ordering.
- Model name, immutable model revision/weight hash, tokenizer revision/hash, max length, paragraph policy, and preprocessing code version.
- Git commit and dirty diff hash, exact CLI invocation, environment/dependency lock, Python/PyTorch/CUDA/cuDNN/driver/GPU details.
- Every stochastic seed and every basis/weight artifact hash.
- Per-shard and final artifact hashes, shapes, dtypes, article ranges, and parent dependencies.

Manifest mismatch, missing fields, wrong order, or hash failure must stop the run. Positional truncation must be prohibited.

### Gate 5: repair the scientific validation design

1. Generate a tracked pilot manifest containing the 60 tensor rows' canonical IDs, tensor hash, source corpus hash, model/query configuration, and label-source hash.
2. Join labels by canonical ID with a one-to-one/bijection assertion. Never combine tensors and metadata by row position unless the same signed manifest binds both orders.
3. Regenerate a manually audited real pilot through the actual production path and persist the semantic, spectral, Hadamard, conformal, and theoretical matrices with hashes.
4. Drop the present pathway-fidelity claim or implement the acknowledged missing geodesic engine. Compare independently constructed metrics and include the trivial Matérn-of-Euclidean baseline.
5. Replace the Fréchet null with neutral/random prompts run through the same encoder, or with centered covariance/spectrum-matched controls across many seeds. Report observer collapse and cumulative collapse separately.
6. Use identical observed and permutation statistics, nested/held-out hyperparameter selection, repeated splits, uncertainty intervals, and explicitly defined null hypotheses.
7. Complete Experiment B ratings with exact ID joins; do not treat skipped evaluations as neutral evidence.
8. Machine-generate both ledgers from signed manifests and quarantine stale JSON results.

### Gate 6: add production observability and resource policy

Before launch, require:

- A disk/RAM/VRAM preflight with at least 20-30% headroom and room for a second recoverable copy.
- Structured durable logs with stage, shard, articles complete, calls/sec, articles/sec, ETA, cache hits/misses, retry counts, RAM/VRAM, disk, NaN/Inf counts, and artifact hashes.
- A heartbeat and explicit stalled-job timeout.
- A measured cost/time projection from canaries, not a theoretical estimate.
- Fail-closed handling for OOM, corrupt shard, checksum mismatch, CUDA loss, model revision drift, and output schema drift.

## Required canary ladder

1. **N=32 functional canary:** execute the exact launch command/path. Verify every required track, exact ID coverage/order, finite tensors, schema, hashes, deterministic same-seed replay, and no fallback warnings.
2. **N=256-500 extraction canary:** measure examples/calls per second, peak RAM/VRAM, shard size, disk growth, and cache reuse. Kill the process mid-shard and prove exact recovery without duplicate or missing IDs.
3. **N≈2,000 scale canary:** demonstrate near-linear or `N log N` growth. Instrument allocations and fail the gate if any production stage creates an `N x N` tensor.
4. **5% stratified canary (`N≈2,938`):** sample by source, date, and text length. Run all intended tracks and controls, calculate output/time projections, inspect distribution drift, and repeat the same seed twice within declared tolerances.
5. **Full run:** only after all prior gates pass, all intended control files exist and are preflighted, corrupt-shard tests fail closed, cache reuse is proven, the exact corpus order is replayable from the manifest, and a human signs the launch manifest.

## Final authoritative opinion

The V3 refactor is a promising **small-sample research prototype**, not a production-safe full-corpus engine. Its strongest achievements are the fail-closed A/B loaders, genuine use of core modules in Experiment B, better local tensor provenance, the honest retraction of synthetic `NMI=0.6205`, and a clean 58,756-record input file.

Its present weaknesses are decisive: broken launch APIs, red tests, frozen random default attention, a disabled temporal track, global quadratic memory, sequential extraction, ineffective caching, unsafe resume semantics, incomplete provenance, synthetic/unjoined publisher labels, an unmatched Fréchet control, a proxy pathway-fidelity test, and contradictory ledgers.

**Recommendation: do not start the 58,756-article sweep.** First make the exact production path green and fail-closed, split and shard extraction, eliminate all full pairwise allocations, prove interruption-safe recovery, bind the corpus/model/query/order in an immutable manifest, and rerun non-circular validation on ID-joined real data through the actual production matrices. Until those conditions are met, a full run would be both operationally hazardous and scientifically non-interpretable.
