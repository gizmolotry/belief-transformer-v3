# Representation-channel audit (2026-09-06)

## Scope and terminology

This audit asks which stored representation each empirical analysis actually consumes. It does
not infer provenance from names such as "contrastive," "axis," or "shear."

- **A-side verdict score:** `s_A = P(entailment | article, h_A) -
  P(contradiction | article, h_A)` from `nli_pairs[*].logits_raw`.
- **Paired scalar contrast:** `c = s_A - s_B`, requiring fresh logits for both hypotheses.
- **Hidden-state contrast:** `mean_pool(article, h_A) - mean_pool(article, h_B)`, stored by the
  current extractor in `cls_per_bot`. Despite the legacy field name, these are mean-pooled
  differences, not literal CLS tokens.

"A-side" below is unrelated to the manuscript's separately named **MBFC Axis A**.

## Audit table

| Analysis / claim | Executable and actual field | Provenance status | Effect of the A-side verdict bug |
|---|---|---|---|
| Exploratory internal PCA: 55.8% + 22.3%; publisher regression; high/low quartiles | `scripts/exploratory_internal_structure_analysis.py` loads `nli_pairs[*].logits_raw`, computes `P(E)-P(C)`, and applies PCA to the resulting N-by-8 matrix. | Confirmed from code, recorded source/input hashes, and exact numerical reconstruction. | **Affected.** These are A-side verdict-score results, not paired-displacement results. The numbers remain reproducible for that narrower measurement. Calling score dispersion "shear" is unsupported. |
| Q0/Q1 correspondence: 78.09%/78.28%, Procrustes 0.173, distance Spearman 0.796 | `scripts/q0_q1_robustness.py` uses the same `logits_raw` score loader. | Confirmed from code and result hashes. Historical artifacts lack article IDs and embedded prompt-bank metadata. | **Affected.** This is correspondence between two A-side score artifacts. It does not validate A-minus-B scalar or hidden-state geometry. Existing row-order and Q1-authorship caveats remain. |
| Original N=60 axis-coherence / permutation lineage | `scripts/run_gpu_validation.py` extracts `cls_per_bot`; `scripts/recompute_pvalue.py` loads `ideo_emb`, `ctrl_emb`, and `rand_emb` from `data/controlled_ideo_embeddings_xlarge.pt`. The current extractor defines `cls_per_bot` as `mean_pool(A)-mean_pool(B)`. | The cache has shape 60x8x1536 for all three conditions and reproduces the observed within-group means in `validation_output.txt`. No extraction manifest was found. Neither a seeded permutation output nor the claimed `p<0.001` is preserved in an auditable result artifact; both scripts use unseeded `torch.randperm`. | **Not contaminated by the A-side verdict bug at the code-channel level.** The p-value is nevertheless not independently auditable from a saved run and should not be called a clean confirmed positive until a frozen, seeded rerun is preserved. |
| MBFC N-Dense Axis A (`r=0.431`) and Axis B (`r=-0.499`) headline claims | No repository script or result artifact was found that emits these exact numbers. `scripts/run_mbfc_continuous_validation.py` preferentially loads `cls_per_bot` and computes one corpus-derived PCA direction, but it neither names two axes nor saves the quoted results. | **Unauditable.** Exact figures currently appear only in `paper/latex_v2/03_results.tex` and arrived in commit `bfdad34a`. | Channel cannot be assigned responsibly. Do not infer that either number is A-side or hidden-state based. Recompute both from explicit definitions before retaining them as empirical claims. |
| Later exploratory MBFC two-PC null (`R2=0.192`, adjusted `R2=0.067`, overall `p=0.251`, LOO predictive `R2=-0.490`) | `scripts/exploratory_internal_structure_analysis.py`; two PCs of the A-side N-by-8 verdict-score matrix. | Confirmed and hash-bound in `paper/exploratory_internal_structure_results_20260828.json`. | **Affected in attribution, not arithmetic.** It is a null for the A-side score structure, not for A-minus-B displacement geometry. |
| Reported full-SPD tensor calibration failure | `scripts/calibrate_full_spd_tensor.py` and `core/full_spd_simulation.py` use artificial coordinates, analytic potentials, and manufactured errors. No NLI field enters this calibration. | Current manuscript correctly says no auditable run artifact survived and makes no positive tensor claim. | **Unaffected.** The failure cannot be blamed on A-side verdict extraction. Its precise failure metric remains unknowable without the missing artifact. |
| Proposed real full-SPD tensor test | Bank preparation provides `T0` as N-by-8-by-3 logits and `T1` as N-by-8-by-1536 hidden representations. The cell builder derives its training target from across-observer variation in `P(E)-P(C)` from `T0`; geometry uses the hidden channel. | **Stopped, not pending.** The contemporaneously reported synthetic calibration failure triggered the frozen stop rule, although its missing run artifact prevents independent verification. No real endpoint was authorized or run. Q1/Q2 also remain paraphrases rather than independently authored banks. | The mixed-channel target is a defect in the dormant design, not an active repair item. It must not be "fixed toward execution." Reviving the question would require an explicitly authorized, newly named preregistration that resolves the channel and independent-bank blockers before any new calibration or real run. |
| Pilot component/kernel ablations (`exp_B_component_necessity.py`) | Loads 60x8x1536 hidden representations and evaluates RKS/fusion/affinity constructions. | `results/exp_B_manifest.json` says all label evaluations were skipped because there were 60 embeddings but only 30 Gemini ratings. Historical extraction provenance is absent. | No direct A-side-logit dependence in inspected code. No label-based kernel necessity result was established. |
| Kernel-necessity preregistration | Specifies centered article-level hidden representations, matched arms, and fresh blinded labels. | Protocol exists; a matching completed runner/result was not located. | Not directly affected, but the input must be described as mean-pooled A-minus-B rather than a literal CLS token. |
| Observer-recentering robustness packet | `core/observer_local_recompute.py` reads `cls_per_bot` plus a persisted RKS basis; the evaluated labels are synthetic. | Conditional synthetic evidence; historical extraction/basis provenance is incomplete. | No direct A-side-verdict dependence in inspected computation. |

## Direct extraction finding

`core/nli_extraction.py` executes both hypotheses, but the two output tracks differ:

1. `logits_raw` appends only `enc_a['logits']`.
2. `cls_per_bot` appends `enc_a['mean_pool'] - enc_b['mean_pool']`.

A controlled B-only intervention against the actual extraction function changed
`cls_per_bot` and produced exactly zero change in `logits_raw`.

## Immediate disposition

1. Preserve and relabel the historical A-side results; do not discard or silently replace them.
2. Treat the fresh paired scalar run as a separate exploratory analysis, not as a retroactive
   validation of hidden-state displacement geometry.
3. Recompute the two headline MBFC axes from explicit, separately named channel definitions.
4. Preserve a seeded output before relying on the original coherence permutation p-value.
5. Keep the tensor experiment stopped. The mixed target/channel design is documented as a blocker,
   not scheduled for repair. Any revival is a new experiment requiring explicit authorization,
   a new preregistration, independent banks, and a coherent hash-bound channel definition.

This audit does not alter prior confirmatory decisions or revive the stopped tensor test.
