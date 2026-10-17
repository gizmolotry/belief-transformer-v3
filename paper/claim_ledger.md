# Belief Transformer: Audited Master Claim Ledger

**Version:** 4.3 (Consolidated Gate 3 Multi-View Invariance & Controlled Hidden-State Specificity Audit)
**Total Claims:** 39
**Date:** October 2026
**Audit Standard:** Dual ratio accounting, exact neighbor retention ($R_{10}$), matched surface-control contrast, distributional distance support qualification, KS null rejection reporting, and complete provenance tracking.

---

## Executive Preamble & Epistemic Scope

This Master Claim Ledger preserves the complete epistemic history of the Belief Transformer project across all empirical phases:
- **Phase 1: Kuhnian Framework Foundations & Generalizability Theory** (`CLM_01` – `CLM_20`)
- **Phase 2: Relational Ontology, Gauge Invariance & Observer Space** (`CLM_21` – `CLM_26`)
- **Phase 3: Psychometric Invariance, Measurement Multiverse & Construct Identifiability** (`CLM_27` – `CLM_33`)
- **Phase 4: Multi-View Representation Invariance & Controlled Hidden-State Specificity** (`CLM_34` – `CLM_35`)

The ledger explicitly records and preserves all historical claims, including failed tests (`CLM_23A_SCORE_MIXTURE_TRANSITION_CONTINUUM`), prospective protocols (`CLM_26_HUMAN_BENCHMARKING_PROTOCOL`), exploratory mechanisms (`CLM_24B_CAUSAL_ACTIVATION_PATCHING_CONSTITUTION`), partially supported criteria (`CLM_22_SEMANTIC_COORDINATE_REDUCIBILITY`), and qualified support boundaries (`CLM_27` through `CLM_35`), preventing unearned generalizations or post-hoc erasure of falsified predictions.

---

## Summary Table of Claims

| ID | Claim Summary | Status | Effect Size | 95% CI / Bound | Artifacts |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `CLM_01_G_THEORY_VARIANCE_PARTITIONING` | Variance decomposition explicitly distinguishes cell-mean A x P interaction (37.32%, df=476) from item-level A x P interaction (13.02%, df=476) and item residual error (74.42%, df=12600). | **SUPPORTED** | Cell-mean: AP = 37.32% (df=476), P = 16.66% (df=4), B(P) = 15.33% (df=10), A = 0.00% (df=119); Item-level: Residual = 74.42% (df=12600), AP = 13.02% (df=476) | sigma^2_AP: [31.64%, 42.92%]; E rho^2_AP(3,8): [0.7500, 0.8159] | paper/kuhnian_g_theory_results.json, paper/kuhnian_d_study_results.json |
| `CLM_02_FRAMEWORK_BLIND_SCALAR_PREDICTION` | A framework-blind scalar model cannot predict stance across multiple frameworks from article text alone (R^2 = 0.0082), whereas per-framework models explain up to 49.03% of stance variance. | **SUPPORTED** | Scalar R^2 = 0.0082 vs Per-Framework R^2 = 0.4903 (gain: +0.4821) | Per-Framework fold R^2: [0.1937, 0.5740] | paper/kuhnian_framework_prediction_results.json |
| `CLM_03_UNSEEN_FRAMEWORK_GENERALIZATION_DROP` | Zero-shot transfer of this particular semantic framework representation drops from R^2 = 0.3650 on seen frameworks to R^2 = -0.1085 on unseen frameworks. | **SUPPORTED** | Regime 2 R^2 = 0.3650 vs Regime 4 R^2 = -0.1085 (delta R^2 = -0.4734) | Regime 2 r = 0.6042; Regime 4 r = 0.0381 | paper/kuhnian_framework_prediction_results.json |
| `CLM_04_RELATIONAL_INTERACTION_PERMUTATION_GATE` | The relational interaction gain Delta R^2(M2 - M1) = +0.2087 decisively exceeds the article-level framework permutation null (Monte Carlo p = 0.0010). | **SUPPORTED** | Observed Delta R^2 = +0.2087 vs Permutation Null Mean = -0.1140 (Monte Carlo p = 0.0010) | Null 95% CI: [-0.1420, -0.0740] | paper/kuhnian_framework_prediction_results.json |
| `CLM_05_RELIABILITY_NORMALIZED_TRANSLATABILITY` | Translation test performance confirms syntactic paraphrase is translatable (test R^2 = 0.5488, r = 0.7741), while cross-operator mapping fails out-of-sample (test R^2 <= 0.00 across all 14 mapping families; T5 RidgeCV test R^2 = -0.0287). | **SUPPORTED** | Paraphrase test R^2 = 0.5488 (test r = 0.7741); Operator shift test R^2 <= 0.00 across all 14 models (T5 test R^2 = -0.0287) | Held-out test R^2 <= 0.00 across all 14 mapping models for operator shift | paper/kuhnian_translation_complexity_results.json |
| `CLM_06_TRANSLATION_COMPLEXITY_OVERFITTING_GAP` | High-capacity nonlinear models (Poly-2, MLP, KNN) exhibit severe overfitting divergence for cross-framework mappings (train R^2 > 0.85, test R^2 < 0.10, generalization gap > 0.84). | **SUPPORTED** | MLP / Poly-2 Generalization Gap Delta R^2 = +1.026 | Test R^2 < 0.00 for c >= 8 across non-paraphrase transitions | paper/kuhnian_translation_complexity_results.json |
| `CLM_07_EXPANDED_BANK_DISTRIBUTION_AND_RANK1` | Strong PC1 variance concentration is not uniquely political (meteorology, culinary, botany all exhibit PC1 > 30%), but political frameworks exhibit higher publisher separation. | **SUPPORTED** | PC1 Var Ratio: Political = 42.0%, Absurd = 31.8%; Distractor separation: inverted = 0.9793 vs orthogonal = 0.3115 | Publisher F: Political = 2.61 vs Absurd = 1.66 | paper/kuhnian_expanded_bank_results.json |
| `CLM_08_POLITICAL_VS_ABSURD_CCA_INFERENCE` | Out-of-sample canonical correlation gap between political frameworks and absurd controls (Delta r = +0.1549) is statistically significant under cluster bootstrap (p = 0.0160) and Monte Carlo permutation (p = 0.0261). | **SUPPORTED** | Observed Delta r = +0.1549; Cluster Bootstrap Delta r = +0.0882 (p = 0.0160) | Cluster Bootstrap 95% CI: [0.0093, 0.1654] | paper/kuhnian_expanded_bank_results.json |
| `CLM_09_FACTORIAL_HYPOTHESIS_OPERATOR_DECOMPOSITION` | In a balanced factorial H x O design, Article x Operator interaction (20.87%) is numerically larger than Article x Framework interaction (16.67%), but their difference (Delta = +4.20%) spans zero under cluster bootstrap (95% CI: [-1.24%, +10.52%], p = 0.0770). | **SUPPORTED** | sigma^2_AO = 20.87%, sigma^2_AH = 16.67%, Delta_{O-H} = +4.20% | Delta_{O-H} 95% CI: [-1.24%, 10.52%] (Monte Carlo p = 0.0770) | paper/kuhnian_factorial_and_replication_results.json, paper/kuhnian_operator_uncertainty_results.json |
| `CLM_10_CROSS_MODEL_REPLICATION` | Relational structure replicates across DeBERTa-v2-xlarge, BART-large-MNLI, and RoBERTa-large-MNLI, with cross-model correlations r = 0.418 to 0.469. | **SUPPORTED** | Cross-model r: DeBERTa->BART = 0.418, DeBERTa->RoBERTa = 0.469 | All cross-model Pearson r in [0.418, 0.469] | paper/kuhnian_factorial_and_replication_results.json |
| `CLM_11_CROSS_TOPIC_REPLICATION_UKRAINE` | The political framing bank on Ukraine War corpus (N=60) decisively passes the matched permutation null gate (PC1 = 26.92% vs null q95 = 22.42%, p < 0.001); comparison to culinary absurd bank reveals generic floor r = 0.2917. | **SUPPORTED** | Ukraine Political PC1 = 26.92% (p < 0.001); Split-half r = 0.5113; Cross-bank CCA floor r = 0.2917 | Column null q95 = 22.42% | paper/kuhnian_ukraine_null_results.json |
| `CLM_12_METRIC_INDEPENDENT_NEIGHBORHOOD_DIVERGENCE` | Local neighborhood divergence between frameworks is robust across 6 distance metrics (Euclidean, Cosine, Correlation, Standardized, Whitened, Rank), with paraphrase J10 consistently exceeding cross-framework J10. | **SUPPORTED** | J10 Paraphrase: 0.23 vs Cross-Framework: 0.09 under Euclidean | Paraphrase > Cross-Framework across 100% of tested metrics | paper/kuhnian_metric_anomaly_results.json |
| `CLM_13_ANOMALY_RANKING_HIERARCHY` | Anomaly rankings follow the strict theoretical hierarchy: r(same fw, diff bank) > r(diff fw, same op) > r(diff op) across Reconstruction Error, Mahalanobis Distance, and LOF. | **SUPPORTED** | Level 1 (0.74) > Level 2 (0.43) > Level 3 (0.02) under Reconstruction Error | Hierarchy holds across all 3 scoring methods | paper/kuhnian_metric_anomaly_results.json |
| `CLM_14_BOUNDED_ADVERSARIAL_ATTAINABILITY` | Probe linear combinations cannot attain arbitrary random geometries out of sample (held-out R^2 < 0.00 across 94-97% of targets), whereas political targets attain held-out r = 0.73 to 0.89. | **SUPPORTED** | Random Targets Held-Out R^2 = -0.0308 (r = -0.1322) vs Political Held-Out r = 0.7257 | Random Targets Held-Out r 95% CI: [-0.3284, 0.0916] | paper/kuhnian_out_of_sample_adversarial_results.json |
| `CLM_15_NUISANCE_VARIABLE_PARTITIONING` | A 10-feature surface nuisance set explains < 0.5% unique stance variance, with document text representations providing 16% to 47% unique variance. | **SUPPORTED** | Unique Nuisance Variance = 0.28% vs Unique Text Variance = 47.10% | Unique text variance > 10x unique nuisance variance across all frameworks | paper/kuhnian_adversarial_nuisance_results.json |
| `CLM_16_ORIENTATION_INVARIANCE_GRASSMANNIAN` | Subspace distances between frameworks are sign- and rotation-invariant under Grassmannian distance, with paraphrase subspaces remaining substantially closer (d_G = 1.36) than cross-framework subspaces (d_G = 1.82). | **SUPPORTED** | d_G(Q0, Q1 paraphrase) = 1.36 vs d_G(Q0, Q2 humanitarian) = 1.82 | d_G paraphrase < d_G cross-framework across all comparisons | paper/kuhnian_adversarial_nuisance_results.json |
| `CLM_17_PUBLISHER_SEPARATION_PERMUTATION_NULL` | Publisher separation on PC1 (ANOVA F = 4.43, eta^2 = 0.352) significantly exceeds the Monte Carlo publisher permutation null distribution (null mean F = 1.02, p = 0.0030). | **SUPPORTED** | Observed F = 4.43, eta^2 = 0.352 vs Null Mean F = 1.02 (Monte Carlo p = 0.0030) | Null 95% CI: [0.38, 2.12] | paper/kuhnian_adversarial_nuisance_results.json |
| `CLM_18_DIMENSIONALITY_AND_MP_NULL` | Political frameworks exhibit low effective dimensionality (Participation Ratio = 3.47, PC1 EVR = 48.3%), rejecting the column permutation null (p < 0.001), with Marchenko-Pastur bound providing a supplemental asymptotic reference. | **SUPPORTED** | Leading eigenvalue ratio = 48.3%, PR = 3.47 | Rejects column permutation null at p < 0.001 | paper/kuhnian_adversarial_nuisance_results.json |
| `CLM_19_FUNCTIONAL_UNTRANSLATABILITY_ACROSS_OPERATORS` | Functional untranslatability across tested operators: No tested mapping family learns a positive generalization from NLI to FrameAxis (test R^2 <= 0.00 across all 14 models). | **SUPPORTED** | Held-out test R^2 <= 0.00 across all 14 mapping families | Test R^2 95% CI contains or is below zero for all tested models | paper/kuhnian_translation_complexity_results.json |
| `CLM_20_PHILOSOPHICAL_STRUCTURAL_REALISM_HYPOTHESIS` | Interpretive reflection on structural invariance: persistent tail rankings and conserved dimensionality suggest semantic measurement is partially stable across perspectives. | **PHILOSOPHICAL_INTERPRETATION** | N/A (Interpretive thesis) | N/A | paper/claim_ledger.json |
| `CLM_21_RELATIONAL_STANCE_REPRODUCIBILITY_AND_NON_RECOVERABILITY` | Relational measurements are substantially more reproducible than scalar article scores under matched aggregation (ICC(3,1)=0.5526 vs 0.3965; ICC(3,3)=0.7875 vs 0.6634); relational stance is not recoverable from tested framework-blind representations alone (R^2<=0.0695). | **SUPPORTED** | Matched single-measure: ICC(3,1) = 0.5526 vs 0.3965; Matched composite: ICC(3,3) = 0.7875 vs 0.6634; Framework-blind R^2 <= 0.0695 vs Relational R^2 = 0.5931; Twin divergence: 2.91 sigma vs 0.027 sigma | ICC(3,3): [0.750, 0.820]; Profile r IQR: [0.232, 0.682] | paper/frontier_relational_supervenience_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_22_SEMANTIC_COORDINATE_REDUCIBILITY` | Syntactic paraphrase fails the preregistered near-gauge criterion (R^2_affine=0.5981 < 0.85), but remains substantially more coordinate-reducible than conceptual shifts (12-21%) or operator shifts (2.9%); spectral rank (5-6) and persistent extremals survive across frameworks. | **PARTIALLY_SUPPORTED** | Paraphrase affine R^2 = 0.5981 (fails >= 0.85 gauge standard); Conceptual affine R^2 = 0.123-0.209; Operator affine R^2 = 0.0289; Conserved rank = 5-6 | Paraphrase Procrustes R^2: [0.47, 0.55]; Spectral entropy: [1.58, 1.84] | paper/frontier_invariance_gauge_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_23A_SCORE_MIXTURE_TRANSITION_CONTINUUM` | The apparent phase transition in operator score mixing is a numerical artifact of 8.75x scale asymmetry; when standardized, a Pettitt test detects no statistically significant change point (p=0.4135), while strong alignment with analytical interpolation (r=0.9965, MSE=0.0041) supports a smooth transition within the tested score-mixing regime. Score mixing represents post-hoc interpolation, not an empirical paradigm phase transition. | **FAILED_HYPOTHESIS** | Standardized score mixture decay tracks analytical interpolation (r = 0.9965, MSE = 0.0041; Pettitt test detects no significant change point, p = 0.4135); Unstandardized cliff is artifact of 8.75x scale ratio (76.5x variance ratio) | MSE to analytical score-mixture formula: 0.0041; Pettitt test p = 0.4135 | paper/frontier_commensurability_indeterminacy_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_23B_TRANSLATION_MODEL_MULTIPLICITY` | Translation-model multiplicity under empirical co-adequacy exhibits a 4.68x surge in normalized prediction dispersion (I_norm = 0.0362 vs 0.0077) over self-translation, confirming an empirical analogue of translation indeterminacy. | **SUPPORTED** | Normalized dispersion surge: 4.68x (I_norm = 0.036153 vs 0.007719); Raw dispersion surge: 9.13x (sigma^2 = 0.028587 vs 0.003132) | Self-translation best R^2 = 0.6031; Conceptual translation best R^2 = 0.2355 | paper/frontier_commensurability_indeterminacy_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_24A_EMPIRICAL_CO_ADEQUACY_UNDERDETERMINATION` | Empirical co-adequacy under tested external diagnostics despite representational non-equivalence: 27 framework pairs share external diagnostic fit (Delta Acc = 0.000, Delta F = 0.13, Delta R^2 = 0.007) while harboring divergent 3D Grassmannian subspaces (d_G = 1.9058) and sign-aligned \|r(PC1)\| = 0.7557. | **SUPPORTED** | Co-adequate diagnostic match: Delta Acc = 0.000, Delta F = 0.13, Delta Sent R^2 = 0.007; Grassmann distance d_G = 1.9058; Sign-aligned \|r(PC1)\| = 0.7557 | Grassmann distance: [1.85, 2.15]; Top-5 Acc: [0.25, 0.31] | paper/frontier_underdetermination_mechanisms_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_24B_CAUSAL_ACTIVATION_PATCHING_CONSTITUTION` | Exploratory activation patching demonstrates depth-dependent article-token shift (surging to +0.306 at layer 22), but is classified as exploratory and non-load-bearing pending negative noise and self-patch controls. | **EXPLORATORY** | Layer 22 article transfer: +0.3060 +/- 0.167; Full transfer: +0.9811 +/- 0.039 (Exploratory / Unaudited baselines) | Layer 22 article transfer: +0.3060 +/- 0.167; Full transfer: +0.9811 +/- 0.039 | paper/surgical_audit_frontier_results.json |
| `CLM_25A_OBSERVER_SPACE_INTRINSIC_DIMENSION` | Empirical stance space over 18 diverse frameworks exhibits a bounded intrinsic dimension of 6.77 (Two-NN) and effective rank 6.74 / 8, confirming constrained observer dimensionality. | **SUPPORTED** | Two-NN intrinsic dimension = 6.77; Effective rank = 6.74 / 8 | Two-NN ID: [6.45, 7.10]; Effective rank: [6.50, 6.95] | paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_25B_ANOMALY_MIGRATION_STRUCTURAL_LAW` | Anomaly ranking agreement decays lawfully with 3D Grassmannian distance under a 10,000-permutation QAP test (r = -0.6902, rho = -0.6578, p_Mantel < 0.0001), exceeding synthetic geodesic rotation nulls by +0.22 to +0.26 across all distance bins and preserving an invariant anomaly core. | **SUPPORTED** | QAP-verified anomaly decay: r = -0.6902, rho = -0.6578 (p_Mantel < 0.0001); Empirical vs Null E[r_A \| d_G] advantage: +0.221 to +0.260 | Pearson r: [-0.75, -0.62]; Spearman rho: [-0.73, -0.58] | paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_25C_OBSERVER_SPACE_CONCEPTUAL_DISSOCIATION` | Static conceptual prompt distance fails to predict observational subspace geometry under dependence-aware QAP testing (DeBERTa embeddings p_Mantel = 0.2522; TF-IDF n-grams p_Mantel = 0.0975), dissociating descriptions of perspectives from observational geometry. | **SUPPORTED** | DeBERTa vs Observational: r = -0.0503 (p_Mantel = 0.2522); TF-IDF vs Observational: r = +0.1116 (p_Mantel = 0.0975) | Mantel p-values strictly non-significant (p >= 0.0975) | paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json |
| `CLM_26_HUMAN_BENCHMARKING_PROTOCOL` | Preregistration-ready protocol formalizing human reader panel benchmarking across 4 interpretive communities (N=60) with power > 0.999 and 4 explicit falsification criteria. | **PROSPECTIVE_PROTOCOL** | Projected D-study profile reliability E rho^2_AP(15) = 0.8929 (design power specification, not empirical finding); Power > 0.999 | Projected E rho^2_AP(15): [0.85, 0.93] | paper/frontier_human_benchmarking_protocol.md, paper/frontier_master_synthesis.json |
| `CLM_27_MTMM_CONSTRUCT_VALIDITY_AND_METHOD_EFFECTS` | Multitrait-multimethod (MTMM) modeling demonstrates convergent construct validity across independently phrased prompt realizations (MTHM r = 0.7707, trait pattern r = 0.8387; 58.03% construct variance on core geopolitical discourse), but reveals that method variance operates primarily through Trait x Method calibration distortion (30.28% in Study 1, 31.65% in Study 2) rather than uniform elevation bias (A x M = 0.00%), explaining cross-operator attenuation (NLI vs FrameAxis MTHM r = 0.1313). | **QUALIFIED_SUPPORT** | Paraphrase MTHM r = 0.7707; Geopolitical Trait Var = 58.03%; Study 1 Trait Var = 31.25% vs Trait x Method = 30.28% (Ratio = 1.032); NLI vs FrameAxis MTHM r = 0.1313; DeBERTa vs BART MTHM r = 0.6077 | Construct Trait F = 4.25 (p = 1.11e-16); Trait x Method F = 95.46 (p = 1.11e-16); Convergent validity p = 4.23e-12 | paper/mtmm_framing_geometry_results.json, scripts/run_mtmm_analysis.py |
| `CLM_28_DIF_AND_MULTI_GROUP_MEASUREMENT_INVARIANCE` | Psychometric Differential Item Functioning (DIF) with Leave-One-Item-Out purified anchors and formal Multi-Group CFA demonstrates that prompt wording preserves exploratory factor-loading pattern congruence (Tucker \|phi\| >= 0.9113) and achieves partial metric invariance under approximate-fit testing, but exact likelihood-ratio and approximate tests decisively reject full metric invariance across model architectures (Delta chi^2 = 97.74, Delta CFI = -0.0744) and operators (Delta chi^2 = 323.15, Delta CFI = -0.1121). A culinary common-item control proves instrument linkability across corpora (75% Class A invariance, zero Class C), clarifying that prior cross-domain failures stemmed from unlinked item banks. | **QUALIFIED_SUPPORT** | Common-Item Linkability Control: 75.0% Class A Invariant, 0% Class C, Partial Metric Supported; Wording Congruence: all \|phi\| >= 0.9113; Architecture MG-CFA Metric Delta chi^2 = 97.74 (p = 1.29e-14, Delta CFI = -0.0744); Operator MG-CFA Metric Delta chi^2 = 323.15 (p < 1e-15, Delta CFI = -0.1121) | Benjamini-Hochberg FDR q <= 0.05 on all purified candidate tests; Delta CFI >= -0.010; Delta RMSEA <= +0.015; Tucker \|phi\| exploratory threshold = 0.90 | paper/dif_measurement_invariance_results.json, scripts/run_dif_analysis.py |
| `CLM_29_DOMAIN_GENERAL_POLITICAL_ANCHOR_INVARIANCE` | Audited domain-general political anchor administration across 5 policy domains (Gaza, Ukraine, Immigration, Climate, Abortion; N=300) establishes that 8 common political contrasts achieve 87.5% pairwise Class A invariance under continuous DIF when Gaza text extraction is audited, but cautions that treating the 8 distinct constructs as indicators of a single general factor is psychometrically uninterpretable due to unacceptable configural baseline fit (RMSEA=0.165, SRMR=0.087); cross-operator divergence (r=0.0642) empirically demonstrates non-interchangeability rather than incommensurability. | **QUALIFIED_SUPPORT** | Audited Gaza sigma = 0.348; Pairwise Class A DIF = 87.5% (210/240); Non-Gaza Class A DIF = 95.1% (137/144); 1-Factor Configural RMSEA = 0.165 (unacceptable baseline fit); Cross-operator r = 0.0642 (non-interchangeability) | Benjamini-Hochberg FDR q <= 0.05 on candidate DIF tests; RMSEA cutoff <= 0.08; Tucker \|phi\| >= 0.90 | paper/political_anchor_cross_domain_results.json, scripts/run_political_anchor_experiment.py, config/prompt_banks/domain_general_political_bank.json |
| `CLM_30_MULTIDIMENSIONAL_POLITICAL_CONSTRUCT_INVARIANCE` | A 24-item multitrait multi-indicator design (8 constructs x 3 realizations) administered across 5 policy domains (N=300) under a split-sample protocol reveals that 5 of 8 political constructs achieve multi-group metric invariance on the Discovery Sample (N=150), whereas 3 constructs (coercion, proportionality, compliance) exhibit domain-dependent calibration shifts; simultaneous 8-factor CFA confirms that full multidimensional metric invariance is underpowered and rejected out-of-sample (N=150). | **QUALIFIED_SUPPORT** | Per-Construct MG-CFA: 5/8 Invariant (Legitimacy p=0.638, Accountability p=0.946, Solidarity p=0.181, Reform p=0.094, Humanitarian p=0.056); 3/8 Non-Invariant (Coercion p=0.0004, Proportionality p=0.0046, Compliance p=0.0459); Pairwise Class A DIF = 87.5% (210/240); Mean Tucker \|phi\| = 0.9364; Global 8-Factor Configural CFI = 0.5171 (Discovery) / 0.5717 (Confirmation); Cross-Operator r = 0.0733 | Benjamini-Hochberg FDR q <= 0.05; LRT alpha = 0.05; Delta CFI >= -0.010, Delta RMSEA <= +0.015; Tucker \|phi\| >= 0.90 | paper/political_anchor_multitrait_results.json, scripts/run_political_anchor_multitrait_experiment.py, config/prompt_banks/domain_general_political_bank_24.json |
| `CLM_31_NATURAL_EDIT_RESPONSE_TOMOGRAPHY` | Natural Edit Response Tomography (NERT) on N=1,000 newsroom edit pairs establishes that linear response operators Rm: z -> Delta y fail as predictive system identification (negative held-out R^2 < 0 across all models on unseen story families). In 5-fold regularized CCA across 58 provenance-hardened story families, MiniLM and RoBERTa dense displacements achieve high out-of-fold correlation (OOF r = 0.9704, deep-residualized r = 0.9462). However, decisive leakage-free non-newsroom baselines reveal that this alignment is driven by generic dense representation geometry rather than edit dynamics: leakage-free unconstrained arbitrary article differences achieve an even higher correlation (OOF r = 0.9790), and magnitude-matched arbitrary pairs achieve r = 0.8892. Subtracting a generic resting representation map T drops cross-model CCA on residuals to r = 0.5532 with diffuse residual dimensionality (real edit PR = 21.51 vs arbitrary PR = 13.67), while whole-document displacements align only modestly with literal added/removed span semantics (cos = 0.2812 to 0.3330). | **QUALIFIED_SUPPORT** | NERT Predictive System Identification Failed (held-out R^2 < 0 all models); Real Newsroom Edit CCA OOF r = 0.9704 (residualized r = 0.9462); Leakage-Free Arbitrary: Unconstrained OOF r = 0.9790, Mag-Matched OOF r = 0.8892 (Delta r = +0.0812); Resting Map T Edit Disp R^2 = 0.3607; Residual CCA r = 0.5532; Real Edit Residual PR = 21.51 vs Arb PR = 13.67; Span Alignment cos = 0.2812 - 0.3330 (Span CCA r = 0.8963) | 58 provenance-hardened story families (max cross-family J <= 0.2810); 5-fold family-partitioned CV; B=500 cluster bootstrap PR CIs | paper/decisive_residual_experiment_results.json, paper/hostile_cca_and_multiverse_audit_results.json, paper/cca_residualization_and_mrqap_results.json, paper/natural_edit_tomography_audited_results.json, paper/natural_edit_response_tomography.md, scripts/decisive_residual_structure_experiment.py, scripts/hostile_cca_and_multiverse_audit.py |
| `CLM_32_MEASUREMENT_MULTIVERSE_STATIC_EDIT_COUPLING` | Deduplicated Measurement Multiverse (M=14 base instruments, N=91 non-redundant pairs) across N=1,000 natural edit pairs demonstrates that resting static agreement couples to edit agreement under QAP matrix permutation (r=0.8473, p=0.000200) and survives 4-factor MRQAP control (partial r = 0.7318, p=0.000200); however, static-dynamic coupling is algebraically constrained by constituent cross-article covariance blocks, Small-Edit Variance Ratios (SEVR) reflect local perturbation sensitivity rather than classical unreliability (refuting theoretical Spearman ceilings, as observed DeBERTa-RoBERTa raw edit correlation is 0.1353), and flash labels ('Superficial Mimics' and 'Perspectival Invariants') are retired as unsupported. | **QUALIFIED_SUPPORT** | Deduplicated Multiverse: S_edit = 0.7316 * S_static + 0.0450 (r = 0.8473, p_QAP = 0.000200); 4-Factor MRQAP: partial r = 0.7318 (p_MRQAP = 0.000200); Algebraic Null: r(S_exp, S_edit) = 1.0000; Observed DeBERTa-RoBERTa raw edit r = 0.1353; Cross-encoder NLI SEVR = 0.85 - 1.00 | MRQAP N=5,000 matrix permutations (null p95 = 0.1932); 58 provenance-hardened story families (max cross-family J <= 0.2810); N=1,000 natural edit pairs | paper/hostile_cca_and_multiverse_audit_results.json, paper/cca_residualization_and_mrqap_results.json, paper/rigorous_multiverse_audit_results.json, paper/measurement_multiverse_synthesis.md, scripts/hostile_cca_and_multiverse_audit.py |
| `CLM_33_CONSTRUCT_ALIASING_AND_SEMANTIC_IDENTIFIABILITY` | A hostile audit across a balanced 6-domain corpus (N=360) refutes persistent cross-construct aliasing under DeBERTa-v2-xlarge NLI (0/11,520 pairs at d_sem >= 0.50, rho_resp >= 0.85) and demonstrates that the apparent sample-size 'frontier' is quantitatively reproduced by ordinary sampling noise (analytical Fisher z null expects 228.8 aliases at N=15 vs 6.6 at N=360). Residualizing an article-level common-mode response propensity (raw absurd-substantive r = 0.5658 -> cross-fitted r = 0.0220) purges antonym bias (+0.2845 -> -0.0120) while preserving the monotonic hierarchy. The Support-Limited Identifiability hypothesis is falsified as a mean effect: fixed-N (N=60) corpus diversification yields flat within-pair distinguishability shifts (mean Delta D = -0.0005 to +0.0045, p >= 0.05) that fail to replicate in held-out confirmation (r = 0.0838). | **QUALIFIED_SUPPORT** | Aliasing at N=360: 0/11,520 (0.00%); Sampling Null Expected Aliases: N=15 (228.8 anal / 389.6 sim) -> N=360 (6.6 anal / 7.3 sim); CM Residualization: absurd r = 0.5658 -> 0.0220; Antonym r = +0.2845 -> -0.0120; Support Diversification: NLI Delta D = -0.0005 (p=0.0589); Centered MiniLM Delta D = +0.0003 (p=0.4753); Held-out Delta D rep r = 0.0838; RoBERTa Anisotropy Correction: D = 0.0966 -> 1.0121; Dyadic MRQAP: R^2 = 0.0780, beta_dsem = -0.7174 (p=0.000200) | N=5,000 QAP permutations; B=300 empirical resampling runs across 6 domains (N=360); Held-out split-half confirmation (180 Discovery vs 180 Confirmation) | paper/support_identifiability_audit_results.json, paper/construct_aliasing_results.json, paper/construct_aliasing_monograph.md, scripts/run_support_identifiability_audit.py |
| `CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH` | Centered Kernel Alignment (CKA) across N=6,248 independent story family representatives reveals that hypothesis-prompt invariance (Q0 <-> Q1 paraphrase stability) is substantially higher at the middle layer (Layer 12 CLS, V7 <-> V8: CKA_RBF = 0.9131, Linear: 0.9073, 95% CI: [0.9119, 0.9176], permutation p < 0.0020) than in final hidden representations (Layer 24 CLS, V3 <-> V4: CKA_RBF = 0.7770, Linear: 0.7740, 95% CI: [0.7720, 0.7857], p < 0.0020) and paired scalar classifier margins (V1 <-> V2: CKA_RBF = 0.7490, Linear: 0.8015, 95% CI: [0.7401, 0.7594], p < 0.0020); sampling three representation depths establishes boundary and midpoint differences rather than monotonic behavior at every intervening layer. | **QUALIFIED_SUPPORT** | Prompt Invariance CKA: Layer 12 CLS = 0.9131, Layer 24 CLS = 0.7770, Scalar Margin = 0.7490; Cross-View Scalar-to-Hidden CKA: V1-V3 = 0.5437, V1-V5 = 0.5034, V1-V7 = 0.1196; Syndication Sensitivity \|Delta CKA\| <= 0.0408 | Layer 12 CKA 95% CI: [0.9119, 0.9176]; Layer 24 CKA 95% CI: [0.7720, 0.7857]; Scalar CKA 95% CI: [0.7401, 0.7594]; 500-permutation null controls (all p < 0.0020); B=200 cluster bootstrap resamples on N=6,248 independent families | deliverables/05_gate3_full_experiment/cka_matrix_6248.json, deliverables/05_gate3_full_experiment/gate3_executive_synthesis.md, deliverables/05_gate3_full_experiment/gate3_full_report.json, deliverables/05_gate3_full_experiment/sensitivity_analysis.json, deliverables/05_gate3_full_experiment/mmd_eligible_conditions.json |
| `CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION` | In a balanced 120-template / N=480 passage controlled intervention suite benchmarked against a 512-article real news panel, final-layer representations (V3 Final-CLS) exhibit high target contrast specificity (ratio of means = 2.01x, p = 2.92e-46) and pronounced relational discrimination in ambient news space (Delta R10 = +66.58 pp, p = 3.75e-47), whereas middle-layer representations (V7 Mid-CLS) fail specificity (ratio of means = 0.92x, p = 0.910) and do not detect relational discrimination (Delta R10 = +0.75 pp, p = 0.360; failure to detect discrimination is not proof of exact insensitivity). This demonstrates an empirical dissociation between prompt stability, proposition specificity, and relational discrimination rather than an inherent universal trade-off. Distance support audits confirm that synthetic passage edits displace outward relative to intra-panel news distributions (Kolmogorov-Smirnov rejection p = 0.005). | **QUALIFIED_SUPPORT** | V3 Final-CLS Specificity: Ratio of Means = 2.01x (Template Mean 2.04x, p = 2.92e-46); V1 Scalar Margin: Ratio of Means = 2.29x (Template Mean 2.49x, p = 2.81e-34); V7 Mid-CLS Specificity: Ratio of Means = 0.92x (p = 0.910); Ambient R10 Retention Drop (Pres - Chan): V3 = +66.58 pp (p = 3.75e-47), V1 = +62.33 pp, V7 = +0.75 pp (p = 0.360); Surface Edit R10: V3 = 22.08% (Reversal < Surface by +17.33 pp, p = 1.25e-8); Distance Support KS Test: 10th NN KS = 0.1732 (p = 0.00505), 1st NN KS = 0.2355 (p = 3.16e-5) | Paired t-test across 120 balanced intervention templates (N=480 passages, 8 dimensions); Kolmogorov-Smirnov continuous distribution tests vs N=512 real news panel; Exact R10 and Jaccard J10 neighbor statistics | deliverables/07_hidden_state_specificity_report.md, deliverables/07_hidden_state_specificity_report_data.npz, results/controlled_hidden_state_diagnostic_audit.json, results/controlled_hidden_state_neighborhood_report.json, data/controlled_semantic_interventions_manifest.json |

---

## Detailed Claim Documentation

### `CLM_01_G_THEORY_VARIANCE_PARTITIONING`
**Summary:** Variance decomposition explicitly distinguishes cell-mean A x P interaction (37.32%, df=476) from item-level A x P interaction (13.02%, df=476) and item residual error (74.42%, df=12600).

**Text:** Generalizability theory across 14,400 observations (120 articles x 5 frameworks x 3 realizations x 8 items) explicitly separates unit levels: at the cell-mean level (1,800 means, df=1,799), A x P interaction accounts for 37.32% of variance (df=476, F=4.65, p < 10^-15) with article main effect at 0.00% (df=119, F=0.88, p=0.806). At the full item level (14,400 observations, df=14,399), item-level residual error accounts for 74.42% (df=12,600), A x P interaction accounts for 13.02% (df=476), wording within framework accounts for 5.35% (df=10), and article main effect is 0.00% (df=119).

**Status:** `SUPPORTED`

**Effect Size:** `Cell-mean: AP = 37.32% (df=476), P = 16.66% (df=4), B(P) = 15.33% (df=10), A = 0.00% (df=119); Item-level: Residual = 74.42% (df=12600), AP = 13.02% (df=476)`

**95% CI:** `sigma^2_AP: [31.64%, 42.92%]; E rho^2_AP(3,8): [0.7500, 0.8159]`

**Artifacts:** paper/kuhnian_g_theory_results.json, paper/kuhnian_d_study_results.json

### `CLM_02_FRAMEWORK_BLIND_SCALAR_PREDICTION`
**Summary:** A framework-blind scalar model cannot predict stance across multiple frameworks from article text alone (R^2 = 0.0082), whereas per-framework models explain up to 49.03% of stance variance.

**Text:** Under grouped 5-fold cross-validation, a framework-blind scalar model M0 achieves held-out pooled R^2 = 0.0082 on DeBERTa, whereas per-framework interaction models M2 achieve pooled R^2 = 0.4903 (r = 0.7009, macro R^2 = 0.2265), confirming that article text alone without framework conditioning cannot predict multi-framework stance.

**Status:** `SUPPORTED`

**Effect Size:** `Scalar R^2 = 0.0082 vs Per-Framework R^2 = 0.4903 (gain: +0.4821)`

**95% CI:** `Per-Framework fold R^2: [0.1937, 0.5740]`

**Artifacts:** paper/kuhnian_framework_prediction_results.json

### `CLM_03_UNSEEN_FRAMEWORK_GENERALIZATION_DROP`
**Summary:** Zero-shot transfer of this particular semantic framework representation drops from R^2 = 0.3650 on seen frameworks to R^2 = -0.1085 on unseen frameworks.

**Text:** Predicting article stance using sentence-transformer semantic framework vectors g(e(x), e(P)) across 4 holdout regimes shows that within-framework prediction on held-out articles achieves R^2 = 0.3650 (r = 0.6042), but drops catastrophically to R^2 = -0.1085 (r = 0.0381) on unseen frameworks. This particular semantic framework representation failed zero-shot transfer.

**Status:** `SUPPORTED`

**Effect Size:** `Regime 2 R^2 = 0.3650 vs Regime 4 R^2 = -0.1085 (delta R^2 = -0.4734)`

**95% CI:** `Regime 2 r = 0.6042; Regime 4 r = 0.0381`

**Artifacts:** paper/kuhnian_framework_prediction_results.json

### `CLM_04_RELATIONAL_INTERACTION_PERMUTATION_GATE`
**Summary:** The relational interaction gain Delta R^2(M2 - M1) = +0.2087 decisively exceeds the article-level framework permutation null (Monte Carlo p = 0.0010).

**Text:** A 1,000-draw article-level framework permutation test preserving repeated-measures marginal distributions yielded null mean Delta R^2 = -0.1140 (null max = -0.0480, 95% CI: [-0.1420, -0.0740]), decisively rejecting the null for the observed relational gain Delta R^2 = +0.2087 at Monte Carlo permutation p = 0.0010.

**Status:** `SUPPORTED`

**Effect Size:** `Observed Delta R^2 = +0.2087 vs Permutation Null Mean = -0.1140 (Monte Carlo p = 0.0010)`

**95% CI:** `Null 95% CI: [-0.1420, -0.0740]`

**Artifacts:** paper/kuhnian_framework_prediction_results.json

### `CLM_05_RELIABILITY_NORMALIZED_TRANSLATABILITY`
**Summary:** Translation test performance confirms syntactic paraphrase is translatable (test R^2 = 0.5488, r = 0.7741), while cross-operator mapping fails out-of-sample (test R^2 <= 0.00 across all 14 mapping families; T5 RidgeCV test R^2 = -0.0287).

**Text:** Evaluating translation across 14 mapping families (T0..T13) demonstrates that paraphrase control translation (Q0 -> Q1) generalizes out-of-sample under RidgeCV (test R^2 = 0.5488, test r = 0.7741, train R^2 = 0.6704), whereas cross-operator translation (paired NLI to FrameAxis) fails to achieve positive test generalization across every single tested mapping family (test R^2 <= 0.00 across all 14 models; T5 RidgeCV test R^2 = -0.0287, test r = 0.2722). No tested model learned a positive cross-operator mapping.

**Status:** `SUPPORTED`

**Effect Size:** `Paraphrase test R^2 = 0.5488 (test r = 0.7741); Operator shift test R^2 <= 0.00 across all 14 models (T5 test R^2 = -0.0287)`

**95% CI:** `Held-out test R^2 <= 0.00 across all 14 mapping models for operator shift`

**Artifacts:** paper/kuhnian_translation_complexity_results.json

### `CLM_06_TRANSLATION_COMPLEXITY_OVERFITTING_GAP`
**Summary:** High-capacity nonlinear models (Poly-2, MLP, KNN) exhibit severe overfitting divergence for cross-framework mappings (train R^2 > 0.85, test R^2 < 0.10, generalization gap > 0.84).

**Text:** Complexity curves Q_train(c) vs Q_test(c) show that while capacity expands training fit up to R^2 = 0.884, test R^2 remains below 0.10 for all non-paraphrase transitions, producing massive generalization gaps (Delta R^2 = +1.026 for Poly-2) and c*(0.10) = None on held-out data.

**Status:** `SUPPORTED`

**Effect Size:** `MLP / Poly-2 Generalization Gap Delta R^2 = +1.026`

**95% CI:** `Test R^2 < 0.00 for c >= 8 across non-paraphrase transitions`

**Artifacts:** paper/kuhnian_translation_complexity_results.json

### `CLM_07_EXPANDED_BANK_DISTRIBUTION_AND_RANK1`
**Summary:** Strong PC1 variance concentration is not uniquely political (meteorology, culinary, botany all exhibit PC1 > 30%), but political frameworks exhibit higher publisher separation.

**Text:** Evaluation across 22 banks in 5 classes shows that absurd structurally matched frameworks (meteorology, culinary, botany, sports) exhibit strong first principal components (PC1 mean = 31.82%), proving that low-dimensional compression is domain-general. However, political frameworks exhibit higher publisher separation (ANOVA F = 2.61 vs 1.66). Decomposing the adversarial distractor mean (r_canon = 0.6454) reveals that adv_inverted_pc1 (r_canon = 0.9793) is a political polarity inversion, whereas true orthogonal noise (adv_orthogonal_distractors) drops to r_canon = 0.3115.

**Status:** `SUPPORTED`

**Effect Size:** `PC1 Var Ratio: Political = 42.0%, Absurd = 31.8%; Distractor separation: inverted = 0.9793 vs orthogonal = 0.3115`

**95% CI:** `Publisher F: Political = 2.61 vs Absurd = 1.66`

**Artifacts:** paper/kuhnian_expanded_bank_results.json

### `CLM_08_POLITICAL_VS_ABSURD_CCA_INFERENCE`
**Summary:** Out-of-sample canonical correlation gap between political frameworks and absurd controls (Delta r = +0.1549) is statistically significant under cluster bootstrap (p = 0.0160) and Monte Carlo permutation (p = 0.0261).

**Text:** Substantive political frameworks average r_canon = 0.6046 out of sample vs 0.3743 for absurd controls (Delta r = +0.1549). Under 4-level statistical inference, the advantage is significant under article bootstrap (p = 0.0030), bank bootstrap (p = 0.0130), combined cluster bootstrap (Delta r = +0.0882, 95% CI: [0.0093, 0.1654], p = 0.0160), and Monte Carlo bank label permutation (p = 0.0261), demonstrating a modest semantic advantage over a substantial shared article floor.

**Status:** `SUPPORTED`

**Effect Size:** `Observed Delta r = +0.1549; Cluster Bootstrap Delta r = +0.0882 (p = 0.0160)`

**95% CI:** `Cluster Bootstrap 95% CI: [0.0093, 0.1654]`

**Artifacts:** paper/kuhnian_expanded_bank_results.json

### `CLM_09_FACTORIAL_HYPOTHESIS_OPERATOR_DECOMPOSITION`
**Summary:** In a balanced factorial H x O design, Article x Operator interaction (20.87%) is numerically larger than Article x Framework interaction (16.67%), but their difference (Delta = +4.20%) spans zero under cluster bootstrap (95% CI: [-1.24%, +10.52%], p = 0.0770).

**Text:** A balanced 120 x 4 x 3 ANOVA decomposes variance into Article x Operator (20.87%) and Article x Framework (16.67%). Under 1,000 cluster bootstrap resamples over articles, Delta_{O-H} yields mean +3.95% with 95% CI [-1.24%, 10.52%] (p = 0.0770). Because the 95% CI spans zero, operator sensitivity is numerically larger in this design but comparable in magnitude to framework sensitivity, noting factor level asymmetry (4 frameworks vs 3 operators).

**Status:** `SUPPORTED`

**Effect Size:** `sigma^2_AO = 20.87%, sigma^2_AH = 16.67%, Delta_{O-H} = +4.20%`

**95% CI:** `Delta_{O-H} 95% CI: [-1.24%, 10.52%] (Monte Carlo p = 0.0770)`

**Artifacts:** paper/kuhnian_factorial_and_replication_results.json, paper/kuhnian_operator_uncertainty_results.json

### `CLM_10_CROSS_MODEL_REPLICATION`
**Summary:** Relational structure replicates across DeBERTa-v2-xlarge, BART-large-MNLI, and RoBERTa-large-MNLI, with cross-model correlations r = 0.418 to 0.469.

**Text:** Evaluation across 3 distinct transformer families on the matched N=60 subset demonstrates consistent PC1 variance concentration (40.7% to 60.1%) and positive cross-model translation (DeBERTa to BART: r = 0.418; DeBERTa to RoBERTa: r = 0.469), confirming that relational geometry is not an idiosyncratic artifact of DeBERTa.

**Status:** `SUPPORTED`

**Effect Size:** `Cross-model r: DeBERTa->BART = 0.418, DeBERTa->RoBERTa = 0.469`

**95% CI:** `All cross-model Pearson r in [0.418, 0.469]`

**Artifacts:** paper/kuhnian_factorial_and_replication_results.json

### `CLM_11_CROSS_TOPIC_REPLICATION_UKRAINE`
**Summary:** The political framing bank on Ukraine War corpus (N=60) decisively passes the matched permutation null gate (PC1 = 26.92% vs null q95 = 22.42%, p < 0.001); comparison to culinary absurd bank reveals generic floor r = 0.2917.

**Text:** On the clean Ukraine War corpus (N=60 across 54 publishers), the geopolitical framing bank exhibits PC1 = 26.92%, decisively rejecting the 1,000-draw column permutation null (null mean = 19.78%, q95 = 22.42%, Monte Carlo p = 0.0000) with split-half reliability r = 0.5113. Comparing against the culinary absurd bank reveals that rank-1 concentration is present in the absurd domain (PC1 = 42.93%), with cross-bank CCA establishing a generic floor of r = 0.2917.

**Status:** `SUPPORTED`

**Effect Size:** `Ukraine Political PC1 = 26.92% (p < 0.001); Split-half r = 0.5113; Cross-bank CCA floor r = 0.2917`

**95% CI:** `Column null q95 = 22.42%`

**Artifacts:** paper/kuhnian_ukraine_null_results.json

### `CLM_12_METRIC_INDEPENDENT_NEIGHBORHOOD_DIVERGENCE`
**Summary:** Local neighborhood divergence between frameworks is robust across 6 distance metrics (Euclidean, Cosine, Correlation, Standardized, Whitened, Rank), with paraphrase J10 consistently exceeding cross-framework J10.

**Text:** Neighborhood overlap sweep k in [3, 50] across 6 distance metrics confirms that paraphrase overlap (J10 approx 0.20-0.27) exceeds cross-framework overlap (J10 approx 0.08-0.12) across 100% of tested metrics and k values, remaining invariant to ZCA whitening and rank transformation.

**Status:** `SUPPORTED`

**Effect Size:** `J10 Paraphrase: 0.23 vs Cross-Framework: 0.09 under Euclidean`

**95% CI:** `Paraphrase > Cross-Framework across 100% of tested metrics`

**Artifacts:** paper/kuhnian_metric_anomaly_results.json

### `CLM_13_ANOMALY_RANKING_HIERARCHY`
**Summary:** Anomaly rankings follow the strict theoretical hierarchy: r(same fw, diff bank) > r(diff fw, same op) > r(diff op) across Reconstruction Error, Mahalanobis Distance, and LOF.

**Text:** Spearman correlation between anomaly rankings strictly adheres to the theoretical hierarchy across all three scoring methods: Reconstruction error yields 0.7433 > 0.4253 > 0.0200; Mahalanobis yields 0.8036 > 0.4706 > 0.0658; LOF yields 0.5963 > 0.2287 > 0.1217.

**Status:** `SUPPORTED`

**Effect Size:** `Level 1 (0.74) > Level 2 (0.43) > Level 3 (0.02) under Reconstruction Error`

**95% CI:** `Hierarchy holds across all 3 scoring methods`

**Artifacts:** paper/kuhnian_metric_anomaly_results.json

### `CLM_14_BOUNDED_ADVERSARIAL_ATTAINABILITY`
**Summary:** Probe linear combinations cannot attain arbitrary random geometries out of sample (held-out R^2 < 0.00 across 94-97% of targets), whereas political targets attain held-out r = 0.73 to 0.89.

**Text:** Under strict nested 5-fold CV across 100 random targets (50 Gaussian, 25 Bimodal, 25 Heavy-tailed), out-of-sample attainability collapses to held-out R^2 = -0.0308 (r = -0.1322, 95% CI: [-0.3284, 0.0916], 97.0% negative) on Q0 probes, and R^2 = -0.0342 on all candidate banks (94.0% negative). In contrast, genuine political targets attain held-out r = 0.7257 (R^2 = 0.5445), definitively proving that arbitrary geometries cannot be manufactured out of sample.

**Status:** `SUPPORTED`

**Effect Size:** `Random Targets Held-Out R^2 = -0.0308 (r = -0.1322) vs Political Held-Out r = 0.7257`

**95% CI:** `Random Targets Held-Out r 95% CI: [-0.3284, 0.0916]`

**Artifacts:** paper/kuhnian_out_of_sample_adversarial_results.json

### `CLM_15_NUISANCE_VARIABLE_PARTITIONING`
**Summary:** A 10-feature surface nuisance set explains < 0.5% unique stance variance, with document text representations providing 16% to 47% unique variance.

**Text:** Commonality analysis against 10 surface nuisance features (length, sentence length, sentiment, uppercase/digit/punct ratios, lexical diversity) shows unique nuisance variance is 0.28% vs unique text variance of 47.10%, confirming that stance geometry reflects substantive semantic content.

**Status:** `SUPPORTED`

**Effect Size:** `Unique Nuisance Variance = 0.28% vs Unique Text Variance = 47.10%`

**95% CI:** `Unique text variance > 10x unique nuisance variance across all frameworks`

**Artifacts:** paper/kuhnian_adversarial_nuisance_results.json

### `CLM_16_ORIENTATION_INVARIANCE_GRASSMANNIAN`
**Summary:** Subspace distances between frameworks are sign- and rotation-invariant under Grassmannian distance, with paraphrase subspaces remaining substantially closer (d_G = 1.36) than cross-framework subspaces (d_G = 1.82).

**Text:** Grassmannian projection distance d_G across 4-dimensional principal subspaces confirms that paraphrase distance d_G(Q0, Q1) = 1.36 is substantially smaller than cross-framework distance d_G(Q0, Q2) = 1.82, proving geometric divergence is invariant to basis alignment.

**Status:** `SUPPORTED`

**Effect Size:** `d_G(Q0, Q1 paraphrase) = 1.36 vs d_G(Q0, Q2 humanitarian) = 1.82`

**95% CI:** `d_G paraphrase < d_G cross-framework across all comparisons`

**Artifacts:** paper/kuhnian_adversarial_nuisance_results.json

### `CLM_17_PUBLISHER_SEPARATION_PERMUTATION_NULL`
**Summary:** Publisher separation on PC1 (ANOVA F = 4.43, eta^2 = 0.352) significantly exceeds the Monte Carlo publisher permutation null distribution (null mean F = 1.02, p = 0.0030).

**Text:** Evaluating publisher separation on PC1 of Q0 yields ANOVA F = 4.43 (z = 6.33), decisively exceeding the 1,000-draw publisher label permutation null (null mean F = 1.02, 95% CI: [0.38, 2.12]) at Monte Carlo permutation p = 0.0030.

**Status:** `SUPPORTED`

**Effect Size:** `Observed F = 4.43, eta^2 = 0.352 vs Null Mean F = 1.02 (Monte Carlo p = 0.0030)`

**95% CI:** `Null 95% CI: [0.38, 2.12]`

**Artifacts:** paper/kuhnian_adversarial_nuisance_results.json

### `CLM_18_DIMENSIONALITY_AND_MP_NULL`
**Summary:** Political frameworks exhibit low effective dimensionality (Participation Ratio = 3.47, PC1 EVR = 48.3%), rejecting the column permutation null (p < 0.001), with Marchenko-Pastur bound providing a supplemental asymptotic reference.

**Text:** Principal component analysis of Q0 yields PR = 3.47 and PC1 EVR = 48.3%, rejecting the empirical column permutation null (p < 0.001), with Marchenko-Pastur spectral upper bound providing a supplemental asymptotic reference.

**Status:** `SUPPORTED`

**Effect Size:** `Leading eigenvalue ratio = 48.3%, PR = 3.47`

**95% CI:** `Rejects column permutation null at p < 0.001`

**Artifacts:** paper/kuhnian_adversarial_nuisance_results.json

### `CLM_19_FUNCTIONAL_UNTRANSLATABILITY_ACROSS_OPERATORS`
**Summary:** Functional untranslatability across tested operators: No tested mapping family learns a positive generalization from NLI to FrameAxis (test R^2 <= 0.00 across all 14 models).

**Text:** Across all 14 tested mapping families, cross-operator translation yields held-out test R^2 <= 0.00 (T5 RidgeCV test R^2 = -0.0287, Kernel Ridge R^2 = -0.0006). This functional untranslatability empirically demonstrates that the two measurement instruments cannot be mapped onto one another using the tested linear, kernel, or neural transformation classes.

**Status:** `SUPPORTED`

**Effect Size:** `Held-out test R^2 <= 0.00 across all 14 mapping families`

**95% CI:** `Test R^2 95% CI contains or is below zero for all tested models`

**Artifacts:** paper/kuhnian_translation_complexity_results.json

### `CLM_20_PHILOSOPHICAL_STRUCTURAL_REALISM_HYPOTHESIS`
**Summary:** Interpretive reflection on structural invariance: persistent tail rankings and conserved dimensionality suggest semantic measurement is partially stable across perspectives.

**Text:** The mathematical observation that relational publisher orderings persist across syntactic paraphrases while collapsing under operator shifts empirically exemplifies epistemic structural realism in semantic representation. This claim is explicitly classified as an interpretive philosophical hypothesis.

**Status:** `PHILOSOPHICAL_INTERPRETATION`

**Effect Size:** `N/A (Interpretive thesis)`

**95% CI:** `N/A`

**Artifacts:** paper/claim_ledger.json

### `CLM_21_RELATIONAL_STANCE_REPRODUCIBILITY_AND_NON_RECOVERABILITY`
**Summary:** Relational measurements are substantially more reproducible than scalar article scores under matched aggregation (ICC(3,1)=0.5526 vs 0.3965; ICC(3,3)=0.7875 vs 0.6634); relational stance is not recoverable from tested framework-blind representations alone (R^2<=0.0695).

**Text:** Evaluating the reproducibility of framing measurements across three independent probe-bank realizations on 120 prospective articles confirms that relational measurements are substantially more reproducible than framework-averaged scalar scores under matched aggregation: single-measure reliability achieves ICC(3,1) = 0.5526 for article-within-framework targets versus ICC(3,1) = 0.3965 for framework-averaged scalar scores (+0.1561 gap); matched composite reliability achieves ICC(3,3) = 0.7875 versus ICC(3,3) = 0.6634 (+0.1241 gap); and within-article profile stability reaches r = 0.4515 (IQR: [0.232, 0.682]). Furthermore, relational stance is not recoverable with useful accuracy from tested framework-blind representations across five feature tiers (all R^2 <= 0.0695), whereas relational interaction models recover substantial variance (R^2 = 0.5931). Concrete document pairs with near-zero distance in dense embedding space (d_cosine = 0.0128) separate by 2.91 sigma under Strategic Security while differing by only 0.027 sigma under International Humanitarian Law.

**Status:** `SUPPORTED`

**Effect Size:** `Matched single-measure: ICC(3,1) = 0.5526 vs 0.3965; Matched composite: ICC(3,3) = 0.7875 vs 0.6634; Framework-blind R^2 <= 0.0695 vs Relational R^2 = 0.5931; Twin divergence: 2.91 sigma vs 0.027 sigma`

**95% CI:** `ICC(3,3): [0.750, 0.820]; Profile r IQR: [0.232, 0.682]`

**Artifacts:** paper/frontier_relational_supervenience_results.json, paper/surgical_audit_frontier_results.json

### `CLM_22_SEMANTIC_COORDINATE_REDUCIBILITY`
**Summary:** Syntactic paraphrase fails the preregistered near-gauge criterion (R^2_affine=0.5981 < 0.85), but remains substantially more coordinate-reducible than conceptual shifts (12-21%) or operator shifts (2.9%); spectral rank (5-6) and persistent extremals survive across frameworks.

**Text:** Evaluating semantic coordinate transformations against the preregistered gauge threshold (R^2_affine >= 0.85) reveals that syntactic paraphrasing (Q0 -> Q1) achieves R^2_affine = 0.5981 (Procrustes R^2 = 0.5133), failing the strict near-gauge hypothesis. However, paraphrase is substantially more coordinate-reducible (59.8% explained by affine transformation) than conceptual shifts (where affine maps explain only 12.3% to 20.9% of variance) or cross-operator comparisons (explaining only 2.89% of variance). Across all tested NLI frameworks, structural invariants persist: spectral effective rank at 90% variance is conserved at 5 to 6 dimensions (spectral entropy H in [1.58, 1.84]), graph clustering coefficients are conserved at 0.28 to 0.45, and a core set of 5 articles remains persistently extremal in distribution tails across >= 3 independent political ontologies.

**Status:** `PARTIALLY_SUPPORTED`

**Effect Size:** `Paraphrase affine R^2 = 0.5981 (fails >= 0.85 gauge standard); Conceptual affine R^2 = 0.123-0.209; Operator affine R^2 = 0.0289; Conserved rank = 5-6`

**95% CI:** `Paraphrase Procrustes R^2: [0.47, 0.55]; Spectral entropy: [1.58, 1.84]`

**Artifacts:** paper/frontier_invariance_gauge_results.json, paper/surgical_audit_frontier_results.json

### `CLM_23A_SCORE_MIXTURE_TRANSITION_CONTINUUM`
**Summary:** The apparent phase transition in operator score mixing is a numerical artifact of 8.75x scale asymmetry; when standardized, a Pettitt test detects no statistically significant change point (p=0.4135), while strong alignment with analytical interpolation (r=0.9965, MSE=0.0041) supports a smooth transition within the tested score-mixing regime. Score mixing represents post-hoc interpolation, not an empirical paradigm phase transition.

**Text:** Audit of continuous transition paths along S(alpha) = (1-alpha) S_A + alpha S_B reveals that the apparent cliff collapse in unstandardized operator mixing (R^2 dropping from 0.9139 at alpha=0.7 to 0.0289 at alpha=1.0) is a numerical artifact of 8.75x scale dominance of raw NLI over FrameAxis (76.5x variance ratio). When standardized prior to mixing, a Pettitt test detects no statistically significant change point (p = 0.4135), while the curve closely follows the analytical expectation for uncorrelated score mixtures R^2(alpha) = (1-alpha)^2 / ((1-alpha)^2 + alpha^2) with r = 0.9965 (MSE = 0.0041), supporting a smooth-transition interpretation within the tested setting. Score-mixture manifolds reflect artificial numerical interpolation through representation space rather than continuous paradigm transitions in language space.

**Status:** `FAILED_HYPOTHESIS`

**Effect Size:** `Standardized score mixture decay tracks analytical interpolation (r = 0.9965, MSE = 0.0041; Pettitt test detects no significant change point, p = 0.4135); Unstandardized cliff is artifact of 8.75x scale ratio (76.5x variance ratio)`

**95% CI:** `MSE to analytical score-mixture formula: 0.0041; Pettitt test p = 0.4135`

**Artifacts:** paper/frontier_commensurability_indeterminacy_results.json, paper/surgical_audit_frontier_results.json

### `CLM_23B_TRANSLATION_MODEL_MULTIPLICITY`
**Summary:** Translation-model multiplicity under empirical co-adequacy exhibits a 4.68x surge in normalized prediction dispersion (I_norm = 0.0362 vs 0.0077) over self-translation, confirming an empirical analogue of translation indeterminacy.

**Text:** Evaluating 15 candidate estimators under empirical co-adequacy (Delta R^2 <= 0.05) reveals that conceptual translation (Q0 -> Q2) exhibits a 4.68x surge in normalized pointwise prediction dispersion (I_norm = Var_m[y_hat] / MSE_ensemble = 0.036153 vs 0.007719; raw dispersion surge 9.13x: sigma^2_trans = 0.028587 vs 0.003132) relative to a self-translation baseline across independent realizations of the same framework. Normalizing by task difficulty confirms that model multiplicity under conceptual divergence substantially exceeds ordinary algorithmic uncertainty.

**Status:** `SUPPORTED`

**Effect Size:** `Normalized dispersion surge: 4.68x (I_norm = 0.036153 vs 0.007719); Raw dispersion surge: 9.13x (sigma^2 = 0.028587 vs 0.003132)`

**95% CI:** `Self-translation best R^2 = 0.6031; Conceptual translation best R^2 = 0.2355`

**Artifacts:** paper/frontier_commensurability_indeterminacy_results.json, paper/surgical_audit_frontier_results.json

### `CLM_24A_EMPIRICAL_CO_ADEQUACY_UNDERDETERMINATION`
**Summary:** Empirical co-adequacy under tested external diagnostics despite representational non-equivalence: 27 framework pairs share external diagnostic fit (Delta Acc = 0.000, Delta F = 0.13, Delta R^2 = 0.007) while harboring divergent 3D Grassmannian subspaces (d_G = 1.9058) and sign-aligned |r(PC1)| = 0.7557.

**Text:** Evaluating external adequacy reveals 27 framework pairs with indistinguishable external fit across multiple diagnostics; for example, Realpolitik and Legal Culpability exhibit identical top-5 publisher accuracy (0.280 vs 0.280, Delta = 0.000), similar all-publisher ANOVA F-statistics (1.86 vs 1.99, Delta F = 0.13), and matched sentiment prediction (R^2 = 0.085 vs 0.092, Delta R^2 = 0.007). However, their internal representations occupy substantially divergent subspaces (3D Grassmann distance d_G = 1.9058 out of a maximum 2.7207, with sign-aligned |r(PC1)| = 0.7557), confirming empirical co-adequacy under tested external diagnostics despite representational non-equivalence.

**Status:** `SUPPORTED`

**Effect Size:** `Co-adequate diagnostic match: Delta Acc = 0.000, Delta F = 0.13, Delta Sent R^2 = 0.007; Grassmann distance d_G = 1.9058; Sign-aligned |r(PC1)| = 0.7557`

**95% CI:** `Grassmann distance: [1.85, 2.15]; Top-5 Acc: [0.25, 0.31]`

**Artifacts:** paper/frontier_underdetermination_mechanisms_results.json, paper/surgical_audit_frontier_results.json

### `CLM_24B_CAUSAL_ACTIVATION_PATCHING_CONSTITUTION`
**Summary:** Exploratory activation patching demonstrates depth-dependent article-token shift (surging to +0.306 at layer 22), but is classified as exploratory and non-load-bearing pending negative noise and self-patch controls.

**Text:** Four-way activation patching across hypothesis and article tokens in DeBERTa-v2-xlarge indicates that article tokens carry increasing causal transfer at late layers (+0.3060 at layer 22; full patch +0.9811). However, because the design lacks self-patch calibration, scrambled/noise baselines, and unrelated prompt controls, this finding is designated as an exploratory methodological lead in Appendix B and is NOT a load-bearing claim of Paper 1.

**Status:** `EXPLORATORY`

**Effect Size:** `Layer 22 article transfer: +0.3060 +/- 0.167; Full transfer: +0.9811 +/- 0.039 (Exploratory / Unaudited baselines)`

**95% CI:** `Layer 22 article transfer: +0.3060 +/- 0.167; Full transfer: +0.9811 +/- 0.039`

**Artifacts:** paper/surgical_audit_frontier_results.json

### `CLM_25A_OBSERVER_SPACE_INTRINSIC_DIMENSION`
**Summary:** Empirical stance space over 18 diverse frameworks exhibits a bounded intrinsic dimension of 6.77 (Two-NN) and effective rank 6.74 / 8, confirming constrained observer dimensionality.

**Text:** Evaluating the observer possibility space across 2,160 stance profiles measured over 18 diverse frameworks reveals a bounded intrinsic dimension of 6.77 (Two-NN estimator vs 7.41 for ambient noise) and an effective rank of 6.74 / 8. The volume fraction of 20.35% relative to an ambient uniform reference in [-1, 1]^8 is documented as exploratory descriptive context.

**Status:** `SUPPORTED`

**Effect Size:** `Two-NN intrinsic dimension = 6.77; Effective rank = 6.74 / 8`

**95% CI:** `Two-NN ID: [6.45, 7.10]; Effective rank: [6.50, 6.95]`

**Artifacts:** paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json

### `CLM_25B_ANOMALY_MIGRATION_STRUCTURAL_LAW`
**Summary:** Anomaly ranking agreement decays lawfully with 3D Grassmannian distance under a 10,000-permutation QAP test (r = -0.6902, rho = -0.6578, p_Mantel < 0.0001), exceeding synthetic geodesic rotation nulls by +0.22 to +0.26 across all distance bins and preserving an invariant anomaly core.

**Text:** Across 153 unique framework pairs, anomaly ranking agreement decays lawfully as a function of 3D Grassmannian distance (Pearson r = -0.6902, Spearman rho = -0.6578); under a 10,000-permutation Mantel / Quadratic Assignment Procedure (QAP) test strictly accounting for dyadic matrix dependence, this relationship survives decisively (p_Mantel < 0.0001 for both Pearson and Spearman). Furthermore, comparing observed anomaly correlations against a synthetic geodesic subspace rotation null confirms that empirical frameworks preserve significantly higher anomaly agreement than pure geometric rotation (Observed E[r_A | d_G] exceeds Null E[r_A | d_G] by +0.22 to +0.26 across all distance bins), preserving an invariant anomaly core despite geometric construct coupling.

**Status:** `SUPPORTED`

**Effect Size:** `QAP-verified anomaly decay: r = -0.6902, rho = -0.6578 (p_Mantel < 0.0001); Empirical vs Null E[r_A | d_G] advantage: +0.221 to +0.260`

**95% CI:** `Pearson r: [-0.75, -0.62]; Spearman rho: [-0.73, -0.58]`

**Artifacts:** paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json

### `CLM_25C_OBSERVER_SPACE_CONCEPTUAL_DISSOCIATION`
**Summary:** Static conceptual prompt distance fails to predict observational subspace geometry under dependence-aware QAP testing (DeBERTa embeddings p_Mantel = 0.2522; TF-IDF n-grams p_Mantel = 0.0975), dissociating descriptions of perspectives from observational geometry.

**Text:** Correlating observational 3D Grassmannian subspace distance with static conceptual prompt distance confirms that static metrics fail to predict observational geometry under dependence-aware QAP testing: DeBERTa hypothesis embeddings yield r = -0.0503 (p_Mantel = 0.2522, rho = -0.0387) and TF-IDF character n-grams yield r = +0.1116 (p_Mantel = 0.0975, rho = +0.0253). This demonstrates a fundamental distinction: similarity between descriptions of perspectives does not equal similarity between what those perspectives do to observations.

**Status:** `SUPPORTED`

**Effect Size:** `DeBERTa vs Observational: r = -0.0503 (p_Mantel = 0.2522); TF-IDF vs Observational: r = +0.1116 (p_Mantel = 0.0975)`

**95% CI:** `Mantel p-values strictly non-significant (p >= 0.0975)`

**Artifacts:** paper/frontier_possibility_observer_space_results.json, paper/surgical_audit_frontier_results.json

### `CLM_26_HUMAN_BENCHMARKING_PROTOCOL`
**Summary:** Preregistration-ready protocol formalizing human reader panel benchmarking across 4 interpretive communities (N=60) with power > 0.999 and 4 explicit falsification criteria.

**Text:** We specify a complete, preregistration-ready protocol to benchmark model-derived perspectival spaces against human interpretive communities. The design encompasses 4 panels (Israeli security analysts, Palestinian human rights advocates, IHL scholars, non-aligned diplomats; M=15 per panel, N_total = 60), specifying power > 0.999 to detect Procrustes manifold alignment R^2 >= 0.50 and design-projected D-study profile reliability E rho^2_AP(15) = 0.8929. The protocol establishes formal evaluation metrics (profile ICC, Procrustes disparity, anomaly Jaccard overlap) and four explicit falsification gates to adjudicate whether cross-encoder NLI models reconstruct authentic human perspectival geometry or machine-idiosyncratic artifacts.

**Status:** `PROSPECTIVE_PROTOCOL`

**Effect Size:** `Projected D-study profile reliability E rho^2_AP(15) = 0.8929 (design power specification, not empirical finding); Power > 0.999`

**95% CI:** `Projected E rho^2_AP(15): [0.85, 0.93]`

**Artifacts:** paper/frontier_human_benchmarking_protocol.md, paper/frontier_master_synthesis.json

### `CLM_27_MTMM_CONSTRUCT_VALIDITY_AND_METHOD_EFFECTS`
**Summary:** Multitrait-multimethod (MTMM) modeling demonstrates convergent construct validity across independently phrased prompt realizations (MTHM r = 0.7707, trait pattern r = 0.8387; 58.03% construct variance on core geopolitical discourse), but reveals that method variance operates primarily through Trait x Method calibration distortion (30.28% in Study 1, 31.65% in Study 2) rather than uniform elevation bias (A x M = 0.00%), explaining cross-operator attenuation (NLI vs FrameAxis MTHM r = 0.1313).

**Text:** Classical Campbell-Fiske Multitrait-Multimethod (MTMM) analysis and two-facet crossed random effects G-theory ANOVA across 8 contrastive framing dimensions demonstrate that semantic framing scores reflect authentic latent constructs across prompt wordings: syntactic paraphrase achieves convergent validity MTHM r = 0.7707 with trait pattern congruence r = 0.8387, and two-facet crossed ANOVA partitions 31.25% of total score variance to Article x Construct systematic variation (sigma^2_AT, F = 4.25, p < 10^-15), explicitly contrasted with Article elevation (sigma^2_A = 0.00%), Construct baseline difficulty (sigma^2_T = 0.00%), and Method elevation (sigma^2_M = 0.00%). On replicated geopolitical prompt banks, construct variance reaches 58.03% (MTHM r = 0.7095). Crucially, full parameterization resolves the apparent 'infinite ratio' artifact: general article-level method bias (sigma^2_AM) is negligible (0.00%, F = 0.55, p = 1.00), but Construct x Method calibration distortion (sigma^2_TM) accounts for 30.28% of total variance (F = 95.46, p < 10^-15), yielding an Article x Construct to Construct-Method ratio of 1.032. Residual error and unmodeled 3-way interaction (sigma^2_ATM,e) account for 38.47%. This demonstrates that method effects operate through construct-specific scale distortion across operators (dropping NLI vs FrameAxis convergent validity to r = 0.1313), rather than uniform stance elevation.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Paraphrase MTHM r = 0.7707; Geopolitical Trait Var = 58.03%; Study 1 Trait Var = 31.25% vs Trait x Method = 30.28% (Ratio = 1.032); NLI vs FrameAxis MTHM r = 0.1313; DeBERTa vs BART MTHM r = 0.6077`

**95% CI:** `Construct Trait F = 4.25 (p = 1.11e-16); Trait x Method F = 95.46 (p = 1.11e-16); Convergent validity p = 4.23e-12`

**Artifacts:** paper/mtmm_framing_geometry_results.json, scripts/run_mtmm_analysis.py

### `CLM_28_DIF_AND_MULTI_GROUP_MEASUREMENT_INVARIANCE`
**Summary:** Psychometric Differential Item Functioning (DIF) with Leave-One-Item-Out purified anchors and formal Multi-Group CFA demonstrates that prompt wording preserves exploratory factor-loading pattern congruence (Tucker |phi| >= 0.9113) and achieves partial metric invariance under approximate-fit testing, but exact likelihood-ratio and approximate tests decisively reject full metric invariance across model architectures (Delta chi^2 = 97.74, Delta CFI = -0.0744) and operators (Delta chi^2 = 323.15, Delta CFI = -0.1121). A culinary common-item control proves instrument linkability across corpora (75% Class A invariance, zero Class C), clarifying that prior cross-domain failures stemmed from unlinked item banks.

**Text:** Audited Differential Item Functioning (DIF) under Jodoin & Gierl (2001) continuous regression criteria (conditioned on Leave-One-Item-Out purified anchors) and formal Multi-Group Confirmatory Factor Analysis (MG-CFA) evaluated under both asymptotic likelihood-ratio tests (LRT) and approximate-fit criteria (Cheung & Rensvold 2002; Chen 2007: Delta CFI >= -0.010, Delta RMSEA <= +0.015, Delta SRMR <= +0.010) reveal the psychometric boundaries of framing geometry: (1) Prompt wording realizations preserve exploratory factor-loading pattern congruence across all pairs (Tucker |phi| >= 0.9113); although exact LRT rejects full metric invariance (Delta chi^2 = 77.25, df = 14, p = 9.13e-11), approximate fit indicates partial preservation with Delta RMSEA = +0.0004. (2) Model architecture shifts exhibit 37.5% Class A invariance, 25.0% Class B DIF, and 37.5% Class C severe DIF, rejecting metric invariance under both LRT (Delta chi^2 = 97.74, df = 14, p = 1.29e-14) and approximate fit (Delta CFI = -0.0744). (3) Shifting semantic operators from NLI to FrameAxis breaks pattern congruence (|phi| = 0.7312 < 0.90) and fails MG-CFA metric invariance (Delta chi^2 = 323.15, df = 7, p < 1e-15, Delta CFI = -0.1121). (4) Cross-corpus evaluation under a common-item design (8 identical Culinary Arts items administered to both Gaza N=120 and Ukraine N=60) yields 75.0% Class A invariance (6/8), zero Class C severe DIF (0/8), and achieves Partial Metric Invariance (6 invariant items, Delta chi^2 = 18.32, p = 0.0106). This culinary control conclusively proves instrument linkability across disparate corpora, refuting the hypothesis of instrument breakdown while leaving open whether substantively meaningful political framing items transport across domains.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Common-Item Linkability Control: 75.0% Class A Invariant, 0% Class C, Partial Metric Supported; Wording Congruence: all |phi| >= 0.9113; Architecture MG-CFA Metric Delta chi^2 = 97.74 (p = 1.29e-14, Delta CFI = -0.0744); Operator MG-CFA Metric Delta chi^2 = 323.15 (p < 1e-15, Delta CFI = -0.1121)`

**95% CI:** `Benjamini-Hochberg FDR q <= 0.05 on all purified candidate tests; Delta CFI >= -0.010; Delta RMSEA <= +0.015; Tucker |phi| exploratory threshold = 0.90`

**Artifacts:** paper/dif_measurement_invariance_results.json, scripts/run_dif_analysis.py

### `CLM_29_DOMAIN_GENERAL_POLITICAL_ANCHOR_INVARIANCE`
**Summary:** Audited domain-general political anchor administration across 5 policy domains (Gaza, Ukraine, Immigration, Climate, Abortion; N=300) establishes that 8 common political contrasts achieve 87.5% pairwise Class A invariance under continuous DIF when Gaza text extraction is audited, but cautions that treating the 8 distinct constructs as indicators of a single general factor is psychometrically uninterpretable due to unacceptable configural baseline fit (RMSEA=0.165, SRMR=0.087); cross-operator divergence (r=0.0642) empirically demonstrates non-interchangeability rather than incommensurability.

**Text:** Administering an 8-item domain-general political anchor bank across 5 policy domains (Gaza Conflict, Ukraine War, US Immigration, Climate Politics, Abortion Rights; N=300 total articles) establishes the psychometric properties of common political framing contrasts: (1) An audit of the scoring pipeline resolved an initial Gaza variance compression defect (where key mismatch 'content' vs 'text' fed empty strings), establishing authentic framing variance across all domains (Gaza margin sigma = 0.348 vs 0.327-0.464 elsewhere) and raising pairwise continuous DIF (Jodoin & Gierl 2001) Class A invariance to 87.5% across all 10 domain pairs (95.1% across non-Gaza pairs). (2) We explicitly correct a prior psychometric misinterpretation: the 8 items operationalize distinct conceptual constructs (legitimacy, coercion, proportionality, compliance, harm vs necessity, accountability, solidarity, reform), not parallel indicators of a single general factor. Furthermore, the 1-factor multi-group CFA configural baseline exhibited unacceptable fit (RMSEA = 0.165, SRMR = 0.087), which standard psychometric theory establishes as an uninterpretable foundation for testing metric invariance or claiming an 'invariant political core'. (3) Cross-operator convergent validity between cross-encoder NLI and bi-encoder FrameAxis remains near zero (mean MTHM r = 0.0642). Following conservative measurement theory, this divergence is framed neutrally across three hypotheses (loss of sensitivity in whole-article mean-pooling, NLI sequence-pair attention calibration variance, or distinct semantic targets); absent human reader benchmarks, it proves lack of operational interchangeability rather than metaphysical incommensurability.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Audited Gaza sigma = 0.348; Pairwise Class A DIF = 87.5% (210/240); Non-Gaza Class A DIF = 95.1% (137/144); 1-Factor Configural RMSEA = 0.165 (unacceptable baseline fit); Cross-operator r = 0.0642 (non-interchangeability)`

**95% CI:** `Benjamini-Hochberg FDR q <= 0.05 on candidate DIF tests; RMSEA cutoff <= 0.08; Tucker |phi| >= 0.90`

**Artifacts:** paper/political_anchor_cross_domain_results.json, scripts/run_political_anchor_experiment.py, config/prompt_banks/domain_general_political_bank.json

### `CLM_30_MULTIDIMENSIONAL_POLITICAL_CONSTRUCT_INVARIANCE`
**Summary:** A 24-item multitrait multi-indicator design (8 constructs x 3 realizations) administered across 5 policy domains (N=300) under a split-sample protocol reveals that 5 of 8 political constructs achieve multi-group metric invariance on the Discovery Sample (N=150), whereas 3 constructs (coercion, proportionality, compliance) exhibit domain-dependent calibration shifts; simultaneous 8-factor CFA confirms that full multidimensional metric invariance is underpowered and rejected out-of-sample (N=150).

**Text:** Administering a frozen 24-item multitrait bank operationalizing 8 distinct political constructs with 3 independently authored contrastive realizations across 5 policy domains (Gaza Conflict, Ukraine War, US Immigration, Climate Politics, Abortion Rights; N=300) under a split-sample confirmation protocol (Discovery N=150, Confirmation N=150) establishes the true dimensionality and invariance boundaries of political framing geometry: (1) Per-construct Multi-Group CFA (3 indicators per construct across 5 groups) on the Discovery Sample (N=150) demonstrates that 5 out of 8 constructs achieve metric invariance: Institutional Legitimacy (Delta chi^2 = 6.08, df=8, p = 0.638, Delta CFI = +0.0005), Democratic Accountability (Delta chi^2 = 2.81, df=8, p = 0.946, Delta CFI = +0.0000), Civic Solidarity (Delta chi^2 = 11.38, df=8, p = 0.181, Delta RMSEA = -0.0120), Reform vs Stability (Delta chi^2 = 13.56, df=8, p = 0.094), and Humanitarian Harm vs Strategic Necessity (Delta chi^2 = 15.15, df=8, p = 0.056). Three constructs reject metric invariance: State Coercion (Delta chi^2 = 28.51, p < 0.001), Tactical Proportionality (Delta chi^2 = 22.15, p = 0.005), and Legal Compliance (Delta chi^2 = 15.76, p = 0.046), reflecting genuine domain shifts in what constitutes coercion, proportionality, or legality across military and domestic contexts. (2) Global 8-factor multidimensional CFA (24 indicators, 8 correlated factors) confirms that estimating 500 parameters simultaneously across 5 groups is severely underpowered at N=150 (Discovery configural CFI = 0.5171, RMSEA = 0.1126, SRMR = 0.2080), decisively rejecting omnibus metric invariance (Delta chi^2 = 172.30, df=64, p = 7.13e-12). (3) The held-out Confirmation Sample (N=150) independently replicates this exact structural pattern (Confirmation configural CFI = 0.5717, RMSEA = 0.1118; metric Delta chi^2 = 271.30, p < 1e-15). (4) Purified continuous DIF across all 240 tests (10 pairs x 24 items) identifies 87.5% Class A invariance (210/240), 4.6% Class B (11/240), and 7.9% Class C (19/240) with mean Tucker |phi| = 0.9364 (range [0.8607, 0.9758]). (5) Cross-operator MTMM on 24 common items yields mean MTHM r = 0.0733 +/- 0.0841 and pattern congruence r = 0.4162, confirming lack of interchangeability across the full multitrait bank.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Per-Construct MG-CFA: 5/8 Invariant (Legitimacy p=0.638, Accountability p=0.946, Solidarity p=0.181, Reform p=0.094, Humanitarian p=0.056); 3/8 Non-Invariant (Coercion p=0.0004, Proportionality p=0.0046, Compliance p=0.0459); Pairwise Class A DIF = 87.5% (210/240); Mean Tucker |phi| = 0.9364; Global 8-Factor Configural CFI = 0.5171 (Discovery) / 0.5717 (Confirmation); Cross-Operator r = 0.0733`

**95% CI:** `Benjamini-Hochberg FDR q <= 0.05; LRT alpha = 0.05; Delta CFI >= -0.010, Delta RMSEA <= +0.015; Tucker |phi| >= 0.90`

**Artifacts:** paper/political_anchor_multitrait_results.json, scripts/run_political_anchor_multitrait_experiment.py, config/prompt_banks/domain_general_political_bank_24.json

### `CLM_31_NATURAL_EDIT_RESPONSE_TOMOGRAPHY`
**Summary:** Natural Edit Response Tomography (NERT) on N=1,000 newsroom edit pairs establishes that linear response operators Rm: z -> Delta y fail as predictive system identification (negative held-out R^2 < 0 across all models on unseen story families). In 5-fold regularized CCA across 58 provenance-hardened story families, MiniLM and RoBERTa dense displacements achieve high out-of-fold correlation (OOF r = 0.9704, deep-residualized r = 0.9462). However, decisive leakage-free non-newsroom baselines reveal that this alignment is driven by generic dense representation geometry rather than edit dynamics: leakage-free unconstrained arbitrary article differences achieve an even higher correlation (OOF r = 0.9790), and magnitude-matched arbitrary pairs achieve r = 0.8892. Subtracting a generic resting representation map T drops cross-model CCA on residuals to r = 0.5532 with diffuse residual dimensionality (real edit PR = 21.51 vs arbitrary PR = 13.67), while whole-document displacements align only modestly with literal added/removed span semantics (cos = 0.2812 to 0.3330).

**Text:** A hostile audit and decisive residual structure analysis of Natural Edit Response Tomography (NERT) across N=1,000 newsroom syndication edit pairs clustered into 58 provenance-hardened story families (ensuring zero cross-family leakage with max cross-family J <= 0.2810; mean Jaccard = 0.8792 +/- 0.063, mean |Delta chars| = 333.5) establishes the empirical boundaries of edit-space NLP: (1) NERT fails as predictive system identification: low-dimensional linear response operators Rm: z -> Delta y yield negative held-out omnibus R^2 across all models on unseen story families (DeBERTa -0.0072, FrameAxis -0.0082, BART -0.0080, RoBERTa -0.0044). (2) In 5-fold regularized CCA, MiniLM and RoBERTa dense displacement vectors achieve high out-of-fold correlation (OOF r = 0.9704, deep-residualized r = 0.9462) with stable loading vectors (|cos theta| = 0.918 - 0.943) that transfer frozen across independent split-half story families (Discovery in-sample r = 0.9846 -> Confirmation frozen held-out r = 0.9700). (3) A strictly leakage-free arbitrary-pair baseline (families pre-partitioned prior to pair sampling) refutes the claim that this alignment is unique to newsroom rewrites: unconstrained cross-family document differences yield OOF r = 0.9790, while displacement-magnitude-matched arbitrary pairs yield OOF r = 0.8892 (real newsroom advantage Delta r = +0.0812). (4) Fitting a generic resting linear mapping T: e_M(x) -> e_R(x) on Discovery resting articles explains 35.42% of held-out resting variance (cosine 0.6280); when applied to displacements, T accounts for 36.07% of real edit displacement variance (cosine 0.3270) versus 31.83% of matched arbitrary displacement variance (cosine 0.5820). (5) Analyzing the residual displacements epsilon = Delta e_R - T Delta e_M reveals that real newsroom edits have more diffuse effective dimensionality (PR = 21.51, top-1 var fraction = 0.1413) than arbitrary pairs (PR = 13.67, top-1 var fraction = 0.1686); cross-model regularized CCA on these residuals drops to r = 0.5532, confirming that the bulk of the initial r ~ 0.97 correlation is driven by generic dense embedding geometry. (6) Literal added/removed text diffs align with whole-article displacements at only cos = 0.3330 (MiniLM) and cos = 0.2812 (RoBERTa), though span differences themselves exhibit strong cross-model alignment (held-out CCA r = 0.8963). (7) Finite-cluster bootstrap (B=500) confirms participation ratio effective rank intervals overlap heavily across architectures (DeBERTa [3.86, 5.65], FrameAxis [4.20, 5.91], BART [4.26, 5.85], RoBERTa [4.05, 5.48]), and Haar random subspace null in G(3,8) confirms FrameAxis vs BART (d_G = 1.7313) and RoBERTa (d_G = 1.7713) match random orthogonal subspaces (p ~ 0.40).

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `NERT Predictive System Identification Failed (held-out R^2 < 0 all models); Real Newsroom Edit CCA OOF r = 0.9704 (residualized r = 0.9462); Leakage-Free Arbitrary: Unconstrained OOF r = 0.9790, Mag-Matched OOF r = 0.8892 (Delta r = +0.0812); Resting Map T Edit Disp R^2 = 0.3607; Residual CCA r = 0.5532; Real Edit Residual PR = 21.51 vs Arb PR = 13.67; Span Alignment cos = 0.2812 - 0.3330 (Span CCA r = 0.8963)`

**95% CI:** `58 provenance-hardened story families (max cross-family J <= 0.2810); 5-fold family-partitioned CV; B=500 cluster bootstrap PR CIs`

**Artifacts:** paper/decisive_residual_experiment_results.json, paper/hostile_cca_and_multiverse_audit_results.json, paper/cca_residualization_and_mrqap_results.json, paper/natural_edit_tomography_audited_results.json, paper/natural_edit_response_tomography.md, scripts/decisive_residual_structure_experiment.py, scripts/hostile_cca_and_multiverse_audit.py

### `CLM_32_MEASUREMENT_MULTIVERSE_STATIC_EDIT_COUPLING`
**Summary:** Deduplicated Measurement Multiverse (M=14 base instruments, N=91 non-redundant pairs) across N=1,000 natural edit pairs demonstrates that resting static agreement couples to edit agreement under QAP matrix permutation (r=0.8473, p=0.000200) and survives 4-factor MRQAP control (partial r = 0.7318, p=0.000200); however, static-dynamic coupling is algebraically constrained by constituent cross-article covariance blocks, Small-Edit Variance Ratios (SEVR) reflect local perturbation sensitivity rather than classical unreliability (refuting theoretical Spearman ceilings, as observed DeBERTa-RoBERTa raw edit correlation is 0.1353), and flash labels ('Superficial Mimics' and 'Perspectival Invariants') are retired as unsupported.

**Text:** An audit and refactoring of the Measurement Multiverse across N=1,000 verified newsroom syndication edit pairs (2,000 texts, 58 provenance-hardened story families, 347 publishers; mean Jaccard = 0.8792) establishes the mathematical and psychometric realities of static-dynamic instrument coupling: (1) Deduplicating the multiverse to M=14 genuinely distinct base instruments (eliminating deterministic z-score, rank, sigmoid, and ensemble clones) yields N=91 non-redundant instrument dyads. Bivariate OLS yields S_edit = 0.7316 * S_static + 0.0450 (r = 0.8473, R^2 = 0.7179, p_QAP = 0.000200, exact 1/5001). (2) Multiple Regression QAP (MRQAP) controlling simultaneously for same encoder, same architecture, same pole set, and same scoring rule confirms that static similarity remains a substantial predictor: beta_static = +0.4986, full model R^2 = 0.8350; the partial correlation between static and edit similarity remains strong at partial r = 0.7318 (p_MRQAP = 0.000200, exact 1/5001). (3) Covariance bilinearity Cov(Delta X, Delta Y) = Cov(XB, YB) + Cov(XA, YA) - Cov(XB, YA) - Cov(XA, YB) demonstrates that edit covariances are algebraically tied to cross-article covariance blocks (r=1.0000, MAE=0.0000); resting static correlation alone does not dictate edit correlation without the cross-terms Cov(XB, YA) and Cov(XA, YB). The algebraic identity does not refute outliers, but flash designations ('Perspectival Invariants' and 'Superficial Mimics') are retired as unsupported by valid counterfactual baselines. (4) Small-Edit Variance Ratios (SEVR = Var(Dy | J >= 0.95) / Var(Dy)) quantify local perturbation sensitivity rather than classical unreliability: NLI models vary strongly even on J >= 0.95 pairs (SEVR = 0.85 - 1.00), which reflect legitimate editorial revisions rather than repeated measurements. The empirical DeBERTa-RoBERTa edit correlation of 0.1353 decisively refutes a naive attenuation ceiling of 0.0106 derived from misinterpreting SEVR as error variance.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Deduplicated Multiverse: S_edit = 0.7316 * S_static + 0.0450 (r = 0.8473, p_QAP = 0.000200); 4-Factor MRQAP: partial r = 0.7318 (p_MRQAP = 0.000200); Algebraic Null: r(S_exp, S_edit) = 1.0000; Observed DeBERTa-RoBERTa raw edit r = 0.1353; Cross-encoder NLI SEVR = 0.85 - 1.00`

**95% CI:** `MRQAP N=5,000 matrix permutations (null p95 = 0.1932); 58 provenance-hardened story families (max cross-family J <= 0.2810); N=1,000 natural edit pairs`

**Artifacts:** paper/hostile_cca_and_multiverse_audit_results.json, paper/cca_residualization_and_mrqap_results.json, paper/rigorous_multiverse_audit_results.json, paper/measurement_multiverse_synthesis.md, scripts/hostile_cca_and_multiverse_audit.py

### `CLM_33_CONSTRUCT_ALIASING_AND_SEMANTIC_IDENTIFIABILITY`
**Summary:** A hostile audit across a balanced 6-domain corpus (N=360) refutes persistent cross-construct aliasing under DeBERTa-v2-xlarge NLI (0/11,520 pairs at d_sem >= 0.50, rho_resp >= 0.85) and demonstrates that the apparent sample-size 'frontier' is quantitatively reproduced by ordinary sampling noise (analytical Fisher z null expects 228.8 aliases at N=15 vs 6.6 at N=360). Residualizing an article-level common-mode response propensity (raw absurd-substantive r = 0.5658 -> cross-fitted r = 0.0220) purges antonym bias (+0.2845 -> -0.0120) while preserving the monotonic hierarchy. The Support-Limited Identifiability hypothesis is falsified as a mean effect: fixed-N (N=60) corpus diversification yields flat within-pair distinguishability shifts (mean Delta D = -0.0005 to +0.0045, p >= 0.05) that fail to replicate in held-out confirmation (r = 0.0838).

**Text:** A hostile audit of construct aliasing and semantic identifiability across a balanced 6-domain corpus (N=360 articles: Gaza, Ukraine, Immigration, Climate, Abortion, General News) and 11,520 cross-construct dyads establishes: (1) Strong construct aliasing is falsified for cross-encoder NLI: exactly zero pairs (0.00%) satisfy d_sem >= 0.50 and rho_resp >= 0.85 on the pooled corpus. (2) The apparent sample-size frontier is an artifact of sampling error: analytical pair-specific Fisher z expectations (N=15: 228.8, N=30: 59.4, N=60: 19.9, N=120: 10.0, N=360: 6.6) and empirical resampling (N=15: 389.6, N=30: 117.8, N=60: 36.8, N=120: 14.4, N=360: 7.3) quantitatively reproduce the decline in extreme correlations. (3) Raw NLI responses are inflated by an article-level common-mode factor (absurd-substantive mean r = 0.5658); common-mode-only residualization eliminates this propensity (cross-fitted r = 0.0220) and purifies the measurement hierarchy (syntactic 0.7903 > lexical 0.5761 > same construct/diff facet 0.2485 > diff construct 0.1990 > antonym -0.0120). (4) Support-Limited Construct Identifiability is falsified as a mean phenomenon: in repeated fixed-N (N=60) composition-balanced experiments across 1, 2, 3, and 6 domains, mean within-pair distinguishability shifts are negligible (residualized NLI Delta D = -0.0005, p = 0.0589; centered FrameAxis MiniLM Delta D = +0.0003, p = 0.4753; centered FrameAxis RoBERTa Delta D = +0.0017) and fail to generalize in split-half held-out confirmation (r = 0.0838). (5) Anisotropy centering resolves the apparent collapse of FrameAxis RoBERTa (raw D = 0.0966, r = 0.9034 -> centered D = 1.0121, r = -0.0121). (6) Dyadic MRQAP (N=5,000 permutations, R^2 = 0.0780) confirms prompt semantic distance beta = -0.7174 (p = 0.000200) and same facet beta = +0.2668 (p = 0.000200) significantly predict residual response agreement, while coarse construct indicators do not (beta = +0.0065, p = 0.4840). (7) In Tier 2 FrameAxis (499,500 pairs), 275 discovery candidates regress out-of-sample to mean r = 0.8347 (28.36% survive strict r >= 0.85; 100% survive r >= 0.70).

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Aliasing at N=360: 0/11,520 (0.00%); Sampling Null Expected Aliases: N=15 (228.8 anal / 389.6 sim) -> N=360 (6.6 anal / 7.3 sim); CM Residualization: absurd r = 0.5658 -> 0.0220; Antonym r = +0.2845 -> -0.0120; Support Diversification: NLI Delta D = -0.0005 (p=0.0589); Centered MiniLM Delta D = +0.0003 (p=0.4753); Held-out Delta D rep r = 0.0838; RoBERTa Anisotropy Correction: D = 0.0966 -> 1.0121; Dyadic MRQAP: R^2 = 0.0780, beta_dsem = -0.7174 (p=0.000200)`

**95% CI:** `N=5,000 QAP permutations; B=300 empirical resampling runs across 6 domains (N=360); Held-out split-half confirmation (180 Discovery vs 180 Confirmation)`

**Artifacts:** paper/support_identifiability_audit_results.json, paper/construct_aliasing_results.json, paper/construct_aliasing_monograph.md, scripts/run_support_identifiability_audit.py

### `CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH`
**Summary:** Centered Kernel Alignment (CKA) across N=6,248 independent story family representatives reveals that hypothesis-prompt invariance (Q0 <-> Q1 paraphrase stability) is substantially higher at the middle layer (Layer 12 CLS, V7 <-> V8: CKA_RBF = 0.9131, Linear: 0.9073, 95% CI: [0.9119, 0.9176], permutation p < 0.0020) than in final hidden representations (Layer 24 CLS, V3 <-> V4: CKA_RBF = 0.7770, Linear: 0.7740, 95% CI: [0.7720, 0.7857], p < 0.0020) and paired scalar classifier margins (V1 <-> V2: CKA_RBF = 0.7490, Linear: 0.8015, 95% CI: [0.7401, 0.7594], p < 0.0020); sampling three representation depths establishes boundary and midpoint differences rather than monotonic behavior at every intervening layer.

**Text:** Multi-view Centered Kernel Alignment (CKA) across N=6,248 independent story family representatives (evaluating against 500-permutation null distributions and B=200 cluster bootstrap iterations) establishes that prompt-invariance geometry varies systematically across network depth: (1) Prompt stability between direct probes (Q0) and paraphrased probes (Q1) is highest at the layer midpoint and lower at the classifier head across the sampled depths: Layer 12 CLS (V7 <-> V8) achieves CKA_RBF = 0.9131 (Linear: 0.9073, 95% CI: [0.9119, 0.9176], p < 0.0020, 0/500 null exceedances), Layer 24 CLS (V3 <-> V4) achieves CKA_RBF = 0.7770 (Linear: 0.7740, 95% CI: [0.7720, 0.7857], p < 0.0020), and the paired scalar margin (V1 <-> V2) achieves CKA_RBF = 0.7490 (Linear: 0.8015, 95% CI: [0.7401, 0.7594], p < 0.0020); we explicitly note that three sampled representations establish differences between midpoint, final layer, and margins rather than monotonic behavior at every unobserved intervening layer. (2) Cross-view geometry between scalar margins (V1) and hidden-state views reveals moderate coupling to final-layer representations (Layer 24 CLS V3: CKA = 0.5437; Layer 24 Mean V5: CKA = 0.5034), but sharp decoupling from middle-layer representations (Layer 12 CLS V7: CKA = 0.1196). (3) Sensitivity analysis on N=6,248 family representatives versus the full N=7,986 selector union confirms that global manifold alignment is resilient to syndicated copies (|Delta CKA| <= 0.0408; Scalar Delta = -0.0080, Mid-CLS Delta = +0.0150, Final-CLS Delta = +0.0408), while local k-NN graph overlap is substantially inflated (+0.076 to +0.099, up to 4.0x) by wire reproduction, validating restriction to independent story families. (4) Quadratic-time unbiased MMD^2 across eligible selector conditions confirms significant distributional divergence across topic selection boundaries (all permutation p < 0.0020).

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `Prompt Invariance CKA: Layer 12 CLS = 0.9131, Layer 24 CLS = 0.7770, Scalar Margin = 0.7490; Cross-View Scalar-to-Hidden CKA: V1-V3 = 0.5437, V1-V5 = 0.5034, V1-V7 = 0.1196; Syndication Sensitivity |Delta CKA| <= 0.0408`

**95% CI:** `Layer 12 CKA 95% CI: [0.9119, 0.9176]; Layer 24 CKA 95% CI: [0.7720, 0.7857]; Scalar CKA 95% CI: [0.7401, 0.7594]; 500-permutation null controls (all p < 0.0020); B=200 cluster bootstrap resamples on N=6,248 independent families`

**Artifacts:** deliverables/05_gate3_full_experiment/cka_matrix_6248.json, deliverables/05_gate3_full_experiment/gate3_executive_synthesis.md, deliverables/05_gate3_full_experiment/gate3_full_report.json, deliverables/05_gate3_full_experiment/sensitivity_analysis.json, deliverables/05_gate3_full_experiment/mmd_eligible_conditions.json

### `CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION`
**Summary:** In a balanced 120-template / N=480 passage controlled intervention suite benchmarked against a 512-article real news panel, final-layer representations (V3 Final-CLS) exhibit high target contrast specificity (ratio of means = 2.01x, p = 2.92e-46) and pronounced relational discrimination in ambient news space (Delta R10 = +66.58 pp, p = 3.75e-47), whereas middle-layer representations (V7 Mid-CLS) fail specificity (ratio of means = 0.92x, p = 0.910) and do not detect relational discrimination (Delta R10 = +0.75 pp, p = 0.360; failure to detect discrimination is not proof of exact insensitivity). This demonstrates an empirical dissociation between prompt stability, proposition specificity, and relational discrimination rather than an inherent universal trade-off. Distance support audits confirm that synthetic passage edits displace outward relative to intra-panel news distributions (Kolmogorov-Smirnov rejection p = 0.005).

**Text:** A controlled semantic intervention audit across 120 base templates (N=480 matched passages across 8 contrast dimensions: original, paraphrase, proposition reversal, and surface control) evaluated against a frozen 512-article real news reference panel establishes: (1) Dual ratio reporting resolves the distinction between aggregate displacement and template-level ratios: for V3 Final-CLS, the ratio of aggregate means is 2.01x (template mean ratio 2.04x, median 1.91x, 100.0% win rate, p = 2.92e-46); for V1 Scalar Margin, the ratio of means is 2.29x (template mean 2.49x, median 2.19x, 93.3% win rate, p = 2.81e-34); whereas V7 Mid-CLS fails specificity with a ratio of aggregate means of 0.92x (< 1.0; template mean 1.14x skewed by small-denominator tail, median 1.03x, win rate 51.7%, p = 0.910). (2) Ambient nearest-neighbor graph retention (R10 in real news space) confirms pronounced relational discrimination in the final layer (V3): meaning-preserving rewrites retain 71.33% (J10 = 0.6121) of news neighbors, while proposition reversals collapse retention to 4.75% (J10 = 0.0312), yielding Delta R10 = +66.58 pp (p = 3.75e-47) and significantly exceeding surface controls (22.08%, Delta R10 = +17.33 pp, p = 1.25e-8). In contrast, middle-layer (V7) retention discrimination is not detected: paraphrase retains 51.67%, surface control retains 47.67%, and proposition reversal retains 50.92% (Delta R10 = +0.75 pp, p = 0.360; failure to detect discrimination under this test does not prove exact insensitivity). (3) The joint data support an empirical dissociation among three distinct measurement properties (prompt stability, proposition specificity, and relational discrimination) rather than an inherent trade-off: Layer 12 CLS exhibits high prompt stability (CKA 0.913) but lacks detected specificity (0.92x) or neighborhood discrimination (+0.75 pp), whereas Layer 24 CLS exhibits moderate prompt stability (CKA 0.777) with strong specificity (2.01x) and high discrimination (+66.58 pp). Maximizing representational invariance alone is therefore an insufficient criterion for instrument validity. (4) Distance support audits reveal that edited passages exhibit outward distributional displacement (1st NN medians rise from 0.9255 to 0.9814 - 1.1098, exceeding the upper quartile of real-to-real distances 1.0278), and two-sample Kolmogorov-Smirnov testing decisively rejects the null hypothesis of distributional identity against the real news panel (10th NN: KS = 0.1732, p = 0.00505; 1st NN: KS = 0.2355, p = 3.16e-5), bounding assertions of manifold membership.

**Status:** `QUALIFIED_SUPPORT`

**Effect Size:** `V3 Final-CLS Specificity: Ratio of Means = 2.01x (Template Mean 2.04x, p = 2.92e-46); V1 Scalar Margin: Ratio of Means = 2.29x (Template Mean 2.49x, p = 2.81e-34); V7 Mid-CLS Specificity: Ratio of Means = 0.92x (p = 0.910); Ambient R10 Retention Drop (Pres - Chan): V3 = +66.58 pp (p = 3.75e-47), V1 = +62.33 pp, V7 = +0.75 pp (p = 0.360); Surface Edit R10: V3 = 22.08% (Reversal < Surface by +17.33 pp, p = 1.25e-8); Distance Support KS Test: 10th NN KS = 0.1732 (p = 0.00505), 1st NN KS = 0.2355 (p = 3.16e-5)`

**95% CI:** `Paired t-test across 120 balanced intervention templates (N=480 passages, 8 dimensions); Kolmogorov-Smirnov continuous distribution tests vs N=512 real news panel; Exact R10 and Jaccard J10 neighbor statistics`

**Artifacts:** deliverables/07_hidden_state_specificity_report.md, deliverables/07_hidden_state_specificity_report_data.npz, results/controlled_hidden_state_diagnostic_audit.json, results/controlled_hidden_state_neighborhood_report.json, data/controlled_semantic_interventions_manifest.json
