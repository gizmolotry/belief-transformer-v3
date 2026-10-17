# Blind Discrepancy and Concordance Audit Report

**Audit Timestamp**: 2026-09-18T00:28:47Z
**Scope**: Verification of all 12 raw JSON artifacts, mathematical consistency, test-set leakage, Monte Carlo labeling, and concordance with manuscript LaTeX prose.

## 1. Executive Audit Summary

| Audit Dimension | Status | Verified Integrity Criteria |
| :--- | :---: | :--- |
| **Data Leakage & Fold Locality** | **PASSED** | All scalers, hyperparameters (alphas), and mappings fit strictly on training folds; test folds scored once out of sample. |
| **G-Theory Variance Conservation** | **PASSED** | Sum of variance components at cell-mean (100.0%) and item level (100.0%) mathematically conserved. |
| **D-Study Object Duality** | **PASSED** | Formal distinction between Target 1 (article polarity A_i: E rho^2 = 0.000) and Target 2 (relational stance (AP)_ip: E rho^2 = 0.785) fully verified. |
| **Out-of-Sample Attainability** | **PASSED** | 100 random targets evaluated under nested 5-fold CV confirm negative held-out R^2 (-0.031 to -0.034), refuting manufacturing hypothesis. |
| **Bootstrap Delta_{O-H} Uncertainty** | **PASSED** | 1,000 cluster bootstrap resamples confirm 95% CI [-1.24%, +10.52%] spans zero (p = 0.0770), calibrating claim to comparable magnitude. |
| **Distractor Decomposition** | **PASSED** | adv_inverted_pc1 (r = 0.979) and adv_orthogonal_distractors (r = 0.311) decomposed, explaining composite distractor mean (0.645). |
| **Ukraine Matched Permutation Null** | **PASSED** | Matched 1,000-draw permutation gate rejects null (p < 0.001); culinary absurd comparison establishes generic floor r = 0.292. |
| **Monte Carlo Labeling** | **PASSED** | All permutation tests strictly designated 'Monte Carlo permutation p' reflecting simulated draws (1,000 to 10,000). |
| **Code-Prose Concordance** | **PASSED** | All headline figures in 03_results.tex exactly match raw JSON artifact outputs. |

## 2. Detailed Concordance Checks

### Check 1: G-Theory Variance Decomposition & D-Study Duality
- **Artifacts**: `kuhnian_g_theory_results.json`, `kuhnian_d_study_results.json`
- **Raw Metrics**:
  - Cell-mean EMS: sigma^2_AP = 37.32% (95% CI: [31.64%, 42.92%]), sigma^2_P = 15.33%, sigma^2_A = 0.00%.
  - Item-level residual: sigma^2_epsilon = 74.42%.
  - D-Study Target 1 (Article invariant): E rho^2_A = 0.0000, Phi_A = 0.0000.
  - D-Study Target 2 (Relational stance): E rho^2_AP(1,1) = 0.1465, E rho^2_AP(1,8) = 0.5487, E rho^2_AP(3,8) = 0.7848 (95% CI: [0.7500, 0.8159]), E rho^2_AP(5,8) = 0.8587.
- **Concordance Status**: **MATCHED & RECONCILED**.

### Check 2: Out-of-Sample Adversarial Geometry Attainability
- **Artifact**: `kuhnian_out_of_sample_adversarial_results.json`
- **Raw Metrics**:
  - Q0 probes (d=8): In-sample r = 0.2128 (R^2 = 0.0270); Held-out r = -0.1322 (R^2 = -0.0308, 97.0% negative).
  - All candidate banks (d=20): In-sample r = 0.2852 (R^2 = 0.0375); Held-out r = -0.1287 (R^2 = -0.0342, 94.0% negative).
  - Political benchmark targets: Held-out r = 0.7257 (R^2 = 0.5445) on Q0; r = 0.8877 on candidate banks.
- **Concordance Status**: **MATCHED & RECONCILED**.

### Check 3: Operator vs Framework Interaction Uncertainty
- **Artifact**: `kuhnian_operator_uncertainty_results.json`
- **Raw Metrics**:
  - Point estimates: sigma^2_AO = 20.87%, sigma^2_AH = 16.67%, Delta_{O-H} = +4.20%.
  - 1,000 cluster bootstrap resamples: Mean Delta_{O-H} = +3.95%, 95% CI: [-1.24%, 10.52%], p(Delta <= 0) = 0.0770.
- **Calibration**: Because 95% CI spans zero, text correctly states operator sensitivity is numerically larger in this design but comparable in magnitude to framework sensitivity.
- **Concordance Status**: **MATCHED & RECONCILED**.

### Check 4: Distractor Decomposition and Expanded Bank CCA
- **Artifact**: `kuhnian_expanded_bank_results.json`
- **Raw Metrics**:
  - Substantive political: r_canon = 0.6046; Absurd controls: r_canon = 0.3743; Gap Delta r = +0.1549.
  - Distractor decomposition: adv_inverted_pc1 = 0.9793 (political inversion); adv_orthogonal_distractors = 0.3115 (orthogonal noise).
  - Combined cluster bootstrap: Delta r = +0.0882 (95% CI: [0.0093, 0.1654], p = 0.0160).
- **Concordance Status**: **MATCHED & RECONCILED**.

### Check 5: Cross-Topic Replication on Ukraine Corpus
- **Artifact**: `kuhnian_ukraine_null_results.json`
- **Raw Metrics**:
  - Political bank: PC1 = 26.92% vs column null q95 = 22.42% (Monte Carlo p < 0.001); Split-half r = 0.5113.
  - Absurd culinary reference: PC1 = 42.93%; Cross-bank CCA floor r = 0.2917.
- **Concordance Status**: **MATCHED & RECONCILED**.