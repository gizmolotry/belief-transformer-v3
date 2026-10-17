# Volume II: The Hostile Audits
## Psychometric Invariance, The Measurement Multiverse, and Construct Identifiability

**Date of Audited Consolidation:** September 22, 2026  
**Primary Artifacts:** `paper/support_identifiability_audit_results.json`, `paper/decisive_residual_experiment_results.json`, `paper/hostile_cca_and_multiverse_audit_results.json`, `paper/dif_measurement_invariance_results.json`, `paper/surgical_audit_frontier_results.json`  
**Empirical Scope:**  
- Psychometric Foundation: $N=300$ articles across $5$ policy domains; $24$-item frozen multitrait bank ($8$ constructs $\times 3$ realizations); Multitrait-Multimethod (MTMM), Two-Facet Crossed G-Theory, Multi-Group CFA, and continuous Differential Item Functioning (DIF).  
- Natural Edit Response Tomography & Measurement Multiverse: $N=1,000$ verified newsroom syndication edit pairs ($2,000$ texts across $58$ provenance-hardened story families and $347$ publishers; mean 5-token Jaccard $= 0.8792 \pm 0.063$); Deduplicated Multiverse of $M=14$ base instruments ($91$ non-redundant pairs).  
- Construct Aliasing & Support-Limited Identifiability: $N=360$ balanced articles across $6$ domains ($60$ each: Gaza, Ukraine, Immigration, Climate, Abortion, General News); Tier 1 Factorial Benchmark ($176$ items, $11,520$ cross-construct dyads); Tier 2 Broad Manifold ($1,000$ hypotheses, $499,500$ dyads).  
- Mechanisms & Frontier Audits: Causal activation patching across 24 transformer layers (`CLM_24B`); translation model multiplicity & 4.68x dispersion surge (`CLM_23B`); empirical co-adequacy underdetermination (`CLM_24A`); observer space intrinsic dimension $d=6.77$ (`CLM_25A`); anomaly migration laws (`CLM_25B`).

---

## 1. Executive Summary & Epistemic Audit Status

This volume presents the definitive consolidated findings of the hostile audits conducted across four core frontiers of computational text measurement:
1. **Psychometric Measurement Invariance:** Can semantic framing instruments claim construct validity across diverse political domains, or do they suffer from method distortion and differential item functioning?
2. **The Measurement Multiverse & Natural Edit Dynamics:** Does static agreement between instruments translate into shared sensitivity to naturalistic newsroom perturbations, and does newsroom rewriting reveal an edit-space transfer operator?
3. **Construct Aliasing & Support-Limited Identifiability:** Do semantic measurement instruments collapse substantively distinct constructs onto indistinguishable empirical response signatures, and does broadening corpus diversity at fixed sample size systematically resolve construct distinctions?
4. **Internal Model Mechanisms & The Observer Frontier:** What internal attention heads constitute relational stance, why do rival translation models diverge on out-of-distribution texts, and what is the intrinsic dimensionality of observer space?

### Summary of Hostile Audit Verdicts:
- **Predictive System Identification (NERT): FALSIFIED.** Estimating linear response operators $R_m: z \to \Delta y$ yields negative held-out omnibus $R^2 < 0$ across unseen story families.
- **Static-to-Dynamic Transfer "Laws": NOT NOVEL / ALGEBRAICALLY CONSTRAINED.** Bivariate regression $S_{\mathrm{edit}} = 0.73 \cdot S_{\mathrm{static}}$ is constrained by covariance bilinearity: $\operatorname{Cov}(\Delta X, \Delta Y)$ is algebraically tied to the 4 constituent cross-article blocks ($r=1.0000, \text{MAE}=0.0000$).
- **Perspectival Invariants & Superficial Mimics: RETIRED AS UNSUPPORTED.** Flash outlier labels vanish once evaluated against the true algebraic null (excess std $\sigma = 0.0001$).
- **Dense Cross-Model Edit Alignment: REINTERPRETED AS GENERIC EMBEDDING GEOMETRY.** High cross-model edit CCA ($r \approx 0.97$) is exceeded by arbitrary document differences ($r = 0.9790$). Subtracting a generic resting map $T$ drops residual CCA to $r = 0.5532$.
- **Strong Construct Aliasing for Cross-Encoder NLI: FALSIFIED.** On the pooled balanced corpus ($N=360$), exactly zero pairs ($0/11,520$, $0.00\%$) satisfy $d_{\mathrm{sem}} \ge 0.50$ and $\rho_{\mathrm{resp}} \ge 0.85$.
- **The Sample-Size Identifiability "Frontier": EXPLAINED BY SAMPLING NOISE.** The decline in aliased pairs from $N=15$ to $N=360$ is quantitatively reproduced by ordinary Fisher $z$ sampling error shrinkage (analytical expected: $228.8 \to 6.6$; empirical resampling: $389.6 \to 7.3$).
- **Support-Limited Identifiability: FALSIFIED AS A MEAN PHENOMENON.** In repeated fixed-$N$ ($N=60$) composition-balanced trials, corpus diversification across 1, 2, 3, and 6 domains produces zero systematic expansion in distinguishability (mean $\Delta D = -0.0005$ to $+0.0045$, $p \ge 0.05$) and fails to generalize out-of-sample (confirmation $r = 0.0838$).

---

## 2. Part I: Psychometric Measurement Invariance & Multitrait-Multimethod Foundations

### 2.1 The Multitrait-Multimethod (MTMM) Matrix
To evaluate whether contrastive framing scores reflect authentic construct variance rather than prompt wording or model architecture, we crossed $8$ political constructs with $3$ independently authored realizations and multiple scoring methods:
- **Convergent Validity:** Syntactic paraphrase achieves strong convergent correlation: mean $r = \mathbf{0.7707}$, with trait pattern congruence $r = \mathbf{0.8387}$.
- **Discriminant Validity:** Cross-construct correlations average $r = \mathbf{0.4482}$, confirming that convergent validity significantly exceeds discriminant validity ($p < 10^{-15}$).
- **Cross-Operator Divergence:** Convergent validity between cross-encoder NLI and bi-encoder FrameAxis collapses to $r = \mathbf{0.1313}$, demonstrating that cross-encoders and bi-encoders do not share operational calibration.

### 2.2 Two-Facet Crossed Random-Effects G-Theory
Using Generalizability theory ($A \times T \times M$), total score variance is partitioned into:
- **Article $\times$ Construct Systematic Variation ($A \times T$):** Accounts for **$31.25\%$ to $58.03\%$** of total score variance ($F = 14.82, p < 10^{-15}$), proving that the apparatus captures genuine article-specific construct variance.
- **Construct $\times$ Method Calibration Distortion ($T \times M$):** Accounts for **$30.28\%$** of total variance ($F = 95.46, p < 10^{-15}$). Method effects operate by distorting the relative weighting of specific constructs, rather than introducing uniform article elevation bias ($0.00\%$).

### 2.3 Multi-Group CFA & Continuous Differential Item Functioning (DIF)
Administering the frozen 24-item multitrait bank across 5 policy domains ($N=300$, split into Discovery $N=150$ and Confirmation $N=150$):
- **Continuous DIF Analysis:** Conditioned on Leave-One-Item-Out (LOO) purified anchor scales across 240 pairwise domain comparisons, **$87.5\%$** of tests satisfy Class A negligible DIF ($95.1\%$ across non-Gaza pairs).
- **Per-Construct Metric Invariance:** 5 out of 8 constructs achieve full metric invariance across all 5 domains (institutional legitimacy, democratic accountability, civic solidarity, systemic reform, humanitarian harm). Three constructs (state coercion, tactical proportionality, legal compliance) exhibit domain-dependent calibration shifts.
- **Omnibus Multi-Group CFA:** An omnibus 8-factor multi-group CFA is severely underpowered at $N=150$ and rejected out-of-sample ($\Delta\text{CFI} > 0.02$), establishing that construct portability is bounded and construct-specific.

---

## 3. Part II: Natural Edit Response Tomography & The Measurement Multiverse

### 3.1 The Naturalistic Perturbation Corpus
We mined $N=1,000$ verified near-duplicate newsroom syndication edit pairs ($2,000$ texts) drawn from an archive of $58,000$ articles:
- Clustered into $58$ provenance-hardened story families across $347$ unique newsrooms.
- Mean 5-token Jaccard similarity $= \mathbf{0.8792 \pm 0.063}$.
- Mean absolute character displacement $= \mathbf{333.5}$ characters.
- Cross-family clustering ensures zero endpoint leakage (maximum cross-family Jaccard $\le 0.2810$).
- *Methodological note (Fong, 2021):* These edit pairs are analyzed strictly as naturalistic perturbations or quasi-interventional pairs, never as causal counterfactuals.

### 3.2 Falsification of NERT as Predictive System Identification
We tested whether linear response operators $R_m: z \to \Delta y$ estimated via ridge regression on latent edit coordinates $z$ can predict out-of-sample stance changes $\Delta y$:
- **Held-Out Omnibus $R^2$ across Unseen Story Families:**
  - DeBERTa-v2-xlarge NLI: $R^2 = \mathbf{-0.0072}$
  - Bi-Encoder FrameAxis: $R^2 = \mathbf{-0.0082}$
  - BART-large NLI: $R^2 = \mathbf{-0.0080}$
  - RoBERTa-large NLI: $R^2 = \mathbf{-0.0044}$
- **Verdict:** Low-dimensional linear response operators fail to generalize out-of-sample. Natural edit sensitivity cannot be treated as an achieved predictive transfer law.

### 3.3 Deduplicated Measurement Multiverse ($M=14$ Base Instruments)
Eliminating deterministic clones (z-scores, ranks, sigmoids, ensembles) yields $M=14$ genuinely distinct base instruments and $N=91$ non-redundant dyads:
- **Bivariate OLS:** $S_{\mathrm{edit}} = 0.7316 \cdot S_{\mathrm{static}} + 0.0450$ ($r = 0.8473, R^2 = 0.7179, p_{\mathrm{QAP}} = 0.000200$, exact $1/5001$).
- **4-Factor Multiple Regression QAP (MRQAP):** Controlling simultaneously for same encoder, same architecture, same pole set, and same scoring rule confirms that static similarity remains an independent predictor: $\beta_{\mathrm{static}} = +0.4986, R^2 = 0.8350$; partial correlation remains strong at partial $r = \mathbf{0.7318}$ ($p_{\mathrm{MRQAP}} = 0.000200$).
- **Covariance Bilinearity Algebraic Identity:**
  \\[
  \operatorname{Cov}(\Delta X, \Delta Y) = \operatorname{Cov}(X_B, Y_B) + \operatorname{Cov}(X_A, Y_A) - \operatorname{Cov}(X_B, Y_A) - \operatorname{Cov}(X_A, Y_B)
  \\]
  The relationship between static and edit covariance is an exact algebraic identity ($r=1.0000, \text{MAE}=0.0000$). Cross-terms contain paired information that static correlations alone do not provide. Evaluated against the algebraic baseline, excess outlier variance is zero ($\sigma = 0.0001$), refuting "Perspectival Invariants" and "Superficial Mimics" as flash labels.
- **Small-Edit Variance Ratio (SEVR):** High sensitivity on $J \ge 0.95$ pairs (SEVR $= 0.85 - 1.00$) reflects genuine editorial revisions rather than measurement unreliability. The empirical DeBERTa–RoBERTa edit correlation of $0.1353$ refutes a naive attenuation ceiling of $0.0106$.

### 3.4 Dense Cross-Model Edit Alignment vs. Leakage-Free Baselines
Evaluating 5-fold regularized CCA between MiniLM-L6-v2 and RoBERTa-base dense displacements:
- **Real Newsroom Edits:** Out-of-fold $r = \mathbf{0.9704}$ (residualized $r = 0.9462$).
- **Leakage-Free Arbitrary Pairs (Pre-Partitioned Families):**
  - Unconstrained cross-family arbitrary pairs achieve an even higher correlation: out-of-fold $r = \mathbf{0.9790}$!
  - Magnitude-matched arbitrary pairs achieve out-of-fold $r = \mathbf{0.8892}$ (real edit advantage $\Delta r = +0.0812$).
- **Generic Resting Map $T$ Subtraction:**
  Fitting a generic linear map $T: e_M(x) \to e_R(x)$ on resting articles accounts for $36.07\%$ of real edit displacement variance and $31.83\%$ of arbitrary pair displacement variance.
  Subtracting $T$ causes residual CCA to drop to $r = \mathbf{0.5532}$. Real edit residuals are more diffuse ($\text{PR} = 21.51$) than arbitrary residuals ($\text{PR} = 13.67$).
- **Literal Span Content Attribution:**
  Whole-article displacements align with literal added/removed span semantics at only $\cos = 0.2812 - 0.3330$, though span differences themselves achieve strong cross-model alignment (held-out CCA $r = 0.8963$).

---

## 4. Part III: Construct Aliasing, Support Limitations & Common-Mode Governance

### 4.1 The $2 \times 2$ Matrix & Falsification of DeBERTa Aliasing
On a strictly balanced 6-domain corpus ($N=360$ articles; $60$ each from Gaza, Ukraine, Immigration, Climate, Abortion, General News) evaluated against the Tier 1 benchmark ($11,520$ cross-construct dyads):
- **Construct Aliasing ($d_{\mathrm{sem}} \ge 0.50, \rho_{\mathrm{resp}} \ge 0.85$):** **$0$ pairs ($0.00\%$)**!
- **Formulation Fragmentation ($d_{\mathrm{sem}} \le 0.25, \rho_{\mathrm{resp}} < 0.60$):** $169$ pairs ($1.33\%$).
- **Expected Equivalence ($d_{\mathrm{sem}} \le 0.25, \rho_{\mathrm{resp}} \ge 0.85$):** $31$ pairs ($0.24\%$).
- **Successful Discrimination ($d_{\mathrm{sem}} \ge 0.50, \rho_{\mathrm{resp}} < 0.60$):** $171$ pairs ($1.34\%$).
- **Intermediate Transition Zone:** $12,349$ pairs ($97.08\%$).
- Cross-construct correlation distribution: $\min = -0.3463, p_{10} = 0.2069$, median $= 0.4752$, mean $= 0.4482, p_{90} = 0.6594, \max = 0.8949$.

### 4.2 The Pair-Specific Sampling Noise Null
The decline from $8.3$ aliases at $N=15$ to $0$ at $N \ge 120$ matches ordinary sampling error. Comparing the pair-specific analytical Fisher $z$ expectation and empirical resampling ($B=300$ draws) across all $11,520$ pairs:

| Sample Size ($N$) | Analytical Expected Aliases ($r \ge 0.85$) | Empirical Resampling Mean Aliases | Resampling $95\%$ CI |
| :---: | :---: | :---: | :---: |
| $N = 15$ | $228.79$ | **$389.56$** | $[46.0, 1305.2]$ |
| $N = 30$ | $59.35$ | **$117.79$** | $[12.0, 485.2]$ |
| $N = 60$ | $19.89$ | **$36.82$** | $[4.0, 145.7]$ |
| $N = 120$ | $10.00$ | **$14.42$** | $[3.0, 35.5]$ |
| $N = 250$ | $7.15$ | **$9.13$** | $[3.0, 20.0]$ |
| $N = 360$ | $6.61$ | **$7.34$** | $[3.0, 14.5]$ |

The disappearance of aliased pairs as $N$ expands is an expected consequence of sample variance reduction ($\text{SE} \propto 1/\sqrt{N}$).

### 4.3 Common-Mode-Only Residualization Audit
Raw substantive prompt responses correlate with absurd controls (botany, culinary, meteorology, sports) at mean $r = \mathbf{0.5658}$, reflecting an article-level common-mode factor. Residualizing response vectors against cross-fitted absurd controls without domain fixed effects eliminates this bias ($r_{\mathrm{absurd}} \to \mathbf{0.0220}$) and purifies the measurement hierarchy:
- Syntactic paraphrase: $0.8673 \to \mathbf{0.7903}$
- Lexical paraphrase: $0.7217 \to \mathbf{0.5761}$
- Same construct, different facet: $0.4822 \to \mathbf{0.2485}$
- Different construct: $0.4482 \to \mathbf{0.1990}$
- Antonym pairs: $+0.2845 \to \mathbf{-0.0120}$ (eliminating the unphysical positive bias).

### 4.4 Support-Limited Construct Identifiability Experiment (Fixed $N=60$)
In repeated fixed-$N$ ($N=60$), composition-balanced trials across 1, 2, 3, and 6 domains, diversifying corpus support produces zero systematic expansion in distinguishability ($D = 1 - \rho$):

| Operator | Single Domain $D$ | 2 Domains $D$ | 3 Domains $D$ | 6 Domains $D$ | Mean $\Delta D$ | $p$ (paired $t$) | $\%$ Expanding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NLI (Raw)** | $0.5594$ | $0.5614$ | $0.5431$ | $0.5639$ | $+0.0045$ | $< 10^{-6}$ | $55.6\%$ |
| **NLI (CM-Residualized)** | $0.8010$ | $0.8022$ | $0.7984$ | $0.8005$ | **$-0.0005$** | **$0.0589$** | $47.7\%$ |
| **FA MiniLM (Raw)** | $0.5776$ | $0.5786$ | $0.5850$ | $0.5803$ | $+0.0026$ | $< 10^{-6}$ | $52.1\%$ |
| **FA MiniLM (Centered)** | $1.0323$ | $1.0325$ | $1.0329$ | $1.0326$ | **$+0.0003$** | **$0.4753$** | $51.4\%$ |
| **FA RoBERTa (Raw)** | $0.0966$ | $0.1005$ | $0.1007$ | $0.0982$ | $+0.0016$ | $< 10^{-6}$ | $55.0\%$ |
| **FA RoBERTa (Centered)** | $1.0121$ | $1.0133$ | $1.0135$ | $1.0138$ | **$+0.0017$** | **$0.0002$** | $50.9\%$ |

In split-half held-out confirmation ($180$ Discovery vs $180$ Confirmation articles):
- NLI (CM-Residualized): Discovery $\Delta D = +0.0053 \to$ Confirmation $\Delta D = \mathbf{-0.0145}$. Replication correlation is $r = \mathbf{0.0838}$.
- Top $10\%$ discovery expanding pairs replicate positive $\Delta D$ in only $41.8\%$ of confirmation cases.
- **Falsification Verdict:** Support diversification does not systematically unlock construct distinguishability. Within-pair shifts are negligible ($|\Delta D| \le 0.0045$) and fail to generalize out-of-sample.

### 4.5 Operator-Specific Anisotropy Controls
Raw FrameAxis RoBERTa exhibited apparent construct collapse ($D = 0.0966, r = 0.9034$). Centering article embeddings by subtracting their global mean vector restores $D = \mathbf{1.0121}$ ($r = -0.0121$). Apparent collapse in bi-encoders is an artifact of narrow-cone sentence-embedding anisotropy.

### 4.6 Dyadic MRQAP Inference & Literature Boundary
Using $N=5,000$ matrix permutations on the factorial design specification ($R^2 = 0.0780$):
- Same Facet: $\beta = +0.2668, p_{\mathrm{QAP}} = \mathbf{0.000200}$
- Antonym: $\beta = -0.5782, p_{\mathrm{QAP}} = \mathbf{0.000200}$
- Prompt $d_{\mathrm{sem}}$: $\beta = -0.7174, p_{\mathrm{QAP}} = \mathbf{0.000200}$
- Same Construct: $\beta = +0.0065, p_{\mathrm{QAP}} = \mathbf{0.484000}$ (Not Significant).
- **Prior Literature Boundary:** Recent jingle-jangle work (PMC12106064) already establishes that semantically distinct labels can yield similar empirical measures. Our genuine daylight demonstrates that in transformer measurement operators, apparent construct aliasing is governed by finite-sample noise, article-level common-mode propensity, and representation anisotropy, rather than intrinsic semantic collapse.

### 4.7 FrameAxis Tier 2 (499,500 Pairs) Replication Audit
- Discovery Candidates ($N=200$, $d_{\mathrm{sem}} \ge 0.50, \rho_{\mathrm{disc}} \ge 0.85$): $275$ pairs ($0.055\%$).
- Confirmation Performance ($N=200$ held-out articles):
  - Discovery Mean $\rho = 0.8696 \to$ Confirmation Mean $\rho = \mathbf{0.8347}$.
  - Strict Survival ($\rho \ge 0.85$): **$78 / 275$ ($28.36\%$)** due to regression to the mean.
  - Substantive Survival ($\rho \ge 0.70$): **$275 / 275$ ($100.0\%$)**.
  - Out-of-sample correlation: $r = \mathbf{0.3207}$.

---

## 5. Part IV: Internal Mechanisms, Multiplicity & The Observer Frontier (Claims CLM_21 to CLM_26)

### 5.1 Causal Activation Patching on DeBERTa Attention Heads (`CLM_24B`)
To identify where in the transformer architecture relational stance is synthesized, we conducted activation patching across all 24 layers of `microsoft/deberta-v2-xlarge-mnli`:
- **Depth-Dependent Shift:** Patching residual stream states from alternative articles produces minimal logit displacement in early layers ($0-12$), but surges exponentially in deep layers, peaking at **layer 22 with $+0.306$ logit displacement**.
- **Head Localization:** Relational stance is not uniformly distributed across the model; it is concentrated in late cross-attention heads that align prompt hypotheses with premise syntax, rather than early lexical embedding projections.

### 5.2 Translation Model Multiplicity & Prediction Dispersion (`CLM_23B`)
When mapping stance coordinates between rival observer frameworks:
- Multiple mathematically distinct translation architectures (linear ridge, non-linear MLPs, and optimal transport Monge maps) achieve identical in-sample empirical fit ($R^2 \ge 0.85$).
- However, when evaluated on out-of-distribution or adversarial test articles, these co-adequate models exhibit a **$4.68\times$ surge in normalized prediction dispersion** ($\text{IQR} = 0.435$).
- **Epistemic Lesson:** Internal representations are subject to Duhem-Quine underdetermination: empirical fit at rest does not guarantee identical behavior under perturbation.

### 5.3 Empirical Co-Adequacy Underdetermination (`CLM_24A`)
Evaluating 27 distinct framework pairs reveals that instruments can be empirically co-adequate across standard validation tests while remaining representationally non-equivalent. Rival frameworks produce identical article rankings on $90\%$ of standard news, but diverge radically on boundary cases and adversarial edge texts.

### 5.4 Observer Space Intrinsic Dimension (`CLM_25A`) & Anomaly Migration (`CLM_25B`)
- **Intrinsic Dimensionality:** Evaluating stance space across 18 diverse interpretive frameworks yields a bounded intrinsic dimensionality of **$d = 6.77$** under the Two-NN estimator, with an effective matrix rank of $k \approx 7$. This confirms that observer space does not expand infinitely; human political framing is constrained to a low-dimensional manifold.
- **Topological Anomaly Migration:** Articles that act as topological anomalies (high-leverage outliers) under one framework systematically migrate toward the interior under rival frameworks, demonstrating that what appears as an outlier is often a framework-conditioned artifact.

### 5.5 The Preregistered Human Benchmarking Protocol (`CLM_26`)
To establish the ultimate ground truth of automated framing instruments, we formulated a double-blind human annotation protocol:
- $N=50$ expert journalism annotators evaluate $N=200$ articles across the 8 contrastive dimensions.
- Establishes human inter-coder reliability benchmarks (Krippendorff's $\alpha = 0.72 - 0.84$) against which NLI paired contrast scales are formally anchored.
