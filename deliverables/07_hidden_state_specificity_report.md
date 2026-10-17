# Hidden-State Contrast Specificity, Ambient Graph Retention, and Distance Support Audit

**Date:** 2026-10-17 06:15:00 UTC  
**Model:** `microsoft/deberta-v2-xlarge-mnli` (Layer midpoint $L/2=12$, Final layer $L=24$)  
**Dataset:** Full balanced 120-template intervention suite ($N=480$ passages across 8 contrast dimensions)  
**Ambient Reference:** Frozen 512-article panel (`deliverables/06_cross_model_panel/deberta_panel_512.npz`)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti (CUDA fp16, batch size 16)  
**Audit Standard:** Dual ratio accounting, exact neighbor retention ($R_{10}$), matched surface-control contrast, distributional distance support qualification, and KS null rejection reporting.

---

## 1. Executive Summary & Audited Invariants

This audit evaluates the relationship between internal hidden-state representations ($V_7$ Mid-CLS, $V_3$ Final-CLS, $V_5$ Final-Mean), scalar margins ($V_1$), and their ambient nearest-neighbor graphs in real news space.

### Core Verified Findings

1. **Exact Neighbor Retention in Ambient Real News Space ($R_{10}$):**
   - In the **final layer ($V_3$ Final-CLS)**, meaning-preserving rewrites retain an average of **71.33%** (7.13 of 10) of the original passage's nearest news articles ($J_{10} = 0.6121$).
   - Reversing the political proposition collapses neighbor retention to **4.75%** (0.47 of 10 articles; $J_{10} = 0.0312$), representing an average retention drop of **$\Delta R_{10} = +66.58\%$** ($p = 3.75 \times 10^{-47}$).
   - In contrast, in the **middle layer ($V_7$ Mid-CLS)**, neighbor retention is virtually identical across all edits: paraphrase retains **51.67%**, surface control retains **47.67%**, and proposition reversal retains **50.92%** ($\Delta R_{10} = +0.75\%$, $p = 0.360$).

2. **Reconciliation of Ratio Metrics (Dual Ratio Reporting):**
   - The discrepancy between displayed aggregate means and reported ratios is resolved by formally distinguishing the **Ratio of Aggregate Means** ($\bar{\Delta}_{\text{on}} / \bar{\Delta}_{\text{off}}$) from the **Mean of Template-Level Ratios** ($\frac{1}{N}\sum_i \frac{\Delta_{\text{on}, i}}{\Delta_{\text{off}, i}}$).
   - For $V_3$ Final-CLS: Ratio of Aggregate Means = **2.01x**; Mean of Template Ratios = **2.04x**; Median Ratio = **1.91x** (100.0% win rate, $p = 2.92 \times 10^{-46}$).
   - For $V_1$ Scalar Margin: Ratio of Aggregate Means = **2.29x**; Mean of Template Ratios = **2.49x**; Median Ratio = **2.19x** (93.3% win rate, $p = 2.81 \times 10^{-34}$).
   - For $V_7$ Mid-CLS: Ratio of Aggregate Means = **0.92x** (< 1.0); Mean of Template Ratios = **1.14x** (skewed by a small-denominator tail); Median Ratio = **1.03x** (win rate = 51.7%, $p = 0.910$).

3. **Linguistic Control and Surface Edit Sensitivity:**
   - Text audit confirms that `meaning_preserving` paraphrases involve a **larger** syntactic restructuring than surface controls (Levenshtein distance $106.6$ vs $79.5$ chars; word Jaccard $0.179$ vs $0.205$), yet preserve $71.3\%$ of news neighbors.
   - Surface controls cause intermediate displacement ($26.77$) and retain $22.08\%$ of neighbors ($R_{10}$).
   - Reversing the political proposition induces **$1.85\times$ greater displacement** along the target contrast ($49.61$ vs $26.77$, $p = 6.98 \times 10^{-35}$) and collapses retention to $4.75\%$ (retaining significantly fewer neighbors than surface control: $\Delta R_{10} = +17.33\%$, $p = 1.25 \times 10^{-8}$).

4. **Corrected Distance Support and Kolmogorov-Smirnov Interpretation:**
   - In the final layer ($V_3$), original synthetic passages have a median distance to their nearest real news article of **0.9255**, which is comparable in magnitude to the intra-panel median of real news articles (**0.9132**).
   - However, edited passages exhibit noticeable outward distributional displacement: median nearest-neighbor distances increase to **0.9814** (paraphrases), **1.0551** (political reversals), and **1.1098** (surface edits). The medians for political reversals and surface edits exceed the upper quartile of real-to-real distances (**1.0278**).
   - The two-sample Kolmogorov-Smirnov test **rejects** the null hypothesis that synthetic original distances and real-to-real distances are drawn from the same continuous distribution ($\text{KS} = 0.173$, $p = 0.005$). This small $p$-value provides evidence of distributional difference, refuting any assertion of identical manifold membership.

---

## 2. Quantitative Results Tables

### Table 1: Dual Specificity Ratio Accounting & Displacements Across 120 Templates

| Representation View | On-Target Displ (Chan) | Off-Target Displ (Chan) | Ratio of Means (On/Off) | Mean Template Ratio | Median Template Ratio | Paraphrase Displ | Paraphrase Ratio (Means) | Surface Displ | Surface Ratio (Means) | Win Rate (On > Off) | $p$-value (Spec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_1$ Scalar Margin** | 1.5312 | 0.6683 | **2.29x** | 2.49x | 2.19x | 0.2662 | **5.75x** | 0.8252 | **1.86x** | 93.3% | $p = 2.81 \times 10^{-34}$ |
| **$V_7$ Mid-CLS** | 0.4370 | 0.4760 | **0.92x** | 1.14x | 1.03x | 0.4670 | **0.94x** | 0.5545 | **0.79x** | 51.7% | $p = 0.910$ (Failed) |
| **$V_3$ Final-CLS** | 49.6103 | 24.6409 | **2.01x** | 2.04x | 1.91x | 12.6263 | **3.93x** | 26.7681 | **1.85x** | 100.0% | $p = 2.92 \times 10^{-46}$ |
| **$V_5$ Final-Mean** | 34.2025 | 17.2430 | **1.98x** | 2.01x | 1.88x | 10.2053 | **3.35x** | 18.6799 | **1.83x** | 95.8% | $p = 1.55 \times 10^{-34}$ |

---

### Table 2: Ambient Nearest-Neighbor Graph Retention in Real News Space ($N=512$ Panel)

| Representation View | $R_{10}$ Preserved (%) | $R_{10}$ Surface (%) | $R_{10}$ Changed (%) | $\Delta R_{10}$ (Pres $-$ Chan) | $\Delta R_{10}$ (Surf $-$ Chan) | $J_{10}$ Preserved | $J_{10}$ Surface | $J_{10}$ Changed | Global Rank $\rho$ (Pres) | Global Rank $\rho$ (Chan) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_1$ Scalar Margin** | **67.58%** | 20.92% | **5.25%** | **+62.33%** | +15.67% ($p = 3.75 \times 10^{-7}$) | 0.5512 | 0.1520 | 0.0312 | +0.9267 | -0.3096 |
| **$V_7$ Mid-CLS** | **51.67%** | 47.67% | **50.92%** | **+0.75%** | -3.25% ($p = 0.943$) | 0.3702 | 0.3364 | 0.3652 | +0.7711 | +0.7956 |
| **$V_3$ Final-CLS** | **71.33%** | 22.08% | **4.75%** | **+66.58%** | +17.33% ($p = 1.25 \times 10^{-8}$) | 0.6121 | 0.1668 | 0.0312 | +0.9157 | -0.2740 |

*Note on statistics:* $R_{10} = |A \cap B| / 10$ measures true neighbor retention. $J_{10} = |A \cap B| / |A \cup B| = |A \cap B| / (20 - |A \cap B|)$.

---

### Table 3: Text-Level Linguistic and Edit Distance Audit Across 120 Templates

| Edit Condition | Word Count Diff ($|\Delta W|$) | Character Length Diff ($|\Delta C|$) | Word Jaccard Similarity | Levenshtein Distance (chars) | Normalized Levenshtein Sim |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Meaning-Preserving Paraphrase** | 1.60 $\pm$ 1.43 | 7.65 $\pm$ 6.68 | 0.1791 $\pm$ 0.086 | 106.6 $\pm$ 12.3 | 0.2725 $\pm$ 0.046 |
| **Surface-Control Edit** | 1.04 $\pm$ 1.02 | 6.94 $\pm$ 5.24 | 0.2046 $\pm$ 0.079 | 79.5 $\pm$ 15.6 | 0.4571 $\pm$ 0.084 |
| **Proposition-Changing Reversal** | 1.23 $\pm$ 1.31 | 7.87 $\pm$ 5.79 | 0.2456 $\pm$ 0.108 | 75.8 $\pm$ 17.0 | 0.4848 $\pm$ 0.101 |

---

### Table 4: Synthetic-to-Real Distance Support Against Panel 512

| Representation View | Population / Version | 1st NN Mean | 1st NN Median [IQR] | 10th NN Mean | 10th NN Median [IQR] | Mean Distance to All 512 Articles |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$V_3$ Final-CLS** | **Real News Articles (Panel)** | **0.9001** | **0.9132** [0.7951, 1.0278] | **1.0203** | **1.0125** [0.9206, 1.1208] | **1.2410** |
| | Synthetic Original | 0.9444 | 0.9255 [0.8241, 1.0540] | 1.0475 | 1.0137 [0.9320, 1.1410] | 1.2912 |
| | Meaning-Preserving | 0.9697 | 0.9814 [0.8650, 1.0820] | 1.0662 | 1.0575 [0.9540, 1.1680] | 1.2932 |
| | Surface Control | 1.1034 | 1.1098 [1.0110, 1.2150] | 1.1969 | 1.2063 [1.1120, 1.2950] | 1.3874 |
| | Proposition-Changing | 1.0205 | 1.0551 [0.9320, 1.1340] | 1.1110 | 1.1422 [1.0250, 1.2210] | 1.3379 |
| **$V_7$ Mid-CLS** | **Real News Articles (Panel)** | **0.9676** | **0.9955** [0.8396, 1.1112] | **1.0779** | **1.0982** [0.9772, 1.1727] | **1.2532** |
| | Synthetic Original | 1.1103 | 1.1089 [1.0210, 1.2040] | 1.3581 | 1.3595 [1.3210, 1.3980] | 1.3865 |
| | Meaning-Preserving | 1.0969 | 1.0860 [1.0110, 1.1890] | 1.3574 | 1.3566 [1.3190, 1.3950] | 1.3864 |

*Diagnostic Test:* Two-sample Kolmogorov-Smirnov test between real-to-real 1st NN distances and synthetic original 1st NN distances yields $\text{KS} = 0.2355, p = 3.16 \times 10^{-5}$ ($10^{\text{th}}$ NN: $\text{KS} = 0.1732, p = 0.00505$). This decisively rejects the null hypothesis of distributional identity ($p < 0.01$).

---

## 3. Methodological Significance and Scientific Nuance

1. **Resolution of Surface Edit Disruption:**
   The linguistic audit demonstrates why surface controls displace by $26.77$ and retain only $22.08\%$ of neighbors: the benchmark's surface edits add contextual clauses (e.g., *"during rainy weather"*), shifting narrative context rather than purely testing spelling or syntax. Consequently, the experiment does not isolate ideological content from all other semantic shifts. However, political proposition reversals displace **$1.85\times$ farther** ($49.61$ vs $26.77$, $p = 6.98 \times 10^{-35}$) and collapse neighbor retention to only $4.75\%$ (significantly lower than surface control, $p = 1.25 \times 10^{-8}$). DeBERTa responds to contextual edits, but proposition inversions produce significantly larger displacement and near-total neighborhood reorganization.

2. **Nuanced Interpretation of Middle-Layer Performance:**
   In layer 12, synthetic passages are substantially more distant from the 512 real news articles (average 10th NN distance $1.358$ vs. real news $1.078$), indicating a poor match with the real-news reference population under this readout. However, this distance does not constitute topological proof of a detached cluster, nor does it fully explain the lack of proposition specificity: $V_7$ also failed the independent within-intervention on-target versus off-target displacement test ($\bar{\Delta}_{\text{on}} / \bar{\Delta}_{\text{off}} = 0.92\times$; win rate $51.7\%$, $p = 0.910$), which does not involve the 512 news panel at all. This confirms that **this particular layer-12 CLS contrast representation fails to discriminate the targeted political propositions**, but does not justify an overbroad conclusion about all middle layers or alternative readouts.

3. **Empirical Dissociation of Measurement Properties (Not a Demonstrated Trade-Off):**
   The findings establish an empirical dissociation among three distinct measurement demands:
   - *Stability:* Saying the same thing differently should not radically change the measurement (e.g., $Q_0 \to Q_1$ hypothesis prompt invariance).
   - *Specificity:* Changing the political proposition should affect the intended framing dimension more than unrelated dimensions.
   - *Relational discrimination:* Meaningful changes should reorganize measured article relationships in a predictable way, rather than merely introducing arbitrary numerical instability.
   These properties are not guaranteed to coincide. In this comparison, the layer-12 CLS representation exhibited high hypothesis-prompt stability ($Q_0 \to Q_1$ CKA $\approx 0.913$), but the experiment **did not detect meaningful neighborhood discrimination** ($\Delta R_{10} = +0.75$ percentage points, $p = 0.360$; failure to detect sensitivity under this test does not prove sensitivity is exactly zero, as an equivalence test would be required to claim a point null). Conversely, the final-layer CLS representation exhibited lower invariance across hypothesis paraphrases ($Q_0 \to Q_1$ CKA $\approx 0.777$), yet demonstrated pronounced proposition specificity (ratio of means $= 2.01\times$) and substantial relational discrimination ($\Delta R_{10} = +66.58$ percentage points).
   Crucially, this dissociation does not establish an inherent trade-off (i.e., that improving stability necessarily degrades sensitivity), particularly because hypothesis paraphrasing and passage rewrites test different components of the apparatus. Rather, it demonstrates that **maximizing representational invariance alone is an insufficient objective for instrument construction**; candidate instruments must be evaluated against stability, specificity, and relational discrimination independently.

4. **Experimental Limitations and Generalization Boundaries:**
   The 120 templates share a structured experimental design (15 templates per contrast dimension). Small $t$-test $p$-values reflect high within-design power, but do not establish broad out-of-distribution generalization to unconstrained natural news discourse without accounting for dependencies across templates and contrast dimensions. The defensible conclusion is bounded: on these controlled passages, DeBERTa's final-layer contrast representation is substantially more sensitive to targeted political-proposition reversals than its layer-12 counterpart, and that sensitivity reorganizes proximity to real news articles.
