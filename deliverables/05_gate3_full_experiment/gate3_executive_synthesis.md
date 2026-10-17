# Gate 3 Full Experiment: Multi-View Relational Geometry and Invariance Synthesis

**Experiment Execution Date**: 2026-10-16  
**Execution Runtime**: 125.70 seconds (PyTorch CUDA, NVIDIA GeForce RTX 5070 Ti)  
**Primary Unit of Analysis**: $N = 6,248$ Independent Story Family Representatives  
**Sensitivity Control Unit**: $N = 7,986$ Articles (Full Selector Union)  
**Permutation Iterations**: 500  
**Cluster Bootstrap Iterations**: $B = 200$ (resampling independent story families)  

---

## 1. Audit Manifest and Artifact Deliverables

All data structures, Gram matrices, permutation distributions, and bootstrap resamples are serialized under `deliverables/05_gate3_full_experiment/`:

| Artifact Name | Format | Primary Population | Content |
| :--- | :---: | :---: | :--- |
| `independent_representatives_6248.json` | JSON | 6,248 families | Canonical ID, family ID, full index, and union family sizes |
| `cka_matrix_6248.json` | JSON | 6,248 families | 8×8 RBF/Linear CKA matrices and 500-permutation null controls |
| `knn_overlap_6248.json` | JSON | 6,248 families | 8×8 exact Euclidean $k$-NN Jaccard overlap matrices ($k=10, 20$) |
| `mmd_eligible_conditions.json` | JSON | Disjoint subsets | Quadratic-time unbiased $\text{MMD}^2$ and 500-permutation tests across eligible corpus selectors |
| `sensitivity_analysis.json` | JSON | 6,248 vs 7,986 | Direct comparison of CKA and $k$-NN metrics with and without syndicated duplicates |
| `gate3_full_report.json` | JSON | Master Summary | Master execution receipt, timestamps, runtime, and parameter tables |

---

## 2. Multi-View Representation Invariance (CKA)

Centered Kernel Alignment (CKA) measures the geometric similarity of the data manifolds induced by different representation views across the 6,248 independent story families. All empirical values are evaluated against a 500-iteration permutation null distribution.

### 2.1 Full 8×8 RBF CKA Similarity Matrix ($N = 6,248$)

| View ID | Feature Description | $V_1$ | $V_2$ | $V_3$ | $V_4$ | $V_5$ | $V_6$ | $V_7$ | $V_8$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_1$** | Scalar $Q_0$ (8D margin) | 1.0000 | 0.7490 | 0.5437 | 0.4875 | 0.5034 | 0.4298 | 0.1196 | 0.1155 |
| **$V_2$** | Scalar $Q_1$ (8D margin) | 0.7490 | 1.0000 | 0.4773 | 0.5466 | 0.4412 | 0.4944 | 0.1087 | 0.1091 |
| **$V_3$** | Layer 24 CLS $Q_0$ ($8\times1536$) | 0.5437 | 0.4773 | 1.0000 | 0.7770 | 0.7924 | 0.6584 | 0.2216 | 0.2149 |
| **$V_4$** | Layer 24 CLS $Q_1$ ($8\times1536$) | 0.4875 | 0.5466 | 0.7770 | 1.0000 | 0.6783 | 0.8183 | 0.2102 | 0.2109 |
| **$V_5$** | Layer 24 Mean $Q_0$ ($8\times1536$) | 0.5034 | 0.4412 | 0.7924 | 0.6783 | 1.0000 | 0.7030 | 0.1713 | 0.1669 |
| **$V_6$** | Layer 24 Mean $Q_1$ ($8\times1536$) | 0.4298 | 0.4944 | 0.6584 | 0.8183 | 0.7030 | 1.0000 | 0.1499 | 0.1508 |
| **$V_7$** | Layer 12 CLS $Q_0$ ($8\times1536$) | 0.1196 | 0.1087 | 0.2216 | 0.2102 | 0.1713 | 0.1499 | 1.0000 | 0.9131 |
| **$V_8$** | Layer 12 CLS $Q_1$ ($8\times1536$) | 0.1155 | 0.1091 | 0.2149 | 0.2109 | 0.1669 | 0.1508 | 0.9131 | 1.0000 |

### 2.2 Invariance Across Contrast Prompt Banks ($Q_0$ vs $Q_1$)

The empirical alignment between prompt banks $Q_0$ (direct probes) and $Q_1$ (paraphrased probes) varies monotonically with network depth:

1. **Middle Representations (Layer 12 CLS)**:
   - $\text{CKA}_{\text{RBF}}(V_7, V_8) = 0.9131$ (Linear: $0.9073$)
   - 500-permutation null: mean $= 0.0000$, $q_{99} = 0.0221$, empirical $p < 0.0020$ ($0/500$ exceedances; resolution bound $\frac{1}{B+1} = \frac{1}{501} \approx 0.0020$).
   - 95% Clustered Bootstrap CI: $[0.9119, 0.9176]$.
2. **Final Hidden Representations (Layer 24 CLS)**:
   - $\text{CKA}_{\text{RBF}}(V_3, V_4) = 0.7770$ (Linear: $0.7740$)
   - 500-permutation null: mean $= 0.0000$, $q_{99} = 0.0128$, empirical $p < 0.0020$ ($0/500$ exceedances).
   - 95% Clustered Bootstrap CI: $[0.7720, 0.7857]$.
3. **Paired Scalar Margins (Classifier Head)**:
   - $\text{CKA}_{\text{RBF}}(V_1, V_2) = 0.7490$ (Linear: $0.8015$)
   - 500-permutation null: mean $= 0.0000$, $q_{99} = 0.0021$, empirical $p < 0.0020$ ($0/500$ exceedances).
   - 95% Clustered Bootstrap CI: $[0.7401, 0.7594]$.

### 2.3 Divergence Across Extraction Depths and Bootstrap Finite-Sample Bias

Cross-depth comparisons show substantial divergence between intermediate representations and downstream outputs:
- **Scalar vs. Final Layer**: $\text{CKA}_{\text{RBF}}(V_1, V_3) = 0.5437$ (95% CI: $[0.5328, 0.5507]$).
- **Final Layer vs. Middle Layer**: $\widehat{\text{CKA}}_{\text{RBF}}(V_3, V_7) = 0.2216$.
  - *Naive Percentile Bootstrap CI*: $[0.2272, 0.2419]$ (Bootstrap mean: $0.2342$, $\text{SE} \approx 0.0035$).
  - *Methodological Note on Bootstrap Shift*: Resampling with replacement on $N=6,248$ articles draws duplicate rows with probability $\approx 36.8\%$. Co-occurring identical pairs in both Gram matrices artificially introduce a small positive finite-sample covariance ($+\sim 0.012$). When true representational alignment is high ($>0.75$), this effect is imperceptible. When true alignment is low ($0.2216$), this $+0.0126$ upward shift exceeds $3.5\times \text{SE}$, shifting the unadjusted percentile interval slightly above the empirical point estimate.
  - *Bias-Corrected Bootstrap CI*: $[0.2160, 0.2292]$ (cleanly bracketing the empirical estimate $\widehat{\text{CKA}} = 0.2216$).
- **Scalar vs. Middle Layer**: $\text{CKA}_{\text{RBF}}(V_1, V_7) = 0.1196$.

---

## 3. Nearest-Neighbor Graph Overlap ($k=10, 20$)

Exact Euclidean distance matrices were computed for all 6,248 independent story families. For each article $i$, the top-$k$ nearest neighbors $\text{NN}_k(i)$ (excluding self) were extracted. Graph overlap is reported as the article-averaged Jaccard index:
$$\bar{J}_k = \frac{1}{N} \sum_{i=1}^N \frac{|\text{NN}_k^u(i) \cap \text{NN}_k^v(i)|}{|\text{NN}_k^u(i) \cup \text{NN}_k^v(i)|}$$

### 3.1 Random Baseline Expectations
For $N = 6,248$:
- $\mathbb{E}[J_{10}] = \frac{10}{2(6247) - 10} \approx 0.000801$
- $\mathbb{E}[J_{20}] = \frac{20}{2(6247) - 20} \approx 0.001603$

### 3.2 Full 8×8 $k$-NN Jaccard Matrix ($k=10$, upper triangle)

| View | $V_1$ | $V_2$ | $V_3$ | $V_4$ | $V_5$ | $V_6$ | $V_7$ | $V_8$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_1$** | 1.0000 | 0.0326 | 0.0332 | 0.0263 | 0.0282 | 0.0232 | 0.0129 | 0.0133 |
| **$V_2$** | 0.0326 | 1.0000 | 0.0262 | 0.0367 | 0.0230 | 0.0311 | 0.0120 | 0.0129 |
| **$V_3$** | 0.0332 | 0.0262 | 1.0000 | 0.1462 | 0.1629 | 0.0880 | 0.0442 | 0.0446 |
| **$V_4$** | 0.0263 | 0.0367 | 0.1462 | 1.0000 | 0.0904 | 0.1776 | 0.0412 | 0.0446 |
| **$V_5$** | 0.0282 | 0.0230 | 0.1629 | 0.0904 | 1.0000 | 0.1027 | 0.0377 | 0.0385 |
| **$V_6$** | 0.0232 | 0.0311 | 0.0880 | 0.1776 | 0.1027 | 1.0000 | 0.0342 | 0.0372 |
| **$V_7$** | 0.0129 | 0.0120 | 0.0442 | 0.0412 | 0.0377 | 0.0342 | 1.0000 | 0.3297 |
| **$V_8$** | 0.0133 | 0.0129 | 0.0446 | 0.0446 | 0.0385 | 0.0372 | 0.3297 | 1.0000 |

### 3.3 Prompt Invariance in Local Neighborhoods
- **Middle Layer ($V_7$ vs $V_8$)**: $\bar{J}_{10} = 0.3297$ ($411.6\times$ random baseline); $\bar{J}_{20} = 0.3642$ ($227.2\times$ random baseline).
- **Final Layer ($V_3$ vs $V_4$)**: $\bar{J}_{10} = 0.1462$ ($182.5\times$ random baseline); $\bar{J}_{20} = 0.1740$ ($108.5\times$ random baseline).
- **Scalar Margins ($V_1$ vs $V_2$)**: $\bar{J}_{10} = 0.0326$ ($40.7\times$ random baseline); $\bar{J}_{20} = 0.0415$ ($25.9\times$ random baseline).

---

## 4. Disjoint-Population Maximum Mean Discrepancy (MMD)

To test whether corpus-selection conditions induce divergent representation distributions, disjoint subpopulations ($S_1 = c_1 \setminus c_2$, $S_2 = c_2 \setminus c_1$) were compared using the unbiased sample $\text{MMD}^2$ under RBF kernels with 500-permutation null tests.

> [!NOTE]
> **Inferential Scope**: MMD evaluates whether two sample populations follow distinct numerical probability distributions in representation space. While this confirms selection-conditioned distributional divergence, it does not isolate ideological framing per se; differences may reflect topical sub-focus, publication period, journalistic register, or publisher composition.

### 4.1 Identified Eligible Conditions ($\ge 50$ Families)
*Correction Note on Threshold Notation*: The filenames `tau010`, `tau025`, and `tau050` correspond to frozen quantile thresholds $\mathbf{\tau = 0.010}$ (top 1.0%), $\mathbf{\tau = 0.025}$ (top 2.5%), and $\mathbf{\tau = 0.050}$ (top 5.0%), respectively. The label `tau=0.50` in initial draft tables was a decimal-point formatting error.

Exactly 11 selection conditions contain $\ge 50$ independent story families:
- `corpus_bart_tau050` ($\tau = 0.050$): 2,557 families ($2,938$ raw articles)
- `corpus_bge_tau050` ($\tau = 0.050$): 2,589 families ($3,012$ raw articles)
- `corpus_topic_tau050` ($\tau = 0.050$): 1,762 families ($2,229$ raw articles)
- `corpus_bart_tau025` ($\tau = 0.025$): 1,261 families ($1,469$ raw articles)
- `corpus_bge_tau025` ($\tau = 0.025$): 1,271 families ($1,506$ raw articles)
- `corpus_topic_tau025` ($\tau = 0.025$): 1,177 families ($1,496$ raw articles)
- `corpus_consensus_tau050` ($\tau = 0.050$): 628 families ($719$ raw articles)
- `corpus_bge_tau010` ($\tau = 0.010$): 522 families ($603$ raw articles)
- `corpus_bart_tau010` ($\tau = 0.010$): 505 families ($588$ raw articles)
- `corpus_topic_tau010` ($\tau = 0.010$): 467 families ($599$ raw articles)
- `corpus_consensus_tau025` ($\tau = 0.025$): 224 families ($256$ raw articles)

### 4.2 Disjoint Comparison Results

#### Comparison 1: Model Selection vs. Topic Filter (`corpus_bart_tau050` vs `corpus_topic_tau050`, $\tau = 0.050$)
- Disjoint Sample Sizes: $S_1 (\text{BART} \setminus \text{Topic}) = 2,361$; $S_2 (\text{Topic} \setminus \text{BART}) = 1,566$.
- $V_1$ (Scalar $Q_0$): $\text{MMD}^2 = 0.008344$ ($\text{dist} = 0.0913$, null $q_{95} = 0.000457$, $p < 0.0020$).
- $V_2$ (Scalar $Q_1$): $\text{MMD}^2 = 0.008751$ ($\text{dist} = 0.0935$, null $q_{95} = 0.000469$, $p < 0.0020$).
- $V_3$ (Final CLS $Q_0$): $\text{MMD}^2 = 0.006583$ ($\text{dist} = 0.0811$, null $q_{95} = 0.000131$, $p < 0.0020$).
- $V_4$ (Final CLS $Q_1$): $\text{MMD}^2 = 0.004888$ ($\text{dist} = 0.0699$, null $q_{95} = 0.000134$, $p < 0.0020$).
- $V_7$ (Mid CLS $Q_0$): $\text{MMD}^2 = 0.007042$ ($\text{dist} = 0.0839$, null $q_{95} = 0.000098$, $p < 0.0020$).

#### Comparison 2: Cross-Model Retrieval (`corpus_bart_tau050` vs `corpus_bge_tau050`, $\tau = 0.050$)
- Disjoint Sample Sizes: $S_1 (\text{BART} \setminus \text{BGE}) = 2,213$; $S_2 (\text{BGE} \setminus \text{BART}) = 2,245$.
- $V_1$ (Scalar $Q_0$): $\text{MMD}^2 = 0.024354$ ($\text{dist} = 0.1561$, null $q_{95} = 0.000427$, $p < 0.0020$).
- $V_2$ (Scalar $Q_1$): $\text{MMD}^2 = 0.018216$ ($\text{dist} = 0.1350$, null $q_{95} = 0.000299$, $p < 0.0020$).
- $V_3$ (Final CLS $Q_0$): $\text{MMD}^2 = 0.006240$ ($\text{dist} = 0.0790$, null $q_{95} = 0.000099$, $p < 0.0020$).
- $V_4$ (Final CLS $Q_1$): $\text{MMD}^2 = 0.005809$ ($\text{dist} = 0.0762$, null $q_{95} = 0.000109$, $p < 0.0020$).
- $V_7$ (Mid CLS $Q_0$): $\text{MMD}^2 = 0.005076$ ($\text{dist} = 0.0712$, null $q_{95} = 0.000076$, $p < 0.0020$).

#### Comparison 3: Consensus Selector vs. Single Selector (`corpus_consensus_tau050` vs `corpus_bart_tau050`, $\tau = 0.050$)
- Disjoint Sample Sizes: $S_1 (\text{Consensus} \setminus \text{BART}) = 120$; $S_2 (\text{BART} \setminus \text{Consensus}) = 2,049$.
- $V_1$ (Scalar $Q_0$): $\text{MMD}^2 = 0.081380$ ($\text{dist} = 0.2853$, null $q_{95} = 0.003633$, $p < 0.0020$).
- $V_2$ (Scalar $Q_1$): $\text{MMD}^2 = 0.087420$ ($\text{dist} = 0.2957$, null $q_{95} = 0.003409$, $p < 0.0020$).
- $V_3$ (Final CLS $Q_0$): $\text{MMD}^2 = 0.038594$ ($\text{dist} = 0.1965$, null $q_{95} = 0.000952$, $p < 0.0020$).
- $V_4$ (Final CLS $Q_1$): $\text{MMD}^2 = 0.040385$ ($\text{dist} = 0.2010$, null $q_{95} = 0.001099$, $p < 0.0020$).
- $V_7$ (Mid CLS $Q_0$): $\text{MMD}^2 = 0.014539$ ($\text{dist} = 0.1206$, null $q_{95} = 0.000747$, $p < 0.0020$).

---

## 5. Syndication Sensitivity Analysis ($N=6,248$ vs $N=7,986$)

The 7,986-article union contains 1,738 syndicated or near-duplicate copies belonging to existing story families. Comparing relational metrics between independent family representatives ($N=6,248$) and the full union ($N=7,986$) isolates the impact of duplicate articles:

| Comparison Metric | View Pair | Independent ($N=6,248$) | Full Union ($N=7,986$) | $\Delta$ |
| :--- | :--- | :---: | :---: | :---: |
| **CKA (RBF)** | $V_1$ vs $V_2$ (Scalar) | 0.7490 | 0.7410 | -0.0080 |
| **CKA (RBF)** | $V_3$ vs $V_4$ (Final CLS) | 0.7770 | 0.8178 | +0.0408 |
| **CKA (RBF)** | $V_7$ vs $V_8$ (Mid CLS) | 0.9131 | 0.9281 | +0.0150 |
| **CKA (RBF)** | $V_1$ vs $V_3$ (Scalar vs Final) | 0.5437 | 0.5426 | -0.0011 |
| **$k$-NN Jaccard-10** | $V_1$ vs $V_2$ (Scalar) | 0.0326 | 0.1308 | **+0.0982 ($4.0\times$ inflation)** |
| **$k$-NN Jaccard-10** | $V_3$ vs $V_4$ (Final CLS) | 0.1462 | 0.2456 | **+0.0994 ($1.7\times$ inflation)** |
| **$k$-NN Jaccard-10** | $V_7$ vs $V_8$ (Mid CLS) | 0.3297 | 0.4056 | **+0.0759 ($1.2\times$ inflation)** |

### Analytical Conclusion on Syndication
1. **Global Manifold Alignment (CKA)** is resilient to syndication duplication, shifting by $\le 0.04$ across all layers.
2. **Local Graph Topology ($k$-NN Overlap)** is distorted by syndication duplicates. Because identical or near-identical syndicated articles have near-zero distance to each other in all views, they form trivial mutual nearest neighbors that artificially inflate Jaccard graph overlap by up to $4.0\times$. 
3. **Methodological Takeaway**: Conducting graph analysis on the **6,248 independent story family units** is essential to avoid reporting inflated topological agreement driven by wire reproduction.
