# Gate 3 Protocol and Relational Geometry Definitions

## 1. Specification of the Eight Representation Views ($V_1..V_8$)

Each article $i \in \{1, \dots, N\}$ is evaluated across eight distinct representation views derived from the authenticated DeBERTa-v2-xlarge-mnli forward extraction.

| View ID | Feature Type | Prompt Bank | Extraction Depth / Layer | Output Geometry | Dimension per Article |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_1$** | Paired NLI scalar margin | $Q_0$ (Direct probe) | Classifier Head | Continuous margins $\Delta s_k = s_{A,k} - s_{B,k}$ | $\mathbb{R}^8$ |
| **$V_2$** | Paired NLI scalar margin | $Q_1$ (Paraphrased probe) | Classifier Head | Continuous margins $\Delta s_k = s_{A,k} - s_{B,k}$ | $\mathbb{R}^8$ |
| **$V_3$** | Hidden-state displacement | $Q_0$ | Layer 24 (Final) | Paragraph-weighted CLS diff: $\mathbf{h}_{A,k}^{\text{cls}} - \mathbf{h}_{B,k}^{\text{cls}}$ | $\mathbb{R}^{8 \times 1536}$ |
| **$V_4$** | Hidden-state displacement | $Q_1$ | Layer 24 (Final) | Paragraph-weighted CLS diff: $\mathbf{h}_{A,k}^{\text{cls}} - \mathbf{h}_{B,k}^{\text{cls}}$ | $\mathbb{R}^{8 \times 1536}$ |
| **$V_5$** | Hidden-state displacement | $Q_0$ | Layer 24 (Final) | Paragraph-weighted Masked Mean diff | $\mathbb{R}^{8 \times 1536}$ |
| **$V_6$** | Hidden-state displacement | $Q_1$ | Layer 24 (Final) | Paragraph-weighted Masked Mean diff | $\mathbb{R}^{8 \times 1536}$ |
| **$V_7$** | Hidden-state displacement | $Q_0$ | Layer 12 (Middle) | Paragraph-weighted CLS diff: $\mathbf{h}_{A,k}^{\text{mid}} - \mathbf{h}_{B,k}^{\text{mid}}$ | $\mathbb{R}^{8 \times 1536}$ |
| **$V_8$** | Hidden-state displacement | $Q_1$ | Layer 12 (Middle) | Paragraph-weighted CLS diff: $\mathbf{h}_{A,k}^{\text{mid}} - \mathbf{h}_{B,k}^{\text{mid}}$ | $\mathbb{R}^{8 \times 1536}$ |

---

## 2. Article Distance Metrics and Normalization Rules

### 2.1 Primary Rule: Contrast-Normalized Unit Concatenation
For a tensor view $V \in \{V_3, \dots, V_8\}$, article $i$ is represented by 8 contrast displacement vectors $\mathbf{v}_{i, k} \in \mathbb{R}^{1536}$ for $k \in \{1, \dots, 8\}$.
1. Each contrast vector is independently $L_2$-normalized to remove arbitrary norm differences across prompt pairs:
   $$\hat{\mathbf{v}}_{i, k} = \frac{\mathbf{v}_{i, k}}{\|\mathbf{v}_{i, k}\|_2 + \epsilon}$$
2. The normalized contrast vectors are concatenated into a $12,288$-dimensional vector scaled by $\frac{1}{\sqrt{8}}$:
   $$\mathbf{z}_i = \frac{1}{\sqrt{8}} \left[ \hat{\mathbf{v}}_{i, 1}^\top, \, \hat{\mathbf{v}}_{i, 2}^\top, \, \dots, \, \hat{\mathbf{v}}_{i, 8}^\top \right]^\top \in \mathbb{R}^{12288}$$
   This guarantees that every article lies on the unit hypersphere:
   $$\|\mathbf{z}_i\|_2^2 = \frac{1}{8} \sum_{k=1}^8 \|\hat{\mathbf{v}}_{i, k}\|_2^2 = 1.0$$
3. The pairwise Euclidean distance between articles $i$ and $j$ under the Primary Rule is mathematically equivalent to the root mean squared cosine distance across all 8 contrast pairs:
   $$d(i, j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2 = \sqrt{2 - 2 \bar{\cos}(i, j)}, \quad \text{where} \quad \bar{\cos}(i, j) = \frac{1}{8} \sum_{k=1}^8 \cos(\mathbf{v}_{i, k}, \mathbf{v}_{j, k})$$

### 2.2 Scalar Views Normalization ($V_1, V_2$)
For scalar margin vectors $\mathbf{s}_i \in [-2, 2]^8$:
$$\mathbf{z}_i = \frac{1}{\sqrt{8}} \mathbf{s}_i \in \mathbb{R}^8, \quad d(i, j) = \frac{1}{\sqrt{8}} \|\mathbf{s}_i - \mathbf{s}_j\|_2$$
Continuous scalar magnitudes are preserved without collapsing margins to signs.

### 2.3 Sensitivity Alternatives
- **Alternative 1 (Global $L_2$-Normalized Concatenation)**: Concatenate raw vectors into $\mathbf{u}_i \in \mathbb{R}^{12288}$, then normalize globally: $\tilde{\mathbf{u}}_i = \mathbf{u}_i / \|\mathbf{u}_i\|_2$.
- **Alternative 2 (Contrast-Averaged Euclidean Distance)**: $\bar{d}(i, j) = \frac{1}{8} \sum_{k=1}^8 \|\mathbf{v}_{i, k} - \mathbf{v}_{j, k}\|_2$.

---

## 3. Kernel Functions and Centered Kernel Alignment (CKA)

### 3.1 Kernel Formulations
1. **Linear Kernel**:
   $$K_{\text{lin}}(i, j) = \langle \mathbf{z}_i, \mathbf{z}_j \rangle$$
2. **Radial Basis Function (RBF) Kernel**:
   $$K_{\text{rbf}}(i, j) = \exp\left( - \frac{\|\mathbf{z}_i - \mathbf{z}_j\|_2^2}{2 \sigma_{\text{med}}^2} \right)$$
   where the bandwidth $\sigma_{\text{med}} = \text{median}(\{d(i, j) : i < j\})$ is determined dynamically from the empirical pairwise distance distribution of the sample.

### 3.2 Centered Kernel Alignment (CKA)
Let $H = I_N - \frac{1}{N} \mathbf{1}\mathbf{1}^\top$ be the centering matrix.
Centered Gram matrices: $\tilde{K} = H K H$, $\tilde{L} = H L H$.
$$\text{CKA}(K, L) = \frac{\langle \tilde{K}, \tilde{L} \rangle_F}{\|\tilde{K}\|_F \|\tilde{L}\|_F} = \frac{\text{tr}(\tilde{K} \tilde{L})}{\sqrt{\text{tr}(\tilde{K}^2) \text{tr}(\tilde{L}^2)}} \in [0, 1]$$

### 3.3 Permutation Null Control
To prevent treating numerical artifacts as structural alignment, an empirical null distribution is generated for every view pair $(u, v)$ by permuting article identities 500 times:
$$\tilde{L}_\pi = H P_\pi L P_\pi^\top H, \quad \text{CKA}_{\text{null}}^{(b)} = \text{CKA}(\tilde{K}, \tilde{L}_{\pi_b})$$
Empirical $p$-value:
$$p = \frac{1 + \sum_{b=1}^{500} \mathbb{I}(\text{CKA}_{\text{null}}^{(b)} \ge \text{CKA}_{\text{emp}})}{501}$$

---

## 4. Nearest-Neighbor Graph Construction and Overlap

1. For each view, compute the full distance matrix $D \in \mathbb{R}^{N \times N}$ and set $D_{ii} = \infty$.
2. For each article $i$, define its $k$-nearest neighbors: $\text{NN}_k^v(i) = \{j_1, \dots, j_k\}$.
3. The graph agreement between views $u$ and $v$ is measured via the article-averaged Jaccard index:
   $$J_k(u, v) = \frac{1}{N} \sum_{i=1}^N \frac{|\text{NN}_k^u(i) \cap \text{NN}_k^v(i)|}{|\text{NN}_k^u(i) \cup \text{NN}_k^v(i)|}$$
4. **Random Baseline**: For $N$ independent nodes and random $k$-NN assignment, the expected Jaccard index is:
   $$\mathbb{E}[J_k] = \frac{k}{2(N - 1) - k}$$
   For $N=512, k=10$: $\mathbb{E}[J_{10}] \approx 0.00988$. For $k=20$: $\mathbb{E}[J_{20}] \approx 0.01996$.

---

## 5. Disjoint-Population Maximum Mean Discrepancy (MMD)

To evaluate whether corpus-selection procedures alter representation distributions:
1. For two selection conditions $c_1, c_2$, define strictly disjoint subpopulations:
   $$S_1 = c_1 \setminus c_2 \quad (|S_1| = m), \qquad S_2 = c_2 \setminus c_1 \quad (|S_2| = n)$$
   Shared articles $c_1 \cap c_2$ are explicitly removed from the test to guarantee independent samples.
2. The unbiased quadratic-time sample $\text{MMD}_u^2$ under kernel $K$ is:
   $$\text{MMD}_u^2(S_1, S_2) = \frac{1}{m(m-1)} \sum_{i \ne j \in S_1} K(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i \ne j \in S_2} K(y_i, y_j) - \frac{2}{mn} \sum_{i \in S_1, j \in S_2} K(x_i, y_j)$$
3. **Statistical Significance**: A 500-permutation test shuffles the pooled partition labels across $S_1 \cup S_2$ to form the empirical null distribution under $H_0: P = Q$.
