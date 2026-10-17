# Deliverable 01: Protocol, Estimands, and Statistical Design

**Project:** Cross-View Invariance of Political Framing Representations  
**Phase:** Gate 1.5 — Adversarial Protocol Audit & Verification Freeze  
**Date:** October 2026  
**Protocol Version:** 1.1 (Adversarially Audited & Reconciled)  
**Status:** FROZEN  

---

### Administrative Change Log

| Version | Date | Section / Item | Nature of Change & Adversarial Audit Reconciliation |
| :--- | :--- | :--- | :--- |
| **v1.0** | 2026-10-14 | Initial Freeze | Baseline protocol specification with 6,691 within-condition deduplicated reference working set. |
| **v1.1** | 2026-10-14 | Global Union ($N=7,986$) | Formally updated evaluation universe to the cryptographic union of all 16 selector conditions ($N=7,986$ unique articles). Reconciled family counts: authoritative universe comprises exactly **6,248 distinct story families** in `story_family_assignments.parquet` (5,964 possess global archive representatives inside the union; 284 are admitted via selector-passed sibling articles). |
| **v1.1** | 2026-10-14 | Section 2 (Eligibility) | Formally certified **12 eligible conditions** meeting the minimum $\ge 50$ family rule (9 individual selectors + 3 consensus conditions, including `consensus_tau010` with 54 strict families). Flagged 4 low-power exploratory conditions (< 50 families: 3 intersection + 1 boundary). Noted 50 families is a minimum eligibility gate, not a guarantee of power. |
| **v1.1** | 2026-10-14 | Section 3.2 (Chunk Aggregation) | Explicitly formalized chunk-level hidden-state aggregation $h(x) = \sum_{j=1}^K w_j h(p_j)$, normalized weighting for $K \in \{1, 2, 3\}$, attention mask padding exclusion, and the distinct non-equivalent nature of logit vs. embedding channels. |
| **v1.1** | 2026-10-14 | Section 5 (MMD Estimand) | Standardized condition membership vectors on $w_c \in \mathbb{R}^{7986}$ and mutual 20-NN graph nodes on $N=7,986$. |
| **v1.1** | 2026-10-14 | Section 6 (Benchmark Claims) | Amended semantic benchmark claims: retracted claim of lexical confounding immunity; established contrast specificity ($\text{Ratio} > 1.0, p < 0.01$) as the primary falsification standard. |

---

## 1. Scientific Objectives and Primary Hypotheses

The central scientific objective is to determine whether geometric and relational structures among political news articles remain reproducible when systematically varying:
1. The **corpus-selection procedure** (dense semantic retrieval, zero-shot topical NLI, unsupervised topic clustering, consensus, and intersection across thresholds $\tau \in \{0.010, 0.025, 0.050\}$).
2. The **hypothesis wording** (primary geopolitical bank $Q_0$ vs. lexical/syntactic paraphrase bank $Q_1$).
3. The **measurement representation readout** (paired NLI probabilities, classification-position hidden states, masked mean-pooled hidden states, and middle-layer hidden states).
4. The **underlying model architecture** (DeBERTa-v2-xlarge, BART-large, RoBERTa-large).

The protocol explicitly adjudicates among three structural possibilities:
- **Shared structure:** Relational structures that persist invariant across alterations in the measurement apparatus.
- **Measurement-dependent structure:** Relational structures that shift systematically as a function of selection threshold, hypothesis syntax, or representation readout.
- **Proposition-sensitive structure:** Relational structures that respond selectively to controlled modifications of underlying political propositions, beyond generic topical or lexical overlap.

### 1.1 Primary Hypotheses & Counter-Evidence Standards

| ID | Scientific Hypothesis | Primary Formal Estimand | Decisive Counter-Evidence (Falsification Criteria) |
| :--- | :--- | :--- | :--- |
| **H1** | Contrast-conditioned hidden states preserve cross-view relationships beyond generic article embeddings. | Held-out CKA and mutual 20-NN edge overlap between $V_3..V_8$ exceeding article-only baselines. | Structure is matched or exceeded by article-only embeddings ($d_{\text{art}}$), empty-prompt controls, or prompt-shuffled baselines ($\Delta \text{CKA} \le 0$). |
| **H2** | Shared representation structure responds selectively to controlled propositional changes. | Paired ranking probability $\text{AUC}_{\text{sem}} = P(d(x, x_{\text{changed}}) > d(x, x_{\text{preserved}}))$ and target contrast specificity. | Surface lexical controls predict distances equally well; or contrast specificity fails (unrelated contrasts react as strongly as the targeted contrast: Specificity Ratio $\le 1.0$). |
| **H3** | Changing corpus selection alters relational measurements, and the extent depends on representation type. | Selection discrepancy interaction $\Delta_{r,s}(c, d) = D_r(c, d) - D_s(c, d)$ across 16 selection conditions. | Discrepancies vanish under composition-matched pseudo-selections, or confidence intervals for $\Delta_{r,s}$ span zero under joint story-cluster bootstrap. |
| **H4** | Multi-view consensus provides information beyond the strongest individual representation. | Increment in held-out semantic selectivity AUC: $\Delta \text{AUC} = \text{AUC}_{\text{consensus}} - \max_r \text{AUC}_r$. | Consensus fails to clear $\Delta \text{AUC} \ge +0.05$, or simple paired-logit / embedding baselines match or exceed consensus performance. |

---

## 2. Fixed Boundaries and Methodological Guardrails

1. **Archive Preservation:** The existing 58,756-article news archive (`data/real_corpus.jsonl`) is frozen with SHA-256 digest `b003a856580bf9bb3ae09f90913426728793525f9da8572b279ea8820ce3e7c0`.
2. **Corpus Conditions:** All 16 selector-conditioned manifests in `data/model_conditioned_corpus/` are preserved. The global union of unique articles across all 16 conditions is verified at 7,986 records, mapping to exactly **6,248 distinct story families** in `story_family_assignments.parquet`. Exactly **12 conditions** satisfy the minimum $\ge 50$ story-family eligibility rule (9 individual-selector and 3 consensus conditions). The remaining 4 conditions (`boundary_top30`, and the 3 `intersection_tau*` conditions) are strictly exploratory and under-powered.
3. **Operator Purity:** Historical single-side $s_A$ extractions are strictly segregated from corrected paired contrast extractions ($s_A - s_B$). Single-side values are retained solely as a mandatory ablation control.
4. **No Dimension Manipulation:** PCA compression is prohibited as a prerequisite for constructing representation kernels. All relational measurements operate directly on raw vectors or full contrast channels.
5. **No Speculative Machinery:** Bespoke Riemannian metric tensors, Christoffel connection tracing, and particle-walker path integrals are excluded from primary measurements.
6. **No Pseudo-Observers:** The eight contrast channels are defined strictly as measurement readouts, never as independent human observers or political agents.
7. **No Supervised Model Tuning:** Models are evaluated strictly zero-shot/frozen. No fine-tuning or prompt gradient optimization is permitted.
8. **Pre-Registration of Readouts:** Layer indices, distance metrics, and graph thresholds are prespecified before inspecting test results. No post-hoc layer sweeping is permitted.

---

## 3. Formal Measurement Design

### 3.1 The Core Tensor Object

For article $i$, hypothesis contrast $k \in \{1,\dots,8\}$, model $m$, prompt bank $b \in \{Q_0, Q_1\}$, and representation readout $r$, the measurement apparatus extracts:
$$X_{i, k, m, b, r}$$

#### A. Scalar NLI Readout
For model $m$ and prompt bank $b$, raw logits $z(x_i, h) \in \mathbb{R}^3$ are mapped to probabilities via softmax:
$$P(c \mid x_i, h) = \frac{\exp(z_c)}{\sum_{c'=0}^2 \exp(z_{c'})}, \quad c \in \{0: \text{Contradiction}, 1: \text{Neutral}, 2: \text{Entailment}\}$$
The audited continuous framing margin is:
$$s(x_i, h) = P(\text{Entailment} \mid x_i, h) - P(\text{Contradiction} \mid x_i, h) \in [-1, 1]$$
The paired contrast score is:
$$y_{ik} = s(x_i, h_{k, A}) - s(x_i, h_{k, B}) \in [-2, 2]$$

#### B. Internal Hidden-State Readout
For hidden-state activation vectors $h^{(m, b, r)}(x_i, h) \in \mathbb{R}^D$, the contrast vector is the exact bilateral displacement:
$$d^{(m, b, r)}_{ik} = h^{(m, b, r)}(x_i, h_{k, A}) - h^{(m, b, r)}(x_i, h_{k, B}) \in \mathbb{R}^D$$

### 3.2 Chunk-Level Aggregation and Pooling Specifications

Because raw news articles vary in length (from hundreds to thousands of words), inputs exceed the single-sequence 512-token context window. To ensure reproducible, standardized extraction without arbitrary truncation:

1. **Paragraph Segmentation & Weighting Rule:**
   Each article text $x$ is segmented into paragraphs $\mathcal{P}(x) = (p_1, \dots, p_K)$ where $K = \min(n_{\text{paras}}, 3)$.
   Base lead-paragraph weights are prespecified as $\mathbf{W} = [0.5, 0.3, 0.2]$.
   Normalized chunk weights are defined as:
   $$w_j = \frac{W_j}{\sum_{m=1}^K W_m}, \quad j \in \{1, \dots, K\}$$
   - **$K = 1$ (Single chunk / unsegmented text):** $w_1 = 1.0$.
   - **$K = 2$ (Two chunks):** $w_1 = \frac{0.5}{0.8} = 0.625, \quad w_2 = \frac{0.3}{0.8} = 0.375$.
   - **$K = 3$ (Three or more chunks):** $w_1 = 0.5, \quad w_2 = 0.3, \quad w_3 = 0.2$.

2. **Padding Token Exclusion in Hidden-State Pooling:**
   Let $H_{L, j, t, :} \in \mathbb{R}^D$ denote the hidden-state activation vector for chunk $p_j$ paired with hypothesis $h$ at layer $L$ and token position $t$, with binary attention mask $M_{j, t} \in \{0, 1\}$.
   - **CLS Token Positions (Layers 12 & 24):** Extracted strictly at position $t=0$ (`[CLS]`), which is guaranteed active and unpadded:
     $$h^{\text{CLS}}_{L}(p_j, h) = H_{L, j, 0, :} \in \mathbb{R}^{1536}$$
   - **Sequence Masked-Mean Pooling (Layer 24):** Pooled strictly across non-padded tokens using the attention mask:
     $$h^{\text{mean}}_{24}(p_j, h) = \frac{\sum_{t=1}^{T} M_{j, t} H_{24, j, t, :}}{\sum_{t=1}^{T} M_{j, t}} \in \mathbb{R}^{1536}$$
     Because $M_{j, t} = 0$ for all padding positions, padding tokens contribute zero to both numerator and denominator.

3. **Article-Level Hidden-State Vector Aggregation:**
   The aggregated article-level hidden state for hypothesis $h$ is the weighted linear combination:
   $$h(x_i, h) = \sum_{j=1}^K w_j h(p_{i, j}, h)$$
   The resulting bilateral contrast displacement vector is:
   $$d_{ik} = h(x_i, h_{kA}) - h(x_i, h_{kB}) = \sum_{j=1}^K w_j \left( h(p_{i, j}, h_{kA}) - h(p_{i, j}, h_{kB}) \right)$$

4. **Channel Separation: Scalar Logits vs. Embedding Manifolds:**
   The scalar logit channel aggregates chunk-level logits before softmax:
   $$z(x_i, h) = \sum_{j=1}^K w_j z(p_{i, j}, h)$$
   Because the NLI classification head consists of an affine dense layer followed by a non-linear activation and softmax, applying the classification head to the averaged hidden state $\sum w_j h(p_j)$ is not mathematically equivalent to $z(x_i, h)$.
   Crucially, these readouts do **not** constitute independent evidence—they originate from the same underlying model forward pass. Rather, they represent **distinct mathematical readouts** of the same underlying computation:
   - *Readout Channel 1 ($V_1, V_2$):* Scaled decision margins derived from task-head projection.
   - *Readout Channel 2 ($V_3..V_8$):* Internal contextual representations derived from encoder hidden states before task-head projection.
   Both readouts are audited, preserved, and evaluated to determine whether geometric structure is preserved across different representation readouts of the same model.

### 3.3 The Primary Eight Views (DeBERTa-v2-xlarge)

| View ID | Prompt Bank | Readout Channel ($r$) | Mathematical Definition | Dimensions | Purpose |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **$V_1$** | $Q_0$ | Paired NLI Probabilities | $y_{ik} = s(x_i, h_{kA}) - s(x_i, h_{kB})$ | 8 | Primary scalar baseline ($Q_0$) |
| **$V_2$** | $Q_1$ | Paired NLI Probabilities | $y_{ik} = s(x_i, h_{kA}) - s(x_i, h_{kB})$ | 8 | Primary scalar baseline ($Q_1$ paraphrase) |
| **$V_3$** | $Q_0$ | Final-Layer CLS State | $d_{ik} = \bar{h}_{L}[0]_{kA} - \bar{h}_{L}[0]_{kB}$ | $8 \times 1536$ | Primary internal representation test ($Q_0$) |
| **$V_4$** | $Q_1$ | Final-Layer CLS State | $d_{ik} = \bar{h}_{L}[0]_{kA} - \bar{h}_{L}[0]_{kB}$ | $8 \times 1536$ | Primary internal representation test ($Q_1$) |
| **$V_5$** | $Q_0$ | Final-Layer Masked Mean | $d_{ik} = \bar{h}_{L, \text{mask}, kA} - \bar{h}_{L, \text{mask}, kB}$ | $8 \times 1536$ | Sequence-pooled representation test ($Q_0$) |
| **$V_6$** | $Q_1$ | Final-Layer Masked Mean | $d_{ik} = \bar{h}_{L, \text{mask}, kA} - \bar{h}_{L, \text{mask}, kB}$ | $8 \times 1536$ | Sequence-pooled representation test ($Q_1$) |
| **$V_7$** | $Q_0$ | Middle-Layer CLS State | $d_{ik} = \bar{h}_{\lfloor L/2 \rfloor}[0]_{kA} - \bar{h}_{\lfloor L/2 \rfloor}[0]_{kB}$ (Layer 12) | $8 \times 1536$ | Prespecified depth ablation ($Q_0$) |
| **$V_8$** | $Q_1$ | Middle-Layer CLS State | $d_{ik} = \bar{h}_{\lfloor L/2 \rfloor}[0]_{kA} - \bar{h}_{\lfloor L/2 \rfloor}[0]_{kB}$ (Layer 12) | $8 \times 1536$ | Prespecified depth ablation ($Q_1$) |

### 3.4 Cross-Model Confirmation Panel ($N=512$)

To test architecture independence beyond DeBERTa, an authenticated $N=512$ panel is evaluated across:
1. `microsoft/deberta-v2-xlarge-mnli`: Sequence token 0 (`[CLS]`), $D=1536$.
2. `facebook/bart-large-mnli`: Decoder sequence classification token (EOS token at position $-1$), $D=1024$.
3. `roberta-large-mnli`: Sequence token 0 (`<s>`), $D=1024$.

---

## 4. Comparable Relational Measurements and Kernels

Because probability scores ($[-2, 2]$) and hidden-state embeddings ($\mathbb{R}^{1536}$) reside in distinct metric spaces, coordinates cannot be directly subtracted. Comparisons are executed strictly in representation kernel space $\mathcal{K}$.

### 4.1 Scalar-Logit Kernel ($K^{\text{logit}}$)
1. For discovery articles $i \in \mathcal{D}$, standardize each coordinate across articles:
   $$z_{ik} = \frac{y_{ik} - \hat{\mu}_k^{\mathcal{D}}}{\hat{\sigma}_k^{\mathcal{D}}}, \quad \mathbf{z}_i \in \mathbb{R}^8$$
2. Compute the RBF kernel:
   $$K^{\text{logit}}_{ij} = \exp\left(-\frac{\|\mathbf{z}_i - \mathbf{z}_j\|^2}{2\sigma_{\text{med}}^2}\right)$$
   where $\sigma_{\text{med}}$ is frozen as the median non-zero Euclidean distance in $\mathcal{D}$. If degenerate, fall back to $\sqrt{8}$.

### 4.2 Hidden-State Contrast Kernel ($K^{\text{hidden}}$)
1. Center each contrast displacement by its discovery mean:
   $$\tilde{d}_{ik} = d_{ik} - \hat{\mu}_k^{\mathcal{D}, \text{hidden}} \in \mathbb{R}^{1536}$$
2. Normalize non-zero vectors to unit sphere:
   $$\tilde{\mathbf{u}}_{ik} = \frac{\tilde{d}_{ik}}{\|\tilde{d}_{ik}\|_2}$$
3. Construct the 8-contrast composite linear similarity kernel:
   $$K^{\text{hidden}}_{ij} = \frac{1}{8}\sum_{k=1}^8 \tilde{\mathbf{u}}_{ik}^\top \tilde{\mathbf{u}}_{jk}$$
4. An uncentered ablation kernel $K^{\text{uncentered}}_{ij} = \frac{1}{8}\sum_{k=1}^8 (\mathbf{u}_{ik}^\top \mathbf{u}_{jk})$ is evaluated alongside $K^{\text{hidden}}$ to test whether mean removal suppresses generic prompt biases without destroying political framing signal.

### 4.3 Graph Construction and Alignment
- **Mutual 20-NN Graphs:** Construct undirected graphs $G^{(r)}$ where edge $(i, j) \in E$ iff $j \in \text{kNN}(i)$ AND $i \in \text{kNN}(j)$ with $k=20$. Deterministic tie-breaking uses ascending article canonical IDs.
- **Centered Kernel Alignment (CKA):**
  $$\text{CKA}(K, L) = \frac{\operatorname{Tr}(K H L H)}{\sqrt{\operatorname{Tr}(K H K H) \operatorname{Tr}(L H L H)}}, \quad H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$
- **Graph Metrics:** Jaccard edge overlap $J(E_r, E_s) = \frac{|E_r \cap E_s|}{|E_r \cup E_s|}$ and rank correlation of nearest-neighbor lists.

---

## 5. Corpus-Selection Dependence: MMD Estimand

For condition $c \in \{1,\dots,16\}$, define membership weight vector $w_c \in \mathbb{R}^N$ over the common $N=7,986$ union:
$$w_{c, i} = \begin{cases} 1/n_c & \text{if article } i \in \mathcal{C}_c \\ 0 & \text{otherwise} \end{cases}$$

For any positive semi-definite representation kernel $K^{(r)}$, the empirical squared Maximum Mean Discrepancy between conditions $c$ and $d$ is:
$$\boxed{D_r(c, d) = (w_c - w_d)^\top K^{(r)} (w_c - w_d)}$$

### 5.1 Selection-by-Representation Interaction
To determine whether selection differences depend systematically on representation readout:
$$\Delta_{r, s}(c, d) = D_r(c, d) - D_s(c, d)$$
- **Inference Null:** Calibrate $\Delta_{r, s}(c, d)$ against a frozen family of composition-matched pseudo-selections preserving condition sample sizes and publisher margins.
- **Story-Family Clustered Bootstrap:** Compute 95% bootstrap confidence intervals via joint story-cluster resampling ($B=1,000$). Resampling units are the **6,248 story families** represented in the union. To prevent leaking excluded articles back into the sample, representatives for condition $c$ are sampled strictly from within the eligible members of condition $c$ belonging to each selected family.

### 5.2 Reference Population Estimands (Eligible Population vs. Excluded Complement)
When evaluating a selected condition $c$ against a broader reference:
1. **Selection Relative to Eligible Population ($D_r(c, \text{pop})$):** Compares condition $c$ against the entire eligible population. While condition $c$ is a subset of the population (which attenuates MMD under mixture distributions), this estimand directly quantifies the representational departure of the selected subset from the broader eligible domain.
2. **Selection Divergence Relative to Excluded Complement ($D_r(c, \neg c)$):** Compares condition $c$ against the disjoint subset of eligible articles explicitly omitted by selector $c$.
Both estimands answer distinct scientific questions and must be reported separately without conflation.

### 5.3 Layer-Scale Invariance & Kernel Normalization
Because internal representations at different network depths reside on different numerical scales (e.g., empirical displacement norms for Layer 24 CLS average $\approx 12.6$ while Layer 12 CLS average $\approx 0.96$), raw Euclidean displacement magnitudes cannot be meaningfully compared across layers. All downstream relational estimands (MMD, graph topology, and CKA) operate strictly within scale-normalized representation kernels where bandwidth $\sigma_{\text{med}}^{(r)}$ is calibrated to the median pairwise distance of readout $r$.

---

## 6. Zero-Human-Label Semantic Falsification Suite

### 6.1 Controlled Intervention Design (480 Passages)
Construct 120 base political passages ($T_1..T_{120}$) across diverse topics, each generating 4 matched variants:
1. **Original ($x_{\text{orig}}$):** Authentic baseline political statement.
2. **Meaning-Preserving ($x_{\text{pres}}$):** Controlled syntactic paraphrase, active-to-passive voice transformation, or synonymous lexical substitution.
3. **Proposition-Changing ($x_{\text{chang}}$):** Explicit alteration of target political proposition via negation, agent-patient role reversal, causal attribution shift, or quantifier inversion.
4. **Surface Control ($x_{\text{surf}}$):** Matched character/token edit distance altering non-propositional adjectives or incidental details without altering the core political claim.

> [!IMPORTANT]
> **Methodological Boundary on Lexical Controls:** Character-level edit distance matching serves as a conservative surface baseline, but does *not* mathematically guarantee freedom from lexical, syntactic, or distributional confounding. A model could still exploit subtle lexical cues or word substitutions. Therefore, the primary, decisive standard for semantic validity is **Contrast Specificity**, not aggregate lexical matching.

### 6.2 Primary Selectivity & Contrast Specificity Estimands
1. **Paired Ranking Probability:**
   $$\text{AUC}_{\text{sem}} = \frac{1}{N_{\text{pairs}}}\sum_{t=1}^{120} \mathbb{1}\left[d_r(x_{\text{orig}}^{(t)}, x_{\text{chang}}^{(t)}) > d_r(x_{\text{orig}}^{(t)}, x_{\text{pres}}^{(t)})\right]$$
2. **Target Contrast Specificity Gate:**
   For an intervention explicitly targeting contrast dimension $k^*$, the on-target displacement must significantly exceed the mean displacement across off-target contrasts:
   $$\text{Specificity Ratio} = \frac{\Delta d_{k^*}}{\frac{1}{7}\sum_{k \ne k^*} \Delta d_k} > 1.0, \quad p < 0.01$$
   where $\Delta d_k = \|d_k(x_{\text{chang}}) - d_k(x_{\text{orig}})\|$. Furthermore, on-target displacement under proposition change must exceed surface control: $\Delta d_{k^*}(x_{\text{chang}}) > \Delta d_{k^*}(x_{\text{surf}})$.

### 6.3 Mandatory Nuisance Baselines
Every representation $V_1..V_8$ is benchmarked against:
1. Single-side $s_A$ logits (testing bilateral cancellation necessity).
2. Article-only embeddings (testing prompt-conditioning necessity).
3. FrameAxis bi-encoder projection comparator.
4. TF-IDF character/word n-gram lexical similarity.
5. Absurd / irrelevant prompt banks (meteorology, culinary).
6. Empty-article / prompt-only controls.
7. Shuffled-hypothesis controls (breaking prompt-article alignment).

---

## 7. Multi-View Graph Consensus (Gate 5)

### 7.1 Linear Consensus Baseline
$$K_{\text{linear}} = \frac{1}{4} \left( \frac{K^{(V_1)} + K^{(V_2)}}{2} + \frac{K^{(V_3)} + K^{(V_4)}}{2} + \frac{K^{(V_5)} + K^{(V_6)}}{2} + \frac{K^{(V_7)} + K^{(V_8)}}{2} \right)$$

### 7.2 Higher-Order Tensor Consensus Benchmark (Peng et al. IJCAI 2024)
Construct high-order transition probability tensor $P_K \in \mathbb{R}^{n \times n \times V}$ across views $v \in \{1,\dots,V\}$:
$$P_K^{(v)} = \sum_{k=1}^K \frac{(P^{(v)})^k + ((P^{(v)})^k)^\top}{2}$$
Solve the CCL-MVC augmented Lagrangian objective:
$$\min_{Z, E, w} \|Z\|_{\circledast} + \lambda \|E\|_{\circledS} + \alpha \sum_{v=1}^V \|Z_0 - w_v Z^{(v)}\|_F^2 + \beta \sum_{i,j=1}^V w_i w_j \operatorname{Tr}((Z^{(i)})^\top Z^{(j)})$$
subject to $P_K = Z + E, w_v \ge 0, \sum_v w_v = 1$.
- Published hyperparameters: $K=3, \lambda=0.01, \alpha=0.001, \beta=1.0, N_{\text{neighbors}}=5, \rho=0.001, \kappa=1.5$.
- **Adjudication Criterion:** Multi-view consensus is certified as scientifically useful iff held-out $\text{AUC}_{\text{sem}}(\text{consensus}) - \max_r \text{AUC}_{\text{sem}}(r) \ge +0.05$ with cluster bootstrap 95% CI excluding zero.

---

## 8. Literature Mapping to Analysis Choices

| Methodological Choice | Authoritative Citation | Concrete Function in Protocol |
| :--- | :--- | :--- |
| **Higher-Order Multi-View Consensus** | Peng et al. (IJCAI 2024) | Benchmark higher-order random walk tensor $P_K$ and auto-weighted diversity $w_v$ against linear baseline. |
| **Similarity Network Fusion (SNF)** | Wang et al. (Nature Methods 2014) | Preserves shared affinity network while capturing view-specific complementary structures. |
| **Centered Kernel Alignment (CKA)** | Kornblith et al. (ICML 2019) | Compares geometry across representation channels without requiring shared coordinate axes. |
| **Multitrait-Multimethod (MTMM)** | Campbell & Fiske (1959) | Separates authentic framing trait variance from measurement method/channel variance. |
| **Contextual Representation Geometry** | Ethayarajh (EMNLP 2019) | Identifies upper-layer anisotropy (cone effect) and mandates mean-centering to avoid artificial cosine inflation. |
| **Control Tasks / Selectivity Probing** | Hewitt & Liang (EMNLP 2019) | Mandates paired semantic selectivity ($\text{AUC}_{\text{sem}}$) and surface lexical controls to verify construct validity. |
| **Unsupervised Ideological Geometry** | Hofmann et al. (ICML 2022) | Benchmark establishing baseline limits of label-free stance extraction from document representations. |
| **Automated Stance Identification (APSI)** | APSI Framework Documentation | Reference implementation for paired contrastive NLI with explicitly authored political hypotheses. |

---

## 9. Stopping and Demotion Rules

1. **Gate 1 Halt:** If extraction fails machine-precision logit parity ($\text{MAE} \ge 10^{-6}$) against the reference scorer, abort before full GPU inference.
2. **H1 Demotion:** If contrast-conditioned hidden states fail to outperform article-only or prompt-shuffled embeddings on held-out CKA, demote internal representation claims.
3. **H2 Demotion:** If semantic intervention ranking AUC fails to exceed 0.60 or fails contrast specificity, classify the representation as topically responsive but proposition-blind.
4. **H3 Demotion:** If selection discrepancies vanish under composition-matched controls, reject selection-dependence claims.
5. **H4 Demotion:** If multi-view consensus fails to achieve a $+0.05$ AUC gain over the strongest individual channel, reject multi-view tensor fusion as unnecessary machinery.
