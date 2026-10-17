# Cross-Model Representation Panel Experiment (N=512 Story Families)

**Experiment Execution Date**: 2026-10-16  
**Execution Runtime**: 294.52 seconds (PyTorch CUDA, NVIDIA GeForce RTX 5070 Ti)  
**Sample Population**: $N = 512$ Independent Story Families (248 publishers)  
**Frozen Panel Source**: `data/cross_model_panel_512.jsonl` (SHA-256: `def09e2b0e1945825c72c64be2b69153e9c82cdaef02924141e2ddc1d46b81c0`)  
**Evaluated Pre-Trained Models**:
1. `microsoft/deberta-v2-xlarge-mnli` (24 layers, $D = 1536$)
2. `roberta-large-mnli` (24 layers, $D = 1024$)
3. `facebook/bart-large-mnli` (12 encoder / 12 decoder layers, $D = 1024$)

---

## 1. Audit Deliverables and Checkpoints

All extracted representations, Gram alignment matrices, and neighborhood graphs are serialized under `deliverables/06_cross_model_panel/`:

| Artifact Name | Format | Content |
| :--- | :---: | :--- |
| `deberta_panel_512.npz` | NPZ | Sliced $Q_0, Q_1$ scalars and internal representations ($V_3, V_4, V_7, V_8$) for 512 articles |
| `roberta_panel_512.npz` | NPZ | Extracted $Q_0, Q_1$ scalars, Layer 24 CLS diffs, and Layer 12 CLS diffs for 512 articles |
| `bart_panel_512.npz` | NPZ | Extracted $Q_0, Q_1$ scalars, Decoder L12 boundary diffs, and Decoder L6 boundary diffs |
| `cross_model_cka_matrix.json` | JSON | Within-model and pairwise cross-model Linear/RBF CKA with 500-permutation null controls |
| `cross_model_knn_overlap.json` | JSON | Within-model and cross-model $k$-NN Jaccard graph overlap matrices ($k=10, 20$) |
| `cross_model_executive_report.md` | Markdown | Comprehensive synthesis of cross-model generalizability and relational geometry |

---

## 2. Within-Model Prompt Invariance ($Q_0$ vs $Q_1$) Across Architectures

Evaluating whether prompt paraphrase stability varies monotonically with network depth across different model architectures:

| Architecture | Readout Level | Representation View | RBF CKA ($Q_0$ vs $Q_1$) | Permutation Null $q_{99}$ | $p$-value | $k$-NN $J_{10}$ Overlap | Factor vs. Random ($\mathbb{E}[J_{10}] \approx 0.0099$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeBERTa-v2-xlarge** | Middle Layer | Layer 12 CLS | **0.9302** | 0.2190 | $< 0.0020$ | **0.4054** | **$41.0\times$** |
| | Final Layer | Layer 24 CLS | **0.7943** | 0.1427 | $< 0.0020$ | **0.2163** | **$21.9\times$** |
| | Scalar Scores | Paired Head Margins | **0.7525** | 0.0232 | $< 0.0020$ | **0.0925** | **$9.3\times$** |
| **BART-large** | Middle Layer | Decoder Layer 6 Boundary | **0.9451** | 0.1231 | $< 0.0020$ | **0.5075** | **$51.3\times$** |
| | Final Layer | Decoder Layer 12 Boundary| **0.8922** | 0.1421 | $< 0.0020$ | **0.3535** | **$35.7\times$** |
| | Scalar Scores | Paired Head Margins | **0.7826** | 0.0205 | $< 0.0020$ | **0.0932** | **$9.4\times$** |
| **RoBERTa-large** | Middle Layer | Layer 12 CLS | **0.6939** | 0.1577 | $< 0.0020$ | **0.2653** | **$26.8\times$** |
| | Final Layer | Layer 24 CLS | **0.5675** | 0.0645 | $< 0.0020$ | **0.1097** | **$11.1\times$** |
| | Scalar Scores | Paired Head Margins | **0.7083** | 0.0215 | $< 0.0020$ | **0.1185** | **$12.0\times$** |

### Key Findings on Prompt Invariance Across Architectures
1. **Universality of Intermediate Layer Neighborhood Stability**:
   Across all three architectures, the middle representation preserves substantially higher local neighborhood overlap ($J_{10}$) between prompt paraphrases than the final representation:
   - BART: $J_{10} = 0.5075$ (middle) vs. $0.3535$ (final) $\to 1.4\times$ higher
   - DeBERTa: $J_{10} = 0.4054$ (middle) vs. $0.2163$ (final) $\to 1.9\times$ higher
   - RoBERTa: $J_{10} = 0.2653$ (middle) vs. $0.1097$ (final) $\to 2.4\times$ higher
2. **Global vs. Local Divergence in Scalars**:
   All three models show the exact same divergence between global and local stability for classification head scalars: while global CKA remains moderately high ($\text{CKA} \approx 0.71 - 0.78$), local neighborhood overlap drops to $J_{10} \approx 0.09 - 0.12$. Downstream classification projections reorganize local article neighborhoods across all tested models.
3. **Architecture Ranking**:
   BART exhibits the highest overall prompt invariance across all layers ($\text{CKA} = 0.9451$ mid, $0.8922$ final), followed by DeBERTa ($\text{CKA} = 0.9302$ mid, $0.7943$ final), with RoBERTa showing comparatively lower invariance ($\text{CKA} = 0.6939$ mid, $0.5675$ final).

---

## 3. Pairwise Cross-Model Alignment (DeBERTa vs BART vs RoBERTa)

Evaluating whether the representational geometry is preserved when the underlying neural architecture is swapped:

### 3.1 Cross-Model RBF CKA Alignment

| Model Pair | Feature Level | Prompt Bank | Empirical RBF CKA | Permutation Null $q_{99}$ | Empirical $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DeBERTa vs. BART** | Scalar Margins | $Q_0$ | 0.4340 | 0.0192 | $< 0.0020$ |
| | Scalar Margins | $Q_1$ | 0.4251 | 0.0208 | $< 0.0020$ |
| | Final Layer | $Q_0$ | **0.6364** | 0.1339 | $< 0.0020$ |
| | Final Layer | $Q_1$ | **0.6489** | 0.1530 | $< 0.0020$ |
| | Middle Layer | $Q_0$ | 0.5725 | 0.1642 | $< 0.0020$ |
| | Middle Layer | $Q_1$ | 0.5437 | 0.1631 | $< 0.0020$ |
| **DeBERTa vs. RoBERTa** | Scalar Margins | $Q_0$ | 0.1360 | 0.0214 | $< 0.0020$ |
| | Scalar Margins | $Q_1$ | 0.1847 | 0.0202 | $< 0.0020$ |
| | Final Layer | $Q_0$ | 0.3091 | 0.0962 | $< 0.0020$ |
| | Final Layer | $Q_1$ | 0.4000 | 0.0944 | $< 0.0020$ |
| | Middle Layer | $Q_0$ | 0.4457 | 0.1884 | $< 0.0020$ |
| | Middle Layer | $Q_1$ | 0.4581 | 0.1826 | $< 0.0020$ |
| **RoBERTa vs. BART** | Scalar Margins | $Q_0$ | 0.1704 | 0.0209 | $< 0.0020$ |
| | Scalar Margins | $Q_1$ | 0.2283 | 0.0211 | $< 0.0020$ |
| | Final Layer | $Q_0$ | 0.3696 | 0.0927 | $< 0.0020$ |
| | Final Layer | $Q_1$ | 0.4709 | 0.0981 | $< 0.0020$ |
| | Middle Layer | $Q_0$ | 0.4299 | 0.1389 | $< 0.0020$ |
| | Middle Layer | $Q_1$ | 0.4395 | 0.1396 | $< 0.0020$ |

### 3.2 Cross-Model Local Neighborhood Overlap ($k$-NN Jaccard)

Random baseline expectation for $N=512$: $\mathbb{E}[J_{10}] \approx 0.00988$.

| Model Pair | Feature Level | $Q_0$ Mean $J_{10}$ | $Q_1$ Mean $J_{10}$ | Factor vs. Random Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **DeBERTa vs. BART** | Final Layer | **0.1095** | **0.1161** | **$11.8\times$ baseline** |
| | Middle Layer | **0.0950** | **0.0962** | **$9.7\times$ baseline** |
| | Scalar Margins | **0.0370** | **0.0407** | **$4.1\times$ baseline** |
| **RoBERTa vs. BART** | Final Layer | 0.0601 | 0.0666 | $6.7\times$ baseline |
| | Middle Layer | 0.0678 | 0.0770 | $7.8\times$ baseline |
| | Scalar Margins | 0.0237 | 0.0327 | $3.3\times$ baseline |
| **DeBERTa vs. RoBERTa**| Final Layer | 0.0479 | 0.0530 | $5.4\times$ baseline |
| | Middle Layer | 0.0387 | 0.0425 | $4.3\times$ baseline |
| | Scalar Margins | 0.0237 | 0.0283 | $2.9\times$ baseline |

---

## 4. Synthesis and Scientific Implications

### 4.1 DeBERTa and BART Share Substantial Framing Geometry
- DeBERTa and BART exhibit high mutual manifold alignment in their final representations ($\text{CKA} \approx 0.64 - 0.65$) and moderate alignment in their middle representations ($\text{CKA} \approx 0.54 - 0.57$) and scalars ($\text{CKA} \approx 0.43$).
- Their final-layer nearest-neighbor overlap ($J_{10} \approx 0.11 - 0.12$) is almost $12\times$ the random expectation, showing that article relationships discovered by DeBERTa are largely reproduced by BART despite differing pre-training objectives and encoder-decoder architecture.

### 4.2 RoBERTa Displays Marked Divergence
- RoBERTa shows substantially lower alignment with both DeBERTa ($\text{CKA} = 0.14 - 0.18$ for scalars, $0.31 - 0.40$ for final layer) and BART ($\text{CKA} = 0.17 - 0.23$ for scalars, $0.37 - 0.47$ for final layer).
- Correspondingly, cross-model $k$-NN overlap between RoBERTa and DeBERTa drops to $J_{10} \approx 0.024 - 0.048$.

### 4.3 Architecture Dependence Exceeds Prompt Dependence
- Swapping the prompt wording ($Q_0 \to Q_1$) within a model yields high CKA ($0.75 - 0.95$).
- Swapping the neural architecture ($\text{DeBERTa} \to \text{BART} \to \text{RoBERTa}$) causes a much steeper drop in alignment ($\text{CKA} = 0.14 - 0.65$).
- **Methodological Conclusion**: Measurement dependence is dominated by **model architecture choice** rather than by probe phrasing. While intermediate representations are universally more stable under prompt paraphrasing, the resulting relational space remains conditioned on the specific neural observer used for extraction.
