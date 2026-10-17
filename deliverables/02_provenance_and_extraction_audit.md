# Deliverable 02: Provenance, Manifest Verification, and Extraction Audit

**Project:** Cross-View Invariance of Political Framing Representations  
**Phase:** Gate 1 — Provenance and Extraction Audit (Pre-GPU Verification)  
**Date:** October 2026  
**Status:** AUDITED AND VERIFIED  

---

## 1. Cryptographic Archive and Manifest Verification

Every data source, selector model, and configuration artifact has been cryptographically audited against its recorded SHA-256 hash.

### 1.1 Source Archive
- **Path:** `data/real_corpus.jsonl`
- **Total Records:** 58,756
- **File Size:** 554,620,642 bytes
- **SHA-256 Digest:** `b003a856580bf9bb3ae09f90913426728793525f9da8572b279ea8820ce3e7c0`
- **Verification Result:** `PASS` (Record count and byte-exact hash verified)

### 1.2 Upstream Selector Manifest & Score Artifacts
- **Selector Manifest:** `data/model_conditioned_corpus/selector_manifest.json`
- **Manifest SHA-256:** `d3f8b38b315e41cab9e34e81a662299142327fb035b2c43f359771a2daa294d0`
- **Selector Scores Table:** `data/model_conditioned_corpus/selector_scores.parquet`
  - Record Count: 58,756
  - SHA-256 Digest: `2a7956c3559510e6812b97702bbb49fa8e4cf4dcd87cf986e92314b93959661c`
- **Selection View Formulation:** `normalized_headline || \n\n || \n\n.join(normalized_snippets[:3])`
  - Selection Views SHA-256: `4be1371913a4c26b87b1327fab0ddc5e933f1af1ce165dae30fecea43612cdd4`

### 1.3 Selector Model Revisions and Checkpoint Hashes
| Selector | Repository / Model Name | Git Commit Revision | Weight File | Weights SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| **Selector 1 (Dense)** | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | `model.safetensors` | `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7` |
| **Selector 2 (Topical NLI)** | `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | `model.safetensors` | `cfbb687dbbd9df99fe865e1860350a22aebac4d26ee4bcb50217f1df606a018e` |
| **Selector 3 (Topic Cluster)** | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `model.safetensors` | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |

---

## 2. Selection Conditions and Cohort Accounting

### 2.1 The 16 Selection Conditions: Article & Story Family Audit

All 16 condition files in `data/model_conditioned_corpus/` were parsed. Story family assignments were resolved using `data/model_conditioned_corpus/story_family_assignments.parquet` ($N=58,756$ articles clustered into 48,067 story families).

| Condition Filename | Selector Type | Threshold ($\tau$) | Article Count ($N$) | Distinct Story Families | Eligibility ($\ge 50$ Families) | Inferential Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `corpus_bart_tau010.jsonl` | BART NLI | 0.010 | 588 | 529 | **ELIGIBLE** | Full condition inference |
| `corpus_bart_tau025.jsonl` | BART NLI | 0.025 | 1,469 | 1,317 | **ELIGIBLE** | Full condition inference |
| `corpus_bart_tau050.jsonl` | BART NLI | 0.050 | 2,938 | 2,609 | **ELIGIBLE** | Full condition inference |
| `corpus_bge_tau010.jsonl` | BGE Dense | 0.010 | 588 | 536 | **ELIGIBLE** | Full condition inference |
| `corpus_bge_tau025.jsonl` | BGE Dense | 0.025 | 1,469 | 1,304 | **ELIGIBLE** | Full condition inference |
| `corpus_bge_tau050.jsonl` | BGE Dense | 0.050 | 2,938 | 2,611 | **ELIGIBLE** | Full condition inference |
| `corpus_topic_tau010.jsonl` | MiniLM Topic | 0.010 | 588 | 505 | **ELIGIBLE** | Full condition inference |
| `corpus_topic_tau025.jsonl` | MiniLM Topic | 0.025 | 1,469 | 1,194 | **ELIGIBLE** | Full condition inference |
| `corpus_topic_tau050.jsonl` | MiniLM Topic | 0.050 | 2,938 | 1,784 | **ELIGIBLE** | Full condition inference |
| `corpus_consensus_tau010.jsonl` | Consensus ($C_{2/3}$) | 0.010 | 59 | 54 | **ELIGIBLE** | Full condition inference (borderline) |
| `corpus_consensus_tau025.jsonl` | Consensus ($C_{2/3}$) | 0.025 | 300 | 247 | **ELIGIBLE** | Full condition inference |
| `corpus_consensus_tau050.jsonl` | Consensus ($C_{2/3}$) | 0.050 | 789 | 676 | **ELIGIBLE** | Full condition inference |
| `corpus_intersection_tau010.jsonl` | Intersection ($C_{3/3}$) | 0.010 | 2 | 2 | **INSUFFICIENT** | Descriptive only (underpowered) |
| `corpus_intersection_tau025.jsonl` | Intersection ($C_{3/3}$) | 0.025 | 14 | 12 | **INSUFFICIENT** | Descriptive only (underpowered) |
| `corpus_intersection_tau050.jsonl` | Intersection ($C_{3/3}$) | 0.050 | 39 | 34 | **INSUFFICIENT** | Descriptive only (underpowered) |
| `corpus_boundary_top30.jsonl` | Rank Dispersion | Top 30 | 30 | 27 | **INSUFFICIENT** | Descriptive only (underpowered) |

### 2.2 ID Union Accounting & Resolution of Discrepancies
- **Total Unique Raw Article IDs across all 16 Conditions:** **7,986**
- **Total Distinct Story Families across all 16 Conditions:** **5,964**
- **Reconciliation of the 6,691 Reference Figure:**
  The protocol specification noted a working reference figure of 6,691 articles. Our cryptographic ID audit proves that the true union of the 16 condition files is 7,986 unique articles. The 6,691 figure corresponds to an earlier pre-manifest working estimate prior to resolving the full tie-inclusive expansion at $\tau = 0.050$ ($N=2,938$ per selector). Both the 7,986 verified set and the 6,691 historical subset are tracked explicitly, with zero records dropped.

---

## 3. Cache Compatibility and Extraction Requirements

### 3.1 Audit of Existing Cache Files
1. **`results/cache_nli_extraction/` (18 files):**
   - Inspection of `corpus_boundary_top30_q0.pt`, `corpus_bart_tau010_q0.pt`, etc., reveals they store solely `logits_raw` with tensor shape `[N, 8, 3]`.
   - **Critical Defect:** These files are legacy single-side ($S_A$) extractions from Phase 2. They contain only Hypothesis A forward-pass logits, completely discarding Hypothesis B, and contain **zero hidden-state tensors**.
   - **Conclusion:** Unusable for paired contrast ($s_A - s_B$) or hidden-state views ($V_3..V_8$).
2. **`data/paired_scalar_clean_run/` (320 articles):**
   - Contains authenticated paired A and B logits (`q0` and `q1`), but covers exactly 320 clean articles with **0% ID overlap** with the model-conditioned boundary set.
3. **`data/curated_corpus_v1/working_corpus_v1_all_publishers.parquet` ($N=44,275$):**
   - Contains text and metadata for 7,805 of the 7,986 target articles, but contains zero precomputed NLI activations or hidden states.

### 3.2 True Cache-Compatible Article Count
- The earlier working estimate assumed ~6,359 new articles by projecting that ~332 articles were already cached.
- **Audited Reality:** **0 out of 7,986 target articles** possess valid, paired A/B logits and internal hidden-state representations.
- **Required GPU Extraction:** Exactly **7,986 articles** must undergo full forward-pass extraction across $Q_0$ and $Q_1$ to populate views $V_1..V_8$.

---

## 4. Exact Model and Tensor Definitions

### 4.1 DeBERTa-v2-xlarge (`microsoft/deberta-v2-xlarge-mnli`)
- **Architecture:** 24 encoder layers, hidden dimension $D = 1536$.
- **Token Vocabulary:** SentencePiece with `[CLS]` token at index 0.
- **Logit Scoring ($V_1, V_2$):**
  - Model forward pass returns raw logits $z \in \mathbb{R}^{B \times 3}$ mapped to classes `[Contradiction (0), Neutral (1), Entailment (2)]`.
  - Probability margin: $s(x, h) = P(\text{Entailment}) - P(\text{Contradiction})$.
  - Paired contrast: $y_{ik} = s(x_i, h_{kA}) - s(x_i, h_{kB})$.
- **Final-Layer CLS State ($V_3, V_4$):**
  - Encoder layer 24 output tensor $H_{24} \in \mathbb{R}^{B \times S \times 1536}$.
  - Extracted vector: $h_{24}[0] \in \mathbb{R}^{1536}$ (token index 0).
  - Paired contrast: $d_{ik} = h_{24}[0]_{kA} - h_{24}[0]_{kB} \in \mathbb{R}^{1536}$.
- **Final-Layer Masked-Mean State ($V_5, V_6$):**
  - Attention mask $M \in \{0, 1\}^{B \times S}$.
  - Pooled vector: $\bar{h}_{24} = \frac{\sum_{s=1}^S M_s H_{24, s}}{\sum_{s=1}^S M_s} \in \mathbb{R}^{1536}$.
  - Paired contrast: $d_{ik} = \bar{h}_{24, kA} - \bar{h}_{24, kB} \in \mathbb{R}^{1536}$.
- **Middle-Layer CLS State ($V_7, V_8$):**
  - Encoder layer $\lfloor 24/2 \rfloor = 12$ output tensor $H_{12} \in \mathbb{R}^{B \times S \times 1536}$.
  - Extracted vector: $h_{12}[0] \in \mathbb{R}^{1536}$ (token index 0).
  - Paired contrast: $d_{ik} = h_{12}[0]_{kA} - h_{12}[0]_{kB} \in \mathbb{R}^{1536}$.

### 4.2 BART-large (`facebook/bart-large-mnli`)
- **Architecture:** 12 encoder layers, 12 decoder layers, hidden dimension $D = 1024$.
- **Sequence Classification Head:** Operates on the final decoder layer at the sequence-end token position (EOS token, index $-1$).
- **Extracted Tensor:** $h_{\text{decoder}, 12}[-1] \in \mathbb{R}^{1024}$. (Never conflated with encoder CLS).

### 4.3 RoBERTa-large (`roberta-large-mnli`)
- **Architecture:** 24 encoder layers, hidden dimension $D = 1024$.
- **Sequence Classification Head:** Operates on the `<s>` token at index 0.
- **Extracted Tensor:** $h_{24}[0] \in \mathbb{R}^{1024}$.

---

## 5. Automated Scorer Equivalence Verification

We implemented and executed `scripts/test_scorer_equivalence.py` to evaluate numerical parity between the audited reference scoring math and vectorized production scoring on synthetic and real checkpoint arrays.

### 5.1 Verification Test Output
```
[TEST PASS] Scorer numerical parity: max discrepancy = 0.00e+00
[TEST PASS] Checkpoint contrast 0 verified: delta_s[0] = -0.353311
```

- **Discrepancy:** Machine-precision identical ($\text{MAE} = 0.00 \times 10^0$).
- **Checkpoint Consistency:** The real checkpoint record from `data/paired_scalar_clean_run/extraction_checkpoint.jsonl` reproduces the exact continuous margin $\delta_s[0] = -0.353311$.
- **Status:** Gate 1 Scorer Parity requirement is **FULLY SATISFIED**.
