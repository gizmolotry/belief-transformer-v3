# Immutable Confirmatory Preregistration Protocol: Q0 + Q1 Replication

**Document Status**: Cryptographically Frozen Confirmatory Specification  
**Timestamp**: 2026-09-13T13:48:00-07:00  
**Project**: Belief Transformer V3 / Perspectival NLI Geometry  
**Repository Path**: `paper/preregistration_confirmatory_q0_q1.md`  

---

## 1. Scientific Context & Confirmatory Hypotheses

This specification establishes an immutable, prospectively frozen protocol for evaluating whether the Belief Transformer measurement apparatus elicits reproducible, low-dimensional stance geometry across both the canonical framing bank ($Q_0$) and the paraphrase bank ($Q_1$) on a genuinely untouched, never-inspected corpus of news articles.

### Pre-Declared Primary Hypotheses
1. **$H_1$ (Dominant Stance Axis Replication on $Q_0$)**:
   Under the audited paired contrast operator ($\Delta s = s_A - s_B$), the first principal component ($\mathrm{PC1}$) explains substantial variance and clears the 10,000-draw publisher-block permutation null gate at $\alpha = 0.05$ (reporting $p \approx 0.0001$).
2. **$H_2$ (Dominant Stance Axis Replication on $Q_1$)**:
   Under identical conditions on the paraphrase bank ($Q_1$), $\mathrm{PC1}$ clears the 10,000-draw publisher-block permutation null gate at $\alpha = 0.05$ (reporting $p \approx 0.0001$).
3. **$H_3$ (Cross-Bank Stance Coordinate Invariance)**:
   Article stance coordinates along $\mathrm{PC1}$ correlate strongly between $Q_0$ and $Q_1$, achieving Pearson $|r| \ge 0.70$ and Spearman $|\rho| \ge 0.70$ ($p < 0.001$), with 2-PC Procrustes disparity $\le 0.35$ and subspace angle $\le 20^\circ$.
4. **$H_4$ (Bank-Level Construct Invariance across Disjoint Partitions)**:
   Primary stance ordering is a distributed property of the 8-axis bank rather than driven by an idiosyncratic prompt pair. Across all $\frac{1}{2}\binom{8}{4} = 35$ unique complementary 4-vs-4 partitions ($S \cap S^c = \emptyset$, zero shared variables), the mean cross-split correlation achieves mean $|r| \ge 0.70$ on both $Q_0$ and $Q_1$.

### Pre-Declared Secondary / Descriptive Outcomes (No Null Failure Gate)
- **Descriptive Outcome 1 (Hidden-State Axis Coherence)**:
  Exhaustive combinatorial upper-tail probability against all $\binom{24}{8} = 735{,}471$ subsets in mean-pooled hidden-state representation space will be computed and reported descriptively regardless of direction. In accordance with established construct dissociation, axis coherence is treated as an erratic, highly conditional property; a non-significant result ($p > 0.05$) is neither a confirmatory failure nor statistical proof of the null, but descriptive documentation of construct separation.

---

## 2. Cryptographic Pre-Commitment & Protocol Hashes

All code, query banks, exclusion registries, and sampling logic are cryptographically frozen:

| Protocol Component | Repository Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Source Candidate Corpus Pool** | `data/real_corpus.jsonl` ($58{,}756$ articles) | `b003a856580bf9bb3ae09f90913426728793525f9da8572b279ea8820ce3e7c0`* |
| **Prospective Sampled Corpus** | `data/prospective_confirmatory_120_corpus.jsonl` ($N=120$) | `397d8aec2114db53b55408983192bbcd82e9d310863208aef9ea63e563c6ad42` |
| **$Q_0$ Canonical Hypothesis Bank** | `config/framing_queries.yaml` | `7e85fcb306be36c50687d19defa3c8be249b82e7eb5be399cf7a82ef75b30c81` |
| **$Q_1$ Paraphrase Hypothesis Bank** | `config/prompt_banks/probe_hypotheses_paraphrase_a.json` | `5a281bfde1784749d6e437d51a13949cd3603e8ee472011e4b4e927bf3bbd236` |
| **Touched Fingerprints Exclusion Registry** | `data/touched_fingerprints_registry.json` ($1{,}290$ entries) | `7680dc6f09e3e83403079ea829634f9b8b79024c8e151eed73c1dc417dc98b4f` |
| **Confirmatory Execution & Sampling Script** | `scripts/exp_prospective_preregistered_replication.py` | `88752e593d90b758c5c243f101699ee4fa74aad675202252a1885f2d578f2944` |
| **Dimensionality Null Test Module** | `scripts/exp_systematic_dimensionality_gate.py` | `02f6a1e00b4c7aa722196e4d5f45e01ee843b29da2725d6b2141da75987f6850` |
| **Core NLI Model Architecture** | HuggingFace Checkpoint | `microsoft/deberta-v2-xlarge-mnli` |

*\*Provenance Note: The source candidate pool (`data/real_corpus.jsonl`) was an untracked local file whose SHA-256 hash was recorded retrospectively after execution at commit `273987bc`; the sampling code, exclusion registry, hypothesis banks, and analysis protocol were prospectively frozen at commit `03be740c`.*

---

## 3. Sampling & Corpus Selection Protocol

To ensure prospective protocol integrity:
1. **Candidate Pool**: Raw pool of 58,756 articles in `data/real_corpus.jsonl` (554,620,642 bytes, SHA-256: `b003a856580bf9bb3ae09f90913426728793525f9da8572b279ea8820ce3e7c0`).
2. **Automated Exclusion Filter**: Every candidate article is checked against `data/touched_fingerprints_registry.json` (containing 1,290 fingerprints across text hashes, titles, and URLs touched in all previous exploratory, held-out, cross-topic, and confirmatory scripts). Any candidate matching an exclusion fingerprint is permanently bypassed.
3. **Keyword Conditioning**: Articles must match core conflict keywords (`palestine`, `palestinian`, `gaza`, `israel`, `israeli`, `hamas`) and contain $\ge 80$ words.
4. **Publisher Diversity**: Maximum cap of $\le 8$ articles per publisher across 70 distinct publishers.
5. **Sample Size**: Exactly $N = 120$ real articles written directly to `data/prospective_confirmatory_120_corpus.jsonl`.
6. **No Prior Inspection**: Zero manual inspection, filtering, screening, or preliminary extraction is permitted prior to execution.

### 3.1 Post-Hoc Content Similarity Audit & Sensitivity Analysis ($N=83$)

While exact registry exclusion achieved zero collisions across all 1,290 fingerprints, titles, and URLs, an adversarial audit using 5-word shingle Jaccard overlap detected that 37 of the 120 articles shared substantial text similarity ($\text{Jaccard} \ge 0.50$) with earlier corpora due to natural wire syndication across differing regional publishers. 

To determine whether syndication drives the empirical findings, a post-hoc sensitivity analysis purges all 37 syndicated records ($N=83$ strictly unique articles):
- $Q_0$ PC1 EVR: **$59.87\%$** ($\text{PR} = 2.52$)
- $Q_1$ PC1 EVR: **$62.03\%$** ($\text{PR} = 2.39$)
- Cross-bank alignment: Pearson $|r| = \mathbf{0.9533}$

This confirms that the phenomenon is robust to syndication filtering, while precisely bounding the claim: the replication represents a **prospective frozen-protocol replication** under exact key exclusion, not 120 entirely novel underlying news texts.

---

## 4. Audited Mathematical Measurement Operator

All evaluations evaluate the audited paired contrast operator:
$$\Delta s_k(x) = s_A^{(k)}(x) - s_B^{(k)}(x)$$
where for each hypothesis pair $k \in \{1, \dots, 8\}$:
$$s_A^{(k)}(x) = P_A^{(k)}(\text{Entailment}) - P_A^{(k)}(\text{Contradiction})$$
$$s_B^{(k)}(x) = P_B^{(k)}(\text{Entailment}) - P_B^{(k)}(\text{Contradiction})$$

Model label mapping for `microsoft/deberta-v2-xlarge-mnli`:
- `Index 0`: CONTRADICTION
- `Index 1`: NEUTRAL
- `Index 2`: ENTAILMENT

Evaluating $\Delta s_k(x)$ performs paired subtraction, removing response components shared between opposing hypotheses and isolating pure differential stance contrast.

---

## 5. Statistical Decision Gates

| Metric | Target / Gate | Confirmatory Criterion | Outcome Interpretation |
| :--- | :--- | :--- | :--- |
| **$Q_0$ PC1 Permutation Gate** | $10{,}000$ publisher draws | $\mathrm{PC1}_{\mathrm{obs}} > q_{95}$ ($p < 0.05$) | Confirmatory replication of dominant stance axis |
| **$Q_1$ PC1 Permutation Gate** | $10{,}000$ publisher draws | $\mathrm{PC1}_{\mathrm{obs}} > q_{95}$ ($p < 0.05$) | Confirmatory replication of paraphrase invariance |
| **$Q_0 \leftrightarrow Q_1$ Stance Correlation** | Pearson $r$ and Spearman $\rho$ | $\|r\| \ge 0.70$, $\|\rho\| \ge 0.70$ ($p < 0.001$) | Article ordering robust to lexical phrasing |
| **Disjoint Partition Invariance** | 35 complementary 4-vs-4 splits | Mean $\|r\| \ge 0.70$ across splits | Ordering is bank-level construct, not single-item artifact |
| **Hidden-State Axis Coherence** | $735{,}471$ subset upper tail | Reported descriptively | Documents construct separation without claiming null proof |
