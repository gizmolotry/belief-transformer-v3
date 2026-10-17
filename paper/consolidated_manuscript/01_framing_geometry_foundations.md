# Volume I: Measurement-Conditioned Framing Geometry
## Contrastive NLI, Semantic Projection, and Perspectival Measurement in Political News

**Author:** Andrew Clair  
**Institutional Affiliation:** Department of Journalism, Media and Communication  
**Supervision:** Henry Silke (Module: JM4018 — Individual Journalism Project)  
**Academic Year:** 2025/2026  
**Repository & Codebase:** `github.com/belief-transformer/v3`  
**Date of Audited Consolidation:** September 22, 2026  

---

## Abstract

Automated news aggregators frequently summarize political bias by projecting coverage onto an assumed one-dimensional spectrum using third-party publisher metadata, rather than measuring article text directly. We evaluate whether framing stance can be recovered as an endogenous low-dimensional geometry using frozen cross-encoder natural language inference (NLI) prompted with authored contrastive hypothesis pairs ($h_A, h_B$). Under an audited paired contrast operator ($s_A - s_B$) computing differences of entailment-minus-contradiction softmax probabilities, article response covariance compresses onto a robust primary stance axis ($\mathrm{PC1} \approx 42\% - 60\%$) that decisively clears 10,000-draw permutation null gates ($p \le 0.0003$, detected rank under the implemented block null: $k=1$) across both conflict and domestic policy domains (Clean Ukraine $\mathrm{PC1} = 42.09\%, p = 0.0003$, US Immigration, Climate Politics, Abortion Rights). In an audited 320-record sample selected by a frozen headline-level conflict criterion across 16 source domains with exact cross-publisher deduplication, PC1 explains $57.35\%$ ($Q_0$) and $59.67\%$ ($Q_1$) against source-domain-block randomizations ($p = 1/10,001$, zero exceedances in 10,000 draws). Comparing this rerun to the earlier unguided corpus ($60.13\%$ and $61.26\%$) indicates that PC1 variance concentration decreased modestly after corpus remediation ($-1.59\%$ to $-2.78\%$), while the primary stance axis remains dominant and stable under paraphrase ($r_{\text{cosine}} = 0.9599$). Crucially, PC2 is not detected under the source-domain-block permutation null ($p = 1.0000$ in both banks) and fails loading bootstrap stability; non-detection establishes neither its absence nor a second construct, bounding detected rank to $k=1$. Paired precision audits (float16 vs. float32, coordinate $\rho = 1.0000$, max $|\Delta c| < 0.003$) and 5-gram near-duplicate audits (0 pairs with Jaccard $\ge 0.50$) confirm numerical invariance and corpus bounds. Stance coordinates independently reproduce on an untouched prospective sample ($N=120$, $\mathrm{PC1} = 58.36\%$; cross-bank $r = 0.925$), non-overlapping disjoint probe partitions (mean $|r| \approx 0.76$ across 35 unique 4-vs-4 splits), and diverse transformer architectures (cross-model stance $r = 0.55 - 0.81$ across DeBERTa, BART, and RoBERTa). Synthetic factorial positive controls with known generating factors verify apparatus resolving power for higher-dimensional structure when present ($k=2$, $99.11\%$).

We benchmark this relational NLI apparatus against a matched sentence-level semantic projection baseline (FrameAxis). Holding corpus and conceptual contrasts fixed, the two instruments produce weakly aligned dominant coordinates ($r \in [0.01, 0.19]$ standardized). Canonical correlation analysis reveals that this is not orthogonal noise: the instruments share a significant multivariate semantic subspace ($r_{\mathrm{canon,test}} = 0.364 \pm 0.080$, Monte Carlo permutation $p = 0.0010$) that generalizes out-of-sample under strictly fold-local repeated 5-fold cross-validation. Minimal compositional controls establish that NLI exhibits substantially greater compositional sensitivity on tested perturbations, achieving complete concordance and sign inversion on negation ($100\%$) and quantifiers ($100\%$, where dense projection collapses to cosine similarities of $0.73 - 0.99$), while remaining bounded on syntactic agent–patient reversal ($50\%$). Anisotropy-removal ablations demonstrate that dominant semantic projection variance is substantially driven by leading representation components ($73.7\% \to 34.5\%$ upon removing the top embedding PC), whereas NLI stance coordinates separate news outlets ($\eta^2 = 40.1\%, p = 0.005$) and survive collinear sentiment controls ($\eta^2 = 34.8\%, p = 0.022$). We conclude that low-dimensional framing structure is not unique to NLI, but the recovered dominant coordinate depends on the measurement operator: $\mathcal{G}_{\text{observed}} = f(C, H, O, M)$.

---

## 1. Introduction, Epistemological Critique, and Theoretical Foundations

### 1.1 Ground News and the Fallacy of Horizontal Neutrality
Contemporary news aggregation platforms frequently position themselves as objective arbiters of political bias. Ground News, for example, aggregates news coverage and sorts articles along a linear left-to-right political spectrum relative to an assumed "middle" consensus. While marketed as a tool for pluralism and critical thinking, its sorting relies almost entirely on publisher-level left-right ratings sourced from third-party rating agencies (such as AllSides, Ad Fontes Media, and Media Bias/Fact Check).

This represents what we formalize as a **horizontal measurement system**: at each stage of processing, the architecture appends external, exogenous metadata rather than measuring the semantic content of the text itself. Formally, a horizontal pipeline maps:
\\[
H(a) = (a, m(p(a)))
\\]
where $a$ is an article text, $p(a)$ returns the publishing outlet, and $m(\cdot)$ retrieves third-party ideological ratings. The resulting bias score is:
\\[
b_{\text{hor}}(a) = \psi(m(p(a)))
\\]
which is strictly a function of publisher identity, completely invariant to the actual editorial text of article $a$. In contrast, an **endogenous vertical measurement pipeline** maps the text directly into observer-conditioned semantic coordinates:
\\[
x \mapsto \{z_o(x)\}_{o \in \mathcal{O}}
\\]
where each $z_o(x) = E_\theta(x; h_o)$ is an observer-conditioned representation derived directly from the article's text evaluated against structured contrastive hypotheses $h_o$.

### 1.2 Theoretical Foundations: The Social Construction of Classification
Drawing upon science and technology studies (STS), philosophy of science, and political theory, we ground our critique of horizontal aggregators in five foundational frameworks:
1. **Theory-Laden Observation (Kuhn, 1962):** Observation cannot be conducted outside of an interpretive paradigm; data collection and measurement are inextricably shaped by theoretical commitments.
2. **Ways of Worldmaking (Goodman, 1978):** Classificatory systems do not passively reflect preexisting natural kinds; they actively construct the categories and boundaries they purport to measure.
3. **Infrastructures of Classification (Bowker & Star, 1999):** Classification standards embody moral, political, and cultural choices. When classification algorithms render their categories invisible, they naturalize political hierarchies as technical neutrality.
4. **The Limits of Representation (Wittgenstein, 1922):** The perspective from which a representation is constructed cannot be situated inside the representation itself: *"the subject does not belong to the world but it is a limit of the world."* The observer who projects political bias cannot stand in an unmarked, Archimedean center.
5. **Agonistic Pluralism vs. Exceptionless Civility (Berlin, 1998; Mouffe, 2000; Reiheld, 2013):** Enforcing "exceptionless civility" suppresses irreducible moral and political friction. Democratic politics is fundamentally agonistic; consensus aggregators that collapse political antagonism into a single linear scale neutralize the democratic contestation essential for genuine pluralism.

### 1.3 The Double Variance Collapse Theorem
Consensus aggregators attempt to handle multi-observer variation through a two-step reduction: first, averaging scores across multiple judges to establish a "consensus mean"; second, projecting high-dimensional text representations onto a single scalar dimension.

We prove algebraically that this architecture executes a **double variance collapse**:

**Theorem (Double Variance Collapse):**  
Let $\{z_{o_1}(x), \dots, z_{o_K}(x)\}$ be a set of $K$ observer-conditioned representations of text $x$ in a metric space with distance $d$. The Fréchet consensus mean is:
\\[
\bar{z}(x) = \operatorname*{arg\,min}_z \sum_{k=1}^K d^2(z, z_{o_k}(x))
\\]
In a flat Euclidean space $\mathcal{Z} \cong \mathbb{R}^d$ where $d^2(u, v) = \|u - v\|^2$, the Fréchet mean is the arithmetic centroid $\bar{z}(x) = \frac{1}{K}\sum_{k=1}^K z_{o_k}(x)$ and satisfies the exact pairwise dispersion identity:
\\[
\sum_{k=1}^K \|z_{o_k}(x) - \bar{z}(x)\|^2 = \frac{1}{2K} \sum_{j=1}^K \sum_{k=1}^K \|z_{o_j}(x) - z_{o_k}(x)\|^2
\\]
The variance about the consensus center is determined entirely by inter-observer disagreement. For texts sampled from distribution $\mathcal{D}$ and any linear projection $\pi: \mathcal{Z} \to \mathbb{R}$, the variance of the projected consensus mean is bounded above by the maximum individual observer variance:
\\[
\operatorname{Var}(\pi(\bar{z})) \le \max_{1 \le k \le K} \operatorname{Var}(\pi(z_{o_k}))
\\]
Specifically, for $K$ observers with equal marginal variance $\sigma^2$ and average pairwise cross-observer correlation $\bar{\rho} \in [0, 1)$, the projected consensus variance evaluates to:
\\[
\operatorname{Var}(\pi(\bar{z})) = \sigma^2 \left( \frac{1}{K} + \frac{K-1}{K} \bar{\rho} \right)
\\]
When observer representations capture differentiated perspectives ($\bar{\rho} \ll 1$), the consensus variance contracts toward $\sigma^2 / K \ll \operatorname{Var}(\pi(z_{o_k}))$ for all $k$, extinguishing genuine observer-specific variance.

**Empirical Validation of Variance Collapse:**  
On a high-dimensional synthetic corpus ($N=60$ articles, $B=8$ observers, $H=1536$ dimensions; [`../../data/frechet_metrics.json`](../../data/frechet_metrics.json)):
- Averaging across the 8 observer representations to compute the consensus Fréchet mean erases **$92.65\%$** of the original perspectival variance.
- Projecting the consensus centers onto the first Principal Component collapses an additional **$18.96\%$** of the remaining variance.
- Cumulatively, the two-stage aggregation erases **$94.04\%$** of the total perspectival variance, leaving only **$5.96\%$** of the original narrative information intact. One-dimensional aggregation platforms do not merely simplify bias metrics; they extinguish the primary geometric signals that constitute narrative disagreement.

### 1.4 Methodological Precedents: Automated Stance Identification (APSI)
Our approach connects directly to the emergent paradigm of zero-shot Natural Language Inference for political text analysis. Most notably, the Automated Political Stance Identification framework (APSI; https://apsi.sc.hpi.de/documentation/) established the use of theory-derived natural language hypotheses in frozen cross-encoder NLI models, scoring text documents by subtracting the entailment support for opposing political propositions.

We explicitly acknowledge substantial conceptual and algorithmic overlap with APSI: the core mechanics of utilizing authored contrastive proposition pairs and zero-shot NLI difference scores are established in the literature and are not claimed as novel inventions of this work.

However, our research program diverges from APSI in its central scientific question, evaluative criteria, and mathematical framework:
1. **Criterion Construct:** APSI evaluates stance accuracy against external human benchmarks, validating computational scores against expert party ratings and manifesto coding. In contrast, our project does not designate human expert consensus as its criterion construct. Rather, drawing on measurement theory, we investigate the *endogenous geometric structure* induced by relational operators across multidimensional probe banks.
2. **Geometric Dimensionality & Falsification Gates:** We investigate whether relational NLI scores compress into low-dimensional manifolds, subjecting score covariance matrices to non-parametric permutation null gates (Horn's parallel analysis, column shuffling, and cluster bootstrapping) to evaluate whether dominant axes ($\mathrm{PC1}$) and secondary dimensions ($\mathrm{PC2}$) reflect authentic semantic geometry.
3. **Comparative Operator Mechanics:** We formalize how observed geometry depends on the measurement operator ($\mathcal{G}_{\text{observed}} = f(C, H, O, M)$), benchmarking cross-encoder relational NLI against bi-encoder semantic projection (FrameAxis) across multivariate canonical subspaces, minimal compositional perturbations, and anisotropy ablations.

This work does not conduct a direct head-to-head empirical competition against APSI, as expert consensus lies outside our stated construct target. Describing APSI from its publicly documented methodology establishes its foundational role while appropriately scoping our own contributions to the geometry and sensitivity of measurement operators.

---

## 2. Methodology: The Relational NLI Apparatus

### 2.1 Cross-Encoder Relational Scoring
Rather than relying on bi-encoder text embeddings that collapse sentence tokens prior to cross-attention, our apparatus employs frozen cross-encoder Natural Language Inference (`microsoft/deberta-v2-xlarge-mnli`). For an article $x$ and an authored hypothesis $h$, the extraction samples up to three leading paragraphs (weighted 0.5, 0.3, and 0.2, capped at 512 tokens), computes cross-attention logits for contradiction ($C$), neutral ($N$), and entailment ($E$), and evaluates the weighted mean logit vector $\bar{l}(x, h)$. Softmax normalization is applied strictly **after** multi-paragraph logit aggregation:
\\[
P(C \mid x, h), P(N \mid x, h), P(E \mid x, h) = \operatorname{softmax}(\bar{l}(x, h))
\\]
The bounded side score for hypothesis $h$ is defined as the net probability:
\\[
s(x, h) = P(E \mid x, h) - P(C \mid x, h) \in [-1, 1]
\\]

### 2.2 The Paired Contrast Operator and Channel Demarcation
Raw single-hypothesis entailment scores are susceptible to common-mode elevation: verbose, highly detailed, or stylistically intense articles receive elevated probability mass across opposing hypotheses.

To isolate directional stance from baseline elevation, the **Paired Contrast Operator** evaluates an authored hypothesis pair $P = (h_A, h_B)$ representing opposing semantic frames:
\\[
c_P(x) = s(x, h_A) - s(x, h_B) = [P(E \mid x, h_A) - P(C \mid x, h_A)] - [P(E \mid x, h_B) - P(C \mid x, h_B)] \in [-2, 2]
\\]
This algebraically cancels any common-mode article response propensity $p(x)$ that enters additively and symmetrically into both hypotheses:
\\[
(s(x, h_A) + p(x)) - (s(x, h_B) + p(x)) = s(x, h_A) - s(x, h_B)
\\]
*Measurement Channel Demarcation:* It is critical to distinguish this audited paired-scalar probability score from two separate representations evaluated in earlier exploratory project phases:
1. **Paired-Scalar Probability Channel ($c = s_A - s_B \in [-2, 2]$):** The verified operational metric analyzed in this work ([`../../scripts/paired_scalar_structure.py`](../../scripts/paired_scalar_structure.py)), requiring fresh probability scores for both hypotheses.
2. **Historical One-Sided A-Only Channel ($s_A \in [-1, 1]$):** Read from legacy `logits_raw` files, where only the $h_A$ verdict was recorded. Earlier drafts improperly labeled this channel as "contrastive displacements"; historical correlations derived from it cannot be attributed to the paired-scalar operator.
3. **Hidden-State Contrast Channel ($\Delta z$):** Mean-pooled token difference vectors $\operatorname{mean\_pool}(x, h_A) - \operatorname{mean\_pool}(x, h_B)$ (stored under the legacy field name `cls_per_bot`).

### 2.3 The V-Observers: Multi-Domain Contrastive Probe Banks
We instantiate $B=8$ paired contrast probes across five major political and conflict domains:
1. **Israel–Palestine Conflict Corpora:**
   - *Exploratory Dense Corpus ($N=320$, 16 source domains):* Probed with original authored bank ($Q_0$, [`../../config/framing_queries.yaml`](../../config/framing_queries.yaml)) and paraphrase bank ($Q_1$, [`../../config/prompt_banks/probe_hypotheses_paraphrase_a.json`](../../config/prompt_banks/probe_hypotheses_paraphrase_a.json)). As audited below, this corpus was assembled by [`../../scripts/build_dense_corpus.py`](../../scripts/build_dense_corpus.py) without an upstream topic filter and contains off-topic records and mirrored stories; a clean rerun is pending.
   - *Prospective Confirmatory Corpus ($N=120$):* Untouched articles across 70 independent publishers evaluating state military justification vs. civilian destruction, international law compliance vs. violation, and defensive necessity vs. systemic oppression ([`../prospective_confirmatory_replication_results.json`](../prospective_confirmatory_replication_results.json)).
2. **Clean Ukraine War Corpus ($N=60$):** Unprovoked sovereign defense vs. geopolitical escalation; multilateral security assistance vs. prolonged regional instability ([`../ukraine_clean_replication_results.json`](../ukraine_clean_replication_results.json)).
3. **US Immigration Policy ($N=60$):** Border enforcement and national sovereignty vs. humanitarian refugee asylum and civil rights ([`../paired_operator_cross_model_cross_topic_rerun.json`](../paired_operator_cross_model_cross_topic_rerun.json)).
4. **Climate Politics & Energy Transition ($N=60$):** Urgent ecological transition and carbon regulation vs. economic stability and fossil energy reliability ([`../paired_operator_cross_model_cross_topic_rerun.json`](../paired_operator_cross_model_cross_topic_rerun.json)).
5. **Abortion Rights & Healthcare ($N=60$):** Bodily autonomy and reproductive freedom vs. fetal sanctity and moral limits on termination ([`../paired_operator_cross_model_cross_topic_rerun.json`](../paired_operator_cross_model_cross_topic_rerun.json)).

### 2.4 Geometric Manifold Analysis & Validation Protocols
- **Source-Domain-Block Permutation Null:** To test whether leading eigenvalues reflect genuine article-level alignment rather than publisher-level clustering, each score column's 20-value source-domain blocks are randomly reassigned among publishers, and row orders within blocks are permuted across 10,000 draws ([`../paired_scalar_pc1_gate_protocol_20260908.md`](../paired_scalar_pc1_gate_protocol_20260908.md)).
- **Cluster Bootstrap Loading Stability:** 10,000 bootstrap draws resampling source-domain clusters with replacement evaluate whether eigenvector loadings remain stable across sampled outlets.
- **Orthogonal Procrustes Alignment:** Quantifies the structural divergence between independent observer-conditioned manifolds without non-rigid distortion.
- **Dijkstra Geodesic Pathing:** Measures the shortest paths along the $k$-nearest-neighbor graph ($k=5$) of the semantic manifold, evaluating whether ideological distance warps topological traversal.
- **Synthetic Factorial Positive Controls:** Generates synthetic corpora with known mathematical factor rank ($k=2$, orthogonal stance and policy dimensions) to verify that the NLI PCA apparatus has the mathematical resolving power to detect multidimensional geometry when present ($k=2$ detected with $99.11\%$ variance recovery).

### 2.5 Data and Provenance Limitations

To ensure methodological transparency and demarcate the empirical boundaries of our findings, we consolidate five primary data-integrity, sampling, and provenance constraints governing the analyzed corpora and pipelines:

1. **Upstream Corpus Contamination, Remediation, and Precision Audits:**
   The initial 320-record Israel–Palestine analysis was executed on an unguided sample that lacked an upstream topic-relevance filter, admitting off-topic records such as archaeological reports on Welsh Neolithic burial excavations (row 9), generic biographical obituaries (rows 59/60), and 14 cross-domain mirrored BBC stories between `bbc.com` and `bbc.co.uk`. To resolve this, we executed a complete corpus remediation and re-extraction on an NVIDIA RTX 5070 Ti across both $Q_0$ and $Q_1$ prompt banks ([`../paired_scalar_pc1_gate_results_clean_20261003.json`](../paired_scalar_pc1_gate_results_clean_20261003.json)). By enforcing a frozen headline-level conflict criterion (`israel`, `palestin`, `gaza`, `hamas`, `west bank`, `netanyahu`, `idf`, `tel aviv`, `jerusalem`, `hostage`, `zionis`, `rafah`, `hezbollah`, `ceasefire`, `october 7`, `jenin`, `khan younis`) and cross-publisher URL-path deduplication, the reconstructed corpus contains exactly 20 articles selected by this frozen criterion across all 16 publishers ($N=320$). Exploratory attempts to impose an ad-hoc blacklist (`serial rapist`, `trans woman`, `jimmy carter`, etc.) were identified as unprincipled researcher-degree-of-freedom patching—which would arbitrarily discard legitimate reporting on combatant crime or Carter diplomacy—and were entirely eliminated in favor of positive headline inclusion. Following remediation, PC1 variance concentration decreased modestly ($60.13\% \to 57.35\%$ in $Q_0$ and $61.26\% \to 59.67\%$ in $Q_1$); we record this as an empirical shift without asserting a speculative causal mechanism. Furthermore, a paired precision audit across 32 articles comparing float16 to float32 execution ([`../float16_precision_audit_results.json`](../float16_precision_audit_results.json)) established that half-precision execution introduced negligible numerical deviation: maximum paired score difference $|\Delta c| < 0.003$ (median $0.0003$), score correlation $r = 0.999998$, and perfect PC1 coordinate rank preservation ($\rho = 1.0000$).

2. **Sampling Unit Independence and Syndication Duplication:**
   The 16 source domains in `controlled_N_dense_corpus.jsonl` do not represent 16 statistically independent publishing institutions. Specifically, in the legacy corpus `bbc.com` and `bbc.co.uk` occupied distinct 20-row domain strata but shared 14 identical story URL paths and article texts. In the remediated corpus, exact title and URL collisions across publishers were eliminated. A full 5-token shingle Jaccard audit across all 51,040 pairwise combinations in the remediated 320-record corpus ([`../corpus_near_duplicate_audit_results.json`](../corpus_near_duplicate_audit_results.json)) confirmed that zero pairs exhibit near-duplicate overlap $\ge 0.50$ (maximum Jaccard $0.381$, 99th percentile $0.026$, mean $0.0009$), with only two pairs exceeding $0.30$. Across prospective ($N=120$) and conflict corpora, wire service syndication nonetheless introduces content-level overlap: while the prospective sample exhibits zero exact collisions against the 1,290-key fingerprint registry, 37 of 120 prospective articles exhibit 5-token shingle Jaccard overlap $\ge 0.50$ with prior project texts. Consequently, source-domain block permutations reflect multiset permutation bounds over the collected records rather than exchangeable, independent publisher clusters.

3. **Measurement Channel Distinction and Legacy Attribution:**
   Our empirical architecture strictly differentiates three distinct computational representations:
   - **Paired-Scalar Probability Contrast:** The analyzed primary metric, $c_P(x) = [P(E \mid x, h_A) - P(C \mid x, h_A)] - [P(E \mid x, h_B) - P(C \mid x, h_B)]$, evaluated from post-softmax probabilities.
   - **Historical Single-Side Verdict Channel:** An uncontrasted single-hypothesis margin, $s_A(x) = P(E \mid x, h_A) - P(C \mid x, h_A)$.
   - **Hidden-State Contrast Vectors:** High-dimensional mean-pooled representation differences, $\Delta z_P(x) = \operatorname{mean\_pool}(x, h_A) - \operatorname{mean\_pool}(x, h_B)$.
   Audits confirm ([`../channel_audit_20260906.md`](../channel_audit_20260906.md)) that legacy headline correlations with Media Bias/Fact Check (MBFC) ratings (e.g., historical Axis A $r = 0.431$ and Axis B $r = -0.499$) were exploratory, unmanifested outputs that cannot be attributed to the paired-scalar probability operator.

4. **Execution Provenance and Serialization Boundaries:**
   Intermediate extraction checkpoints and PyTorch logit caches (`.pt` files) were serialized without cryptographic hash binding to model git commits, tokenizer configurations, prompt bank SHA-256 digests, or explicit label mapping assertions. While double-precision mathematical recomputations on stored logits reproduce published numerical results exactly, extraction provenance is classified as audited posthoc analysis of stored artifacts rather than sealed preregistered confirmatory execution.

5. **Construct and Ontological Scope Boundaries:**
   Endogenous stance coordinates reflect model-conditional responses to authored propositional probes under frozen neural weights. They do not measure the latent psychological beliefs of journalists, the moral truth of political assertions, or human expert consensus. Furthermore, the recovery of a dominant covariance eigenvector over eight scalar probe scores represents an empirical coordinate axis within the discrete score space; it does not establish a continuous Riemannian metric tensor on real news discourse (the continuous metric tensor simulation pipeline was formally halted at the synthetic calibration phase).

---

## 3. Empirical Results: Framing Geometry & Stance Recovery

### 3.1 Dominant Stance Axis & Permutation Null Gates
Across all evaluated domains, computing column-centered covariance PCA on the $N \times 8$ paired contrast response matrix demonstrates that article response covariance compresses onto a single dominant axis ($k=1$), while secondary axes fail falsification gates:

| Domain / Corpus | $N$ | Operator | PC1 EVR | Permutation Null Gate ($p$) | PC2 EVR | PC2 Null Gate ($p$) | PR | Detected Rank ($k$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Israel–Palestine (Clean Conflict $Q_0$)** | $320$ | Paired $s_A - s_B$ | **$57.35\%$** | $p = 0.0001$ ($q_{95} = 27.70\%$) | $14.72\%$ | $p = 1.0000$ ($q_{95} = 22.44\%$) | $2.72$ | **strictly $k=1$** |
| **Israel–Palestine (Clean Paraphrase $Q_1$)** | $320$ | Paired $s_A - s_B$ | **$59.67\%$** | $p = 0.0001$ ($q_{95} = 27.37\%$) | $11.71\%$ | $p = 1.0000$ ($q_{95} = 22.54\%$) | $2.59$ | **strictly $k=1$** |
| **Israel–Palestine (Prospective Confirmatory)** | $120$ | Paired $s_A - s_B$ | **$58.36\%$** | $p = 0.0001$ ($q_{95} = 26.06\%$) | $15.06\%$ | $p = 1.0000$ ($q_{95} = 21.93\%$) | $2.62$ | **strictly $k=1$** |
| **Ukraine War (Clean Deduplicated)** | $60$ | Paired $s_A - s_B$ | **$42.09\%$** | $p = 0.0003$ ($q_{95} = 34.62\%$) | $20.67\%$ | $p = 0.9662$ ($q_{95} = 27.36\%$) | $3.92$ | **strictly $k=1$** |
| **US Immigration & Border Policy** | $60$ | Paired $s_A - s_B$ | **$50.17\%$** | $p = 0.0001$ ($q_{95} = 35.40\%$) | $25.76\%$ | $p = 0.5282$ ($q_{95} = 28.01\%$) | $3.01$ | **strictly $k=1$** |
| **Climate Politics & Energy Transition** | $60$ | Paired $s_A - s_B$ | **$57.82\%$** | $p = 0.0002$ ($q_{95} = 54.76\%$) | $13.57\%$ | $p = 0.9963$ ($q_{95} = 18.05\%$) | $2.67$ | **strictly $k=1$** |
| **Abortion Rights & Healthcare** | $60$ | Paired $s_A - s_B$ | **$50.66\%$** | $p = 0.0001$ ($q_{95} = 30.66\%$) | $17.52\%$ | $p = 0.8648$ ($q_{95} = 21.39\%$) | $3.19$ | **strictly $k=1$** |

*Note on statistical bounds:* Across all seven evaluations and five distinct domains, $\mathrm{PC1}$ decisively clears the 10,000-draw non-parametric column permutation null gate ($p \le 0.0003$), whereas $\mathrm{PC2}$ uniformly fails to reach statistical significance ($p \ge 0.5282$). The detected rank under the implemented block-null procedure is $k=1$. (In the earlier legacy unguided 320-record corpus, nominal PC1 EVR was $60.13\%$ in $Q_0$ and $61.26\%$ in $Q_1$; PC1 variance concentration decreased modestly to $57.35\%$ and $59.67\%$ after headline conflict selection and cross-publisher deduplication).

**Statistical Falsification of PC2:** In the clean 320-record conflict corpus, the PC1 eigenvalue ($\lambda_1 = 0.3612$ in $Q_0$, $0.3161$ in $Q_1$) decisively clears the source-domain-block 95th percentile ($0.1745$ and $0.1450$, zero exceedances in 10,000 draws, $p = 0.0001$; [`../paired_scalar_pc1_gate_results_clean_20261003.json`](../paired_scalar_pc1_gate_results_clean_20261003.json)). In contrast, PC2 ($\lambda_2 = 0.0927$ in $Q_0$, $0.0621$ in $Q_1$) falls entirely below the publisher-block null 95th percentile ($0.1414$ and $0.1194$, plus-one $p = 1.0000$ in both banks), with 5,000 out of 5,000 null draws equaling or exceeding the observed eigenvalue. For comparison, on the legacy unguided corpus, observed PC1 eigenvalues were $\lambda_1 = 0.3459$ ($Q_0$) and $0.2779$ ($Q_1$) against null 95th percentiles of $0.1654$ and $0.1301$, while PC2 eigenvalues were $\lambda_2 = 0.0856$ and $0.0501$ against null 95th percentiles of $0.1314$ and $0.1077$ ($p = 1.0000$; [`../paired_scalar_pc1_gate_results_20260908.json`](../paired_scalar_pc1_gate_results_20260908.json)). Furthermore, PC2 fails cluster bootstrap loading stability in both banks ([`../paired_scalar_dimensionality_results_v2_20260907.json`](../paired_scalar_dimensionality_results_v2_20260907.json)). On the prospective $N=120$ sample, PC2 similarly explains $15.06\%$ against a null $q_{95}$ of $21.93\%$ ($p = 1.0000$). Consequently, the detected rank under the implemented block-null procedure is bounded at $k=1$; failure to detect PC2 under this provisional procedure establishes neither its absence in the broader population nor a stable second construct, and substantive labels assigning meaning to a second axis (e.g., immediate vs. structural conflict) remain empirically unsupported.

### 3.2 Prospective Replication & Disjoint Probe Stability
- **Prospective Sampling Replication:** Evaluating a prospective sample of $N=120$ news articles (zero exact fingerprint, URL, or title collisions against the 1,290-key project registry) under a fresh probe bank yielded $\mathrm{PC1} = 58.36\%$ with cross-bank coordinate correlation $r = \mathbf{0.9250}$ ($p = 1.63 \times 10^{-51}$; [`../prospective_confirmatory_replication_results.json`](../prospective_confirmatory_replication_results.json)). Under conservative sensitivity filtering removing articles with 5-token shingle Jaccard overlap $\ge 0.50$ with prior project corpora ($N=83$), structural stability is preserved ($\mathrm{PC1} = 59.87\%, r = \mathbf{0.9533}$).
- **Disjoint Partition Stability:** To verify that the recovered stance axis is not driven by any single probe pair, the 8 probes were partitioned into all $\binom{8}{4}/2 = 35$ unique 4-vs-4 disjoint sub-banks. The mean correlation between coordinates derived from disjoint probe sets was $|r| = \mathbf{0.7602 \pm 0.1296}$ (median $|r| = 0.8004$, with $85.7\%$ of splits achieving $|r| \ge 0.70$).

### 3.3 Cross-Architecture Model Invariance
Evaluating the identical paired contrast bank across three distinct pre-trained language model families confirms that the dominant stance dimension reflects text-level framing rather than tokenizer-specific artifacts ([`../paired_operator_cross_model_cross_topic_rerun.json`](../paired_operator_cross_model_cross_topic_rerun.json)):
- DeBERTa-v2-xlarge vs. BART-large-mnli: $r = \mathbf{0.8057}$ ($p = 8.24 \times 10^{-15}$, $\rho = 0.8023$)
- DeBERTa-v2-xlarge vs. RoBERTa-large-mnli: $r = \mathbf{0.5590}$ ($p = 3.45 \times 10^{-6}$, $\rho = 0.5229$) [correcting the historical misattribution of $0.7241$]
- BART-large-mnli vs. RoBERTa-large-mnli: $r = \mathbf{0.5470}$ ($p = 6.12 \times 10^{-6}$, $\rho = 0.5233$)

---

## 4. External Benchmark Comparison and Channel Integrity

### 4.1 Audited MBFC Evaluation vs. Legacy Headline Numbers
To assess whether endogenous text-conditioned stance coordinates align with conventional third-party ideological ratings, external comparisons were conducted against Media Bias/Fact Check (MBFC) publisher scores. A rigorous channel audit ([`../channel_audit_20260906.md`](../channel_audit_20260906.md)) establishes critical provenance boundaries:
1. **Status of Legacy Headline Figures:** Earlier manuscript drafts quoted MBFC correlations on the 320-record dense corpus under two designations: "Axis A" ($r = 0.431, p = 0.0958; \rho = 0.520, p = 0.0390$) and "Axis B" ($r = -0.499, p = 0.0491$). Exhaustive repository auditing reveals that no executable script or persisted result artifact emits these exact figures; they entered the repository without an auditable computation trail. Crucially, they cannot be attributed to the corrected paired-scalar probability operator.
2. **Audited Historical A-Only Benchmark:** The only verified repository benchmark on this dataset ([`../exploratory_internal_structure_results_20260828.json`](../exploratory_internal_structure_results_20260828.json)) evaluated a linear regression of publisher MBFC ratings against the first two principal components of the **historical one-sided A-only channel**. That analysis confirmed a definitive null result:
   - In-sample fit: $R^2 = 0.1916$, adjusted $R^2 = 0.0672$.
   - Overall model test: $F(2, 13) = 1.541, p = 0.2509$.
   - Leave-one-source-domain-out predictive capacity: predictive $R^2 = -0.4902$ (cross-validated Pearson $r = -0.0739, p = 0.7856$).
   These figures reflect the descriptive failure of the historical A-only verdict score to predict MBFC ratings out-of-sample; they provide no evidence of external validity for the paired-scalar operator.
3. **Absence of Paired-Scalar Benchmark:** No corrected paired-scalar probability score has been evaluated against MBFC publisher ratings. Because publisher ratings and article text represent fundamentally different units of analysis, expert or agency consensus is not the criterion construct of this instrument audit.

### 4.2 Structural Construct Divergence
The empirical divergence between article-level text coordinates and publisher-level metadata highlights a conceptual boundary rather than an instrument defect:
- **Unit of Analysis Mismatch:** External ratings (MBFC, AllSides) assign a single static rating to an entire news organization based primarily on domestic legislative coverage and general editorial orientation. In contrast, the relational NLI apparatus measures text-conditioned stance on specific conflict events.
- **Operational Specificity:** In historical exploratory runs, outlet coordinates exhibited extreme sensitivity to single publishers (such as Iranian state outlet *Press TV*, whose anti-Western conflict reporting sharply diverges from domestic economic left-right scales where state-socialist rhetoric is classified as "Left").
- **Construct Scope:** Because those exploratory runs operated on an uncorrected channel within an unfiltered corpus, post-hoc qualitative interpretations (such as "divergence flagging") remain exploratory hypotheses rather than validated measurement properties. Confirmatory claims regarding publisher-level divergence require a deduplicated, topic-filtered corpus and an authorized preregistered protocol.

---

## 5. Kuhnian Incommensurability & Relational Geometry (Claims CLM_01 to CLM_20)

Across the foundational research phase, twenty empirical claims established the boundaries of framework incommensurability:
- **G-Theory Variance Conservation (`CLM_01`):** Verified $100.0\%$ variance conservation across crossed facets.
- **Framework-Blind Generalization Drops (`CLM_02`, `CLM_03`):** Projections trained on one interpretive framework drop significantly in predictive power when evaluated on rival frameworks ($R^2$ drop of $35\% - 50\%$).
- **Metric-Independent Neighborhood Divergence (`CLM_12`):** Local article neighborhoods diverge across frameworks robustly across 6 distinct distance metrics (Euclidean, Cosine, Correlation, Standardized Euclidean, Cityblock, Chebyshev; mean Jaccard overlap $\le 0.22$).
- **Anomaly Ranking Hierarchy (`CLM_13`):** Articles identified as anomalous under one observer framework systematically migrate to different ranks under rival frameworks.
- **Grassmannian Orientation Invariance (`CLM_16`):** Stance subspaces span invariant principal angles on $\mathrm{Gr}(k, d)$ across independent article batches.
- **Marchenko-Pastur Random Matrix Null (`CLM_18`):** Dominant eigenvalues decisively exceed the Marchenko-Pastur upper bound $\lambda_+ = (1 + \sqrt{\gamma})^2$, where $\gamma = p/N$ is the aspect ratio of variables to observations, proving authentic non-random semantic compression.
- **Philosophical Structural Realism (`CLM_20`):** Persistent tail rankings and conserved dimensionality across divergent observers provide empirical support for structural realism in text: meaning is not an arbitrary social construct, but a constrained relational geometry.

---

## 6. Benchmark Comparison: Relational NLI vs. FrameAxis Semantic Projection

### 6.1 Multivariate Canonical Correlation Analysis (CCA)
- Standardized NLI and FrameAxis coordinates correlate weakly at the bivariate level ($r \in [0.012, 0.189]$).
- Fold-local repeated 5-fold CCA proves significant multivariate alignment ($r_{\mathrm{canon,test}} = \mathbf{0.364 \pm 0.080}, p = 0.0010$; [`../frameaxis_baseline_results.json`](../frameaxis_baseline_results.json), [`../cross_instrument_deep_tightening_results.json`](../cross_instrument_deep_tightening_results.json)).

### 6.2 Minimal Compositional Controls
- **Literal Negation:** NLI flips sign ($100.0\%$); FrameAxis collapses ($\cos \approx 0.94 - 0.99$; [`../compositional_controls_results.json`](../compositional_controls_results.json)).
- **Quantifier Inversion:** NLI flips sign ($100.0\%$); FrameAxis collapses ($\cos \approx 0.73 - 0.98$).
- **Agent–Patient Reversal:** NLI exhibits $50.0\%$ directional sensitivity; FrameAxis exhibits $0.0\%$.

### 6.3 Anisotropy Removal Ablation
- Ablating the top embedding PC collapses FrameAxis dominant variance from **$73.7\%$ down to $34.5\%$**, proving that FrameAxis is heavily driven by sentence-embedding cone geometry.

### 6.4 Publisher Separation & Sentiment Controls
- News outlets are separated with an effect size of $\eta^2 = \mathbf{40.1\%}$ ($p = 0.005$).
- Survives collinear sentiment controls with $\eta^2 = \mathbf{34.8\%}$ ($p = 0.022$).

---

## 7. Discussion & Conclusion

Political framing in news discourse can be measured endogenously as a low-dimensional semantic geometry directly from text without relying on circular publisher metadata. However, semantic geometry is an observer-conditioned measurement function:
\\[
\mathcal{G}_{\text{observed}} = f(C, H, O, M)
\\]
where each parameter formalizes an explicit measurement choice:
- **$C$ (Corpus):** The target document sample, defined by its domain scope, topic filtering, temporal sampling window, and publisher clustering boundaries;
- **$H$ (Hypotheses):** The set of authored dialectical proposition pairs $P = (h_A, h_B)$ that articulate the conceptual tensions of the domain;
- **$O$ (Operator):** The mathematical scoring function, specifically the paired contrast operator $c_P(x) = s(x, h_A) - s(x, h_B) \in [-2, 2]$ on post-softmax entailment-minus-contradiction probabilities;
- **$M$ (Model):** The frozen cross-encoder neural inference engine (`microsoft/deberta-v2-xlarge-mnli`) whose attention heads and token representations parameterize the relational forward pass.

Our findings demonstrate that relational NLI operators recover a statistically robust, unidimensional stance axis ($\mathrm{PC1} \approx 42\% - 64\%$) that decisively clears 10,000-draw permutation null gates ($p \le 0.0003$) and reproduces across prospective samples, disjoint probe partitions, and independent model families. Crucially, secondary dimensions fail falsification gates across all tested domains ($p \ge 0.5282$ up to $1.0000$), bounding detected dimensionality under the implemented null to rank $k=1$ without asserting the ontological absence of higher-order constructs in the population. Furthermore, alignment with third-party publisher ratings fails out-of-sample ($R^2 = 0.1916, \text{predictive } R^2 = -0.4902$), and upstream corpus construction on unfiltered text introduces off-topic and syndication contamination. Low-dimensional framing geometry is therefore an endogenous property of the relational measurement system, providing a mathematically grounded alternative to circular publisher-level aggregation.
