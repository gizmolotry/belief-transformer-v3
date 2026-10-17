# Benchmarking Model-Derived Perspectival Spaces Against Human Interpretive Communities: Preregistration-Ready Protocol

## 1. Objective and Theoretical Premise
This document specifies a mathematically rigorous, **preregistration-ready** experimental protocol (Status: `PROSPECTIVE_PROTOCOL / NOT_YET_TESTED`). No empirical human data has yet been collected; this protocol defines the sample size derivations, measurement instrumentation, and falsification gates prior to field execution.

The objective is to test the preregistered empirical hypothesis:
$$\mathcal{H}_{\text{persp}}: \text{Cross-encoder NLI models reconstruct authentic human perspectival geometry rather than idiosyncratic artifactual geometries.}$$
Specifically, we evaluate whether human interpretive communities reading identical articles under explicit framing directives span observational manifolds that align with the model-derived manifolds $\mathcal{G}_P$.

---

## 2. Experimental Design & Factorial Structure

### 2.1 Target Corpus
- **Fixed Article Sample**: $N = 120$ prospectively sampled Middle East news articles from 70 global publishers (identical to the prospective corpus analyzed herein).

### 2.2 Interpretive Communities (Human Reader Panels)
Four distinct human reader panels ($K = 4$), each consisting of $M = 15$ qualified evaluators ($N_{\text{total}} = 60$ annotators):
1. **Panel 1: Israeli Strategic/Security Analysts**: Professionals or accredited researchers affiliated with Israeli defense/security institutions or strategic think tanks.
2. **Panel 2: Palestinian Human Rights Advocates**: Legal officers, field researchers, or advocates affiliated with Palestinian human rights NGOs or civil society organizations.
3. **Panel 3: International Humanitarian Law (IHL) Scholars**: Academic faculty, legal practitioners, or ICRC-certified jurists specializing in the Geneva Conventions and international law of armed conflict.
4. **Panel 4: Non-Aligned / Diplomatic Generalists**: Career foreign service officers, multilateral observers, or professional news editors from non-belligerent jurisdictions (e.g., Swiss, Nordic, or East Asian diplomatic institutions).

### 2.3 Experimental Task
- Every annotator evaluates a balanced subset of articles across $H = 8$ bipolar framing axes.
- Axes mirror the canonical sets: Geopolitical ($Q_0$), Humanitarian Rights ($Q_2$), Realpolitik ($Q_3$), and IHL ($P_{\text{IHL}}$).
- Rating scale: Bipolar continuous slider $[-100, +100]$, mapped to $[-1.0, +1.0]$, identical to the normalized model stance targets.
- Design: Replicated cross-classified G-theory design $(Articles \times Annotators \times Perspectives)$.

---

## 3. Statistical Power & Sample Size Derivation

### 3.1 Power Analysis for Subspace Alignment (Procrustes / Grassmann Distance)
- To detect a true Procrustes alignment $R^2 \ge 0.50$ between human panel consensus manifolds $\mathcal{G}_{\text{human}}$ and model manifolds $\mathcal{G}_{\text{model}}$ at $\alpha = 0.01$ with power $1 - \beta = 0.95$:
  - Degrees of freedom in target subspace ($k = 3$ dimensions across $N = 120$ articles):
    $$N_{\text{eff}} = 120, \quad F = \frac{R^2 / k}{(1 - R^2) / (N - k - 1)} = \frac{0.50 / 3}{0.50 / 116} = 38.67 \gg F_{\text{crit}} = 3.95$$
  - Power exceeds $0.999$, ensuring decisive resolution against null alignments ($R^2 < 0.15$).

### 3.2 Generalizability Theory Sample Size Specification
- Applying the D-study formula for $(AP)$ cell-mean reliability:
  $$E\rho^2_{AP} = \frac{\sigma^2_{AP}}{\sigma^2_{AP} + \frac{\sigma^2_{AP \times R}}{n_r}}$$
  With $n_r = 15$ raters per panel, given typical inter-annotator agreement $\sigma^2_{AP \times R} / \sigma^2_{AP} \approx 1.8$, expected profile reliability is:
  $$E\rho^2_{AP}(15) = \frac{1.0}{1.0 + 1.8 / 15} = \mathbf{0.8929},$$
  exceeding the psychometric gold standard ($0.80$).

---

## 4. Formal Evaluation Metrics

### 4.1 Profile Concordance (Intraclass Correlation)
- Compute two-way random intraclass correlation $\text{ICC}(2, k)$ between the aggregated human profile vector $\mathbf{y}_{\text{human}} \in \mathbb{R}^{120 \times 8}$ and the model profile $\mathbf{y}_{\text{model}} \in \mathbb{R}^{120 \times 8}$:
  $$\text{Concordance}(P) = \text{Corr}\left(\mathbf{y}_{\text{human}}^{(P)}, \mathbf{y}_{\text{model}}^{(P)}\right)$$

### 4.2 Manifold Alignment via Orthogonal Procrustes
- For human manifold matrix $\mathbf{H} \in \mathbb{R}^{120 \times k}$ and model manifold matrix $\mathbf{M} \in \mathbb{R}^{120 \times k}$:
  $$\min_{\mathbf{R} \in \mathcal{O}(k)} \|\mathbf{H}\mathbf{R} - \mathbf{M}\|_F^2$$
- Alignment statistic: $R^2_{\text{Procrustes}} = 1 - \frac{\|\mathbf{H}\hat{\mathbf{R}} - \mathbf{M}\|_F^2}{\|\mathbf{H}\|_F^2}$.

### 4.3 Anomaly Set Concordance
- Identify top 10% anomalies in human ratings $\mathcal{A}_{\text{human}}$ vs model ratings $\mathcal{A}_{\text{model}}$ via Mahalanobis distance.
- Evaluate overlap via Jaccard index: $J_{10} = \frac{|\mathcal{A}_{\text{human}} \cap \mathcal{A}_{\text{model}}|}{|\mathcal{A}_{\text{human}} \cup \mathcal{A}_{\text{model}}|}$.

---

## 5. Preregistered Falsification Criteria

The claim that *NLI models reflect authentic human perspectival geometry* is considered **FALSIFIED** if any of the following occur:
1. **Manifold Independence**: $R^2_{\text{Procrustes}}(\mathcal{G}_{\text{human}}, \mathcal{G}_{\text{model}}) < 0.20$ ($p > 0.05$ against permutation null).
2. **Within-Panel Disconnection**: The correlation between model-derived perspective distance and human inter-panel distance fails to reach significance ($r(D_{\text{model}}, D_{\text{human}}) < 0.30$, $p > 0.05$).
3. **Anomaly Divergence**: The Jaccard similarity of anomalous articles between human panels and model predictions is indistinguishable from random chance ($J_{10} \le 0.05$).
4. **Machine Idiosyncrasy**: Model-human agreement is lower than model-absurd agreement (i.e., $\text{ICC}(\text{model}, \text{human}) \le \text{ICC}(\text{model}, \text{culinary})$).
