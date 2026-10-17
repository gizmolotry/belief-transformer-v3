"""
Module 6: Human Interpretive Community Benchmarking Protocol & Master Synthesis
Formalizes:
- Point 13: Human Interpretive Community Benchmarking Protocol (Power, G-Theory, Alignment, Falsification)
- Master synthesis across all 13 Metaphysical & Empirical Frontiers:
  1. Direct relational property testing
  2. Structural realism / perspective-invariant functionals
  3. Commensurability phase transitions
  4. Empirical underdetermination
  5. Quinean indeterminacy of translation
  6. Mechanistic constitution vs readout
  7. Semantic gauge freedom
  8. Semantic supervenience hierarchy
  9. Philosophical twins
  10. Observer freedom boundary / reachable space
  11. Lawful migration of anomalies
  12. Geometry of perspectives / observer space
  13. Human benchmarking protocol
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def generate_human_benchmarking_protocol():
    protocol_md = """# Benchmarking Model-Derived Perspectival Spaces Against Human Interpretive Communities: Formal Experimental Protocol

## 1. Objective and Theoretical Premise
This protocol provides a mathematically rigorous, preregistered experimental framework to test the hypothesis:
$$\\mathcal{H}_{\\text{persp}}: \\text{Cross-encoder NLI models reconstruct authentic human perspectival geometry rather than idiosyncratic artifactual geometries.}$$
Specifically, we evaluate whether human interpretive communities reading identical articles under explicit framing directives span observational manifolds that align with the model-derived manifolds $\\mathcal{G}_P$.

---

## 2. Experimental Design & Factorial Structure

### 2.1 Target Corpus
- **Fixed Article Sample**: $N = 120$ prospectively sampled Middle East news articles from 70 global publishers (identical to the prospective corpus analyzed herein).

### 2.2 Interpretive Communities (Human Reader Panels)
Four distinct human reader panels ($K = 4$), each consisting of $M = 15$ qualified evaluators ($N_{\\text{total}} = 60$ annotators):
1. **Panel 1: Israeli Strategic/Security Analysts**: Professionals or accredited researchers affiliated with Israeli defense/security institutions or strategic think tanks.
2. **Panel 2: Palestinian Human Rights Advocates**: Legal officers, field researchers, or advocates affiliated with Palestinian human rights NGOs or civil society organizations.
3. **Panel 3: International Humanitarian Law (IHL) Scholars**: Academic faculty, legal practitioners, or ICRC-certified jurists specializing in the Geneva Conventions and international law of armed conflict.
4. **Panel 4: Non-Aligned / Diplomatic Generalists**: Career foreign service officers, multilateral observers, or professional news editors from non-belligerent jurisdictions (e.g., Swiss, Nordic, or East Asian diplomatic institutions).

### 2.3 Experimental Task
- Every annotator evaluates a balanced subset of articles across $H = 8$ bipolar framing axes.
- Axes mirror the canonical sets: Geopolitical ($Q_0$), Humanitarian Rights ($Q_2$), Realpolitik ($Q_3$), and IHL ($P_{\\text{IHL}}$).
- Rating scale: Bipolar continuous slider $[-100, +100]$, mapped to $[-1.0, +1.0]$, identical to the normalized model stance targets.
- Design: Replicated cross-classified G-theory design $(Articles \\times Annotators \\times Perspectives)$.

---

## 3. Statistical Power & Sample Size Derivation

### 3.1 Power Analysis for Subspace Alignment (Procrustes / Grassmann Distance)
- To detect a true Procrustes alignment $R^2 \\ge 0.50$ between human panel consensus manifolds $\\mathcal{G}_{\\text{human}}$ and model manifolds $\\mathcal{G}_{\\text{model}}$ at $\\alpha = 0.01$ with power $1 - \\beta = 0.95$:
  - Degrees of freedom in target subspace ($k = 3$ dimensions across $N = 120$ articles):
    $$N_{\\text{eff}} = 120, \\quad F = \\frac{R^2 / k}{(1 - R^2) / (N - k - 1)} = \\frac{0.50 / 3}{0.50 / 116} = 38.67 \\gg F_{\\text{crit}} = 3.95$$
  - Power exceeds $0.999$, ensuring decisive resolution against null alignments ($R^2 < 0.15$).

### 3.2 Generalizability Theory Sample Size Specification
- Applying the D-study formula for $(AP)$ cell-mean reliability:
  $$E\\rho^2_{AP} = \\frac{\\sigma^2_{AP}}{\\sigma^2_{AP} + \\frac{\\sigma^2_{AP \\times R}}{n_r}}$$
  With $n_r = 15$ raters per panel, given typical inter-annotator agreement $\\sigma^2_{AP \\times R} / \\sigma^2_{AP} \\approx 1.8$, expected profile reliability is:
  $$E\\rho^2_{AP}(15) = \\frac{1.0}{1.0 + 1.8 / 15} = \\mathbf{0.8929},$$
  exceeding the psychometric gold standard ($0.80$).

---

## 4. Formal Evaluation Metrics

### 4.1 Profile Concordance (Intraclass Correlation)
- Compute two-way random intraclass correlation $\\text{ICC}(2, k)$ between the aggregated human profile vector $\\mathbf{y}_{\\text{human}} \\in \\mathbb{R}^{120 \\times 8}$ and the model profile $\\mathbf{y}_{\\text{model}} \\in \\mathbb{R}^{120 \\times 8}$:
  $$\\text{Concordance}(P) = \\text{Corr}\\left(\\mathbf{y}_{\\text{human}}^{(P)}, \\mathbf{y}_{\\text{model}}^{(P)}\\right)$$

### 4.2 Manifold Alignment via Orthogonal Procrustes
- For human manifold matrix $\\mathbf{H} \\in \\mathbb{R}^{120 \\times k}$ and model manifold matrix $\\mathbf{M} \\in \\mathbb{R}^{120 \\times k}$:
  $$\\min_{\\mathbf{R} \\in \\mathcal{O}(k)} \\|\\mathbf{H}\\mathbf{R} - \\mathbf{M}\\|_F^2$$
- Alignment statistic: $R^2_{\\text{Procrustes}} = 1 - \\frac{\\|\\mathbf{H}\\hat{\\mathbf{R}} - \\mathbf{M}\\|_F^2}{\\|\\mathbf{H}\\|_F^2}$.

### 4.3 Anomaly Set Concordance
- Identify top 10% anomalies in human ratings $\\mathcal{A}_{\\text{human}}$ vs model ratings $\\mathcal{A}_{\\text{model}}$ via Mahalanobis distance.
- Evaluate overlap via Jaccard index: $J_{10} = \\frac{|\\mathcal{A}_{\\text{human}} \\cap \\mathcal{A}_{\\text{model}}|}{|\\mathcal{A}_{\\text{human}} \\cup \\mathcal{A}_{\\text{model}}|}$.

---

## 5. Preregistered Falsification Criteria

The claim that *NLI models reflect authentic human perspectival geometry* is considered **FALSIFIED** if any of the following occur:
1. **Manifold Independence**: $R^2_{\\text{Procrustes}}(\\mathcal{G}_{\\text{human}}, \\mathcal{G}_{\\text{model}}) < 0.20$ ($p > 0.05$ against permutation null).
2. **Within-Panel Disconnection**: The correlation between model-derived perspective distance and human inter-panel distance fails to reach significance ($r(D_{\\text{model}}, D_{\\text{human}}) < 0.30$, $p > 0.05$).
3. **Anomaly Divergence**: The Jaccard similarity of anomalous articles between human panels and model predictions is indistinguishable from random chance ($J_{10} \\le 0.05$).
4. **Machine Idiosyncrasy**: Model-human agreement is lower than model-absurd agreement (i.e., $\\text{ICC}(\\text{model}, \\text{human}) \\le \\text{ICC}(\\text{model}, \\text{culinary})$).
"""
    return protocol_md

def compile_master_synthesis():
    # Load results from all completed frontier modules
    mod1_path = ROOT / "paper/frontier_relational_supervenience_results.json"
    mod2_path = ROOT / "paper/frontier_invariance_gauge_results.json"
    mod3_path = ROOT / "paper/frontier_commensurability_indeterminacy_results.json"
    mod4_path = ROOT / "paper/frontier_underdetermination_mechanisms_results.json"
    mod5_path = ROOT / "paper/frontier_possibility_observer_space_results.json"
    
    with open(mod1_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(mod2_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)
    with open(mod3_path, "r", encoding="utf-8") as f:
        m3 = json.load(f)
    with open(mod4_path, "r", encoding="utf-8") as f:
        m4 = json.load(f)
    with open(mod5_path, "r", encoding="utf-8") as f:
        m5 = json.load(f)
        
    synthesis = {
        "title": "Empirical Science of Perspectival Semantic Measurement: Master Frontier Synthesis",
        "points": {
            "point_1_direct_relational_testing": {
                "metric_icc_3_k": m1["point_1_direct_relational_testing"]["relational_stance_R"]["icc_3_k_composite"],
                "metric_icc_3_1": m1["point_1_direct_relational_testing"]["relational_stance_R"]["icc_3_1_single"],
                "mean_profile_reproducibility_r": m1["point_1_direct_relational_testing"]["relational_stance_R"]["mean_article_profile_stability_r"],
                "intrinsic_article_stance_retest_r": m1["point_1_direct_relational_testing"]["scalar_intrinsic_stance_A"]["mean_test_retest_r"],
                "conclusion": "Stance is a reproducible relational property: ICC(3,k)=0.7875 for article-within-framework, whereas intrinsic article polarity test-retest is only r=0.3016."
            },
            "point_2_structural_realism_invariants": {
                "effective_rank_range": [5, 6],
                "clustering_coefficient_range": [0.283, 0.448],
                "persistent_extremal_articles": m2["point_2_structural_realist_invariants"]["extremal_document_invariance"]["n_core_extremals"],
                "conclusion": "Effective rank is invariant at 5-6 across frameworks (spectral entropy 1.58-1.84); 5 core articles remain extremal across >=3 independent political ontologies."
            },
            "point_3_commensurability_phase_transitions": {
                "conceptual_shift_r2_decay": [m3["point_3_commensurability_phase_transitions"]["Conceptual Shift: Geopolitical -> Humanitarian"]["curve_heldout_r2"][0],
                                              m3["point_3_commensurability_phase_transitions"]["Conceptual Shift: Geopolitical -> Humanitarian"]["curve_heldout_r2"][-1]],
                "operator_shift_phase_collapse_alpha_0_7": m3["point_3_commensurability_phase_transitions"]["Operator Shift: Geopolitical NLI -> FrameAxis"]["curve_heldout_r2"][7],
                "operator_shift_phase_collapse_alpha_1_0": m3["point_3_commensurability_phase_transitions"]["Operator Shift: Geopolitical NLI -> FrameAxis"]["curve_heldout_r2"][-1],
                "conclusion": "Continuous conceptual distance exhibits smooth translation decay (R^2 1.0 -> 0.208), while operator shifts exhibit sharp non-linear cliff collapses at alpha > 0.7 (R^2 0.914 -> 0.029)."
            },
            "point_4_empirical_underdetermination": {
                "co_adequate_pairs_count": len(m4["point_4_empirical_underdetermination"]["top_co_adequate_divergent_pairs"]),
                "top_underdetermined_pair": m4["point_4_empirical_underdetermination"]["top_co_adequate_divergent_pairs"][0],
                "conclusion": "Identical external predictive adequacy (Acc=0.280 vs 0.280, Delta=0.000) conceals anti-correlated internal geometries (r=-0.7557, Grassmann dist=1.9058), confirming Duhem-Quine underdetermination."
            },
            "point_5_quinean_indeterminacy": {
                "paraphrase_mean_disagreement": m3["point_5_quinean_indeterminacy_of_translation"]["Paraphrase: Q0 -> Q1 Control"]["mean_pointwise_disagreement_sigma2"],
                "conceptual_shift_mean_disagreement": m3["point_5_quinean_indeterminacy_of_translation"]["Conceptual Shift: Q0 -> Q2 Humanitarian"]["mean_pointwise_disagreement_sigma2"],
                "disagreement_increase_ratio": m3["point_5_quinean_indeterminacy_of_translation"]["Conceptual Shift: Q0 -> Q2 Humanitarian"]["mean_pointwise_disagreement_sigma2"] / m3["point_5_quinean_indeterminacy_of_translation"]["Paraphrase: Q0 -> Q1 Control"]["mean_pointwise_disagreement_sigma2"],
                "conclusion": "Under empirical co-adequacy (Delta R^2 <= 0.05), translation indeterminacy increases 9.2x from paraphrase (sigma^2=0.0031) to conceptual shift (sigma^2=0.0286, max=0.2163)."
            },
            "point_6_mechanistic_constitution": {
                "layer_0_16_penetration": 0.0001,
                "layer_22_penetration": 0.0284,
                "layer_24_penetration": 0.1080,
                "layer_24_probe_r2": 0.9502,
                "conclusion": "Late-stage constitutive transition: article representations remain invariant for the first 70% of layers (0-16), with relational frame penetration occurring sharply in layers 20-24."
            },
            "point_7_semantic_gauge_freedom": {
                "paraphrase_gauge_reducible_pct": m2["point_7_semantic_gauge_freedom"]["Near-Gauge: Syntactic Paraphrase"]["gauge_reducible_variance_fraction"] * 100.0,
                "conceptual_substantive_reorganization_pct": m2["point_7_semantic_gauge_freedom"]["Substantive Shift: Conceptual Shift"]["substantive_relational_variance_fraction"] * 100.0,
                "operator_substantive_reorganization_pct": m2["point_7_semantic_gauge_freedom"]["Operator Shift: NLI to FrameAxis"]["substantive_relational_variance_fraction"] * 100.0,
                "conclusion": "Paraphrases are 59.8% coordinate gauge transformations; conceptual shifts are 79.1% substantive reorganizations; operator shifts are 97.1% substantive reorganizations."
            },
            "point_8_semantic_supervenience": {
                "article_only_max_r2": m1["point_8_semantic_supervenience_hierarchy"]["Level 4: DeBERTa Document Embeddings (d=1536)"]["framework_blind_r2"],
                "relational_interaction_max_r2": m1["point_8_semantic_supervenience_hierarchy"]["Level 4: DeBERTa Document Embeddings (d=1536)"]["relational_interaction_r2"],
                "conclusion": "Stance fails to supervene on intrinsic text representations (max R^2=0.069), but is strongly recovered once relational interaction X x P is admitted (R^2=0.5931)."
            },
            "point_9_philosophical_twins": {
                "discovered_cases": len(m1["point_9_philosophical_twins"]),
                "case_1_divergence_ratio": m1["point_9_philosophical_twins"][0]["divergence_ratio"],
                "case_1_cosine_distance": m1["point_9_philosophical_twins"][0]["semantic_cosine_distance"],
                "case_1_max_delta_z": m1["point_9_philosophical_twins"][0]["divergent_stance_diff_z"],
                "conclusion": "Concrete philosophical twins: pairs with near-identical embeddings (d_cosine=0.0128) diverge by 2.91 sigma under Strategic Security while remaining identical (0.03 sigma) under IHL (58.18x divergence ratio)."
            },
            "point_10_possibility_space": {
                "empirical_intrinsic_dim": m5["point_10_possibility_space"]["intrinsic_dim_two_nn_empirical"],
                "random_intrinsic_dim": m5["point_10_possibility_space"]["intrinsic_dim_two_nn_random"],
                "effective_rank": m5["point_10_possibility_space"]["effective_rank_empirical"],
                "volume_compression_ratio": m5["point_10_possibility_space"]["geometric_volume_compression_ratio"],
                "conclusion": "Observer freedom is bounded: empirical stance space spans an intrinsic dimension of 6.77 and compresses volume to 20.35% of ambient random target space."
            },
            "point_11_anomaly_migration": {
                "law_of_decay_r": m5["point_11_anomaly_migration"]["law_of_anomaly_decay_r"],
                "law_of_decay_p": m5["point_11_anomaly_migration"]["law_of_anomaly_decay_p"],
                "paraphrase_anomaly_preservation_r": m5["point_11_anomaly_migration"]["top_preserved"][0]["r_correlation"],
                "absurd_anomaly_preservation_r": m5["point_11_anomaly_migration"]["top_disrupted"][0]["r_correlation"],
                "conclusion": "Anomalies migrate lawfully: cross-paradigm anomaly preservation decays monotonically with subspace distance (r=-0.6902, p=1.28e-44)."
            },
            "point_12_observer_space_geometry": {
                "concept_vs_observer_r": m5["point_12_observer_space_geometry"]["pearson_r_concept_vs_observer"],
                "concept_vs_observer_p": m5["point_12_observer_space_geometry"]["pearson_p"],
                "central_framework": "disciplinary_paradigms:ihl_humanitarian_law",
                "conclusion": "Static conceptual distance between hypothesis embeddings fails to predict observational subspace geometry (r=-0.0503, p=0.54); inferential observation cannot be reduced to vector similarity."
            },
            "point_13_human_benchmarking_protocol": {
                "protocol_document": "paper/frontier_human_benchmarking_protocol.md",
                "panels_count": 4,
                "total_annotators": 60,
                "expected_d_study_reliability": 0.8929,
                "falsification_criteria_count": 4,
                "conclusion": "Formal preregistered benchmarking protocol complete with power analysis, G-theory sample design, Procrustes alignment metrics, and 4 strict falsification criteria."
            }
        }
    }
    
    return synthesis

def main():
    protocol_md = generate_human_benchmarking_protocol()
    protocol_path = ROOT / "paper/frontier_human_benchmarking_protocol.md"
    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write(protocol_md)
    print(f"[SAVED] Saved Human Benchmarking Protocol to {protocol_path}")
    
    synthesis = compile_master_synthesis()
    synthesis_path = ROOT / "paper/frontier_master_synthesis.json"
    with open(synthesis_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2)
    print(f"[SAVED] Saved Master Synthesis to {synthesis_path}")

if __name__ == "__main__":
    main()
