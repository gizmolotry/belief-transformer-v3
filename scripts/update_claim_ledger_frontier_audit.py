"""
Audit and update claim_ledger.json and claim_ledger.md with surgical epistemic precision.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ledger_json_path = ROOT / "paper/claim_ledger.json"
ledger_md_path = ROOT / "paper/claim_ledger.md"

with open(ledger_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find and surgically update CLM_21 through CLM_26
for c in data["claims"]:
    cid = c["claim_id"]
    
    if cid == "CLM_21_RELATIONAL_ONTOLOGY_AND_SUPERVENIENCE":
        c["claim_summary"] = "Relational measurements are more reproducible than framework-averaged scalars (ICC(3,k)=0.7875 vs ICC(3,1)=0.3965); relational stance is not recoverable from tested framework-blind text representations alone (R^2<=0.0695)."
        c["claim_text"] = "Evaluating the reproducibility of framing measurements across three independent probe-bank realizations on 120 prospective articles confirms that relational measurements are substantially more reproducible than framework-averaged scalar scores: article-within-framework targets achieve composite reliability ICC(3,k) = 0.7875 and mean profile stability r = 0.4515, whereas the framework-averaged scalar article score exhibits low test-retest reliability (r = 0.3016, ICC(3,1) = 0.3965). Furthermore, relational stance is not recoverable with useful accuracy from the tested framework-blind text representations across five increasingly expressive feature spaces (surface lexical, TF-IDF, MiniLM, 1536-dim DeBERTa, Poly-2; all R^2 <= 0.0695), whereas relational interaction models recover substantial variance (R^2 = 0.5931). Concrete document pairs with near-zero distance in dense embedding space (d_cosine = 0.0128) separate by 2.91 sigma under a Strategic Security framework while differing by only 0.027 sigma under an International Humanitarian Law framework, demonstrating that dense semantic proximity does not constrain framework-conditioned stance."
        c["status"] = "SUPPORTED"
        c["effect_size"] = "Relational ICC(3,k) = 0.7875 vs Scalar ICC(3,1) = 0.3965; Framework-blind R^2 <= 0.0695 vs Relational R^2 = 0.5931; Twin divergence: 2.91 sigma vs 0.027 sigma"
        c["ci_95"] = "ICC(3,k): [0.750, 0.820]; Profile r IQR: [0.232, 0.682]"
        c["primary_artifacts"] = ["paper/frontier_relational_supervenience_results.json"]
        
    elif cid == "CLM_22_STRUCTURAL_INVARIANTS_AND_GAUGE_FREEDOM":
        c["claim_summary"] = "Syntactic paraphrase fails the preregistered near-gauge criterion (R^2_affine=0.598 < 0.85), but remains substantially more coordinate-reducible than conceptual (12-21%) or operator (2.9%) comparisons; spectral rank (5-6) and persistent extremals survive across frameworks."
        c["claim_text"] = "Evaluating semantic coordinate transformations against the preregistered gauge threshold (R^2_affine >= 0.85) reveals that syntactic paraphrasing (Q0 -> Q1) achieves R^2_affine = 0.5981 (Procrustes R^2 = 0.5133), failing the strict near-gauge hypothesis. However, paraphrase is substantially more coordinate-reducible (59.8% explained by affine transformation) than conceptual shifts (where affine maps explain only 12.3% to 20.9% of variance) or cross-operator comparisons (explaining only 2.89% of variance). Across all tested NLI frameworks, structural invariants persist: spectral effective rank at 90% variance is conserved at 5 to 6 dimensions (spectral entropy H in [1.58, 1.84]), graph clustering coefficients are conserved at 0.28 to 0.45, and a core set of 5 articles remains persistently extremal in distribution tails across >= 3 independent political ontologies."
        c["status"] = "PARTIALLY_SUPPORTED" # Falsified on strict gauge threshold, supported on relative reducibility and structural invariants
        c["effect_size"] = "Paraphrase affine R^2 = 0.5981 (fails >= 0.85 gauge standard); Conceptual affine R^2 = 0.123-0.209; Operator affine R^2 = 0.0289; Conserved rank = 5-6"
        c["ci_95"] = "Paraphrase Procrustes R^2: [0.47, 0.55]; Spectral entropy: [1.58, 1.84]"
        c["primary_artifacts"] = ["paper/frontier_invariance_gauge_results.json"]
        
    elif cid == "CLM_23_COMMENSURABILITY_TRANSITIONS_AND_QUINEAN_INDETERMINACY":
        c["claim_summary"] = "Apparent cliff in operator interpolation is an artifact of 8.75x scale asymmetry in numerical score mixing; translation-model multiplicity exhibits a 9.13x surge in prediction dispersion as an empirical analogue of indeterminacy."
        c["claim_text"] = "We distinguish valid linguistic interventions from post-hoc numerical interpolation through representation space: score-mixture paths S(alpha) = (1-alpha) S_A + alpha S_B represent artificial numerical interpolation rather than continuous paradigm transitions in language space. Furthermore, audit demonstrates that the apparent cliff collapse in unstandardized operator mixing (R^2 dropping from 0.9139 at alpha=0.7 to 0.0289 at alpha=1.0) is a numerical artifact of the 8.75x scale dominance of raw NLI over FrameAxis (76.5x variance ratio). When standardized prior to mixing, the curve decays smoothly without a change point (p = 0.4135 under Pettitt test), strictly following the analytical score-mixture expectation R^2(alpha) = (1-alpha)^2 / ((1-alpha)^2 + alpha^2) with r = 0.9965 (MSE = 0.0041). Turning to translation-model multiplicity as an empirical analogue of translation indeterminacy, evaluating 15 candidate estimators under empirical co-adequacy (Delta R^2 <= 0.05) reveals a 9.13x surge in pointwise prediction dispersion when translating between conceptual frameworks (mean variance sigma^2_trans = 0.0286, max = 0.5191) relative to self-translation across independent realizations of the same framework (mean variance sigma^2_self = 0.0031, max = 0.1045)."
        c["status"] = "SUPPORTED"
        c["effect_size"] = "Standardized score mixture decays smoothly (Pettitt p = 0.4135, r_analytical = 0.9965); Translation dispersion surge: 9.13x (sigma^2 = 0.0286 vs 0.0031)"
        c["ci_95"] = "Self-translation best R^2 = 0.6031; Conceptual translation best R^2 = 0.2355"
        c["primary_artifacts"] = ["paper/frontier_commensurability_indeterminacy_results.json", "paper/surgical_audit_frontier_results.json"]
        
    elif cid == "CLM_24_EMPIRICAL_UNDERDETERMINATION_AND_MECHANISMS":
        c["claim_summary"] = "Empirical co-adequacy across multiple external criteria conceals large subspace divergence (d_Grassmann=1.9058), while causal activation patching in DeBERTa demonstrates that hypothesis conditioning is concentrated in intermediate and late self-attention layers."
        c["claim_text"] = "Evaluating external adequacy reveals 27 framework pairs with indistinguishable external fit; for example, Realpolitik and Legal Culpability exhibit identical top-5 publisher accuracy (0.280 vs 0.280, Delta = 0.000), similar all-publisher ANOVA F-statistics (1.86 vs 1.99, Delta F = 0.13), and matched sentiment prediction (R^2 = 0.085 vs 0.092, Delta R^2 = 0.007). However, their internal representations occupy substantially divergent subspaces (3D Grassmann distance d_G = 1.9058 out of a maximum 2.7207, with sign-aligned |r(PC1)| = 0.7557). Causal activation patching in DeBERTa-v2-xlarge across layers 8 to 24 reveals that transplanting hypothesis token hidden states into a counterfactual forward pass shifts the final stance output by over 85% when intervened at layers 8 to 20 (causal transfer ratio = +0.851 to +0.959), tapering to +0.571 at layer 22 and 0.000 at layer 24 (where [CLS] has already integrated hypothesis context). This demonstrates that relational conditioning is causally active across intermediate and late joint self-attention layers."
        c["status"] = "SUPPORTED"
        c["effect_size"] = "Co-adequate external match: Delta Acc = 0.000, Delta F = 0.13, Delta Sent R^2 = 0.007; Grassmann distance d_G = 1.9058; Causal patch ratio: +0.959 (L8) to +0.869 (L20)"
        c["ci_95"] = "Grassmann distance: [1.85, 2.15]; Causal patch std: [0.21, 0.37]"
        c["primary_artifacts"] = ["paper/frontier_underdetermination_mechanisms_results.json", "paper/surgical_audit_frontier_results.json"]
        
    elif cid == "CLM_25_POSSIBILITY_SPACE_AND_OBSERVER_GEOMETRY":
        c["claim_summary"] = "Sampled probe banks occupy a compressed possibility volume (20.35% of ambient random reference); anomaly agreement decays lawfully with subspace distance under a dependence-aware QAP test (p_Mantel < 0.0001)."
        c["claim_text"] = "The sampled cloud of 18 plausible probe banks occupies an intrinsic dimension of 6.77 (Two-NN) and an estimated geometric volume of 20.35% relative to an ambient random reference in [-1, 1]^8, demonstrating bounded observer variation within this candidate set. Across 153 unique framework pairs, anomaly agreement decays lawfully as a function of 3D Grassmannian distance (Pearson r = -0.6902, Spearman rho = -0.6578); under a 10,000-permutation Mantel / Quadratic Assignment Procedure (QAP) test strictly accounting for dyadic matrix dependence, this decay remains highly significant (p_Mantel < 0.0001 for both Pearson and Spearman), establishing that geometrically more distant perspectives agree less about which articles are anomalous. Furthermore, static conceptual distance metrics on hypothesis prompts fail to predict observational subspace geometry under QAP testing (DeBERTa embedding distance: r = -0.0503, p_Mantel = 0.2522; TF-IDF n-grams: r = +0.1116, p_Mantel = 0.0975)."
        c["status"] = "SUPPORTED"
        c["effect_size"] = "QAP-verified anomaly decay: r = -0.6902, rho = -0.6578 (p_Mantel < 0.0001); Conceptual vs observational QAP: p_Mantel = 0.098 to 0.252; Possibility volume = 20.35%"
        c["ci_95"] = "Pearson r: [-0.75, -0.62]; Spearman rho: [-0.73, -0.58]"
        c["primary_artifacts"] = ["paper/frontier_possibility_observer_space_results.json", "paper/surgical_audit_frontier_results.json"]
        
    elif cid == "CLM_26_HUMAN_BENCHMARKING_PROTOCOL_AND_SYNTHESIS":
        c["claim_summary"] = "Preregistration-ready protocol formalizing human reader panel benchmarking across 4 interpretive communities (N=60) with power > 0.999 and 4 explicit falsification criteria."
        c["claim_text"] = "We specify a complete, preregistration-ready protocol to benchmark model-derived perspectival spaces against human interpretive communities. The design encompasses 4 panels (Israeli security analysts, Palestinian human rights advocates, IHL scholars, non-aligned diplomats; M=15 per panel, N_total = 60), specifying power > 0.999 to detect Procrustes manifold alignment R^2 >= 0.50 and design-projected D-study profile reliability E rho^2_AP(15) = 0.8929. The protocol establishes formal evaluation metrics (profile ICC, Procrustes disparity, anomaly Jaccard overlap) and four explicit falsification gates to adjudicate whether cross-encoder NLI models reconstruct authentic human perspectival geometry or machine-idiosyncratic artifacts."
        c["status"] = "PROSPECTIVE_PROTOCOL" # Corrected from SUPPORTED to PROSPECTIVE_PROTOCOL
        c["effect_size"] = "Projected D-study profile reliability E rho^2_AP(15) = 0.8929 (design power specification, not empirical finding); Power > 0.999"
        c["ci_95"] = "Projected E rho^2_AP(15): [0.85, 0.93]"
        c["primary_artifacts"] = ["paper/frontier_human_benchmarking_protocol.md", "paper/frontier_master_synthesis.json"]

with open(ledger_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Updated claim_ledger.json successfully.")

# Regenerate claim_ledger.md
lines = [
    "# Master Claim Ledger: Belief-Transformer Perspectival Measurement",
    "",
    f"**Total Registered Claims:** {len(data['claims'])}",
    f"**Last Updated:** {data.get('timestamp', '2026-09-18')}",
    "",
    "| Claim ID | Summary | Status | Effect Size | 95% CI | Primary Artifact |",
    "|---|---|---|---|---|---|"
]

for c in data["claims"]:
    cid = c["claim_id"]
    summ = c["claim_summary"].replace("|", "\\|")
    stat = c["status"]
    eff = c["effect_size"].replace("|", "\\|")
    ci = c["ci_95"].replace("|", "\\|")
    art = ", ".join(f"`{a}`" for a in c["primary_artifacts"])
    lines.append(f"| `{cid}` | {summ} | **{stat}** | {eff} | {ci} | {art} |")

lines.append("\n## Detailed Claim Text\n")
for c in data["claims"]:
    lines.append(f"### `{c['claim_id']}`")
    lines.append(f"**Summary:** {c['claim_summary']}")
    lines.append(f"**Text:** {c['claim_text']}")
    lines.append(f"**Status:** `{c['status']}`")
    lines.append(f"**Effect Size:** `{c['effect_size']}`")
    lines.append(f"**95% CI:** `{c['ci_95']}`")
    lines.append(f"**Artifacts:** {', '.join(f'`{a}`' for a in c['primary_artifacts'])}\n")

with open(ledger_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Regenerated claim_ledger.md successfully at: {ledger_md_path}")
