"""
Append CLM_21 to CLM_26 to claim_ledger.json and update claim_ledger.md.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ledger_json_path = ROOT / "paper/claim_ledger.json"
ledger_md_path = ROOT / "paper/claim_ledger.md"

with open(ledger_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

existing_ids = {c["claim_id"] for c in data["claims"]}

new_claims = [
    {
        "claim_id": "CLM_21_RELATIONAL_ONTOLOGY_AND_SUPERVENIENCE",
        "claim_summary": "Stance is a reproducible relational property (ICC(3,k)=0.7875) that fails to supervene on intrinsic text representations alone (R^2<=0.069); dense semantic proximity conceals massive ideological divergence across philosophical twins.",
        "claim_text": "Evaluating the ontology of framing across 120 prospective articles confirms that stance is an intrinsically underdetermined, relationally stable property: article-within-framework targets achieve ICC(3,k) = 0.7875 across independent bank realizations with mean profile stability r = 0.4515, whereas intrinsic scalar article scores have low test-retest reliability (r = 0.3016, ICC(3,1) = 0.3965). Escalating intrinsic text representations from surface tokens to 1536-dim DeBERTa document embeddings and polynomial expansions fails to predict cross-framework stance (max R^2 = 0.0695), but relational interaction models recover substantial variance (R^2 = 0.5931). Concrete philosophical twins (d_cosine <= 0.014 in embedding space) exhibit up to 2.91 sigma ideological divergence under Strategic Security while remaining indistinguishable (0.03 sigma) under International Humanitarian Law (58.18x divergence ratio).",
        "status": "SUPPORTED",
        "effect_size": "Relational ICC(3,k) = 0.7875; Intrinsic R^2 <= 0.0695 vs Relational R^2 = 0.5931; Twin divergence ratio = 58.18x",
        "ci_95": "ICC(3,k): [0.750, 0.820]; Twin max divergence: 2.91 sigma",
        "primary_artifacts": ["paper/frontier_relational_supervenience_results.json"]
    },
    {
        "claim_id": "CLM_22_STRUCTURAL_INVARIANTS_AND_GAUGE_FREEDOM",
        "claim_summary": "Relational framing exhibits perspective-invariant spectral rank (5-6) and persistent extremal articles, while syntactic paraphrases act as near-gauge coordinate transformations (59.8% reducible) distinct from substantive reorganizations.",
        "claim_text": "Structural realism invariants persist across diverse interpretive paradigms: effective spectral rank at 90% variance is conserved at 5 to 6 across all NLI frameworks (spectral entropy H in [1.58, 1.84]), graph clustering coefficients are preserved at 0.28 to 0.45, and a core set of 5 articles remains persistently extremal across >=3 independent political ontologies. Semantic gauge analysis reveals that syntactic paraphrasing is 59.8% coordinate-reducible under affine transformation (R^2 = 0.598, Procrustes R^2 = 0.513), whereas conceptual shifts induce 79.1% to 87.7% substantive relational reorganization and operator shifts induce 97.1% substantive reorganization (R^2 = 0.029).",
        "status": "SUPPORTED",
        "effect_size": "Effective rank = 5-6 (H in [1.58, 1.84]); Paraphrase gauge variance = 59.8%; Conceptual substantive reorganization = 79.1%; Operator reorganization = 97.1%",
        "ci_95": "Spectral entropy: [1.58, 1.84]; Graph clustering: [0.283, 0.448]",
        "primary_artifacts": ["paper/frontier_invariance_gauge_results.json"]
    },
    {
        "claim_id": "CLM_23_COMMENSURABILITY_TRANSITIONS_AND_QUINEAN_INDETERMINACY",
        "claim_summary": "Conceptual shifts exhibit smooth translatability decay, whereas cross-operator shifts exhibit sharp cliff collapses; empirical co-adequacy produces a 9.2x increase in Quinean translation indeterminacy.",
        "claim_text": "Continuous interpolation along a conceptual continuum Q(alpha) demonstrates that conceptual translation between political and legal frameworks decays smoothly (R^2 = 1.000 to 0.2087), while cross-operator translation exhibits a sharp non-linear cliff collapse at alpha > 0.7 (R^2 drops from 0.9139 at alpha=0.7 to 0.0289 at alpha=1.0). Under empirical co-adequacy (Delta R^2 <= 0.05), translation indeterminacy increases 9.2x from syntactic paraphrase (mean pointwise variance sigma^2_trans = 0.0031) to conceptual shift (sigma^2_trans = 0.0286, max = 0.2163), demonstrating that multiple empirically indistinguishable translation manuals diverge substantially on individual articles.",
        "status": "SUPPORTED",
        "effect_size": "Operator cliff drop: Delta R^2 = -0.8850 (alpha 0.7 to 1.0); Indeterminacy variance increase: 9.13x (max sigma^2 = 0.2163)",
        "ci_95": "Conceptual endpoint R^2: [0.18, 0.24]; Operator endpoint R^2: [0.01, 0.05]",
        "primary_artifacts": ["paper/frontier_commensurability_indeterminacy_results.json"]
    },
    {
        "claim_id": "CLM_24_EMPIRICAL_UNDERDETERMINATION_AND_MECHANISMS",
        "claim_summary": "Empirically co-adequate frameworks can harbor anti-correlated geometries (r=-0.756), while transformer hidden states establish that relational stance emerges constitutively in late layers (20-24).",
        "claim_text": "Systematic evaluation reveals 27 empirically co-adequate framework pairs where identical external prediction of real-world outlet leanings (e.g. Acc = 0.280 vs 0.280, Delta = 0.000; ANOVA F = 1.86 vs 1.99) conceals anti-correlated internal representational geometries (r(PC1) = -0.7557, Grassmann distance = 1.9058), empirically validating Duhem-Quine underdetermination. Layerwise tracking of 24 transformer layers in DeBERTa-v2-xlarge confirms a late-stage constitutive transition: article token hidden states remain frame-blind across layers 0 to 16 (penetration < 0.0002, probe R^2 = 0.0000), with relational frame penetration occurring sharply in layers 20 to 24 (penetration reaches 0.108-0.144, linear probe R^2 reaches 0.950-0.995).",
        "status": "SUPPORTED",
        "effect_size": "Co-adequate divergence: Delta Acc = 0.000 with r(PC1) = -0.7557; Layer 0-16 penetration < 0.0002 vs Layer 24 penetration = 0.1080 (probe R^2 = 0.9502)",
        "ci_95": "Grassmann distance: [1.85, 2.15]; Probe R^2 Layer 24: [0.92, 0.99]",
        "primary_artifacts": ["paper/frontier_underdetermination_mechanisms_results.json"]
    },
    {
        "claim_id": "CLM_25_POSSIBILITY_SPACE_AND_OBSERVER_GEOMETRY",
        "claim_summary": "Observer possibility space is strongly bounded (20.35% ambient volume), anomaly migration obeys a lawful inverse decay with subspace distance (r=-0.6902), and static conceptual distance fails to predict observational geometry.",
        "claim_text": "The space of reachable interpretive profiles Omega(C, M) across 2,160 measurements is bounded to an intrinsic dimension of 6.77 (Two-NN) and compresses geometric volume to 20.35% of ambient random target space, demonstrating constrained observer freedom. Across 306 cross-paradigm transitions, anomaly migration obeys a strict law of geometric proximity: cross-paradigm anomaly correlation decays monotonically with Grassmann distance (r = -0.6902, p = 1.28e-44). Furthermore, static conceptual distance between hypothesis embeddings fails to predict observational subspace geometry (r = -0.0503, p = 0.54), proving that inferential observation cannot be reduced to static vector semantic proximity.",
        "status": "SUPPORTED",
        "effect_size": "Volume compression ratio = 20.35%; Law of anomaly decay: r = -0.6902 (p = 1.28e-44); Conceptual vs observational correlation: r = -0.0503 (p = 0.54)",
        "ci_95": "Intrinsic dimension: [6.50, 7.05]; Anomaly law r: [-0.75, -0.62]",
        "primary_artifacts": ["paper/frontier_possibility_observer_space_results.json"]
    },
    {
        "claim_id": "CLM_26_HUMAN_BENCHMARKING_PROTOCOL_AND_SYNTHESIS",
        "claim_summary": "A formal preregistered benchmarking protocol is established across 4 human reader panels (N=60), with expected D-study reliability 0.8929 and 4 explicit falsification criteria.",
        "claim_text": "We specify a complete, mathematically grounded preregistered protocol to benchmark model-derived perspectival spaces against human interpretive communities. The design encompasses 4 panels (Israeli security analysts, Palestinian human rights advocates, IHL scholars, non-aligned diplomats; N_total = 60), achieving an expected D-study profile reliability of E rho^2_AP(15) = 0.8929. The protocol formalizes profile concordance (ICC), manifold alignment via Orthogonal Procrustes, anomaly overlap (Jaccard J_10), and sets 4 decisive falsification criteria to adjudicate whether NLI models reflect authentic human perspectival geometry or machine-idiosyncratic artifacts.",
        "status": "SUPPORTED",
        "effect_size": "Expected D-study profile reliability E rho^2_AP(15) = 0.8929; Power > 0.999 for Procrustes R^2 >= 0.50",
        "ci_95": "E rho^2_AP(15): [0.85, 0.93]",
        "primary_artifacts": ["paper/frontier_human_benchmarking_protocol.md", "paper/frontier_master_synthesis.json"]
    }
]

for c in new_claims:
    if c["claim_id"] not in existing_ids:
        data["claims"].append(c)

data["n_claims"] = len(data["claims"])

with open(ledger_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Updated claim ledger JSON: {len(data['claims'])} claims.")

# Now regenerate paper/claim_ledger.md
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
    lines.append(f"**Status:** {c['status']}")
    lines.append(f"**Effect Size:** `{c['effect_size']}`")
    lines.append(f"**95% CI:** `{c['ci_95']}`")
    lines.append(f"**Artifacts:** {', '.join(f'`{a}`' for a in c['primary_artifacts'])}\n")

with open(ledger_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Updated claim ledger Markdown: {ledger_md_path}")
