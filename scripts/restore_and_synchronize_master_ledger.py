# -*- coding: utf-8 -*-
"""
Restoration and synchronization script for Master Claim Ledger v4.3.
Restores historical claims CLM_01 through CLM_33 and integrates Gate 3 findings
(CLM_34 and CLM_35) across all authoritative Master Claim Ledger locations.
"""

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. Load claim_ledger.json
json_path = ROOT / 'paper/claim_ledger.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Initial claims in json: {len(data['claims'])}")

# 2. Gate 3 Claims Definitions
clm_34 = {
    "claim_id": "CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH",
    "claim_summary": "Centered Kernel Alignment (CKA) across N=6,248 independent story family representatives reveals that hypothesis-prompt invariance (Q0 <-> Q1 paraphrase stability) is substantially higher at the middle layer (Layer 12 CLS, V7 <-> V8: CKA_RBF = 0.9131, Linear: 0.9073, 95% CI: [0.9119, 0.9176], permutation p < 0.0020) than in final hidden representations (Layer 24 CLS, V3 <-> V4: CKA_RBF = 0.7770, Linear: 0.7740, 95% CI: [0.7720, 0.7857], p < 0.0020) and paired scalar classifier margins (V1 <-> V2: CKA_RBF = 0.7490, Linear: 0.8015, 95% CI: [0.7401, 0.7594], p < 0.0020); sampling three representation depths establishes boundary and midpoint differences rather than monotonic behavior at every intervening layer.",
    "claim_text": "Multi-view Centered Kernel Alignment (CKA) across N=6,248 independent story family representatives (evaluating against 500-permutation null distributions and B=200 cluster bootstrap iterations) establishes that prompt-invariance geometry varies systematically across network depth: (1) Prompt stability between direct probes (Q0) and paraphrased probes (Q1) is highest at the layer midpoint and lower at the classifier head across the sampled depths: Layer 12 CLS (V7 <-> V8) achieves CKA_RBF = 0.9131 (Linear: 0.9073, 95% CI: [0.9119, 0.9176], p < 0.0020, 0/500 null exceedances), Layer 24 CLS (V3 <-> V4) achieves CKA_RBF = 0.7770 (Linear: 0.7740, 95% CI: [0.7720, 0.7857], p < 0.0020), and the paired scalar margin (V1 <-> V2) achieves CKA_RBF = 0.7490 (Linear: 0.8015, 95% CI: [0.7401, 0.7594], p < 0.0020); we explicitly note that three sampled representations establish differences between midpoint, final layer, and margins rather than monotonic behavior at every unobserved intervening layer. (2) Cross-view geometry between scalar margins (V1) and hidden-state views reveals moderate coupling to final-layer representations (Layer 24 CLS V3: CKA = 0.5437; Layer 24 Mean V5: CKA = 0.5034), but sharp decoupling from middle-layer representations (Layer 12 CLS V7: CKA = 0.1196). (3) Sensitivity analysis on N=6,248 family representatives versus the full N=7,986 selector union confirms that global manifold alignment is resilient to syndicated copies (|Delta CKA| <= 0.0408; Scalar Delta = -0.0080, Mid-CLS Delta = +0.0150, Final-CLS Delta = +0.0408), while local k-NN graph overlap is substantially inflated (+0.076 to +0.099, up to 4.0x) by wire reproduction, validating restriction to independent story families. (4) Quadratic-time unbiased MMD^2 across eligible selector conditions confirms significant distributional divergence across topic selection boundaries (all permutation p < 0.0020).",
    "status": "QUALIFIED_SUPPORT",
    "effect_size": "Prompt Invariance CKA: Layer 12 CLS = 0.9131, Layer 24 CLS = 0.7770, Scalar Margin = 0.7490; Cross-View Scalar-to-Hidden CKA: V1-V3 = 0.5437, V1-V5 = 0.5034, V1-V7 = 0.1196; Syndication Sensitivity |Delta CKA| <= 0.0408",
    "ci_95": "Layer 12 CKA 95% CI: [0.9119, 0.9176]; Layer 24 CKA 95% CI: [0.7720, 0.7857]; Scalar CKA 95% CI: [0.7401, 0.7594]; 500-permutation null controls (all p < 0.0020); B=200 cluster bootstrap resamples on N=6,248 independent families",
    "primary_artifacts": [
        "deliverables/05_gate3_full_experiment/cka_matrix_6248.json",
        "deliverables/05_gate3_full_experiment/gate3_executive_synthesis.md",
        "deliverables/05_gate3_full_experiment/gate3_full_report.json",
        "deliverables/05_gate3_full_experiment/sensitivity_analysis.json",
        "deliverables/05_gate3_full_experiment/mmd_eligible_conditions.json"
    ]
}

clm_35 = {
    "claim_id": "CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION",
    "claim_summary": "In a balanced 120-template / N=480 passage controlled intervention suite benchmarked against a 512-article real news panel, final-layer representations (V3 Final-CLS) exhibit high target contrast specificity (ratio of means = 2.01x, p = 2.92e-46) and pronounced relational discrimination in ambient news space (Delta R10 = +66.58 pp, p = 3.75e-47), whereas middle-layer representations (V7 Mid-CLS) fail specificity (ratio of means = 0.92x, p = 0.910) and do not detect relational discrimination (Delta R10 = +0.75 pp, p = 0.360; failure to detect discrimination is not proof of exact insensitivity). This demonstrates an empirical dissociation between prompt stability, proposition specificity, and relational discrimination rather than an inherent universal trade-off. Distance support audits confirm that synthetic passage edits displace outward relative to intra-panel news distributions (Kolmogorov-Smirnov rejection p = 0.005).",
    "claim_text": "A controlled semantic intervention audit across 120 base templates (N=480 matched passages across 8 contrast dimensions: original, paraphrase, proposition reversal, and surface control) evaluated against a frozen 512-article real news reference panel establishes: (1) Dual ratio reporting resolves the distinction between aggregate displacement and template-level ratios: for V3 Final-CLS, the ratio of aggregate means is 2.01x (template mean ratio 2.04x, median 1.91x, 100.0% win rate, p = 2.92e-46); for V1 Scalar Margin, the ratio of means is 2.29x (template mean 2.49x, median 2.19x, 93.3% win rate, p = 2.81e-34); whereas V7 Mid-CLS fails specificity with a ratio of aggregate means of 0.92x (< 1.0; template mean 1.14x skewed by small-denominator tail, median 1.03x, win rate 51.7%, p = 0.910). (2) Ambient nearest-neighbor graph retention (R10 in real news space) confirms pronounced relational discrimination in the final layer (V3): meaning-preserving rewrites retain 71.33% (J10 = 0.6121) of news neighbors, while proposition reversals collapse retention to 4.75% (J10 = 0.0312), yielding Delta R10 = +66.58 pp (p = 3.75e-47) and significantly exceeding surface controls (22.08%, Delta R10 = +17.33 pp, p = 1.25e-8). In contrast, middle-layer (V7) retention discrimination is not detected: paraphrase retains 51.67%, surface control retains 47.67%, and proposition reversal retains 50.92% (Delta R10 = +0.75 pp, p = 0.360; failure to detect discrimination under this test does not prove exact insensitivity). (3) The joint data support an empirical dissociation among three distinct measurement properties (prompt stability, proposition specificity, and relational discrimination) rather than an inherent trade-off: Layer 12 CLS exhibits high prompt stability (CKA 0.913) but lacks detected specificity (0.92x) or neighborhood discrimination (+0.75 pp), whereas Layer 24 CLS exhibits moderate prompt stability (CKA 0.777) with strong specificity (2.01x) and high discrimination (+66.58 pp). Maximizing representational invariance alone is therefore an insufficient criterion for instrument validity. (4) Distance support audits reveal that edited passages exhibit outward distributional displacement (1st NN medians rise from 0.9255 to 0.9814 - 1.1098, exceeding the upper quartile of real-to-real distances 1.0278), and two-sample Kolmogorov-Smirnov testing decisively rejects the null hypothesis of distributional identity against the real news panel (10th NN: KS = 0.1732, p = 0.00505; 1st NN: KS = 0.2355, p = 3.16e-5), bounding assertions of manifold membership.",
    "status": "QUALIFIED_SUPPORT",
    "effect_size": "V3 Final-CLS Specificity: Ratio of Means = 2.01x (Template Mean 2.04x, p = 2.92e-46); V1 Scalar Margin: Ratio of Means = 2.29x (Template Mean 2.49x, p = 2.81e-34); V7 Mid-CLS Specificity: Ratio of Means = 0.92x (p = 0.910); Ambient R10 Retention Drop (Pres - Chan): V3 = +66.58 pp (p = 3.75e-47), V1 = +62.33 pp, V7 = +0.75 pp (p = 0.360); Surface Edit R10: V3 = 22.08% (Reversal < Surface by +17.33 pp, p = 1.25e-8); Distance Support KS Test: 10th NN KS = 0.1732 (p = 0.00505), 1st NN KS = 0.2355 (p = 3.16e-5)",
    "ci_95": "Paired t-test across 120 balanced intervention templates (N=480 passages, 8 dimensions); Kolmogorov-Smirnov continuous distribution tests vs N=512 real news panel; Exact R10 and Jaccard J10 neighbor statistics",
    "primary_artifacts": [
        "deliverables/07_hidden_state_specificity_report.md",
        "deliverables/07_hidden_state_specificity_report_data.npz",
        "results/controlled_hidden_state_diagnostic_audit.json",
        "results/controlled_hidden_state_neighborhood_report.json",
        "data/controlled_semantic_interventions_manifest.json"
    ]
}

existing_ids = {c['claim_id']: idx for idx, c in enumerate(data['claims'])}

if 'CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH' not in existing_ids:
    data['claims'].append(clm_34)
else:
    data['claims'][existing_ids['CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH']] = clm_34

if 'CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION' not in existing_ids:
    data['claims'].append(clm_35)
else:
    data['claims'][existing_ids['CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION']] = clm_35

data['n_claims'] = len(data['claims'])
data['timestamp'] = '2026-10-17T12:00:00Z'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated claim_ledger.json with {len(data['claims'])} claims.")

def generate_markdown(title: str, version_tag: str, is_deliverable: bool = False) -> str:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Version:** {version_tag}")
    lines.append(f"**Total Claims:** {len(data['claims'])}")
    lines.append(f"**Date:** October 2026")
    lines.append("**Audit Standard:** Dual ratio accounting, exact neighbor retention ($R_{10}$), matched surface-control contrast, distributional distance support qualification, KS null rejection reporting, and complete provenance tracking.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Preamble & Epistemic Scope")
    lines.append("")
    lines.append("This Master Claim Ledger preserves the complete epistemic history of the Belief Transformer project across all empirical phases:")
    lines.append("- **Phase 1: Kuhnian Framework Foundations & Generalizability Theory** (`CLM_01` \u2013 `CLM_20`)")
    lines.append("- **Phase 2: Relational Ontology, Gauge Invariance & Observer Space** (`CLM_21` \u2013 `CLM_26`)")
    lines.append("- **Phase 3: Psychometric Invariance, Measurement Multiverse & Construct Identifiability** (`CLM_27` \u2013 `CLM_33`)")
    lines.append("- **Phase 4: Multi-View Representation Invariance & Controlled Hidden-State Specificity** (`CLM_34` \u2013 `CLM_35`)")
    lines.append("")
    lines.append("The ledger explicitly records and preserves all historical claims, including failed tests (`CLM_23A_SCORE_MIXTURE_TRANSITION_CONTINUUM`), prospective protocols (`CLM_26_HUMAN_BENCHMARKING_PROTOCOL`), exploratory mechanisms (`CLM_24B_CAUSAL_ACTIVATION_PATCHING_CONSTITUTION`), partially supported criteria (`CLM_22_SEMANTIC_COORDINATE_REDUCIBILITY`), and qualified support boundaries (`CLM_27` through `CLM_35`), preventing unearned generalizations or post-hoc erasure of falsified predictions.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary Table of Claims")
    lines.append("")
    lines.append("| ID | Claim Summary | Status | Effect Size | 95% CI / Bound | Artifacts |")
    lines.append("| :--- | :--- | :---: | :--- | :--- | :--- |")
    
    for c in data['claims']:
        cid = c['claim_id']
        summ = c['claim_summary'].replace('|', '\\|')
        stat = c['status']
        eff = c['effect_size'].replace('|', '\\|')
        ci = c['ci_95'].replace('|', '\\|')
        art = ', '.join(c['primary_artifacts'])
        lines.append(f"| `{cid}` | {summ} | **{stat}** | {eff} | {ci} | {art} |")
        
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Claim Documentation")
    lines.append("")
    
    for c in data['claims']:
        cid = c['claim_id']
        lines.append(f"### `{cid}`")
        lines.append(f"**Summary:** {c['claim_summary']}")
        lines.append("")
        lines.append(f"**Text:** {c['claim_text']}")
        lines.append("")
        lines.append(f"**Status:** `{c['status']}`")
        lines.append("")
        lines.append(f"**Effect Size:** `{c['effect_size']}`")
        lines.append("")
        lines.append(f"**95% CI:** `{c['ci_95']}`")
        lines.append("")
        art = ', '.join(c['primary_artifacts'])
        lines.append(f"**Artifacts:** {art}")
        lines.append("")
        
    return "\n".join(lines)

# Write targets
md_root = generate_markdown(
    title="Belief Transformer: Audited Master Claim Ledger",
    version_tag="4.3 (Consolidated Gate 3 Multi-View Invariance & Controlled Hidden-State Specificity Audit)",
    is_deliverable=False
)
with open(ROOT / 'paper/claim_ledger.md', 'w', encoding='utf-8') as f:
    f.write(md_root)

md_v4 = generate_markdown(
    title="Belief Transformer: Master Audited Claim Ledger (v4.3)",
    version_tag="4.3 (Consolidated Gate 3 Multi-View Invariance & Controlled Hidden-State Specificity Audit)",
    is_deliverable=True
)
with open(ROOT / 'deliverables/08_master_audited_claim_ledger_v4.md', 'w', encoding='utf-8') as f:
    f.write(md_v4)

with open(ROOT / 'paper/consolidated_manuscript/03_master_audited_claim_ledger.md', 'w', encoding='utf-8') as f:
    f.write(md_v4)

# 5. Verification & Hash Computation
h_deliv = hashlib.sha256((ROOT / 'deliverables/08_master_audited_claim_ledger_v4.md').read_bytes()).hexdigest()
h_manus = hashlib.sha256((ROOT / 'paper/consolidated_manuscript/03_master_audited_claim_ledger.md').read_bytes()).hexdigest()
h_exec = hashlib.sha256((ROOT / 'deliverables/07_hidden_state_specificity_report.md').read_bytes()).hexdigest()

print(f"deliverables/08 SHA-256: {h_deliv}")
print(f"manuscript/03   SHA-256: {h_manus}")
print(f"exec report 07  SHA-256: {h_exec}")

assert h_deliv == h_manus, "Ledger files mismatch!"
assert h_deliv != h_exec, "Ledger matches executive report! Must be distinct!"

# Exhaustive integrity checks
cids = [c['claim_id'] for c in data['claims']]
assert len(cids) == 39, f"Expected 39 claims, got {len(cids)}"
assert cids[0] == 'CLM_01_G_THEORY_VARIANCE_PARTITIONING'
assert cids[36] == 'CLM_33_CONSTRUCT_ALIASING_AND_SEMANTIC_IDENTIFIABILITY'
assert cids[37] == 'CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH'
assert cids[38] == 'CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION'

status_map = {c['claim_id']: c['status'] for c in data['claims']}
assert status_map['CLM_01_G_THEORY_VARIANCE_PARTITIONING'] == 'SUPPORTED'
assert status_map['CLM_20_PHILOSOPHICAL_STRUCTURAL_REALISM_HYPOTHESIS'] == 'PHILOSOPHICAL_INTERPRETATION'
assert status_map['CLM_22_SEMANTIC_COORDINATE_REDUCIBILITY'] == 'PARTIALLY_SUPPORTED'
assert status_map['CLM_23A_SCORE_MIXTURE_TRANSITION_CONTINUUM'] == 'FAILED_HYPOTHESIS'
assert status_map['CLM_24B_CAUSAL_ACTIVATION_PATCHING_CONSTITUTION'] == 'EXPLORATORY'
assert status_map['CLM_26_HUMAN_BENCHMARKING_PROTOCOL'] == 'PROSPECTIVE_PROTOCOL'
assert status_map['CLM_27_MTMM_CONSTRUCT_VALIDITY_AND_METHOD_EFFECTS'] == 'QUALIFIED_SUPPORT'
assert status_map['CLM_31_NATURAL_EDIT_RESPONSE_TOMOGRAPHY'] == 'QUALIFIED_SUPPORT'
assert status_map['CLM_32_MEASUREMENT_MULTIVERSE_STATIC_EDIT_COUPLING'] == 'QUALIFIED_SUPPORT'
assert status_map['CLM_33_CONSTRUCT_ALIASING_AND_SEMANTIC_IDENTIFIABILITY'] == 'QUALIFIED_SUPPORT'
assert status_map['CLM_34_MULTI_VIEW_REPRESENTATION_INVARIANCE_AND_DEPTH'] == 'QUALIFIED_SUPPORT'
assert status_map['CLM_35_HIDDEN_STATE_SPECIFICITY_GRAPH_RETENTION_AND_DISSOCIATION'] == 'QUALIFIED_SUPPORT'

for p in [ROOT / 'paper/claim_ledger.md', ROOT / 'deliverables/08_master_audited_claim_ledger_v4.md', ROOT / 'paper/consolidated_manuscript/03_master_audited_claim_ledger.md']:
    t = p.read_text(encoding='utf-8')
    for cid in cids:
        assert f'`{cid}`' in t, f'{cid} missing from table in {p.name}'
        assert f'### `{cid}`' in t, f'{cid} missing from detail in {p.name}'
    assert '**Total Claims:** 39' in t

exec_text = (ROOT / 'deliverables/07_hidden_state_specificity_report.md').read_text(encoding='utf-8')
assert exec_text.startswith('# Hidden-State Contrast Specificity, Ambient Graph Retention, and Distance Support Audit')
assert 'CLM_01' not in exec_text, 'Executive report should not contain claim ledger table!'

print("ALL 12 INTEGRITY AND PROVENANCE CHECKS PASSED PERFECTLY!")

