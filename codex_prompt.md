# Codex Independent Audit Task

You are an adversarial, skeptical code and empirical auditor.
Your job is to independently inspect, verify, and stress-test the latest empirical results in \d:\belief-transformer\V3\.
Do NOT trust any summaries or reports. Check the actual code, raw cached arrays, JSON results, and data files directly.

Specifically audit the following 3 critical items:

1. THE PROSPECTIVE CONFIRMATORY REPLICATION (Q0 + Q1 on N=120):
   - Inspect \data/prospective_confirmatory_120_corpus.jsonl\ and compare its article fingerprints against \data/touched_fingerprints_registry.json\. Is there ANY overlap or leakage with the 1,290 excluded fingerprints?
   - Inspect \paper/prospective_confirmatory_replication_results.json\ and \data/cache_prospective_120_q0_q1_logits.pt\.
   - Independently recalculate the PCA EVR, participation ratio, and cross-bank correlation between Q0 and Q1 from the raw data. Do the numbers match the reported values (Q0 PC1=58.36%, Q1 PC1=56.09%, Pearson r=0.9253)?
   - Verify the audited paired operator: is it strictly computing (P_A(E) - P_A(C)) - (P_B(E) - P_B(C))?

2. THE UKRAINE NULL RESULT:
   - Inspect \scripts/exp_paired_operator_reruns.py\ and \data/ukraine_N60_corpus.jsonl\.
   - Inspect the cached logits in \data/cache_paired_logits_russia_ukraine_war_N60.pt\.
   - Check whether the null result (PC1=48.02%, p=0.3414) is a genuine absence of shared rank-one structure or a bug / degenerate probe bank:
     * Do the individual hypothesis pairs show variance across articles?
     * What are the pairwise correlations between the 8 Ukraine axes?
     * Why did PC2 nominally clear (20.87%, p=0.0328) while PC1 failed? Is this an artifact of ordered eigenvalue null testing?
     * Inspect \data/ukraine_N60_corpus.jsonl\ for syndicated wire duplicates or irrelevant texts.

3. THE CROSS-MODEL CORRELATION NUMBERS:
   - Inspect \paper/paired_operator_cross_model_cross_topic_rerun.json\ and the cached logits for DeBERTa, BART, and RoBERTa on N=60 (\data/cache_paired_logits_{model}_N60.pt\).
   - Independently compute the pairwise correlations (DeBERTa vs BART, DeBERTa vs RoBERTa, BART vs RoBERTa) from the cached logits under the paired operator (s_A - s_B).
   - Verify whether BART vs RoBERTa is genuinely r = 0.5470 (p = 6.12e-6) and whether DeBERTa vs BART is genuinely r = 0.8057.
   - Check if there are any sign-flip ambiguities or data leakage between models.

Write your complete, independent findings to \paper/independent_codex_audit_report.md\. Include your raw calculations and a clear audit verdict on each target.
