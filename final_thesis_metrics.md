# Final Thesis Metrics

Generated: 2026-07-07

## Evidence Inputs

- Fresh main canonical matrix: `outputs/experiments/runs/experiments_20260707_013511`
- Completed synthetic validation manifest: `outputs/experiments/runs/experiments_20260506_192553`
- Thesis evidence bundle: `outputs/thesis_validation/final_20260707_013511`
- Track 4 action summary: `outputs/track4_action_graph/observer_state_repair3_matched500_20260522/summary_null_calibrated_hysteresis/track4_observer_state_ablation_summary.json`

## Validation

- Canonical main coverage: 12 suite leaves, 36 protocol cells observed, 0 missing protocol cells.
- Synthetic coverage: 9 synthetic cells observed, 0 missing synthetic cells.
- JSON metric scan: 74 evidence/leaf JSON files scanned, 0 parse errors, 0 non-finite numeric values.
- Evidence JSON scan: 0 NaN/Inf values.

## Headline Metrics

- Semantic signal status: `mixed_positive_not_real_control_safe`.
- Positive evidence claims: procrustes control separation, synthetic recoverability, observer relativity, Track 4 work barrier, Track 4 observer-state action separation.
- Remaining blockers: control destruction, stochastic variance separation, Track 4 traversal validity, verification provenance, procrustes verification provenance, narrative registry freeze.

### Synthetic Recoverability

- Mean NMI: 0.7433, std 0.0203.
- Mean ARI: 0.4836, std 0.0477.
- Per-kernel mean NMI: rbf 0.7390, matern 0.7658, imq 0.7252.
- Claim status: pass, thesis safe.

### Procrustes / Gauge Alignment

- Procrustes control-separation pass rate: 1.000.
- Mean procrustes ratio: 2.6237.
- Mean minimum per-control ratio: 1.4163.
- Per-kernel procrustes ratios: rbf 2.7386, matern 2.4981, imq 2.6343.
- Observer relativity mean coordinate delta: 0.1739.
- Observer relativity max rotation: 113.6812 degrees.
- Observer relativity pass rate: 1.000.

### Track 4

- Traversal aggregate: 12 runs, mean survival rate 0.6722, mean work 74.0984.
- Terrain traversal validity pass rate: 0.0833, not thesis safe.
- Work barrier signal: mean real-vs-control work gap 284.2823, pass rate 1.000, thesis safe.
- Observer-state action point estimate: 3.5629, pass, thesis safe.
- Action ratios by kernel: rbf 4.0243, matern 2.7952, imq 4.1238.
- Action ratios by seed: 42 -> 3.5751, 420 -> 3.7076, 4200 -> 3.4101.
- Action ratios by basis: track2 3.6366, integrated 3.5210.

## Notes

- The fresh synthetic run started on 2026-07-07 but did not complete within the tool timeout because the synthetic path recomputes all 60 observer universes per seed. The final evidence bundle therefore uses the latest completed canonical 3x3 synthetic validation run.
- `RESULTS.md` does not yet cite `experiments_20260707_013511`, so the narrative audit remains false even though protocol coverage is complete in the generated evidence.
- The generated evidence does not expose a direct `classification_accuracy` field; planted-label recovery is represented by NMI and ARI.
