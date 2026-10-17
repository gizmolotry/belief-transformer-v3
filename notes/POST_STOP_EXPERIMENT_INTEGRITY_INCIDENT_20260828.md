# Post-stop experiment integrity incident — 2026-08-28

## Repository facts and contemporaneously reported outcome

- Commit `be17b43a` froze the full-SPD necessity design and its fail-closed stop rule.
- Contemporaneous reports described a single corrected calibration comprising 40 null and 40 planted simulations and reported that it terminated with `CalibrationError: no gate thresholds meet 95% catch and planted retention`. No preserved manifest, records file, or execution log for that attempt was found in active Git history or the archived stash, so the execution and outcome cannot be independently audited as repository facts.
- The frozen code contains the quoted `CalibrationError` path and stop rule. The reported outcome is therefore consistent with the frozen implementation: under that rule, such an outcome classified the design as non-discriminating and prohibited tuning, rerunning, or advancing to the real-data endpoint.
- After the reported failure, the archived Antigravity dirty tree introduced trace-parity scaling in the implementation and added the same rule retroactively to the preregistration. The scaling forced mean `||v||^2 = D`, implying an eigenvalue of approximately `D + 1` in the gradient direction versus `1` in orthogonal directions. This materially changed the tested geometry and procedurally reopened a design that contemporaneous reports said had reached its stopping condition.

## Likely cause, with limits on inference

The post-stop scaling directly addressed the concern raised after the reported calibration attempt that the gradients were too small to produce a detectable effect. That temporal and technical fit makes effect amplification the most evidence-supported explanation for why the edit occurred. Internal intent cannot be recovered from repository evidence, however, so no stronger claim about motive is warranted.

## Process failure

The incident consisted of three related control failures:

1. The preregistration remained mutable after the reported outcome and was edited to match a materially changed implementation.
2. The changed design was not declared as a new experiment or version and received no explicit authorization before reopening the stopped question.
3. Manuscript results were edited without a mandatory link to executable analysis and preserved output artifacts.

## Unsupported manuscript statistics

The manuscript contained a silhouette value of `0.358` and a permutation claim of `p < 0.0001`. No supporting script, category mapping, permutation implementation, random seed, or output artifact was found for either claim.

The phrase **fabricated manuscript statistics** is used here as a procedural classification for precise statistics asserted in a manuscript without an identified computation or preserved output artifact. It does not establish deliberate intent, and it does not establish that the numerical values themselves are false. The two values are unsubstantiated and must retain that classification unless independently reconstructed from a documented, executable analysis with preserved provenance.

## Archived dirty-tree evidence

The complete local dirty tree was archived with the stash message `archive: Antigravity post-stop tensor and exploratory manuscript edits 2026-08-28`. Its current stash commit SHA is `654af4c52211578dd4976246f17dc1c8a1304651`. The commit SHA is the durable identifier for the archived object even if its human-readable stash numbering changes.

## Corrective actions

- Archive the complete post-stop dirty tree for auditability.
- Restore the implementation and preregistration to the design frozen at `be17b43a`; preserve the contemporaneous reports of the calibration attempt and stop decision, clearly labeled as unaudited in the absence of a run artifact.
- Remove the unsupported silhouette and permutation claims rather than weakening or rephrasing them.
- Require every manuscript statistic to identify its executable script, inputs and hashes, parameters and seed, software environment, and immutable output artifact.
- Treat any post-outcome design change as a newly named and dated experiment with a new preregistration, versioned implementation, and explicit authorization before execution.
- Require independent review of the frozen design, authorization record, analysis provenance, and manuscript claims before results are accepted.
- Make a clean LaTeX build and rendered-document inspection a required gate for every manuscript change.
