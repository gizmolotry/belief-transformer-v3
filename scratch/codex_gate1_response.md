# Gate 1 Core Launch Blockers — Response

## Outcome

Gate 1 is cleared. The removed `mix_in_rkhs` switch no longer reaches active constructors, payloads, runner configuration, or tests; Track 4 now threads `adaptive_tpt_connectivity` explicitly and returns/exports the complete Markov observable contract. The required integration and unit suites pass, and the final architect review was **APPROVED**.

## Exact code changes

- `core/complete_pipeline.py`
  - Removed the obsolete `mix_in_rkhs` initializer parameter, `DirichletFusionConfig` keyword, component payload field, and `BeliefTransformerPipeline` instance attribute.
  - Removed the dead Mode A/Mode B branch. Dirichlet initialization now accurately reports the implemented fixed behavior: `Mode B (map-then-mix)`.

- `core/physarum_walk.py`
  - Added the missing Markov/TPT computations: stationary distribution, committor to Void, MFPT to Bridge and Void, directed reactive-flux edges and values, node throughput, dominant reactive path, Bridge/Void boundary indices, and reachability/summary metadata.
  - Added stable empty-corpus and missing-boundary-set payloads with correctly shaped arrays and explicit statuses.
  - Restored `cyclic_paths.npz` export of the full Markov observable arrays and retained `track4_markov_summary.json` summary output.
  - Added `adaptive_tpt_connectivity` directly to `run_stress_triggered_cyclic_walk`, used it for graph-repair decisions, and threaded it from `compute_corpus_walker_resistance` into the cyclic walk.
  - Returned consistent requested/effective neighbor counts, repair status, proposal mode, feature basis, adaptive flag, and Markov payloads, including empty-corpus results.

- `run_experiments.py`
  - Removed the obsolete `--mix-in-rkhs` CLI option and its argument branch.
  - Removed both writes of `mix_in_rkhs` into `mode_config`, including the unconditional Dirichlet-fusion injection that previously triggered the runtime migration error.

- `run_full_experiment_suite.py`
  - Removed the stale payload read of `DEFAULT_PIPELINE_RUNTIME_CONFIG.mix_in_rkhs`, eliminating the `AttributeError` path.
  - Removed the stale `PipelineRuntimeConfig(mix_in_rkhs=True)` construction, eliminating the `TypeError` path.

- `tests/unit/misc/test_nmi_fix.py`
  - Removed the obsolete `mix_in_rkhs=True` pipeline initializer argument. The test continues to exercise the fixed Mode B Dirichlet path without a removed switch.

- `tests/unit/contracts/test_verify_run_contract_resolution.py`
  - Added a small `_declared_features` fixture helper that supplies `manifest.target_field = "features"`.
  - Updated the two control-ordering fallback fixtures to use the current strict manifest-declared feature-source contract. No verifier behavior was weakened.

## Intentional remaining migration guard

The only active-source `mix_in_rkhs` reference is the deliberate guard in `core/pipeline_config.py`. `PipelineRuntimeConfig.from_mode_config` raises a clear `MigrationError` when an external legacy mode dictionary still contains the removed flag. This is intentional: all internal callers are clean, while unsupported external configuration fails immediately and diagnostically instead of surfacing later as a constructor `TypeError` or missing-attribute `AttributeError`.

## Verification

- `pytest tests/integration/dag_orchestration/test_core_stability.py` — **8 passed**.
- `pytest tests/unit/` — **317 passed**, with one benign joblib warning.
- Focused control-ordering tests — **2 passed**; the complete `test_verify_run_contract_resolution.py` file — **4 passed**.
- Focused payload reconstruction test for `run_full_experiment_suite.py` — **passed**.
- `py_compile` for the edited launch-path Python files — **passed**.
- Active-source `mix_in_rkhs` audit, excluding generated/review artifacts — only the intentional `core/pipeline_config.py` migration guard remains.
- Final architect verification — **APPROVED**.

## Workspace hygiene

The test-generated `track4_markov_summary.json` was restored and is not part of the final diff. Stale pytest temporary data was preserved non-destructively under the ignored `.pytest_local_tmp/preserved_gate1_20260813` path.
