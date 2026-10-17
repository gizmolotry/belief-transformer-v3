import hashlib
from copy import deepcopy

import numpy as np
import pytest

from core.full_spd_experiment import (
    BANKS,
    NEGATIVE_CONTROLS,
    ArmStatistics,
    BankArtifact,
    BankManifest,
    CalibrationError,
    ContainerValidationError,
    Decision,
    EccentricityContainer,
    FoldAssignment,
    GateEvaluation,
    GateMetrics,
    GateThresholds,
    ManifestValidationError,
    NegativeControlResult,
    RankingCollapseError,
    SimulationSummary,
    arm_statistics,
    calibrate_thresholds,
    decide,
    deterministic_publisher_folds,
    evaluate_cached,
    holm_adjust_three,
    midranks,
    negative_candidate_substitution,
    permutation_inference,
    publisher_cluster_bootstrap_indices,
    q_within_publisher_permutation,
    robustness_fraction,
    run_real,
    validate_bank_manifest,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(*, same_raw=False):
    ids = tuple(f"a{i:03d}" for i in range(320))
    publishers = {article_id: f"p{i // 20:02d}" for i, article_id in enumerate(ids)}
    artifacts = {
        bank: BankArtifact(bank, _hash("prompt" + bank), _hash("rawQ0" if same_raw else "raw" + bank), ids)
        for bank in BANKS
    }
    return BankManifest(artifacts, publishers)


def _container():
    # Five folds, two publishers/fold, four articles/publisher.  The statistical
    # engine is size-agnostic; the strict 320/16x20 gate belongs to the manifest.
    ids = tuple(f"a{i:03d}" for i in range(40))
    publishers = {article_id: f"p{i // 4:02d}" for i, article_id in enumerate(ids)}
    folds = deterministic_publisher_folds(ids, publishers)
    seeds = (42, 420)
    arms = ("full", "identity", "oof_conformal", "scalar_trace", "diagonal", *NEGATIVE_CONTROLS)
    base = np.arange(8, dtype=float)
    reverse = base[::-1]
    rotate = np.roll(base, 2)
    patterns = {
        "full": (base, base, base),
        "identity": (base, reverse, rotate),
        "oof_conformal": (base, rotate, reverse),
        "scalar_trace": (base, reverse, reverse),
        "diagonal": (base, rotate, rotate),
        "article_shuffled": (base, reverse, reverse),
        "observer_shuffled": (base, reverse, rotate),
        "spectrum_matched_random": (base, rotate, reverse),
    }
    values = {}
    for fold in range(5):
        assert len(folds.heldout_ids[fold]) == 8
        for seed in seeds:
            for arm, banks in patterns.items():
                for bank, vector in zip(BANKS, banks):
                    values[(fold, seed, bank, arm)] = vector + seed * 1e-6
    return EccentricityContainer(ids, publishers, folds, seeds, arms, values)


def _negative_results():
    return tuple(NegativeControlResult(name, .01, .1, Decision.FAIL) for name in NEGATIVE_CONTROLS)


def _thresholds():
    return GateThresholds(.01, .99, -1.0, 10.0, 10.0, .01)


def _threshold_mapping():
    return {"tau_disp": .01, "tau_tie": .99, "tau_gen": -1.0, "tau_gap": 10.0, "tau_amp": 10.0, "delta_min": .01}


def _cached_cells(*, provenance=False):
    container = _container()
    cells = []
    gate = {"dispersion": 1.0, "tie_fraction": 0.0, "generalization": 1.0, "generalization_gap": 0.0, "amplification": 1.0}
    required = ("full", "identity", "oof_conformal", "scalar_trace", "diagonal")
    for fold in range(5):
        for seed in container.seeds:
            for bank in BANKS:
                cell = {
                    "ordered_ids": list(container.ordered_ids), "publisher_by_id": dict(container.publisher_by_id),
                    "fold_for_publisher": dict(container.folds.fold_for_publisher), "fold": fold, "seed": seed, "bank": bank,
                    "eccentricities": {arm: container.values[(fold, seed, bank, arm)].tolist() for arm in container.arms},
                    "gate_metrics": {arm: dict(gate) for arm in required}, "connected": {arm: True for arm in required},
                    "construction_valid": True, "provenance_valid": True, "corpus_control_passed": False,
                }
                if provenance:
                    cell["provenance"] = {"q_bank_hash": _hash("prompt" + bank), "target_source_kind": "observer_derived"}
                cells.append(cell)
    return cells


def test_bank_manifest_requires_exact_distinct_bound_arrays_and_real_gate():
    assert len(validate_bank_manifest(_manifest())) == 320
    with pytest.raises(ManifestValidationError, match="distinct raw-array"):
        validate_bank_manifest(_manifest(same_raw=True))


def test_deterministic_five_publisher_grouped_folds_keep_publishers_together():
    container = _container()
    folds_again = deterministic_publisher_folds(container.ordered_ids, container.publisher_by_id)
    assert folds_again == container.folds
    for fold, article_ids in container.folds.heldout_ids.items():
        assert all(container.folds.fold_for_publisher[container.publisher_by_id[i]] == fold for i in article_ids)


def test_primary_statistics_reselects_actual_strongest_comparator():
    stats = arm_statistics(_container())
    assert stats.strongest_comparator in {"identity", "oof_conformal", "scalar_trace", "diagonal"}
    assert stats.t_by_arm[stats.strongest_comparator] == max(stats.t_by_arm[a] for a in ("identity", "oof_conformal", "scalar_trace", "diagonal"))
    assert stats.delta == pytest.approx(stats.t_by_arm["full"] - stats.t_by_arm[stats.strongest_comparator])


def test_permutations_hold_q0_fixed_and_share_map_across_all_arms():
    container = _container()
    mapping = q_within_publisher_permutation(container, np.random.default_rng(7))
    for fold in range(5):
        assert np.array_equal(mapping[(fold, "Q0")], np.arange(8))
        ids = container.folds.heldout_ids[fold]
        for bank in ("Q1", "Q2"):
            for dest, source in enumerate(mapping[(fold, bank)]):
                assert container.publisher_by_id[ids[dest]] == container.publisher_by_id[ids[source]]
    result = permutation_inference(container, n_replicates=4, seed=4)
    assert result.p_value == pytest.approx((1 + np.count_nonzero(result.null_deltas >= result.observed.delta)) / 5)
    assert all(np.array_equal(m[(0, "Q0")], np.arange(8)) for m in result.mappings)


def test_bootstrap_resamples_publisher_clusters_with_duplicate_article_multiplicity():
    container = _container()
    duplicate = None
    for seed in range(100):
        sample = publisher_cluster_bootstrap_indices(container, np.random.default_rng(seed))
        if len(np.unique(sample[0])) < len(sample[0]):
            duplicate = sample[0]
            break
    assert duplicate is not None
    assert len(duplicate) == 8
    assert len(np.unique(duplicate)) < 8


def test_true_ties_collapse_and_never_use_article_ids_as_inferential_ranks():
    with pytest.raises(RankingCollapseError):
        midranks([1.0, 1.0, 1.0])
    ranks = midranks([1.0, 1.0, 2.0, 2.0])
    assert np.array_equal(ranks, [1.5, 1.5, 3.5, 3.5])


def test_holm_is_exactly_three_controls_and_monotone_step_down():
    adjusted = holm_adjust_three({"article_shuffled": .01, "observer_shuffled": .02, "spectrum_matched_random": .04})
    assert adjusted == {"article_shuffled": .03, "observer_shuffled": .04, "spectrum_matched_random": .04}
    with pytest.raises(ContainerValidationError, match="exactly"):
        holm_adjust_three({"article_shuffled": .01})


def test_negative_candidate_substitution_uses_control_as_candidate_not_full():
    container = _container()
    substituted = negative_candidate_substitution(container, "article_shuffled")
    assert "article_shuffled" in substituted.t_by_arm
    assert substituted.delta == pytest.approx(substituted.t_by_arm["article_shuffled"] - substituted.t_by_arm[substituted.strongest_comparator])


def test_decision_uses_strict_p_and_fail_vs_inconclusive_vs_invalid_semantics():
    stats = arm_statistics(_container())
    deltas = {(42, "Q0", "Q1"): .1, (42, "Q0", "Q2"): .1, (42, "Q1", "Q2"): .1, (420, "Q0", "Q1"): .1, (420, "Q0", "Q2"): .1, (420, "Q1", "Q2"): .1}
    valid = GateEvaluation(True, ())
    strict = decide(construction_valid=True, gate=valid, statistics=stats, p_value=.05, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=_negative_results())
    assert strict.decision is Decision.FAIL and "strict_permutation" in strict.reasons
    inconclusive = decide(construction_valid=True, gate=GateEvaluation(False, ("ties",)), statistics=stats, p_value=.001, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=_negative_results())
    assert inconclusive.decision is Decision.INCONCLUSIVE
    invalid = decide(construction_valid=False, gate=valid, statistics=stats, p_value=.001, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=_negative_results())
    assert invalid.decision is Decision.INVALID


def test_threshold_failures_are_inconclusive_and_robustness_is_exact_fraction():
    from core.full_spd_experiment import evaluate_gates
    report = evaluate_gates(GateMetrics(.0, 1.0, -2.0, 11.0, 11.0), _thresholds())
    assert not report.valid
    assert set(report.failures) == {"dispersion", "ties", "generalization", "generalization_gap", "amplification"}
    assert robustness_fraction({(1, "Q0", "Q1"): 1.0, (1, "Q0", "Q2"): 0.0, (1, "Q1", "Q2"): 1.0, (2, "Q0", "Q1"): 1.0, (2, "Q0", "Q2"): 1.0}) == .8


def test_calibration_requires_95_percent_gate_separation_and_selects_smallest_margin():
    records = []
    records += [SimulationSummary("no_effect", 1, 0, 1, 0, 1, 0.0) for _ in range(20)]
    records += [SimulationSummary("planted", 1, 0, 1, 0, 1, .2) for _ in range(20)]
    records += [SimulationSummary("collapse", 0, 1, -1, 2, 9, 0) for _ in range(10)]
    result = calibrate_thresholds(records)
    assert result.catch_rate >= .95 and result.planted_retention >= .95
    assert result.thresholds.delta_min == .01
    with pytest.raises(CalibrationError):
        calibrate_thresholds(records[:40])


def test_evaluate_cached_builds_pure_container_and_reports_all_required_statistics_deterministically():
    cells = _cached_cells()
    first = evaluate_cached(cells=cells, thresholds=_threshold_mapping(), n_replicates=2, permutation_seed=9, bootstrap_seed=10)
    second = evaluate_cached(cells=cells, thresholds=_threshold_mapping(), n_replicates=2, permutation_seed=9, bootstrap_seed=10)
    assert first == second
    assert first["replicates"] == {
        "permutations": 2, "bootstraps": 2, "permutation_seed": 9, "bootstrap_seed": 10,
        "shared_mapping_per_replicate": True, "shared_bootstrap_sample_per_replicate": True,
    }
    assert set(first["primary"]["statistics"]["t_by_arm"]) == {"full", "identity", "oof_conformal", "scalar_trace", "diagonal"}
    assert set(first["negative_controls"]) == set(NEGATIVE_CONTROLS)
    assert len(first["pairwise_spearman_deltas"]) == 6
    assert set(first["inference"]["permutation_deltas"]) == {"primary", *(f"direct:{name}" for name in NEGATIVE_CONTROLS), *(f"candidate:{name}" for name in NEGATIVE_CONTROLS)}
    assert set(first["inference"]["bootstrap_deltas"]) == set(first["inference"]["permutation_deltas"])
    assert all(len(values) == 2 for values in first["inference"]["permutation_deltas"].values())
    assert all(len(values) == 2 for values in first["inference"]["bootstrap_deltas"].values())


def test_evaluate_cached_refuses_missing_arm_or_gate_instead_of_weakening_analysis():
    cells = _cached_cells()
    incomplete = deepcopy(cells)
    del incomplete[0]["eccentricities"]["diagonal"]
    with pytest.raises(ContainerValidationError, match="eight frozen arms"):
        evaluate_cached(cells=incomplete, thresholds=_threshold_mapping(), n_replicates=1)
    del cells[0]["gate_metrics"]["full"]
    with pytest.raises(ContainerValidationError, match="gate_metrics"):
        evaluate_cached(cells=cells, thresholds=_threshold_mapping(), n_replicates=1)


def test_run_real_rejects_any_replicate_weakening_before_statistics_run():
    contract = {
        "experiment": {
            "seeds": [42, 420, 4200], "k": 10, "permutations": 10000, "bootstraps": 10000,
            "arms": ["full", "identity", "oof_conformal", "scalar_trace", "diagonal", *NEGATIVE_CONTROLS],
            "thresholds": _threshold_mapping(),
        }
    }
    calibration = {"calibration": _threshold_mapping()}
    preflight = {"banks": {bank: {"prompt_hash": _hash("prompt" + bank)} for bank in BANKS}}
    with pytest.raises(ManifestValidationError, match="non-contractual"):
        run_real(cells=_cached_cells(provenance=True), contract=contract, calibration=calibration, preflight=preflight, n_replicates=1)


def test_run_real_rejects_contract_missing_required_negative_control_before_real_execution():
    contract = {
        "experiment": {
            "seeds": [42, 420, 4200], "k": 10, "permutations": 10000, "bootstraps": 10000,
            "arms": ["full", "identity", "oof_conformal", "scalar_trace", "diagonal"],
            "thresholds": _threshold_mapping(),
        }
    }
    with pytest.raises(ManifestValidationError, match="eight frozen arms"):
        run_real(cells=[], contract=contract, calibration={"calibration": _threshold_mapping()}, preflight={"banks": {}}, n_replicates=10000)


def test_required_arm_gates_are_labelled_and_negative_control_metadata_cannot_invalidate():
    cells = _cached_cells()
    cells[0]["negative_control_gate_metrics"] = {"article_shuffled": {"generalization": -999.0}}
    harmless = evaluate_cached(cells=cells, thresholds=_threshold_mapping(), n_replicates=1)
    assert harmless["gates"]["valid"]
    required_bad = deepcopy(cells)
    required_bad[0]["gate_metrics"]["full"]["generalization"] = -999.0
    result = evaluate_cached(cells=required_bad, thresholds=_threshold_mapping(), n_replicates=1)
    assert result["decision"] == "inconclusive"
    assert "full:generalization" in result["gates"]["failures"]


def test_comparator_amplification_is_audited_but_only_full_amplification_is_gating():
    comparator_high = _cached_cells()
    comparator_high[0]["gate_metrics"]["oof_conformal"]["amplification"] = 11.0
    result = evaluate_cached(cells=comparator_high, thresholds=_threshold_mapping(), n_replicates=1)
    assert result["gates"]["valid"]
    assert result["gates"]["scope"]["full_only"] == [
        "generalization", "generalization_gap", "amplification",
    ]
    record = next(
        item for item in result["gates"]["metrics"]
        if item["fold"] == 0 and item["seed"] == 42 and item["bank"] == "Q0"
        and item["arm"] == "oof_conformal"
    )
    assert record["amplification"] == 11.0

    full_high = _cached_cells()
    full_high[0]["gate_metrics"]["full"]["amplification"] = 11.0
    failed = evaluate_cached(cells=full_high, thresholds=_threshold_mapping(), n_replicates=1)
    assert failed["decision"] == "inconclusive"
    assert failed["gates"]["failures"] == ["full:amplification"]


def test_dispersion_and_ties_still_gate_every_required_arm():
    cells = _cached_cells()
    cells[0]["gate_metrics"]["identity"]["dispersion"] = 0.0
    cells[1]["gate_metrics"]["diagonal"]["tie_fraction"] = 1.0
    result = evaluate_cached(cells=cells, thresholds=_threshold_mapping(), n_replicates=1)
    assert result["decision"] == "inconclusive"
    assert result["gates"]["failures"] == ["diagonal:ties", "identity:dispersion"]


def test_missing_grid_cell_and_disconnected_required_arm_fail_closed():
    cells = _cached_cells()
    with pytest.raises(ContainerValidationError, match="complete 5"):
        evaluate_cached(cells=cells[:-1], thresholds=_threshold_mapping(), n_replicates=1)
    disconnected = deepcopy(cells)
    disconnected[0]["connected"]["identity"] = False
    result = evaluate_cached(cells=disconnected, thresholds=_threshold_mapping(), n_replicates=1)
    assert result["decision"] == "inconclusive"
    assert "identity:connectivity" in result["gates"]["failures"]


def test_pairwise_rule_uses_three_seed_averaged_means_not_all_individual_cells():
    stats = arm_statistics(_container())
    deltas = {
        (42, "Q0", "Q1"): -.1, (420, "Q0", "Q1"): .2, (4200, "Q0", "Q1"): .2,
        (42, "Q0", "Q2"): .2, (420, "Q0", "Q2"): .2, (4200, "Q0", "Q2"): .2,
        (42, "Q1", "Q2"): .2, (420, "Q1", "Q2"): .2, (4200, "Q1", "Q2"): .2,
    }
    decision = decide(construction_valid=True, gate=GateEvaluation(True, ()), statistics=stats, p_value=.01, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=_negative_results())
    assert "pairwise" not in decision.reasons
    assert "robustness" not in decision.reasons
    assert robustness_fraction(deltas) == pytest.approx(8 / 9)


def test_pairwise_rule_rejects_one_pair_with_missing_seed_even_if_its_mean_is_positive():
    stats = arm_statistics(_container())
    deltas = {(seed, left, right): .1 for seed in (42, 420, 4200) for left, right in (("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2"))}
    del deltas[(4200, "Q1", "Q2")]
    with pytest.raises(ContainerValidationError, match="same nonempty seed set"):
        decide(construction_valid=True, gate=GateEvaluation(True, ()), statistics=stats, p_value=.01, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=_negative_results())


def test_threshold_mapping_rejects_unfrozen_extra_key():
    from core.full_spd_experiment import _thresholds_from_mapping
    malformed = dict(_threshold_mapping())
    malformed["extra"] = 1
    with pytest.raises(ContainerValidationError, match="exactly the six"):
        _thresholds_from_mapping(malformed)


def test_negative_control_candidate_pass_is_invalidating():
    stats = arm_statistics(_container())
    deltas = {(seed, left, right): .1 for seed in (42, 420, 4200) for left, right in (("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2"))}
    controls = list(_negative_results())
    controls[0] = NegativeControlResult("article_shuffled", .01, .1, Decision.PASS)
    result = decide(construction_valid=True, gate=GateEvaluation(True, ()), statistics=stats, p_value=.01, bootstrap_lower=.1, thresholds=_thresholds(), pairwise_deltas=deltas, negative_controls=controls)
    assert result.decision is Decision.INVALID
    assert result.reasons == ("negative_control_passed_primary",)


def test_contract_inference_reuses_one_mapping_and_sample_per_replicate_without_retaining_them(monkeypatch):
    import core.full_spd_experiment as experiment

    container = _container()
    original_stats = experiment._arm_statistics_unchecked
    original_validate = experiment.validate_eccentricity_container
    seen, validations = [], []

    def traced_stats(*args, mapping=None, indices_by_fold=None, **kwargs):
        if mapping is not None:
            seen.append(("mapping", id(mapping)))
        if indices_by_fold is not None:
            seen.append(("sample", id(indices_by_fold)))
        return original_stats(*args, mapping=mapping, indices_by_fold=indices_by_fold, **kwargs)

    def traced_validate(value):
        validations.append(value)
        return original_validate(value)

    monkeypatch.setattr(experiment, "_arm_statistics_unchecked", traced_stats)
    monkeypatch.setattr(experiment, "validate_eccentricity_container", traced_validate)
    inference = experiment._contract_inference(container, n_replicates=2, permutation_seed=3, bootstrap_seed=4)
    mapping_ids = [identity for kind, identity in seen if kind == "mapping"]
    sample_ids = [identity for kind, identity in seen if kind == "sample"]
    assert len(mapping_ids) == len(sample_ids) == 14  # 7 comparisons x 2 replicates
    assert len(set(mapping_ids[:7])) == len(set(mapping_ids[7:])) == 1
    assert len(set(sample_ids[:7])) == len(set(sample_ids[7:])) == 1
    assert len(validations) == 1
    assert not hasattr(inference, "mappings") and not hasattr(inference, "sampled_indices")
    assert all(values.size == 2 for values in inference.permutation_deltas.values())
    assert all(values.size == 2 for values in inference.bootstrap_deltas.values())


def _real_cells():
    ids = tuple(f"r{i:03d}" for i in range(320))
    publishers = {article_id: f"p{i // 20:02d}" for i, article_id in enumerate(ids)}
    folds = deterministic_publisher_folds(ids, publishers)
    required = ("full", "identity", "oof_conformal", "scalar_trace", "diagonal")
    arms = ("full", "identity", "oof_conformal", "scalar_trace", "diagonal", *NEGATIVE_CONTROLS)
    cells = []
    for fold in range(5):
        n = len(folds.heldout_ids[fold])
        for seed in (42, 420, 4200):
            for bank in BANKS:
                cells.append({
                    "ordered_ids": list(ids), "publisher_by_id": publishers, "fold_for_publisher": dict(folds.fold_for_publisher),
                    "fold": fold, "seed": seed, "bank": bank,
                    "eccentricities": {arm: (np.arange(n, dtype=float) + 1 + .001 * (seed + len(bank) + len(arm))).tolist() for arm in arms},
                    "gate_metrics": {arm: {"dispersion": 1, "tie_fraction": 0, "generalization": 1, "generalization_gap": 0, "amplification": 1} for arm in required},
                    "connected": {arm: True for arm in required}, "construction_valid": True, "provenance_valid": True, "corpus_control_passed": False,
                    "provenance": {"q_bank_hash": _hash("prompt" + bank), "target_source_kind": "observer_derived", "git_commit": "abc"},
                })
    return cells


def test_real_schema_is_exactly_45_cells_and_all_14_contract_arrays_are_10000(monkeypatch):
    import core.full_spd_experiment as experiment

    captured = {}
    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return {"decision": "fail", "replicates": {"permutations": kwargs["n_replicates"], "bootstraps": kwargs["n_replicates"]}}
    monkeypatch.setattr(experiment, "_evaluate_container", fake_evaluate)
    contract = {"experiment": {"seeds": [42, 420, 4200], "k": 10, "permutations": 10000, "bootstraps": 10000, "arms": ["full", "identity", "oof_conformal", "scalar_trace", "diagonal", *NEGATIVE_CONTROLS], "thresholds": _threshold_mapping()}}
    preflight = {"git_dirty": False, "git_commit": "abc", "banks": {bank: {"prompt_hash": _hash("prompt" + bank)} for bank in BANKS}}
    result = run_real(cells=_real_cells(), contract=contract, calibration={"calibration": _threshold_mapping()}, preflight=preflight)
    assert len(_real_cells()) == 45 and captured["n_replicates"] == 10_000
    assert result["contractual_replicates_enforced"] is True


def test_run_real_rejects_cell_git_commit_mismatch_before_numerical_evaluation(monkeypatch):
    import core.full_spd_experiment as experiment

    monkeypatch.setattr(experiment, "_evaluate_container", lambda *args, **kwargs: {"decision": "fail"})
    cells = _real_cells()
    cells[0]["provenance"]["git_commit"] = "wrong"
    contract = {"experiment": {"seeds": [42, 420, 4200], "k": 10, "permutations": 10000, "bootstraps": 10000, "arms": ["full", "identity", "oof_conformal", "scalar_trace", "diagonal", *NEGATIVE_CONTROLS], "thresholds": _threshold_mapping()}}
    preflight = {"git_dirty": False, "git_commit": "abc", "banks": {bank: {"prompt_hash": _hash("prompt" + bank)} for bank in BANKS}}
    with pytest.raises(ManifestValidationError, match="git commit"):
        run_real(cells=cells, contract=contract, calibration={"calibration": _threshold_mapping()}, preflight=preflight)
