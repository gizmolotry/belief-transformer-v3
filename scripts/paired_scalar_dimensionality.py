from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import paired_scalar_structure as paired


SEED = 20_260_907
CV_REPEATS = 16
CV_FOLDS = 4
EIGENVALUE_PERMUTATIONS = 10_000
BOOTSTRAP_REPLICATES = 10_000
K_VALUES = (1, 2, 3)
ALPHA = 0.05
DIRECT_COSINE_THRESHOLD = 0.90
DIRECT_COSINE_REQUIRED_FRACTION = 0.95
MAX_SWAP_RATE = 0.05
MAX_BOOTSTRAP_FAILURE_RATE = 0.01
ROTATIONAL_CONTAINMENT_MEDIAN = 0.90
EIGEN_TIE_RELATIVE_TOLERANCE = 1e-10

DEFAULT_CHECKPOINT = ROOT / "outputs" / "fresh_scalar_structure_run" / "extraction_checkpoint.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "controlled_N_dense_corpus.jsonl"
DEFAULT_Q0 = ROOT / "config" / "framing_queries.yaml"
DEFAULT_Q1 = ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
DEFAULT_CONTEXT_RESULT = ROOT / "paper" / "paired_scalar_structure_results_20260906.json"
PROTOCOL = ROOT / "paper" / "paired_scalar_dimensionality_protocol_20260907.md"


class DimensionalityInputError(ValueError):
    """Raised when the frozen dimensionality-analysis contract is not satisfied."""


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DimensionalityInputError(f"{label} is not numeric") from exc
    if array.ndim != 2 or array.shape[1] != 8 or array.shape[0] < 4:
        raise DimensionalityInputError(f"{label} must have shape (n>=4, 8), got {array.shape}")
    if not np.isfinite(array).all():
        raise DimensionalityInputError(f"{label} contains nonfinite values")
    return array


def _pca_fit(value: Any) -> dict[str, np.ndarray]:
    matrix = _matrix(value, "PCA matrix")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    loadings = right_vectors.T
    eigenvalues = singular_values**2 / (matrix.shape[0] - 1)
    if eigenvalues.shape != (8,) or eigenvalues[0] <= 0.0:
        raise DimensionalityInputError("PCA matrix lacks finite positive variance")

    # Make serialized reference/loadings deterministic without changing any subspace.
    pivots = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[pivots, np.arange(loadings.shape[1])])
    signs[signs == 0.0] = 1.0
    loadings = loadings * signs
    return {"mean": mean, "eigenvalues": eigenvalues, "loadings": loadings}


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise DimensionalityInputError("cannot summarize an empty or nonfinite distribution")
    return {
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
        "mean": float(array.mean()),
    }


def grouped_split_schedule(
    publishers: Sequence[str],
    *,
    repeats: int,
    folds: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if repeats < 2 or folds < 2:
        raise DimensionalityInputError("grouped CV requires at least two repeats and two folds")
    labels = np.asarray(publishers, dtype=object)
    groups = np.asarray(sorted(set(labels.tolist())), dtype=object)
    if groups.size < folds or groups.size % folds != 0:
        raise DimensionalityInputError("publisher count must be divisible by the number of folds")

    schedule: list[dict[str, Any]] = []
    fold_width = groups.size // folds
    for repeat in range(repeats):
        shuffled = rng.permutation(groups)
        held_out_across_repeat: list[str] = []
        for fold in range(folds):
            test_groups = sorted(shuffled[fold * fold_width : (fold + 1) * fold_width].tolist())
            train_groups = sorted(set(groups.tolist()) - set(test_groups))
            train_indices = np.flatnonzero(np.isin(labels, train_groups))
            test_indices = np.flatnonzero(np.isin(labels, test_groups))
            if train_indices.size == 0 or test_indices.size == 0:
                raise DimensionalityInputError("grouped CV produced an empty train or test set")
            if set(labels[train_indices]).intersection(set(labels[test_indices])):
                raise DimensionalityInputError("publisher leakage detected in grouped CV")
            held_out_across_repeat.extend(test_groups)
            schedule.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_publishers": train_groups,
                    "test_publishers": test_groups,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                }
            )
        if sorted(held_out_across_repeat) != groups.tolist():
            raise DimensionalityInputError("each publisher must be held out once per repeat")
    return schedule


def grouped_reconstruction(
    value: Any,
    schedule: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    folds: int,
) -> dict[str, Any]:
    matrix = _matrix(value, "grouped-CV matrix")
    if len(schedule) != repeats * folds:
        raise DimensionalityInputError("grouped-CV schedule has the wrong number of cells")

    per_fold: list[dict[str, Any]] = []
    for split in schedule:
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        fit = _pca_fit(matrix[train_indices])
        centered_test = matrix[test_indices] - fit["mean"]
        denominator = float(np.sum(centered_test**2))
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise DimensionalityInputError("held-out reconstruction denominator is not positive")

        errors: dict[str, dict[str, float]] = {}
        for k in K_VALUES:
            basis = fit["loadings"][:, :k]
            reconstructed = (centered_test @ basis) @ basis.T
            residual = centered_test - reconstructed
            sse = float(np.sum(residual**2))
            errors[str(k)] = {
                "mean_squared_error": sse / residual.size,
                "relative_squared_error": sse / denominator,
            }
        per_fold.append(
            {
                "repeat": int(split["repeat"]),
                "fold": int(split["fold"]),
                "train_publishers": list(split["train_publishers"]),
                "test_publishers": list(split["test_publishers"]),
                "n_train_articles": int(train_indices.size),
                "n_test_articles": int(test_indices.size),
                "errors": errors,
                "paired_relative_error_decreases": {
                    "k1_to_k2": errors["1"]["relative_squared_error"]
                    - errors["2"]["relative_squared_error"],
                    "k2_to_k3": errors["2"]["relative_squared_error"]
                    - errors["3"]["relative_squared_error"],
                },
            }
        )

    candidates: dict[str, dict[str, Any]] = {}
    for k in K_VALUES:
        repeat_means = []
        for repeat in range(repeats):
            cells = [
                row["errors"][str(k)]["relative_squared_error"]
                for row in per_fold
                if row["repeat"] == repeat
            ]
            if len(cells) != folds:
                raise DimensionalityInputError("grouped-CV repeat lacks the declared number of folds")
            repeat_means.append(float(np.mean(cells)))
        repeat_array = np.asarray(repeat_means, dtype=np.float64)
        candidates[str(k)] = {
            "grand_mean_relative_squared_error": float(repeat_array.mean()),
            "standard_error_across_repeat_means": float(
                repeat_array.std(ddof=1) / math.sqrt(repeats)
            ),
            "repeat_mean_relative_squared_errors": repeat_means,
            "repeat_mean_distribution": _quantiles(repeat_array),
        }

    best_k = min(
        K_VALUES,
        key=lambda k: (candidates[str(k)]["grand_mean_relative_squared_error"], k),
    )
    threshold = (
        candidates[str(best_k)]["grand_mean_relative_squared_error"]
        + candidates[str(best_k)]["standard_error_across_repeat_means"]
    )
    selected_k = min(
        k
        for k in K_VALUES
        if candidates[str(k)]["grand_mean_relative_squared_error"] <= threshold
    )

    improvement_summary: dict[str, Any] = {}
    for name in ("k1_to_k2", "k2_to_k3"):
        values = [row["paired_relative_error_decreases"][name] for row in per_fold]
        improvement_summary[name] = {
            **_quantiles(values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "interpretation": "descriptive_only_nested_components_mechanically_reduce_error",
        }

    return {
        "selection_rule": "smallest k within one standard error of the lowest mean repeat-level relative error",
        "selected_k": selected_k,
        "best_mean_error_k": best_k,
        "one_standard_error_threshold": threshold,
        "candidates": candidates,
        "paired_improvement_summary": improvement_summary,
        "per_fold": per_fold,
    }


def eigenvalue_permutation_null(
    value: Any,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if permutations < 1:
        raise DimensionalityInputError("permutation count must be positive")
    matrix = _matrix(value, "eigenvalue-null matrix")
    observed_fit = _pca_fit(matrix)
    observed = observed_fit["eigenvalues"]
    null = np.empty((permutations, 2), dtype=np.float64)
    permuted = np.empty_like(matrix)
    for draw in range(permutations):
        for column in range(matrix.shape[1]):
            permuted[:, column] = rng.permutation(matrix[:, column])
        null[draw] = _pca_fit(permuted)["eigenvalues"][1:3]

    components: dict[str, Any] = {}
    for offset, component in enumerate((2, 3)):
        component_null = null[:, offset]
        observed_value = float(observed[component - 1])
        threshold = float(np.quantile(component_null, 0.95, method="higher"))
        exceedances = int(np.count_nonzero(component_null >= observed_value))
        plus_one_p = (exceedances + 1) / (permutations + 1)
        components[f"pc{component}"] = {
            "observed_eigenvalue": observed_value,
            "observed_explained_variance_ratio": float(observed_value / observed.sum()),
            "null_mean": float(component_null.mean()),
            "null_95th_percentile_method_higher": threshold,
            "null_exceedances_greater_or_equal": exceedances,
            "plus_one_p_value": plus_one_p,
            "passes": bool(observed_value > threshold and plus_one_p <= ALPHA),
        }
    return {
        "permutations": permutations,
        "null_scheme": "independently permute article rows within each score column and refit covariance PCA",
        "alpha": ALPHA,
        "full_observed_eigenvalues": [float(value) for value in observed],
        "full_observed_explained_variance_ratio": [
            float(value / observed.sum()) for value in observed
        ],
        "components": components,
    }


def publisher_cluster_bootstrap(
    value: Any,
    publishers: Sequence[str],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if replicates < 1:
        raise DimensionalityInputError("bootstrap replicate count must be positive")
    matrix = _matrix(value, "bootstrap matrix")
    labels = np.asarray(publishers, dtype=object)
    if labels.shape != (matrix.shape[0],):
        raise DimensionalityInputError("publisher labels do not match matrix rows")
    groups = np.asarray(sorted(set(labels.tolist())), dtype=object)
    if groups.size < 3:
        raise DimensionalityInputError("publisher-cluster bootstrap needs at least three publishers")
    group_indices = {group: np.flatnonzero(labels == group) for group in groups}

    reference = _pca_fit(matrix)
    reference_pc2 = reference["loadings"][:, 1]
    reference_pc3 = reference["loadings"][:, 2]
    reference_subspace = reference["loadings"][:, 1:3]

    direct_cosines: list[float] = []
    reference_containment: list[float] = []
    replicate_containment: list[float] = []
    principal_cosine_1: list[float] = []
    principal_cosine_2: list[float] = []
    relative_gaps: list[float] = []
    sign_aligned_loadings: list[np.ndarray] = []
    reference_to_replicate_pc3_swaps = 0
    replicate_to_reference_pc3_swaps = 0
    failures: Counter[str] = Counter()

    for _ in range(replicates):
        drawn_groups = rng.choice(groups, size=groups.size, replace=True)
        if np.unique(drawn_groups).size < 3:
            failures["fewer_than_three_distinct_publishers"] += 1
            continue
        sampled_indices = np.concatenate([group_indices[group] for group in drawn_groups])
        try:
            fit = _pca_fit(matrix[sampled_indices])
        except (DimensionalityInputError, np.linalg.LinAlgError):
            failures["nonfinite_or_failed_pca"] += 1
            continue
        eigenvalues = fit["eigenvalues"]
        rank_tolerance = eigenvalues[0] * max(matrix.shape) * np.finfo(np.float64).eps
        if int(np.count_nonzero(eigenvalues > rank_tolerance)) < 3:
            failures["rank_below_three"] += 1
            continue
        relative_gap = float(abs(eigenvalues[1] - eigenvalues[2]) / eigenvalues[1])
        if relative_gap <= EIGEN_TIE_RELATIVE_TOLERANCE:
            failures["pc2_pc3_eigenvalue_tie"] += 1
            continue

        replicate_pc2 = fit["loadings"][:, 1]
        replicate_pc3 = fit["loadings"][:, 2]
        replicate_subspace = fit["loadings"][:, 1:3]
        direct = float(abs(np.dot(reference_pc2, replicate_pc2)))
        direct_cosines.append(direct)
        reference_containment.append(
            float(np.clip(np.linalg.norm(replicate_subspace.T @ reference_pc2), 0.0, 1.0))
        )
        replicate_containment.append(
            float(np.clip(np.linalg.norm(reference_subspace.T @ replicate_pc2), 0.0, 1.0))
        )
        subspace_cosines = np.linalg.svd(
            reference_subspace.T @ replicate_subspace, compute_uv=False
        )
        subspace_cosines = np.clip(np.sort(subspace_cosines)[::-1], 0.0, 1.0)
        principal_cosine_1.append(float(subspace_cosines[0]))
        principal_cosine_2.append(float(subspace_cosines[1]))
        relative_gaps.append(relative_gap)

        if abs(np.dot(reference_pc2, replicate_pc3)) > direct:
            reference_to_replicate_pc3_swaps += 1
        if abs(np.dot(replicate_pc2, reference_pc3)) > direct:
            replicate_to_reference_pc3_swaps += 1
        if np.dot(reference_pc2, replicate_pc2) < 0.0:
            replicate_pc2 = -replicate_pc2
        sign_aligned_loadings.append(replicate_pc2)

    valid = len(direct_cosines)
    failed = replicates - valid
    if valid == 0:
        raise DimensionalityInputError("all publisher-cluster bootstrap replicates failed")
    direct_fraction = float(np.mean(np.asarray(direct_cosines) >= DIRECT_COSINE_THRESHOLD))
    reference_swap_rate = reference_to_replicate_pc3_swaps / valid
    replicate_swap_rate = replicate_to_reference_pc3_swaps / valid
    failure_rate = failed / replicates
    stability_conditions = {
        "direct_cosine_fraction": bool(direct_fraction >= DIRECT_COSINE_REQUIRED_FRACTION),
        "reference_pc2_swap_rate": bool(reference_swap_rate <= MAX_SWAP_RATE),
        "replicate_pc2_swap_rate": bool(replicate_swap_rate <= MAX_SWAP_RATE),
        "failure_rate": bool(failure_rate <= MAX_BOOTSTRAP_FAILURE_RATE),
    }
    stability_passes = all(stability_conditions.values())
    reference_containment_summary = _quantiles(reference_containment)
    replicate_containment_summary = _quantiles(replicate_containment)
    rotational_ambiguity = bool(
        reference_containment_summary["median"] >= ROTATIONAL_CONTAINMENT_MEDIAN
        and replicate_containment_summary["median"] >= ROTATIONAL_CONTAINMENT_MEDIAN
        and (
            direct_fraction < DIRECT_COSINE_REQUIRED_FRACTION
            or reference_swap_rate > MAX_SWAP_RATE
            or replicate_swap_rate > MAX_SWAP_RATE
        )
    )

    loading_array = np.stack(sign_aligned_loadings, axis=0)
    loading_intervals = []
    for coordinate in range(loading_array.shape[1]):
        values = loading_array[:, coordinate]
        loading_intervals.append(
            {
                "coordinate": coordinate,
                "reference_loading": float(reference_pc2[coordinate]),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )

    return {
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "failed_replicates": failed,
        "failure_rate": failure_rate,
        "failure_reasons": dict(sorted(failures.items())),
        "reference_pc2_loadings": [float(value) for value in reference_pc2],
        "reference_pc3_loadings": [float(value) for value in reference_pc3],
        "reference_eigenvalues": [float(value) for value in reference["eigenvalues"]],
        "direct_abs_cosine": {
            **_quantiles(direct_cosines),
            "threshold": DIRECT_COSINE_THRESHOLD,
            "fraction_at_or_above_threshold": direct_fraction,
            "required_fraction": DIRECT_COSINE_REQUIRED_FRACTION,
        },
        "reference_pc2_in_replicate_pc2_pc3_subspace": reference_containment_summary,
        "replicate_pc2_in_reference_pc2_pc3_subspace": replicate_containment_summary,
        "pc2_pc3_subspace_principal_cosines": {
            "larger_cosine_smaller_angle": _quantiles(principal_cosine_1),
            "smaller_cosine_larger_angle": _quantiles(principal_cosine_2),
        },
        "swap_diagnostics": {
            "reference_pc2_closer_to_replicate_pc3_count": reference_to_replicate_pc3_swaps,
            "reference_pc2_closer_to_replicate_pc3_rate": reference_swap_rate,
            "replicate_pc2_closer_to_reference_pc3_count": replicate_to_reference_pc3_swaps,
            "replicate_pc2_closer_to_reference_pc3_rate": replicate_swap_rate,
        },
        "relative_pc2_pc3_eigenvalue_gap": _quantiles(relative_gaps),
        "sign_aligned_pc2_loading_percentile_intervals": loading_intervals,
        "stability_conditions": stability_conditions,
        "stability_passes": stability_passes,
        "rotational_ambiguity_pattern": rotational_ambiguity,
    }


def dimensionality_decision(
    *, pc2_passes: bool, pc3_passes: bool, pc2_stable: bool, selected_k: int
) -> str:
    if pc2_passes and not pc2_stable:
        return "inconclusive_pc2_unstable"
    if not pc2_passes:
        if not pc3_passes and selected_k == 1:
            return "supports_no_more_than_one"
        return "inconclusive_mixed_evidence"
    if pc3_passes:
        if pc2_stable and selected_k == 3:
            return "supports_at_least_three"
        return "inconclusive_mixed_evidence"
    if pc2_stable and selected_k == 2:
        return "supports_two_over_one_not_three"
    return "inconclusive_mixed_evidence"


def _hash_records(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DimensionalityInputError(f"missing required files: {', '.join(missing)}")
    return {name: paired.sha256_file(path) for name, path in paths.items()}


def _validate_context_result(path: Path, input_hashes: Mapping[str, str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DimensionalityInputError("could not parse contextual paired-scalar result") from exc
    try:
        bound = payload["provenance"]["inputs"]
        primary = payload["claim_boundaries"]["primary_measurement"]
    except (KeyError, TypeError) as exc:
        raise DimensionalityInputError("contextual result lacks paired-scalar provenance") from exc
    for name in ("checkpoint", "corpus", "q0", "q1"):
        if str(bound.get(name, "")).lower() != input_hashes[name].lower():
            raise DimensionalityInputError(f"contextual result does not bind current {name}")
    if primary != "paired scalar c=s_A-s_B":
        raise DimensionalityInputError("contextual result does not declare paired scalar as primary")


def build_analysis(
    *,
    checkpoint_path: Path,
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    context_result_path: Path,
    protocol_path: Path = PROTOCOL,
    expected_n: int = 320,
    expected_publishers: int = 16,
    expected_per_publisher: int = 20,
    seed: int = SEED,
    cv_repeats: int = CV_REPEATS,
    cv_folds: int = CV_FOLDS,
    permutations: int = EIGENVALUE_PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    input_paths = {
        "checkpoint": checkpoint_path,
        "corpus": corpus_path,
        "q0": q0_path,
        "q1": q1_path,
        "contextual_paired_scalar_result": context_result_path,
    }
    source_paths = {
        "runner": Path(__file__).resolve(),
        "paired_scalar_scoring_runner": Path(paired.__file__).resolve(),
        "frozen_protocol": protocol_path,
    }
    initial_input_hashes = _hash_records(input_paths)
    initial_source_hashes = _hash_records(source_paths)
    _validate_context_result(context_result_path, initial_input_hashes)
    paired.load_prompt_pairs(q0_path, q1_path)

    ids, logits = paired.load_checkpoint(checkpoint_path)
    articles = paired.load_articles(corpus_path, expected_n=expected_n)
    corpus_ids = [article["canonical_id"] for article in articles]
    if ids != corpus_ids:
        raise DimensionalityInputError("checkpoint IDs do not exactly match corpus order")
    publishers = [article["_publisher"] for article in articles]
    counts = Counter(publishers)
    if len(counts) != expected_publishers or set(counts.values()) != {expected_per_publisher}:
        raise DimensionalityInputError(
            "publisher balance differs from the frozen expected publisher/sample counts"
        )

    scores: dict[str, np.ndarray] = {}
    for bank in ("q0", "q1"):
        _, _, contrast = paired.scores_from_logits(logits[bank]["A"], logits[bank]["B"])
        scores[bank] = _matrix(contrast, f"{bank} paired scalar")

    seed_sequence = np.random.SeedSequence(seed)
    child_names = (
        "grouped_split_schedule",
        "q0_eigenvalue_null",
        "q1_eigenvalue_null",
        "q0_publisher_bootstrap",
        "q1_publisher_bootstrap",
    )
    children = seed_sequence.spawn(len(child_names))
    child_sequences = dict(zip(child_names, children, strict=True))
    schedule = grouped_split_schedule(
        publishers,
        repeats=cv_repeats,
        folds=cv_folds,
        rng=np.random.default_rng(child_sequences["grouped_split_schedule"]),
    )

    results: dict[str, Any] = {}
    for bank in ("q0", "q1"):
        cv = grouped_reconstruction(
            scores[bank], schedule, repeats=cv_repeats, folds=cv_folds
        )
        null = eigenvalue_permutation_null(
            scores[bank],
            permutations=permutations,
            rng=np.random.default_rng(child_sequences[f"{bank}_eigenvalue_null"]),
        )
        bootstrap = publisher_cluster_bootstrap(
            scores[bank],
            publishers,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(child_sequences[f"{bank}_publisher_bootstrap"]),
        )
        pc2_passes = bool(null["components"]["pc2"]["passes"])
        pc3_passes = bool(null["components"]["pc3"]["passes"])
        pc2_stable = bool(bootstrap["stability_passes"])
        verdict = dimensionality_decision(
            pc2_passes=pc2_passes,
            pc3_passes=pc3_passes,
            pc2_stable=pc2_stable,
            selected_k=int(cv["selected_k"]),
        )
        results[bank] = {
            "decision_inputs": {
                "pc2_eigenvalue_null_passes": pc2_passes,
                "pc3_eigenvalue_null_passes": pc3_passes,
                "pc2_loading_stability_passes": pc2_stable,
                "grouped_cv_selected_k": int(cv["selected_k"]),
            },
            "dimensionality_verdict": verdict,
            "pc2_axis_interpretation": (
                "eligible_only_for_blind_followup_interpretation"
                if pc2_passes and pc2_stable
                else "inconclusive_or_not_supported; do_not_name_axis"
            ),
            "grouped_cross_validation": cv,
            "eigenvalue_permutation_null": null,
            "publisher_cluster_bootstrap": bootstrap,
        }

    verdicts = {bank: results[bank]["dimensionality_verdict"] for bank in ("q0", "q1")}
    cross_bank = (
        f"paraphrase_stable_{verdicts['q0']}"
        if verdicts["q0"] == verdicts["q1"]
        else "cross_bank_inconclusive_different_outcomes"
    )

    final_input_hashes = _hash_records(input_paths)
    final_source_hashes = _hash_records(source_paths)
    if initial_input_hashes != final_input_hashes or initial_source_hashes != final_source_hashes:
        raise DimensionalityInputError("analyzed inputs or bound sources changed during analysis")

    return {
        "schema": {
            "name": "belief_transformer.paired_scalar_dimensionality_selection",
            "version": "1.0.0",
        },
        "status": "prospective_specification_applied_to_previously_inspected_data",
        "claim_boundaries": {
            "measurement": "paired scalar c=s_A-s_B only",
            "not_preregistered": True,
            "not_confirmatory": True,
            "q1_is_paraphrase_not_independent_replication": True,
            "no_component_names_or_manuscript_claims_authorized": True,
            "grouped_cv": "descriptive one-standard-error model selection, not significance evidence",
        },
        "analysis": {
            "root_seed": seed,
            "child_seed_spawn_keys": {
                name: list(sequence.spawn_key) for name, sequence in child_sequences.items()
            },
            "n_articles": len(ids),
            "publisher_counts": dict(sorted(counts.items())),
            "cv_repeats": cv_repeats,
            "cv_folds": cv_folds,
            "candidate_dimensions": list(K_VALUES),
            "eigenvalue_permutations_per_bank": permutations,
            "publisher_bootstrap_replicates_per_bank": bootstrap_replicates,
        },
        "frozen_thresholds": {
            "eigenvalue_alpha": ALPHA,
            "direct_abs_cosine": DIRECT_COSINE_THRESHOLD,
            "direct_abs_cosine_required_fraction": DIRECT_COSINE_REQUIRED_FRACTION,
            "maximum_each_swap_rate": MAX_SWAP_RATE,
            "maximum_bootstrap_failure_rate": MAX_BOOTSTRAP_FAILURE_RATE,
            "rotational_ambiguity_containment_median": ROTATIONAL_CONTAINMENT_MEDIAN,
            "pc2_pc3_relative_tie_tolerance": EIGEN_TIE_RELATIVE_TOLERANCE,
        },
        "provenance": {
            "paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256": initial_input_hashes,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_sha256": initial_source_hashes,
            "article_ids_exactly_match_corpus_order": True,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        "cross_bank_summary": {
            "q0_verdict": verdicts["q0"],
            "q1_verdict": verdicts["q1"],
            "outcome": cross_bank,
            "interpretation": "phrasing stability only; not independent replication",
        },
        "results": results,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen paired-scalar dimensionality-selection analysis"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--q0", type=Path, default=DEFAULT_Q0)
    parser.add_argument("--q1", type=Path, default=DEFAULT_Q1)
    parser.add_argument("--context-result", type=Path, default=DEFAULT_CONTEXT_RESULT)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_analysis(
        checkpoint_path=args.checkpoint,
        corpus_path=args.corpus,
        q0_path=args.q0,
        q1_path=args.q1,
        context_result_path=args.context_result,
    )
    _write_json_exclusive(args.output_json, payload)
    print(f"Dimensionality analysis written to {args.output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DimensionalityInputError, paired.PairedScalarInputError) as exc:
        print(f"DimensionalityInputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
