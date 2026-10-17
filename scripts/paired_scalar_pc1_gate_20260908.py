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

from scripts import paired_scalar_dimensionality_v2_20260907 as v2


paired = v2.paired
dimensionality_v1 = v2.v1

PROTOCOL_DATE = "2026-09-08"
ROOT_SEED = 20_260_908
PERMUTATIONS = 10_000
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05
COSINE_THRESHOLD = 0.80
RHO_THRESHOLD = 0.50
EXPECTED_PUBLISHERS = 16
EXPECTED_PER_PUBLISHER = 20
EXPECTED_N = EXPECTED_PUBLISHERS * EXPECTED_PER_PUBLISHER

DEFAULT_CHECKPOINT = ROOT / "outputs" / "fresh_scalar_structure_run" / "extraction_checkpoint.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "controlled_N_dense_corpus.jsonl"
DEFAULT_Q0 = ROOT / "config" / "framing_queries.yaml"
DEFAULT_Q1 = ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
DEFAULT_PAIRED_STRUCTURE_PROTOCOL = ROOT / "paper" / "paired_scalar_structure_protocol_20260905.md"
DEFAULT_PAIRED_STRUCTURE_RESULT = ROOT / "paper" / "paired_scalar_structure_results_20260906.json"
DEFAULT_DIMENSIONALITY_V2_PROTOCOL = (
    ROOT / "paper" / "paired_scalar_dimensionality_protocol_v2_20260907.md"
)
DEFAULT_DIMENSIONALITY_V2_RESULT = (
    ROOT / "paper" / "paired_scalar_dimensionality_results_v2_20260907.json"
)
DEFAULT_PROTOCOL = ROOT / "paper" / "paired_scalar_pc1_gate_protocol_20260908.md"
DEFAULT_OUTPUT = ROOT / "paper" / "paired_scalar_pc1_gate_results_20260908.json"


class PC1GateInputError(ValueError):
    """Raised when the frozen PC1-gate contract cannot be satisfied."""


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PC1GateInputError(f"{label} is not numeric") from exc
    if matrix.shape != (EXPECTED_N, 8):
        raise PC1GateInputError(f"{label} must have shape ({EXPECTED_N}, 8), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise PC1GateInputError(f"{label} contains nonfinite values")
    return matrix


def _vector(value: Any, label: str, *, length: int = EXPECTED_N) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PC1GateInputError(f"{label} is not numeric") from exc
    if vector.shape != (length,):
        raise PC1GateInputError(f"{label} must have shape ({length},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise PC1GateInputError(f"{label} contains nonfinite values")
    return vector


def _pca_fit(value: Any) -> dict[str, np.ndarray]:
    matrix = _matrix(value, "covariance PCA matrix")
    fit = dimensionality_v1._pca_fit(matrix)
    return {
        "mean": np.asarray(fit["mean"], dtype=np.float64),
        "eigenvalues": np.asarray(fit["eigenvalues"], dtype=np.float64),
        "loadings": np.asarray(fit["loadings"], dtype=np.float64),
    }


def _balanced_groups(
    publishers: Sequence[str], n_rows: int = EXPECTED_N
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        groups, group_indices = v2._balanced_group_indices(publishers, n_rows)
    except v2.DimensionalityV2InputError as exc:
        raise PC1GateInputError(str(exc)) from exc
    return groups, group_indices


def _align_loading(reference: Any, candidate: Any) -> tuple[np.ndarray, float]:
    reference_vector = np.asarray(reference, dtype=np.float64)
    candidate_vector = np.asarray(candidate, dtype=np.float64)
    if reference_vector.shape != (8,) or candidate_vector.shape != (8,):
        raise PC1GateInputError("PC1 loadings must each have shape (8,)")
    if not np.isfinite(reference_vector).all() or not np.isfinite(candidate_vector).all():
        raise PC1GateInputError("PC1 loadings must be finite")
    denominator = float(np.linalg.norm(reference_vector) * np.linalg.norm(candidate_vector))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PC1GateInputError("PC1 loadings must have positive norm")
    cosine = float(np.dot(reference_vector, candidate_vector) / denominator)
    if cosine < 0.0:
        candidate_vector = -candidate_vector
        cosine = -cosine
    return candidate_vector, float(np.clip(cosine, 0.0, 1.0))


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | str]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PC1GateInputError("cannot summarize an empty or nonfinite distribution")
    return {
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "interval_quantile_method": "linear",
    }


def publisher_block_pc1_null(
    value: Any,
    publishers: Sequence[str],
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if isinstance(permutations, bool) or permutations < 1:
        raise PC1GateInputError("permutation count must be a positive integer")
    matrix = _matrix(value, "publisher-block PC1-null matrix")
    _balanced_groups(publishers, matrix.shape[0])
    observed_fit = _pca_fit(matrix)
    observed = float(observed_fit["eigenvalues"][0])
    null_values = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted = v2.publisher_block_permutation(matrix, publishers, rng)
        null_values[draw] = float(_pca_fit(permuted)["eigenvalues"][0])
    threshold = float(np.quantile(null_values, 0.95, method="higher"))
    exceedances = int(np.count_nonzero(null_values >= observed))
    plus_one_p = (exceedances + 1) / (permutations + 1)
    detected = bool(observed > threshold and plus_one_p <= ALPHA)
    return {
        "statistic": "lambda1",
        "pca": "covariance PCA, column-centered and unstandardized, denominator n-1",
        "null_draws": permutations,
        "null_scheme": (
            "independently reassign each column's complete 20-value publisher blocks among "
            "16 target publishers, then independently permute row order inside each assigned block"
        ),
        "observed_lambda1": observed,
        "null_mean_lambda1": float(null_values.mean()),
        "null_q95_lambda1": threshold,
        "null_q95_method": "higher",
        "null_exceedances_greater_or_equal": exceedances,
        "plus_one_p_value": plus_one_p,
        "decision_rule": "observed_lambda1 > null_q95_lambda1 and plus_one_p_value <= 0.05",
        "detected": detected,
        "outcome": "detected_under_this_null" if detected else "not_detected_under_this_null",
    }


def paired_publisher_cluster_bootstrap(
    q0_value: Any,
    q1_value: Any,
    publishers: Sequence[str],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if isinstance(replicates, bool) or replicates < 1:
        raise PC1GateInputError("bootstrap replicate count must be a positive integer")
    q0 = _matrix(q0_value, "Q0 bootstrap matrix")
    q1 = _matrix(q1_value, "Q1 bootstrap matrix")
    groups, group_indices = _balanced_groups(publishers, q0.shape[0])

    q0_reference = _pca_fit(q0)["loadings"][:, 0]
    q1_reference_raw = _pca_fit(q1)["loadings"][:, 0]
    q1_reference, full_cosine = _align_loading(q0_reference, q1_reference_raw)

    cosines: list[float] = []
    failures: Counter[str] = Counter()
    for _ in range(replicates):
        drawn_groups = rng.choice(groups, size=groups.size, replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in drawn_groups])
        try:
            q0_replicate = _pca_fit(q0[sampled_indices])["loadings"][:, 0]
        except (PC1GateInputError, dimensionality_v1.DimensionalityInputError, np.linalg.LinAlgError):
            failures["q0_pca_failure"] += 1
            continue
        try:
            q1_replicate = _pca_fit(q1[sampled_indices])["loadings"][:, 0]
        except (PC1GateInputError, dimensionality_v1.DimensionalityInputError, np.linalg.LinAlgError):
            failures["q1_pca_failure"] += 1
            continue
        q0_replicate, _ = _align_loading(q0_reference, q0_replicate)
        _, cosine = _align_loading(q0_replicate, q1_replicate)
        cosines.append(cosine)

    valid = len(cosines)
    failed = replicates - valid
    if valid == 0:
        raise PC1GateInputError("all paired publisher-cluster bootstrap replicates failed")
    return {
        "scheme": (
            "sample 16 publisher clusters with replacement, include all 20 rows per sampled "
            "cluster, and use identical sampled row indices for Q0 and Q1"
        ),
        "alignment": (
            "align replicate Q0 PC1 to full-data Q0 PC1, then align replicate Q1 PC1 to "
            "replicate Q0 PC1"
        ),
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "failed_replicates": failed,
        "failure_rate": failed / replicates,
        "failure_reasons": dict(sorted(failures.items())),
        "full_data_q0_pc1_loadings": [float(item) for item in q0_reference],
        "full_data_q1_pc1_loadings_sign_aligned_to_q0": [
            float(item) for item in q1_reference
        ],
        "full_data_q0_q1_loading_cosine": full_cosine,
        "bootstrap_q0_q1_loading_cosine": _distribution(cosines),
    }


def center_scores_within_publisher(
    scores: Any, publishers: Sequence[str]
) -> tuple[np.ndarray, float]:
    vector = _vector(scores, "article PC1 scores")
    groups, group_indices = _balanced_groups(publishers, vector.size)
    centered = np.empty_like(vector)
    for group in groups:
        indices = group_indices[group]
        centered[indices] = vector[indices] - vector[indices].mean()
    maximum_abs_mean = max(
        abs(float(centered[group_indices[group]].mean())) for group in groups
    )
    return centered, maximum_abs_mean


def within_publisher_permute(
    values: Any, publishers: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    vector = _vector(values, "within-publisher permutation values")
    groups, group_indices = _balanced_groups(publishers, vector.size)
    permuted = np.empty_like(vector)
    for group in groups:
        indices = group_indices[group]
        permuted[indices] = rng.permutation(vector[indices])
    return permuted


def _average_ranks(values: Any) -> np.ndarray:
    vector = _vector(values, "rank values")
    order = np.argsort(vector, kind="mergesort")
    ranks = np.empty(vector.size, dtype=np.float64)
    start = 0
    while start < vector.size:
        stop = start + 1
        while stop < vector.size and vector[order[stop]] == vector[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(left: Any, right: Any) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PC1GateInputError("Spearman correlation is undefined for constant ranks")
    return float(np.dot(left_centered, right_centered) / denominator)


def within_publisher_q1_correspondence_null(
    q0_scores: Any,
    q1_scores: Any,
    publishers: Sequence[str],
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if isinstance(permutations, bool) or permutations < 1:
        raise PC1GateInputError("permutation count must be a positive integer")
    q0 = _vector(q0_scores, "publisher-centered Q0 PC1 scores")
    q1 = _vector(q1_scores, "publisher-centered Q1 PC1 scores")
    _balanced_groups(publishers, q0.size)
    observed = spearman_rho(q0, q1)
    null_values = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted_q1 = within_publisher_permute(q1, publishers, rng)
        null_values[draw] = spearman_rho(q0, permuted_q1)
    threshold = float(np.quantile(null_values, 0.95, method="higher"))
    exceedances = int(np.count_nonzero(null_values >= observed))
    plus_one_p = (exceedances + 1) / (permutations + 1)
    return {
        "scheme": (
            "independently permute Q1 row correspondence within each publisher while keeping "
            "Q0 fixed and preserving each publisher's Q1 score multiset and mean"
        ),
        "null_draws": permutations,
        "observed_spearman_rho": observed,
        "null_mean_spearman_rho": float(null_values.mean()),
        "null_q95_spearman_rho": threshold,
        "null_q95_method": "higher",
        "null_exceedances_greater_or_equal": exceedances,
        "plus_one_p_value": plus_one_p,
        "alternative": "positive Q0-Q1 correspondence",
    }


def phrasing_stability_outcome(*, cosine_q025: float, rho: float, p_value: float) -> str:
    if p_value > ALPHA:
        return "not_detected"
    if cosine_q025 >= COSINE_THRESHOLD and rho >= RHO_THRESHOLD:
        return "strong_phrasing_stability"
    return "detected_below_practical_threshold"


def gate1_joint_outcome(q0_detected: bool, q1_detected: bool) -> str:
    return (
        "detected_under_this_null"
        if q0_detected and q1_detected
        else "not_detected_under_this_null"
    )


def paired_phrasing_stability(
    q0_value: Any,
    q1_value: Any,
    publishers: Sequence[str],
    *,
    bootstrap_replicates: int,
    permutations: int,
    bootstrap_rng: np.random.Generator,
    correspondence_rng: np.random.Generator,
) -> dict[str, Any]:
    q0 = _matrix(q0_value, "Q0 phrasing-stability matrix")
    q1 = _matrix(q1_value, "Q1 phrasing-stability matrix")
    bootstrap = paired_publisher_cluster_bootstrap(
        q0,
        q1,
        publishers,
        replicates=bootstrap_replicates,
        rng=bootstrap_rng,
    )
    q0_fit = _pca_fit(q0)
    q1_fit = _pca_fit(q1)
    q0_loading = q0_fit["loadings"][:, 0]
    q1_loading, _ = _align_loading(q0_loading, q1_fit["loadings"][:, 0])
    q0_article_scores = (q0 - q0_fit["mean"]) @ q0_loading
    q1_article_scores = (q1 - q1_fit["mean"]) @ q1_loading
    q0_centered, q0_centering_error = center_scores_within_publisher(
        q0_article_scores, publishers
    )
    q1_centered, q1_centering_error = center_scores_within_publisher(
        q1_article_scores, publishers
    )
    correspondence = within_publisher_q1_correspondence_null(
        q0_centered,
        q1_centered,
        publishers,
        permutations=permutations,
        rng=correspondence_rng,
    )
    cosine_q025 = float(bootstrap["bootstrap_q0_q1_loading_cosine"]["q025"])
    rho = float(correspondence["observed_spearman_rho"])
    p_value = float(correspondence["plus_one_p_value"])
    outcome = phrasing_stability_outcome(
        cosine_q025=cosine_q025,
        rho=rho,
        p_value=p_value,
    )
    return {
        "executed": True,
        "scope": "Q1 paraphrase phrasing stability only; not independent replication",
        "paired_publisher_cluster_bootstrap": bootstrap,
        "publisher_centered_article_scores": {
            "score_definition": "full-data PC1 score, then centered within publisher",
            "q0_maximum_absolute_publisher_mean_after_centering": q0_centering_error,
            "q1_maximum_absolute_publisher_mean_after_centering": q1_centering_error,
            "spearman_rho": rho,
        },
        "within_publisher_q1_correspondence_null": correspondence,
        "thresholds": {
            "bootstrap_cosine_q025_minimum": COSINE_THRESHOLD,
            "spearman_rho_minimum": RHO_THRESHOLD,
            "plus_one_p_value_maximum": ALPHA,
        },
        "outcome": outcome,
    }


def run_gates(
    scores: Mapping[str, Any],
    publishers: Sequence[str],
    *,
    seed: int,
    permutations: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if set(scores) != {"q0", "q1"}:
        raise PC1GateInputError("scores must contain exactly q0 and q1")
    stream_names = (
        "q0_gate1_publisher_block_null",
        "q1_gate1_publisher_block_null",
        "gate2_paired_publisher_cluster_bootstrap",
        "gate2_within_publisher_q1_correspondence_null",
    )
    children = np.random.SeedSequence(seed).spawn(len(stream_names))
    streams = dict(zip(stream_names, children, strict=True))
    gate1_banks: dict[str, Any] = {}
    for bank in ("q0", "q1"):
        gate1_banks[bank] = publisher_block_pc1_null(
            scores[bank],
            publishers,
            permutations=permutations,
            rng=np.random.default_rng(streams[f"{bank}_gate1_publisher_block_null"]),
        )
    q0_detected = bool(gate1_banks["q0"]["detected"])
    q1_detected = bool(gate1_banks["q1"]["detected"])
    joint_outcome = gate1_joint_outcome(q0_detected, q1_detected)
    gate1_passed = q0_detected and q1_detected
    if gate1_passed:
        gate2 = paired_phrasing_stability(
            scores["q0"],
            scores["q1"],
            publishers,
            bootstrap_replicates=bootstrap_replicates,
            permutations=permutations,
            bootstrap_rng=np.random.default_rng(
                streams["gate2_paired_publisher_cluster_bootstrap"]
            ),
            correspondence_rng=np.random.default_rng(
                streams["gate2_within_publisher_q1_correspondence_null"]
            ),
        )
    else:
        gate2 = {
            "executed": False,
            "reason": "gate1_not_detected_under_this_null",
        }
    return {
        "root_seed": seed,
        "child_seed_spawn_keys": {
            name: list(sequence.spawn_key) for name, sequence in streams.items()
        },
        "gate1": {
            "passes_only_if_both_banks_detected": True,
            "passed": gate1_passed,
            "outcome": joint_outcome,
            "banks": gate1_banks,
        },
        "gate2": gate2,
    }


def _hash_records(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PC1GateInputError(f"missing required files: {', '.join(missing)}")
    return {name: paired.sha256_file(path) for name, path in paths.items()}


def _assert_hashes_unchanged(
    before: Mapping[str, str], after: Mapping[str, str], label: str
) -> None:
    if dict(before) != dict(after):
        changed = sorted(
            set(before).union(after)
            - {name for name in set(before).intersection(after) if before[name] == after[name]}
        )
        raise PC1GateInputError(f"{label} changed during analysis: {', '.join(changed)}")


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PC1GateInputError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise PC1GateInputError(f"{label} must contain a JSON object")
    return payload


def _require_hash(actual: Any, expected: str, label: str) -> None:
    if str(actual).lower() != expected.lower():
        raise PC1GateInputError(f"{label} hash binding does not match the current file")


def _validate_paired_structure_result(
    path: Path, input_hashes: Mapping[str, str], source_hashes: Mapping[str, str]
) -> None:
    payload = _load_json_object(path, "paired structure result")
    try:
        schema = payload["schema"]
        claim_boundaries = payload["claim_boundaries"]
        provenance = payload["provenance"]
        bound_inputs = provenance["inputs"]
        bound_sources = provenance["source_sha256"]
    except (KeyError, TypeError) as exc:
        raise PC1GateInputError("paired structure result is malformed") from exc
    if schema.get("name") != "belief_transformer.paired_scalar_structure":
        raise PC1GateInputError("paired structure result schema is not recognized")
    if claim_boundaries.get("primary_measurement") != "paired scalar c=s_A-s_B":
        raise PC1GateInputError("paired structure result does not declare the corrected measurement")
    for name in ("checkpoint", "corpus", "q0", "q1"):
        _require_hash(bound_inputs.get(name), input_hashes[name], f"paired structure {name}")
    structure_sources = {
        "protocol": input_hashes["paired_structure_protocol"],
        "runner": source_hashes["imported_paired_scalar_scoring_runner"],
        "pca_helper": source_hashes["imported_pca_helper"],
        "robustness_helper": source_hashes["imported_robustness_helper"],
        "data_helper": source_hashes["imported_data_helper"],
        "extraction_helper": source_hashes["imported_extraction_helper"],
    }
    for name, expected in structure_sources.items():
        _require_hash(bound_sources.get(name), expected, f"paired structure source {name}")


def _validate_dimensionality_v2_result(
    path: Path, input_hashes: Mapping[str, str], source_hashes: Mapping[str, str]
) -> None:
    payload = _load_json_object(path, "dimensionality v2 result")
    try:
        schema = payload["schema"]
        claim_boundaries = payload["claim_boundaries"]
        provenance = payload["provenance"]
        bound_inputs = provenance["input_sha256"]
        bound_sources = provenance["source_sha256"]
    except (KeyError, TypeError) as exc:
        raise PC1GateInputError("dimensionality v2 result is malformed") from exc
    if schema.get("name") != "belief_transformer.paired_scalar_dimensionality_selection_v2":
        raise PC1GateInputError("dimensionality v2 result schema is not recognized")
    if claim_boundaries.get("measurement") != "paired scalar c=s_A-s_B only":
        raise PC1GateInputError("dimensionality v2 result does not declare the corrected measurement")
    for name in ("checkpoint", "corpus", "q0", "q1"):
        _require_hash(bound_inputs.get(name), input_hashes[name], f"dimensionality v2 {name}")
    _require_hash(
        bound_inputs.get("contextual_paired_scalar_result"),
        input_hashes["paired_structure_result"],
        "dimensionality v2 paired structure result",
    )
    dimensionality_sources = {
        "v2_frozen_protocol": input_hashes["dimensionality_v2_protocol"],
        "v2_runner": source_hashes["imported_dimensionality_v2_runner"],
        "imported_v1_runner": source_hashes["imported_dimensionality_v1_runner"],
        "paired_scalar_scoring_runner": source_hashes["imported_paired_scalar_scoring_runner"],
    }
    for name, expected in dimensionality_sources.items():
        _require_hash(bound_sources.get(name), expected, f"dimensionality v2 source {name}")


def _global_rng_states_equal(
    before: tuple[Any, ...], after: tuple[Any, ...]
) -> bool:
    if len(before) != len(after):
        return False
    for left, right in zip(before, after, strict=True):
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def build_analysis(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    corpus_path: Path = DEFAULT_CORPUS,
    q0_path: Path = DEFAULT_Q0,
    q1_path: Path = DEFAULT_Q1,
    paired_structure_protocol_path: Path = DEFAULT_PAIRED_STRUCTURE_PROTOCOL,
    paired_structure_result_path: Path = DEFAULT_PAIRED_STRUCTURE_RESULT,
    dimensionality_v2_protocol_path: Path = DEFAULT_DIMENSIONALITY_V2_PROTOCOL,
    dimensionality_v2_result_path: Path = DEFAULT_DIMENSIONALITY_V2_RESULT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    seed: int = ROOT_SEED,
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if isinstance(permutations, bool) or permutations < 1:
        raise PC1GateInputError("permutations must be a positive integer")
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 1:
        raise PC1GateInputError("bootstrap_replicates must be a positive integer")
    global_rng_before = np.random.get_state()
    input_paths = {
        "checkpoint": checkpoint_path.resolve(),
        "corpus": corpus_path.resolve(),
        "q0": q0_path.resolve(),
        "q1": q1_path.resolve(),
        "paired_structure_protocol": paired_structure_protocol_path.resolve(),
        "paired_structure_result": paired_structure_result_path.resolve(),
        "dimensionality_v2_protocol": dimensionality_v2_protocol_path.resolve(),
        "dimensionality_v2_result": dimensionality_v2_result_path.resolve(),
        "pc1_gate_protocol": protocol_path.resolve(),
    }
    source_paths = {
        "pc1_gate_runner": Path(__file__).resolve(),
        "imported_dimensionality_v2_runner": Path(v2.__file__).resolve(),
        "imported_dimensionality_v1_runner": Path(dimensionality_v1.__file__).resolve(),
        "imported_paired_scalar_scoring_runner": Path(paired.__file__).resolve(),
        "imported_pca_helper": Path(paired.expl.__file__).resolve(),
        "imported_robustness_helper": Path(paired.rob.__file__).resolve(),
        "imported_data_helper": ROOT / "core" / "data_utils.py",
        "imported_extraction_helper": ROOT / "core" / "nli_extraction.py",
    }
    input_hashes_before = _hash_records(input_paths)
    source_hashes_before = _hash_records(source_paths)
    _validate_paired_structure_result(
        input_paths["paired_structure_result"], input_hashes_before, source_hashes_before
    )
    _validate_dimensionality_v2_result(
        input_paths["dimensionality_v2_result"], input_hashes_before, source_hashes_before
    )
    paired.load_prompt_pairs(input_paths["q0"], input_paths["q1"])

    canonical_ids, logits = paired.load_checkpoint(input_paths["checkpoint"])
    articles = paired.load_articles(input_paths["corpus"], expected_n=EXPECTED_N)
    corpus_ids = [article["canonical_id"] for article in articles]
    if canonical_ids != corpus_ids:
        raise PC1GateInputError("checkpoint IDs do not exactly match corpus order")
    if len(canonical_ids) != EXPECTED_N or len(set(canonical_ids)) != EXPECTED_N:
        raise PC1GateInputError("analysis requires exactly 320 unique canonical IDs")
    publishers = [article["_publisher"] for article in articles]
    _balanced_groups(publishers, len(canonical_ids))

    scores: dict[str, np.ndarray] = {}
    for bank in ("q0", "q1"):
        _, _, contrast = paired.scores_from_logits(logits[bank]["A"], logits[bank]["B"])
        scores[bank] = _matrix(contrast, f"{bank} corrected paired scalar c=s_A-s_B")

    gates = run_gates(
        scores,
        publishers,
        seed=seed,
        permutations=permutations,
        bootstrap_replicates=bootstrap_replicates,
    )
    input_hashes_after = _hash_records(input_paths)
    source_hashes_after = _hash_records(source_paths)
    _assert_hashes_unchanged(input_hashes_before, input_hashes_after, "bound inputs")
    _assert_hashes_unchanged(source_hashes_before, source_hashes_after, "bound sources")
    global_rng_after = np.random.get_state()
    if not _global_rng_states_equal(global_rng_before, global_rng_after):
        raise PC1GateInputError("NumPy global RNG state changed during analysis")

    publisher_counts = dict(sorted(Counter(publishers).items()))
    return {
        "schema": {
            "name": "belief_transformer.paired_scalar_pc1_gate",
            "version": "1.0.0",
        },
        "protocol_date": PROTOCOL_DATE,
        "status": "posthoc_exploratory_analysis_of_inspected_data",
        "claim_boundaries": {
            "not_preregistered": True,
            "not_confirmatory": True,
            "not_independent_replication": True,
            "measurement": "corrected paired scalar c=s_A-s_B only",
            "q1_role": "paraphrase bank only",
            "qualitative_coding_performed": False,
            "semantic_axis_name_assigned": False,
            "metadata_analysis_performed": False,
            "publisher_aware_pc1_previously_passed": False,
            "gate1_allowed_outcomes": [
                "detected_under_this_null",
                "not_detected_under_this_null",
            ],
        },
        "analysis": {
            "n_articles": len(canonical_ids),
            "n_unique_canonical_ids": len(set(canonical_ids)),
            "canonical_ids": canonical_ids,
            "article_ids_exactly_match_corpus_order": True,
            "publisher_counts": publisher_counts,
            "balanced_publisher_design": "16x20",
            "pca": "covariance PCA, column-centered and unstandardized, denominator n-1",
            "primary_null_draws_per_bank": permutations,
            "conditional_bootstrap_replicates": bootstrap_replicates,
            "conditional_correspondence_null_draws": permutations,
            "numpy_global_rng_state_preserved": True,
        },
        "provenance": {
            "input_paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256_before": input_hashes_before,
            "input_sha256_after": input_hashes_after,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_sha256_before": source_hashes_before,
            "source_sha256_after": source_hashes_after,
            "all_bound_files_unchanged_during_analysis": True,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        "results": gates,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen bounded PC1 gate")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--q0", type=Path, default=DEFAULT_Q0)
    parser.add_argument("--q1", type=Path, default=DEFAULT_Q1)
    parser.add_argument(
        "--paired-structure-protocol", type=Path, default=DEFAULT_PAIRED_STRUCTURE_PROTOCOL
    )
    parser.add_argument(
        "--paired-structure-result", type=Path, default=DEFAULT_PAIRED_STRUCTURE_RESULT
    )
    parser.add_argument(
        "--dimensionality-v2-protocol", type=Path, default=DEFAULT_DIMENSIONALITY_V2_PROTOCOL
    )
    parser.add_argument(
        "--dimensionality-v2-result", type=Path, default=DEFAULT_DIMENSIONALITY_V2_RESULT
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_analysis(
        checkpoint_path=args.checkpoint,
        corpus_path=args.corpus,
        q0_path=args.q0,
        q1_path=args.q1,
        paired_structure_protocol_path=args.paired_structure_protocol,
        paired_structure_result_path=args.paired_structure_result,
        dimensionality_v2_protocol_path=args.dimensionality_v2_protocol,
        dimensionality_v2_result_path=args.dimensionality_v2_result,
        protocol_path=args.protocol,
    )
    _write_json_exclusive(args.output_json, payload)
    print(f"Bounded PC1 gate written to {args.output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PC1GateInputError,
        v2.DimensionalityV2InputError,
        dimensionality_v1.DimensionalityInputError,
        paired.PairedScalarInputError,
    ) as exc:
        print(f"PC1GateInputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
