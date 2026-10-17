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

from scripts import paired_scalar_dimensionality as v1


paired = v1.paired
SEED = 20_260_907
PERMUTATIONS = 10_000
BOOTSTRAP_REPLICATES = 10_000
K_VALUES = (1, 2, 3)
ALPHA = 0.05
EXPECTED_PUBLISHERS = 16
EXPECTED_PER_PUBLISHER = 20
EXPECTED_N = EXPECTED_PUBLISHERS * EXPECTED_PER_PUBLISHER

DEFAULT_CHECKPOINT = ROOT / "outputs" / "fresh_scalar_structure_run" / "extraction_checkpoint.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "controlled_N_dense_corpus.jsonl"
DEFAULT_Q0 = ROOT / "config" / "framing_queries.yaml"
DEFAULT_Q1 = ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
DEFAULT_CONTEXT_RESULT = ROOT / "paper" / "paired_scalar_structure_results_20260906.json"
V1_PROTOCOL = ROOT / "paper" / "paired_scalar_dimensionality_protocol_20260907.md"
V1_RESULT = ROOT / "paper" / "paired_scalar_dimensionality_results_20260907.json"
V2_PROTOCOL = ROOT / "paper" / "paired_scalar_dimensionality_protocol_v2_20260907.md"


class DimensionalityV2InputError(ValueError):
    """Raised when the frozen corrected-v2 analysis contract is not satisfied."""


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DimensionalityV2InputError(f"{label} is not numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 8 or matrix.shape[0] < 16:
        raise DimensionalityV2InputError(f"{label} must have shape (n>=16, 8), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise DimensionalityV2InputError(f"{label} contains nonfinite values")
    return matrix


def _mle_pca_fit(value: Any) -> dict[str, np.ndarray]:
    matrix = _matrix(value, "PPCA training matrix")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    loadings = right_vectors.T
    eigenvalues = singular_values**2 / matrix.shape[0]
    if eigenvalues.shape != (8,) or eigenvalues[0] <= 0.0:
        raise DimensionalityV2InputError("PPCA training covariance lacks positive variance")
    pivots = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[pivots, np.arange(loadings.shape[1])])
    signs[signs == 0.0] = 1.0
    return {"mean": mean, "eigenvalues": eigenvalues, "loadings": loadings * signs}


def fit_closed_form_ppca(value: Any, k: int) -> dict[str, Any]:
    if k not in K_VALUES:
        raise DimensionalityV2InputError(f"PPCA k must be one of {K_VALUES}")
    fit = _mle_pca_fit(value)
    raw_eigenvalues = fit["eigenvalues"]
    raw_retained = raw_eigenvalues[:k]
    raw_sigma2 = float(np.mean(raw_eigenvalues[k:]))
    lambda1 = float(raw_eigenvalues[0])
    numeric_floor = max(
        lambda1 * 1e-12,
        np.finfo(np.float64).eps * max(lambda1, 1.0),
    )
    retained = np.maximum(raw_retained, numeric_floor)
    sigma2 = max(raw_sigma2, numeric_floor)
    covariance_eigenvalues = np.concatenate(
        [retained, np.full(8 - k, sigma2, dtype=np.float64)]
    )
    if not np.isfinite(covariance_eigenvalues).all() or np.any(covariance_eigenvalues <= 0.0):
        raise DimensionalityV2InputError("floored PPCA covariance is not positive definite")
    return {
        "k": k,
        "mean": fit["mean"],
        "retained_loadings": fit["loadings"][:, :k],
        "raw_full_mle_covariance_eigenvalues": raw_eigenvalues,
        "raw_retained_eigenvalues": raw_retained,
        "floored_retained_eigenvalues": retained,
        "raw_sigma2": raw_sigma2,
        "floored_sigma2": sigma2,
        "numeric_floor": numeric_floor,
        "covariance_eigenvalues": covariance_eigenvalues,
    }


def ppca_negative_log_likelihood(value: Any, fit: Mapping[str, Any]) -> np.ndarray:
    matrix = _matrix(value, "PPCA held-out matrix")
    mean = np.asarray(fit["mean"], dtype=np.float64)
    basis = np.asarray(fit["retained_loadings"], dtype=np.float64)
    retained = np.asarray(fit["floored_retained_eigenvalues"], dtype=np.float64)
    sigma2 = float(fit["floored_sigma2"])
    k = int(fit["k"])
    if mean.shape != (8,) or basis.shape != (8, k) or retained.shape != (k,):
        raise DimensionalityV2InputError("PPCA fit has inconsistent dimensions")
    if sigma2 <= 0.0 or np.any(retained <= 0.0):
        raise DimensionalityV2InputError("PPCA fit covariance is not positive definite")

    centered = matrix - mean
    projected = centered @ basis
    residual = centered - projected @ basis.T
    quadratic = np.sum(projected**2 / retained, axis=1) + np.sum(residual**2, axis=1) / sigma2
    log_determinant = float(np.sum(np.log(retained)) + (8 - k) * math.log(sigma2))
    nll = 0.5 * (8 * math.log(2.0 * math.pi) + log_determinant + quadratic)
    if not np.isfinite(nll).all():
        raise DimensionalityV2InputError("PPCA held-out NLL is nonfinite")
    return nll


def leave_one_publisher_out_ppca(value: Any, publishers: Sequence[str]) -> dict[str, Any]:
    matrix = _matrix(value, "LOPO PPCA matrix")
    labels = np.asarray(publishers, dtype=object)
    if labels.shape != (matrix.shape[0],):
        raise DimensionalityV2InputError("publisher labels do not match PPCA matrix rows")
    groups = sorted(set(labels.tolist()))
    if len(groups) != EXPECTED_PUBLISHERS:
        raise DimensionalityV2InputError("LOPO PPCA requires exactly 16 publishers")
    counts = Counter(labels.tolist())
    if set(counts.values()) != {EXPECTED_PER_PUBLISHER}:
        raise DimensionalityV2InputError("LOPO PPCA requires exactly 20 articles per publisher")

    folds: list[dict[str, Any]] = []
    nll_by_k: dict[str, list[float]] = {str(k): [] for k in K_VALUES}
    for held_out in groups:
        test_indices = np.flatnonzero(labels == held_out)
        train_indices = np.flatnonzero(labels != held_out)
        if set(labels[train_indices]).intersection(set(labels[test_indices])):
            raise DimensionalityV2InputError("publisher leakage detected in LOPO PPCA")
        candidates: dict[str, Any] = {}
        for k in K_VALUES:
            fit = fit_closed_form_ppca(matrix[train_indices], k)
            mean_nll = float(ppca_negative_log_likelihood(matrix[test_indices], fit).mean())
            nll_by_k[str(k)].append(mean_nll)
            candidates[str(k)] = {
                "held_out_mean_nll_per_article": mean_nll,
                "numeric_floor": float(fit["numeric_floor"]),
                "raw_full_mle_covariance_eigenvalues": [
                    float(item) for item in fit["raw_full_mle_covariance_eigenvalues"]
                ],
                "raw_retained_eigenvalues": [float(item) for item in fit["raw_retained_eigenvalues"]],
                "floored_retained_eigenvalues": [
                    float(item) for item in fit["floored_retained_eigenvalues"]
                ],
                "raw_sigma2": float(fit["raw_sigma2"]),
                "floored_sigma2": float(fit["floored_sigma2"]),
                "covariance_eigenvalues": [
                    float(item) for item in fit["covariance_eigenvalues"]
                ],
            }
        folds.append(
            {
                "held_out_publisher": held_out,
                "n_train_publishers": 15,
                "n_train_articles": int(train_indices.size),
                "n_test_articles": int(test_indices.size),
                "candidates": candidates,
            }
        )

    summaries: dict[str, Any] = {}
    for k in K_VALUES:
        values = np.asarray(nll_by_k[str(k)], dtype=np.float64)
        summaries[str(k)] = {
            "publisher_mean_nll_values": [float(item) for item in values],
            "mean_publisher_nll": float(values.mean()),
            "standard_error_across_publishers": float(
                values.std(ddof=1) / math.sqrt(EXPECTED_PUBLISHERS)
            ),
            "publisher_nll_distribution": v1._quantiles(values),
        }
    best_k = min(K_VALUES, key=lambda candidate: (summaries[str(candidate)]["mean_publisher_nll"], candidate))
    threshold = (
        summaries[str(best_k)]["mean_publisher_nll"]
        + summaries[str(best_k)]["standard_error_across_publishers"]
    )
    selected_k = min(
        candidate
        for candidate in K_VALUES
        if summaries[str(candidate)]["mean_publisher_nll"] <= threshold
    )
    return {
        "selection_scope": "descriptive_model_selection_not_a_hypothesis_test",
        "selection_rule": "smallest k within one standard error of best mean publisher-level held-out PPCA NLL",
        "best_mean_nll_k": best_k,
        "one_standard_error_threshold": threshold,
        "selected_k": selected_k,
        "candidate_summaries": summaries,
        "folds": folds,
    }


def _balanced_group_indices(
    publishers: Sequence[str], n_rows: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = np.asarray(publishers, dtype=object)
    if labels.shape != (n_rows,):
        raise DimensionalityV2InputError("publisher labels do not match matrix rows")
    groups = np.asarray(sorted(set(labels.tolist())), dtype=object)
    if groups.size != EXPECTED_PUBLISHERS:
        raise DimensionalityV2InputError("publisher-block null requires exactly 16 publishers")
    indices = {group: np.flatnonzero(labels == group) for group in groups}
    if any(rows.size != EXPECTED_PER_PUBLISHER for rows in indices.values()):
        raise DimensionalityV2InputError("publisher-block null requires balanced 16x20 blocks")
    return groups, indices


def publisher_block_permutation(
    value: Any, publishers: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    matrix = _matrix(value, "publisher-block permutation matrix")
    groups, group_indices = _balanced_group_indices(publishers, matrix.shape[0])
    permuted = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        source_groups = rng.permutation(groups)
        for target_group, source_group in zip(groups, source_groups, strict=True):
            source_values = matrix[group_indices[source_group], column]
            permuted[group_indices[target_group], column] = rng.permutation(source_values)
    return permuted


def _same_rank_eigenvalue_summary(
    observed: np.ndarray, null: np.ndarray, permutations: int
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for offset, component in enumerate((2, 3)):
        values = null[:, offset]
        observed_value = float(observed[component - 1])
        threshold = float(np.quantile(values, 0.95, method="higher"))
        exceedances = int(np.count_nonzero(values >= observed_value))
        plus_one_p = (exceedances + 1) / (permutations + 1)
        detected = bool(observed_value > threshold and plus_one_p <= ALPHA)
        status = (
            f"pc{component}_detected_under_cluster_null"
            if detected
            else (
                "pc2_not_detected_under_cluster_null"
                if component == 2
                else "pc3_not_detected"
            )
        )
        components[f"pc{component}"] = {
            "observed_eigenvalue": observed_value,
            "observed_explained_variance_ratio": float(observed_value / observed.sum()),
            "null_mean": float(values.mean()),
            "null_95th_percentile_method_higher": threshold,
            "null_exceedances_greater_or_equal": exceedances,
            "plus_one_p_value": plus_one_p,
            "detected": detected,
            "status": status,
        }
    return components


def publisher_block_eigenvalue_null(
    value: Any,
    publishers: Sequence[str],
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if permutations < 1:
        raise DimensionalityV2InputError("permutation count must be positive")
    matrix = _matrix(value, "publisher-block eigenvalue-null matrix")
    _balanced_group_indices(publishers, matrix.shape[0])
    observed = v1._pca_fit(matrix)["eigenvalues"]
    null = np.empty((permutations, 2), dtype=np.float64)
    for draw in range(permutations):
        permuted = publisher_block_permutation(matrix, publishers, rng)
        null[draw] = v1._pca_fit(permuted)["eigenvalues"][1:3]
    return {
        "role": "primary_cluster_aware_null",
        "permutations": permutations,
        "scheme": (
            "independently reassign complete 20-value publisher blocks per column, then "
            "independently permute row order inside every assigned block"
        ),
        "full_observed_eigenvalues": [float(item) for item in observed],
        "full_observed_explained_variance_ratio": [
            float(item / observed.sum()) for item in observed
        ],
        "components": _same_rank_eigenvalue_summary(observed, null, permutations),
    }


def article_level_sensitivity_null(
    value: Any, *, permutations: int, rng: np.random.Generator
) -> dict[str, Any]:
    result = v1.eigenvalue_permutation_null(value, permutations=permutations, rng=rng)
    components: dict[str, Any] = {}
    for name, record in result["components"].items():
        copied = dict(record)
        detected = bool(copied.pop("passes"))
        components[name] = {
            **copied,
            "detected": detected,
        }
    return {
        **result,
        "role": "article_level_sensitivity_only_not_used_for_primary_verdict",
        "components": components,
    }


def dimensionality_verdict(
    *, pc2_detected: bool, pc3_detected: bool, pc2_stable: bool, selected_k: int
) -> str:
    if not pc2_detected:
        return "pc2_not_detected_under_cluster_null"
    if not pc2_stable or selected_k == 1:
        return "inconclusive_mixed_evidence"
    if pc3_detected and selected_k == 3:
        return "evidence_for_at_least_three"
    return "evidence_for_at_least_two"


def _hash_records(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DimensionalityV2InputError(f"missing required files: {', '.join(missing)}")
    return {name: paired.sha256_file(path) for name, path in paths.items()}


def _validate_context_result(path: Path, input_hashes: Mapping[str, str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bound = payload["provenance"]["inputs"]
        primary = payload["claim_boundaries"]["primary_measurement"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DimensionalityV2InputError("context paired-scalar result is malformed") from exc
    for name in ("checkpoint", "corpus", "q0", "q1"):
        if str(bound.get(name, "")).lower() != input_hashes[name].lower():
            raise DimensionalityV2InputError(f"context result does not bind current {name}")
    if primary != "paired scalar c=s_A-s_B":
        raise DimensionalityV2InputError("context result does not declare paired scalar as primary")


def _validate_v1_result(path: Path, input_hashes: Mapping[str, str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload["schema"]
        bound = payload["provenance"]["input_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DimensionalityV2InputError("v1 result is malformed") from exc
    if schema.get("name") != "belief_transformer.paired_scalar_dimensionality_selection":
        raise DimensionalityV2InputError("v1 result schema is not recognized")
    for name in ("checkpoint", "corpus", "q0", "q1"):
        if str(bound.get(name, "")).lower() != input_hashes[name].lower():
            raise DimensionalityV2InputError(f"v1 result does not bind current {name}")


def build_analysis(
    *,
    checkpoint_path: Path,
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    context_result_path: Path,
    v1_protocol_path: Path = V1_PROTOCOL,
    v1_result_path: Path = V1_RESULT,
    v2_protocol_path: Path = V2_PROTOCOL,
    seed: int = SEED,
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    input_paths = {
        "checkpoint": checkpoint_path,
        "corpus": corpus_path,
        "q0": q0_path,
        "q1": q1_path,
        "contextual_paired_scalar_result": context_result_path,
        "v1_protocol": v1_protocol_path,
        "v1_result": v1_result_path,
    }
    source_paths = {
        "v2_runner": Path(__file__).resolve(),
        "v2_frozen_protocol": v2_protocol_path,
        "imported_v1_runner": Path(v1.__file__).resolve(),
        "paired_scalar_scoring_runner": Path(paired.__file__).resolve(),
    }
    initial_inputs = _hash_records(input_paths)
    initial_sources = _hash_records(source_paths)
    _validate_context_result(context_result_path, initial_inputs)
    _validate_v1_result(v1_result_path, initial_inputs)
    paired.load_prompt_pairs(q0_path, q1_path)

    ids, logits = paired.load_checkpoint(checkpoint_path)
    articles = paired.load_articles(corpus_path, expected_n=EXPECTED_N)
    if ids != [article["canonical_id"] for article in articles]:
        raise DimensionalityV2InputError("checkpoint IDs do not exactly match corpus order")
    publishers = [article["_publisher"] for article in articles]
    _balanced_group_indices(publishers, len(ids))
    scores: dict[str, np.ndarray] = {}
    for bank in ("q0", "q1"):
        _, _, contrast = paired.scores_from_logits(logits[bank]["A"], logits[bank]["B"])
        scores[bank] = _matrix(contrast, f"{bank} paired scalar")

    stream_names = (
        "q0_primary_publisher_block_null",
        "q1_primary_publisher_block_null",
        "q0_article_sensitivity_null",
        "q1_article_sensitivity_null",
        "q0_publisher_bootstrap",
        "q1_publisher_bootstrap",
    )
    children = np.random.SeedSequence(seed).spawn(len(stream_names))
    streams = dict(zip(stream_names, children, strict=True))

    results: dict[str, Any] = {}
    for bank in ("q0", "q1"):
        ppca = leave_one_publisher_out_ppca(scores[bank], publishers)
        primary = publisher_block_eigenvalue_null(
            scores[bank],
            publishers,
            permutations=permutations,
            rng=np.random.default_rng(streams[f"{bank}_primary_publisher_block_null"]),
        )
        sensitivity = article_level_sensitivity_null(
            scores[bank],
            permutations=permutations,
            rng=np.random.default_rng(streams[f"{bank}_article_sensitivity_null"]),
        )
        bootstrap = v1.publisher_cluster_bootstrap(
            scores[bank],
            publishers,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(streams[f"{bank}_publisher_bootstrap"]),
        )
        pc2 = primary["components"]["pc2"]
        pc3 = primary["components"]["pc3"]
        verdict = dimensionality_verdict(
            pc2_detected=bool(pc2["detected"]),
            pc3_detected=bool(pc3["detected"]),
            pc2_stable=bool(bootstrap["stability_passes"]),
            selected_k=int(ppca["selected_k"]),
        )
        results[bank] = {
            "decision_inputs": {
                "pc2_primary_status": pc2["status"],
                "pc3_primary_status": pc3["status"],
                "pc2_loading_stability_passes": bool(bootstrap["stability_passes"]),
                "ppca_selected_k": int(ppca["selected_k"]),
            },
            "dimensionality_verdict": verdict,
            "ppca_leave_one_publisher_out": ppca,
            "primary_publisher_block_eigenvalue_null": primary,
            "article_level_eigenvalue_null_sensitivity": sensitivity,
            "publisher_cluster_bootstrap": bootstrap,
            "interpretation_boundary": (
                "bootstrap instability is not evidence that PC2 is absent"
                if not bootstrap["stability_passes"]
                else "loading stability alone does not establish external validity"
            ),
        }

    verdicts = {bank: results[bank]["dimensionality_verdict"] for bank in ("q0", "q1")}
    cross_bank = (
        f"paraphrase_stable_{verdicts['q0']}"
        if verdicts["q0"] == verdicts["q1"]
        else "cross_bank_inconclusive_different_outcomes"
    )
    final_inputs = _hash_records(input_paths)
    final_sources = _hash_records(source_paths)
    if initial_inputs != final_inputs or initial_sources != final_sources:
        raise DimensionalityV2InputError("bound inputs or sources changed during analysis")

    return {
        "schema": {
            "name": "belief_transformer.paired_scalar_dimensionality_selection_v2",
            "version": "2.0.0",
        },
        "status": "post_review_corrected_exploratory_reanalysis_of_inspected_data",
        "claim_boundaries": {
            "not_preregistered": True,
            "not_confirmatory": True,
            "not_independent_replication": True,
            "measurement": "paired scalar c=s_A-s_B only",
            "non_detection_is_not_absence_or_equivalence": True,
            "no_component_names_or_manuscript_changes_authorized": True,
        },
        "analysis": {
            "root_seed": seed,
            "child_seed_spawn_keys": {
                name: list(sequence.spawn_key) for name, sequence in streams.items()
            },
            "n_articles": len(ids),
            "publisher_counts": dict(sorted(Counter(publishers).items())),
            "ppca_candidates": list(K_VALUES),
            "lopo_folds": EXPECTED_PUBLISHERS,
            "primary_permutations_per_bank": permutations,
            "sensitivity_permutations_per_bank": permutations,
            "bootstrap_replicates_per_bank": bootstrap_replicates,
        },
        "provenance": {
            "input_paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256": initial_inputs,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_sha256": initial_sources,
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
    parser = argparse.ArgumentParser(description="Run the frozen corrected v2 dimensionality analysis")
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
    print(f"Corrected v2 dimensionality analysis written to {args.output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DimensionalityV2InputError, v1.DimensionalityInputError, paired.PairedScalarInputError) as exc:
        print(f"DimensionalityV2InputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
