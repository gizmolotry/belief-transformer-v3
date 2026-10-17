"""Fail-closed Q0/Q1 paraphrase-robustness analysis.

The two NLI banks have no authenticated article identifiers.  They are aligned
by row order only, and this limitation is recorded in every result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

if __name__ == "__main__":
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts import exploratory_internal_structure_analysis as helper


CLASS_ORDER = helper.CLASS_ORDER
N_OBSERVERS = helper.N_OBSERVERS
SCHEMA_NAME = "belief_transformer.q0_q1_paraphrase_robustness"
SCHEMA_VERSION = "1.0.0"
AnalysisInputError = helper.AnalysisInputError
write_json_atomic = helper.write_json_atomic


def _finite(value: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"{label} is not numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise AnalysisInputError(f"{label} must have {ndim} dimensions")
    if array.size == 0 or not np.isfinite(array).all():
        raise AnalysisInputError(f"{label} must be nonempty and finite")
    return array


def load_scores(source: Path | bytes) -> np.ndarray:
    """Load one artifact and return its n-article-by-8 polarity matrix."""

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    logits = helper.load_nli_logits(source)
    if logits.ndim != 3 or logits.shape[1:] != (N_OBSERVERS, len(CLASS_ORDER)):
        raise AnalysisInputError("NLI artifact must have shape (n_articles, 8, 3)")
    scores = helper.observer_polarity(logits)
    if scores.shape[1:] != (N_OBSERVERS,) or not np.isfinite(scores).all():
        raise AnalysisInputError("NLI polarity matrix is invalid")
    return scores


def _p_display(value: float) -> str:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise AnalysisInputError("p-value is invalid")
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def _permuted_columns(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    matrix = _finite(matrix, "permutation matrix", ndim=2)
    return np.column_stack(
        [matrix[rng.permutation(matrix.shape[0]), column] for column in range(matrix.shape[1])]
    )


def _variance_null(
    matrix: np.ndarray, observed: float, permutations: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        pca = helper.deterministic_pca(_permuted_columns(matrix, rng))
        values[index] = float(np.sum(pca["explained_variance_ratio"]))
    if not np.isfinite(values).all():
        raise AnalysisInputError("variance null produced nonfinite values")
    return values


def _procrustes_disparity(scores0: np.ndarray, scores1: np.ndarray) -> float:
    scores0 = _finite(scores0, "Q0 projected scores", ndim=2)
    scores1 = _finite(scores1, "Q1 projected scores", ndim=2)
    if scores0.shape != scores1.shape or scores0.shape[1] != 2:
        raise AnalysisInputError("projected score clouds must have matching n-by-2 shapes")
    try:
        _, _, disparity = procrustes(scores0, scores1)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise AnalysisInputError("Procrustes disparity could not be computed") from exc
    if not math.isfinite(float(disparity)):
        raise AnalysisInputError("Procrustes disparity is nonfinite")
    return float(disparity)


def _procrustes_null(
    scores0: np.ndarray, scores1: np.ndarray, permutations: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        values[index] = _procrustes_disparity(scores0, scores1[rng.permutation(scores1.shape[0])])
    return values


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = _finite(values, "null distribution", ndim=1)
    return {
        "null_mean": float(np.mean(values)),
        "null_quantiles": {
            "2.5%": float(np.quantile(values, 0.025)),
            "97.5%": float(np.quantile(values, 0.975)),
        },
    }


def _variance_result(observed: float, null: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(null >= observed))
    p_value = (1.0 + count) / (null.size + 1.0)
    return {
        "observed": float(observed),
        **_summary(null),
        "count_null_ge_observed": count,
        "p_value": float(p_value),
        "p_display": _p_display(p_value),
        "tail": "upper",
    }


def _procrustes_result(observed: float, null: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(null <= observed))
    p_value = (1.0 + count) / (null.size + 1.0)
    rank = count + 1
    denominator = null.size + 1
    return {
        "observed": float(observed),
        **_summary(null),
        "count_null_le_observed": count,
        "p_value": float(p_value),
        "p_display": _p_display(p_value),
        "tail": "lower",
        "lower_tail_rank": rank,
        "lower_tail_rank_out_of": denominator,
        "lower_tail_percentile": float(100.0 * rank / denominator),
    }


def _distance_result(scores0: np.ndarray, scores1: np.ndarray) -> dict[str, Any]:
    distances0 = pdist(scores0)
    distances1 = pdist(scores1)
    if distances0.size < 3 or np.ptp(distances0) == 0.0 or np.ptp(distances1) == 0.0:
        raise AnalysisInputError("pairwise projected distances must be nonconstant")
    result = spearmanr(distances0, distances1)
    rho, p_value = float(result.statistic), float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(p_value):
        raise AnalysisInputError("pairwise-distance Spearman result is nonfinite")
    p_value = max(p_value, np.finfo(np.float64).tiny)
    return {
        "rho": rho,
        "scipy_p_value_naive": p_value,
        "scipy_p_display_naive": _p_display(p_value),
        "n_pairwise_distances": int(distances0.size),
    }


def _verify_exact_head_source(script_path: Path) -> dict[str, Any]:
    script_path = script_path.resolve()
    try:
        top_level_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=script_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = Path(top_level_result.stdout.strip()).resolve()
        if not top_level_result.stdout.strip() or not script_path.is_relative_to(repo_root):
            raise AnalysisInputError("running script is outside the Git top level")
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = commit_result.stdout.strip()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise AnalysisInputError("git rev-parse returned an invalid commit")
        repo_relative_path = script_path.relative_to(repo_root).as_posix()
        head_result = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{repo_relative_path}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=False,
        )
        head_bytes = bytes(head_result.stdout)
        running_bytes = script_path.read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisInputError("code provenance command failed") from exc

    running_sha256 = hashlib.sha256(running_bytes).hexdigest()
    head_sha256 = hashlib.sha256(head_bytes).hexdigest()
    if running_bytes != head_bytes:
        raise AnalysisInputError(
            "running script bytes differ from the script recorded at git HEAD; "
            "commit the exact analysis code before running the CLI"
        )
    return {
        "git_commit": commit,
        "script": {
            "repo_relative_path": repo_relative_path,
            "sha256": running_sha256,
            "git_head_sha256": head_sha256,
            "exact_head_match": True,
        },
    }


def _code_provenance() -> dict[str, Any]:
    harness_record = _verify_exact_head_source(Path(__file__))
    helper_record = _verify_exact_head_source(Path(helper.__file__))
    if harness_record.get("git_commit") != helper_record.get("git_commit"):
        raise AnalysisInputError("harness and imported helper provenance commits differ")
    for label, record in (("harness", harness_record), ("helper", helper_record)):
        source = record.get("script")
        if not isinstance(source, Mapping) or source.get("exact_head_match") is not True:
            raise AnalysisInputError(f"{label} source is not an exact Git-HEAD match")
    return {
        "git_commit": str(harness_record["git_commit"]),
        "harness": dict(harness_record["script"]),
        "helper": dict(helper_record["script"]),
    }


def build_analysis(
    q0_artifact: Path,
    q1_artifact: Path,
    *,
    permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.perf_counter()
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, (int, np.integer))
        or int(permutations) < 1
    ):
        raise AnalysisInputError("permutations must be a positive integer")
    permutations = int(permutations)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise AnalysisInputError("seed must be a deterministic integer")
    seed = int(seed)
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisInputError("seed must be a deterministic integer") from exc

    q0_artifact = helper._require_file(Path(q0_artifact), "Q0 artifact")
    q1_artifact = helper._require_file(Path(q1_artifact), "Q1 artifact")
    if q0_artifact == q1_artifact:
        raise AnalysisInputError("Q0 and Q1 artifacts must be different files")
    input_paths = {"q0_artifact": q0_artifact, "q1_artifact": q1_artifact}
    snapshots: dict[str, bytes] = {}
    initial_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        try:
            snapshot = path.read_bytes()
        except OSError as exc:
            raise AnalysisInputError(f"could not read {name}: {path}") from exc
        snapshots[name] = snapshot
        initial_hashes[name] = hashlib.sha256(snapshot).hexdigest()

    code = _code_provenance()

    q0 = load_scores(snapshots["q0_artifact"])
    q1 = load_scores(snapshots["q1_artifact"])
    if q0.shape[0] != q1.shape[0]:
        raise AnalysisInputError(f"Q0/Q1 row counts differ: {q0.shape[0]} versus {q1.shape[0]}")
    if q0.shape[0] < 3:
        raise AnalysisInputError("at least three paired article rows are required")

    pca0 = helper.deterministic_pca(q0)
    pca1 = helper.deterministic_pca(q1)
    observed_variance0 = float(np.sum(pca0["explained_variance_ratio"]))
    observed_variance1 = float(np.sum(pca1["explained_variance_ratio"]))
    variance0 = _variance_null(q0, observed_variance0, permutations, rng)
    variance1 = _variance_null(q1, observed_variance1, permutations, rng)
    observed_disparity = _procrustes_disparity(pca0["scores"], pca1["scores"])
    correspondence_null = _procrustes_null(pca0["scores"], pca1["scores"], permutations, rng)
    distances = _distance_result(pca0["scores"], pca1["scores"])
    angles = np.asarray(subspace_angles(pca0["loadings"].T, pca1["loadings"].T), dtype=np.float64)
    if angles.shape != (2,) or not np.isfinite(angles).all():
        raise AnalysisInputError("principal angles are invalid")

    final_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise AnalysisInputError(f"could not reread {name}: {path}") from exc
        final_hashes[name] = hashlib.sha256(current).hexdigest()
    changed = [name for name in input_paths if initial_hashes[name] != final_hashes[name]]
    if changed:
        raise AnalysisInputError("input artifacts changed while analysis was running: " + ", ".join(changed))

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "provenance": {
            **code,
            "runtime_environment": helper.runtime_environment_versions(),
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256_before": initial_hashes[name],
                    "sha256_after": final_hashes[name],
                }
                for name, path in input_paths.items()
            },
            "class_order": list(CLASS_ORDER),
            "alignment": {
                "mode": "row_order_only",
                "article_ids_independently_authenticated": False,
                "limitation": (
                    "Q0 and Q1 rows are paired by position and equal row counts only; article IDs "
                    "are absent and semantic alignment cannot be independently authenticated."
                ),
            },
        },
        "analysis": {
            "seed": int(seed),
            "permutations": permutations,
            "runtime_seconds": float(time.perf_counter() - started),
            "confirmatory_decision": None,
        },
        "formulas_and_null_interpretations": {
            "observer_polarity": "softmax(logits_raw)[entailment] - softmax(logits_raw)[contradiction]",
            "pca": "independent SVD of each column-centered article-by-eight polarity matrix",
            "variance_null": (
                "For each bank, independently permute each observer column across rows and refit PCA; "
                "this tests cross-axis dependence/non-independence, not validity of a specific 2D structure."
            ),
            "variance_p_value": "(1 + count(null >= observed)) / (B + 1)",
            "procrustes": "scipy.spatial.procrustes disparity on the paired projected article rows",
            "procrustes_null": (
                "Hold both projected clouds and PCA fits fixed; independently permute only Q1 row "
                "correspondence before the same scipy.spatial.procrustes operation."
            ),
            "procrustes_p_value": "(1 + count(null <= observed)) / (B + 1)",
            "pairwise_distance_spearman": (
                "Spearman correlation of upper-triangular pairwise distances; distances are dependent, "
                "so the scipy p-value is descriptive/naive only."
            ),
            "artifact_snapshot_and_hash_limitation": (
                "Artifacts are snapshotted once, those exact bytes are analyzed, and final path hashes "
                "must match the snapshot hashes. This prevents unobserved replacement bytes from being "
                "analyzed but cannot prove that an unobservable A→B→A filesystem event did not occur."
            ),
            "scope": "robustness analysis only; no confirmatory decision label is emitted",
        },
        "results": {
            "n_articles": int(q0.shape[0]),
            "pca": {
                "q0": {
                    "two_pc_variance_ratio": observed_variance0,
                    "explained_variance_ratio": [
                        float(value) for value in pca0["explained_variance_ratio"]
                    ],
                },
                "q1": {
                    "two_pc_variance_ratio": observed_variance1,
                    "explained_variance_ratio": [
                        float(value) for value in pca1["explained_variance_ratio"]
                    ],
                },
            },
            "variance_null": {
                "q0": _variance_result(observed_variance0, variance0),
                "q1": _variance_result(observed_variance1, variance1),
            },
            "principal_angles": {
                "radians": [float(value) for value in angles],
                "degrees": [float(np.degrees(value)) for value in angles],
            },
            "procrustes": _procrustes_result(observed_disparity, correspondence_null),
            "pairwise_projected_distance_spearman": distances,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fail-closed Q0/Q1 robustness analysis.")
    parser.add_argument("--q0-artifact", required=True, type=Path)
    parser.add_argument("--q1-artifact", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output_json.resolve()
    protected_paths = {
        Path(args.q0_artifact).resolve(),
        Path(args.q1_artifact).resolve(),
        Path(__file__).resolve(),
        Path(helper.__file__).resolve(),
    }
    if output_path in protected_paths:
        raise AnalysisInputError("output JSON must not overwrite an input or source artifact")
    payload = build_analysis(
        args.q0_artifact,
        args.q1_artifact,
        permutations=args.permutations,
        seed=args.seed,
    )
    write_json_atomic(payload, output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisInputError as exc:
        print(f"AnalysisInputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
