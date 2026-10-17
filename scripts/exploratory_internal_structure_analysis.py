"""Deterministic exploratory analysis of observer-score internal structure.

The NLI artifact stores no article identifiers.  Its rows are therefore aligned
to the corpus by row order only; this limitation is recorded in every output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import torch
from scipy.stats import f as f_distribution
from scipy.stats import pearsonr


CLASS_ORDER = ("contradiction", "neutral", "entailment")
N_OBSERVERS = 8
SCHEMA_NAME = "belief_transformer.exploratory_internal_structure"
SCHEMA_VERSION = "1.1.0"


class AnalysisInputError(ValueError):
    """Raised when an input cannot support the prespecified analysis."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AnalysisInputError(f"{label} is not a file: {resolved}")
    return resolved


def _finite_array(value: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"{label} is not numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise AnalysisInputError(f"{label} must have {ndim} dimensions, got {array.ndim}")
    if array.size == 0 or not np.isfinite(array).all():
        raise AnalysisInputError(f"{label} must be nonempty and finite")
    return array


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = _finite_array(logits, "logits", ndim=3)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(axis=-1, keepdims=True)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise AnalysisInputError("softmax denominator is invalid")
    probabilities = exponentials / denominator
    if not np.isfinite(probabilities).all():
        raise AnalysisInputError("softmax produced nonfinite probabilities")
    return probabilities


def observer_polarity(logits: np.ndarray) -> np.ndarray:
    """Return P(entailment) - P(contradiction) for each article/observer."""

    probabilities = stable_softmax(logits)
    return probabilities[..., 2] - probabilities[..., 0]


def load_nli_logits(path: Path) -> np.ndarray:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AnalysisInputError(f"could not load NLI artifact: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisInputError("NLI artifact must be a mapping")
    pairs = payload.get("nli_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)) or not pairs:
        raise AnalysisInputError("NLI artifact must contain a nonempty nli_pairs sequence")

    rows: list[np.ndarray] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or "logits_raw" not in pair:
            raise AnalysisInputError(f"nli_pairs[{index}] lacks logits_raw")
        if any(key in pair for key in ("canonical_id", "article_id", "id")):
            raise AnalysisInputError(
                "NLI rows unexpectedly contain article IDs; this row-order-only analysis "
                "does not implement ID-based alignment"
            )
        logits = pair["logits_raw"]
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        array = _finite_array(logits, f"nli_pairs[{index}].logits_raw", ndim=2)
        if array.shape != (N_OBSERVERS, len(CLASS_ORDER)):
            raise AnalysisInputError(
                f"nli_pairs[{index}].logits_raw must have shape "
                f"({N_OBSERVERS}, {len(CLASS_ORDER)}), got {array.shape}"
            )
        rows.append(array)

    declared_count = payload.get("n_articles")
    if declared_count is not None:
        if isinstance(declared_count, bool) or not isinstance(declared_count, (int, np.integer)):
            raise AnalysisInputError("NLI artifact n_articles must be an integer")
        if int(declared_count) != len(rows):
            raise AnalysisInputError("NLI artifact n_articles disagrees with nli_pairs length")
    return np.stack(rows, axis=0)


def _parse_published_date(article: Mapping[str, Any], line_number: int) -> date:
    raw = article.get("published_at") or article.get("published_date")
    if not isinstance(raw, str) or not raw.strip():
        raise AnalysisInputError(f"corpus line {line_number} lacks a published date")
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError as exc:
        raise AnalysisInputError(
            f"corpus line {line_number} has an invalid published date"
        ) from exc


def load_corpus(path: Path) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AnalysisInputError(f"corpus line {line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisInputError(f"corpus line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise AnalysisInputError(f"corpus line {line_number} must be a JSON object")

            canonical_id = value.get("canonical_id")
            if not isinstance(canonical_id, str) or not canonical_id.strip():
                raise AnalysisInputError(f"corpus line {line_number} lacks canonical_id")
            canonical_id = canonical_id.strip()
            if canonical_id in seen_ids:
                raise AnalysisInputError(f"duplicate corpus canonical_id: {canonical_id}")
            seen_ids.add(canonical_id)

            publisher = value.get("publisher") or value.get("source")
            if not isinstance(publisher, str) or not publisher.strip():
                raise AnalysisInputError(f"corpus line {line_number} lacks publisher/source")
            articles.append(
                {
                    "canonical_id": canonical_id,
                    "publisher": publisher.strip(),
                    "published_date": _parse_published_date(value, line_number),
                }
            )
    if not articles:
        raise AnalysisInputError("corpus is empty")
    return articles


def load_mbfc(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisInputError("MBFC input is invalid JSON") from exc
    if not isinstance(payload, dict) or not payload:
        raise AnalysisInputError("MBFC input must be a nonempty publisher-to-score mapping")
    result: dict[str, float] = {}
    for publisher, raw_score in payload.items():
        if not isinstance(publisher, str) or not publisher.strip():
            raise AnalysisInputError("MBFC publisher names must be nonempty strings")
        if isinstance(raw_score, bool):
            raise AnalysisInputError(f"MBFC score for {publisher!r} is not numeric")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise AnalysisInputError(f"MBFC score for {publisher!r} is not numeric") from exc
        if not math.isfinite(score):
            raise AnalysisInputError(f"MBFC score for {publisher!r} is not finite")
        result[publisher.strip()] = score
    return result


def deterministic_pca(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = _finite_array(matrix, "observer polarity matrix", ndim=2)
    if matrix.shape[1] != N_OBSERVERS:
        raise AnalysisInputError(f"observer polarity matrix must have {N_OBSERVERS} columns")
    if matrix.shape[0] < 3:
        raise AnalysisInputError("PCA requires at least three articles")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / (matrix.shape[0] - 1)
    total_variance = float(eigenvalues.sum())
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise AnalysisInputError("PCA total variance must be positive")
    if eigenvalues.size < 2 or eigenvalues[1] <= np.finfo(np.float64).eps * total_variance:
        raise AnalysisInputError("PCA requires a nondegenerate second component")
    separation_tolerance = np.finfo(np.float64).eps * max(matrix.shape) * total_variance
    if abs(eigenvalues[0] - eigenvalues[1]) <= separation_tolerance:
        raise AnalysisInputError("PCA first and second components are not uniquely ordered")
    if eigenvalues.size > 2 and abs(eigenvalues[1] - eigenvalues[2]) <= separation_tolerance:
        raise AnalysisInputError("PCA second component is not uniquely oriented")

    loadings = vh[:2].copy()
    for component in range(2):
        pivot = int(np.argmax(np.abs(loadings[component])))
        if loadings[component, pivot] < 0.0:
            loadings[component] *= -1.0
    scores = centered @ loadings.T
    return {
        "center": matrix.mean(axis=0),
        "loadings": loadings,
        "scores": scores,
        "explained_variance": eigenvalues[:2],
        "explained_variance_ratio": eigenvalues[:2] / total_variance,
    }


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = _finite_array(x, "regression predictors", ndim=2)
    y = _finite_array(y, "regression outcome", ndim=1)
    if x.shape[0] != y.shape[0]:
        raise AnalysisInputError("regression predictor/outcome counts differ")
    n, p = x.shape
    if p < 1 or n <= p + 1:
        raise AnalysisInputError("regression has insufficient residual degrees of freedom")
    design = np.column_stack([np.ones(n, dtype=np.float64), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != p + 1:
        raise AnalysisInputError("regression design matrix is rank deficient")
    fitted = design @ coefficients
    residuals = y - fitted
    sse = float(residuals @ residuals)
    centered_y = y - y.mean()
    sst = float(centered_y @ centered_y)
    if not math.isfinite(sst) or sst <= 0.0:
        raise AnalysisInputError("regression outcome must have positive variance")
    r2 = 1.0 - sse / sst
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    return {
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
        "fitted": fitted,
        "residuals": residuals,
        "sse": sse,
        "sst": sst,
        "r2": float(r2),
        "adjusted_r2": float(adjusted_r2),
        "n": n,
        "predictor_count": p,
    }


def overall_f_test(fit: Mapping[str, Any]) -> dict[str, float | int]:
    n = int(fit["n"])
    p = int(fit["predictor_count"])
    df_residual = n - p - 1
    explained = float(fit["sst"]) - float(fit["sse"])
    if explained < 0.0 and abs(explained) <= 1e-12 * float(fit["sst"]):
        explained = 0.0
    if explained < 0.0:
        raise AnalysisInputError("regression explained sum of squares is negative")
    mse = float(fit["sse"]) / df_residual
    statistic = math.inf if mse == 0.0 and explained > 0.0 else (explained / p) / mse
    if mse == 0.0 and explained == 0.0:
        statistic = 0.0
    p_value = 0.0 if math.isinf(statistic) else float(f_distribution.sf(statistic, p, df_residual))
    return {
        "f_statistic": float(statistic),
        "numerator_df": p,
        "denominator_df": df_residual,
        "p_value": p_value,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray, label: str) -> dict[str, float]:
    x = _finite_array(x, f"{label} x", ndim=1)
    y = _finite_array(y, f"{label} y", ndim=1)
    if x.shape != y.shape or x.size < 3:
        raise AnalysisInputError(f"{label} requires equally sized vectors of length at least three")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise AnalysisInputError(f"{label} requires nonconstant vectors")
    result = pearsonr(x, y)
    correlation, p_value = float(result.statistic), float(result.pvalue)
    if not math.isfinite(correlation) or not math.isfinite(p_value):
        raise AnalysisInputError(f"{label} produced nonfinite output")
    return {"r": correlation, "p_value": p_value}


def leave_one_publisher_out(
    x: np.ndarray, y: np.ndarray, publishers: Sequence[str]
) -> dict[str, Any]:
    x = _finite_array(x, "LOO predictors", ndim=2)
    y = _finite_array(y, "LOO outcome", ndim=1)
    if x.shape[0] != y.shape[0] or len(publishers) != y.size:
        raise AnalysisInputError("LOO inputs have inconsistent lengths")
    predictions = np.empty_like(y)
    rows: list[dict[str, Any]] = []
    for held_out in range(y.size):
        keep = np.arange(y.size) != held_out
        fit = _fit_linear(x[keep], y[keep])
        prediction = float(fit["intercept"] + x[held_out] @ fit["coefficients"])
        if not math.isfinite(prediction):
            raise AnalysisInputError("LOO produced a nonfinite prediction")
        predictions[held_out] = prediction
        rows.append(
            {
                "publisher": str(publishers[held_out]),
                "observed": float(y[held_out]),
                "predicted": prediction,
                "residual": float(y[held_out] - prediction),
            }
        )
    centered = y - y.mean()
    sst = float(centered @ centered)
    if sst <= 0.0:
        raise AnalysisInputError("LOO outcome must have positive variance")
    predictive_r2 = 1.0 - float(np.sum((y - predictions) ** 2)) / sst
    return {
        "predictive_r2": float(predictive_r2),
        "pearson": _safe_pearson(predictions, y, "LOO prediction correlation"),
        "predictions": rows,
    }


def pc2_leave_one_out(x: np.ndarray, y: np.ndarray, publishers: Sequence[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for held_out in range(y.size):
        keep = np.arange(y.size) != held_out
        fit = _fit_linear(x[keep, 1:2], y[keep])
        correlation = _safe_pearson(x[keep, 1], y[keep], "PC2 leave-one-out correlation")
        records.append(
            {
                "held_out_publisher": str(publishers[held_out]),
                "slope": float(fit["coefficients"][0]),
                "r2": float(fit["r2"]),
                "adjusted_r2": float(fit["adjusted_r2"]),
                "pearson": correlation,
            }
        )
    return {
        "records": records,
        "ranges": {
            "slope": [min(row["slope"] for row in records), max(row["slope"] for row in records)],
            "r2": [min(row["r2"] for row in records), max(row["r2"] for row in records)],
            "pearson_r": [
                min(row["pearson"]["r"] for row in records),
                max(row["pearson"]["r"] for row in records),
            ],
        },
    }


def _publisher_entropy(publishers: Sequence[str]) -> float:
    counts = np.asarray(list(Counter(publishers).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def shear_quartile_descriptives(
    polarity: np.ndarray, articles: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    shear = np.std(polarity, axis=1, ddof=0)
    low_threshold, high_threshold = np.quantile(shear, [0.25, 0.75])
    ordinals = np.asarray([article["published_date"].toordinal() for article in articles])

    def describe(mask: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise AnalysisInputError("a shear quartile is empty")
        publishers = [str(articles[index]["publisher"]) for index in indices]
        return {
            "n_articles": int(indices.size),
            "date_sd_days_population": float(np.std(ordinals[indices], ddof=0)),
            "publisher_entropy_natural_log": _publisher_entropy(publishers),
        }

    return {
        "shear_definition": "population standard deviation across eight observer polarities",
        "selection_rule": "inclusive empirical 25th/75th percentile thresholds",
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "low": describe(shear <= low_threshold),
        "high": describe(shear >= high_threshold),
    }


def _json_fit(fit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_publishers": int(fit["n"]),
        "intercept": float(fit["intercept"]),
        "coefficients": [float(value) for value in fit["coefficients"]],
        "r2": float(fit["r2"]),
        "adjusted_r2": float(fit["adjusted_r2"]),
    }


def _run_git(repo_root: Path, arguments: Sequence[str], *, text: bool) -> Any:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        command = " ".join(["git", *arguments])
        raise AnalysisInputError(f"code provenance command failed: {command}") from exc


def verify_code_provenance(script_path: Path, repo_root: Path) -> dict[str, Any]:
    """Require the executing source bytes to equal the version stored at HEAD."""

    script_path = script_path.resolve()
    repo_root = repo_root.resolve()
    try:
        relative_script = script_path.relative_to(repo_root)
    except ValueError as exc:
        raise AnalysisInputError("running script is outside the repository root") from exc

    top_level_result = _run_git(repo_root, ["rev-parse", "--show-toplevel"], text=True)
    discovered_root = Path(top_level_result.stdout.strip()).resolve()
    if discovered_root != repo_root:
        raise AnalysisInputError("provided repository root is not the Git top level")

    commit_result = _run_git(repo_root, ["rev-parse", "HEAD"], text=True)
    commit = commit_result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise AnalysisInputError("git rev-parse returned an invalid commit")

    repo_relative_path = relative_script.as_posix()
    head_result = _run_git(repo_root, ["show", f"{commit}:{repo_relative_path}"], text=False)
    head_bytes = bytes(head_result.stdout)
    running_bytes = script_path.read_bytes()
    running_sha256 = _sha256_bytes(running_bytes)
    head_sha256 = _sha256_bytes(head_bytes)
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


def runtime_environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "torch": str(torch.__version__),
    }


def _hash_inputs(input_paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in input_paths.items()}


def _verify_input_hashes(
    initial_hashes: Mapping[str, str], final_hashes: Mapping[str, str]
) -> dict[str, str]:
    if set(initial_hashes) != set(final_hashes):
        raise AnalysisInputError("input hash sets differ")
    changed = sorted(
        name for name in initial_hashes if initial_hashes[name] != final_hashes[name]
    )
    if changed:
        raise AnalysisInputError(
            "input files changed while analysis was running: " + ", ".join(changed)
        )
    return {name: final_hashes[name] for name in sorted(final_hashes)}


def build_analysis(
    nli_artifact: Path,
    corpus_jsonl: Path,
    mbfc_json: Path,
    *,
    repo_root: Path | None = None,
    script_path: Path | None = None,
    provenance_verifier: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    nli_artifact = _require_file(nli_artifact, "NLI artifact")
    corpus_jsonl = _require_file(corpus_jsonl, "corpus JSONL")
    mbfc_json = _require_file(mbfc_json, "MBFC JSON")
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    script_path = (script_path or Path(__file__)).resolve()
    verifier = provenance_verifier or verify_code_provenance
    code_provenance = dict(verifier(script_path, repo_root))
    if "git_commit" not in code_provenance or "script" not in code_provenance:
        raise AnalysisInputError("code provenance verifier returned an incomplete record")

    input_paths = {
        "corpus_jsonl": corpus_jsonl,
        "mbfc_json": mbfc_json,
        "nli_artifact": nli_artifact,
    }
    initial_input_hashes = _hash_inputs(input_paths)

    logits = load_nli_logits(nli_artifact)
    articles = load_corpus(corpus_jsonl)
    mbfc = load_mbfc(mbfc_json)
    if logits.shape[0] != len(articles):
        raise AnalysisInputError(
            f"NLI/corpus row count mismatch: {logits.shape[0]} versus {len(articles)}"
        )

    polarity = observer_polarity(logits)
    if polarity.shape != (len(articles), N_OBSERVERS) or not np.isfinite(polarity).all():
        raise AnalysisInputError("observer polarity matrix is invalid")
    pca = deterministic_pca(polarity)

    publisher_indices: dict[str, list[int]] = defaultdict(list)
    for index, article in enumerate(articles):
        publisher_indices[str(article["publisher"])].append(index)

    centroids: list[dict[str, Any]] = []
    for publisher in sorted(publisher_indices):
        indices = np.asarray(publisher_indices[publisher], dtype=int)
        centroids.append(
            {
                "publisher": publisher,
                "n_articles": int(indices.size),
                "observer_polarity": [float(value) for value in polarity[indices].mean(axis=0)],
                "pc1": float(pca["scores"][indices, 0].mean()),
                "pc2": float(pca["scores"][indices, 1].mean()),
            }
        )

    overlapping = [row for row in centroids if row["publisher"] in mbfc]
    if len(overlapping) < 5:
        raise AnalysisInputError("at least five publishers must overlap the MBFC mapping")
    publishers = [str(row["publisher"]) for row in overlapping]
    predictors = np.asarray([[row["pc1"], row["pc2"]] for row in overlapping], dtype=np.float64)
    outcome = np.asarray([mbfc[publisher] for publisher in publishers], dtype=np.float64)

    pc1_fit = _fit_linear(predictors[:, 0:1], outcome)
    pc2_fit = _fit_linear(predictors[:, 1:2], outcome)
    two_dimensional_fit = _fit_linear(predictors, outcome)
    shear_descriptives = shear_quartile_descriptives(polarity, articles)
    verified_input_hashes = _verify_input_hashes(
        initial_input_hashes, _hash_inputs(input_paths)
    )

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "provenance": {
            **code_provenance,
            "runtime_environment": runtime_environment_versions(),
            "inputs": {
                name: {"path": str(path), "sha256": verified_input_hashes[name]}
                for name, path in input_paths.items()
            },
            "class_order": list(CLASS_ORDER),
            "alignment": {
                "mode": "row_order_only",
                "artifact_row_ids_present": False,
                "limitation": (
                    "NLI rows have no article IDs; equal counts are verified, but semantic row "
                    "alignment cannot be independently authenticated."
                ),
            },
        },
        "formulas_and_limitations": {
            "observer_polarity": (
                "softmax(logits_raw)[entailment] - softmax(logits_raw)[contradiction]"
            ),
            "pca": "SVD of the column-centered article-by-eight observer-polarity matrix",
            "publisher_centroid": "unweighted arithmetic mean across articles from each publisher",
            "adjusted_r2": "1 - (1 - R2) * (n - 1) / (n - p - 1)",
            "overall_two_predictor_f_test": (
                "F=((SST-SSE)/2)/(SSE/(n-3)); upper-tail F distribution with df=(2,n-3)"
            ),
            "predictive_r2": "1 - sum((observed-LOO_prediction)^2)/sum((observed-mean_observed)^2)",
            "analysis_scope": (
                "exploratory and descriptive; no confirmatory decision labels are emitted"
            ),
            "temporal_inference": {
                "status": "not_estimated",
                "reason": (
                    "Articles are nested within publishers; the prior article-level bootstrap "
                    "treated dependent articles as independent and was invalid."
                ),
            },
            "mbfc": "publisher-level external scores are observational targets, not ground truth",
        },
        "results": {
            "n_articles": len(articles),
            "n_publishers": len(centroids),
            "pca": {
                "explained_variance": [float(value) for value in pca["explained_variance"]],
                "explained_variance_ratio": [
                    float(value) for value in pca["explained_variance_ratio"]
                ],
                "loadings": [[float(value) for value in row] for row in pca["loadings"]],
            },
            "publisher_centroids": centroids,
            "mbfc": {
                "publisher_order": publishers,
                "pc1_fit": _json_fit(pc1_fit),
                "pc2_fit": _json_fit(pc2_fit),
                "two_dimensional_fit": {
                    **_json_fit(two_dimensional_fit),
                    "overall_f_test": overall_f_test(two_dimensional_fit),
                },
                "leave_one_publisher_out": leave_one_publisher_out(
                    predictors, outcome, publishers
                ),
                "pc2_leave_one_out_sensitivity": pc2_leave_one_out(
                    predictors, outcome, publishers
                ),
            },
            "shear_quartiles": shear_descriptives,
        },
    }


def write_json_atomic(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_path.parent, delete=False, newline="\n"
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed exploratory observer-score analysis."
    )
    parser.add_argument("--nli-artifact", required=True, type=Path)
    parser.add_argument("--corpus-jsonl", required=True, type=Path)
    parser.add_argument("--mbfc-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_analysis(args.nli_artifact, args.corpus_jsonl, args.mbfc_json)
    write_json_atomic(payload, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
