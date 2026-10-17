"""Deterministic, hash-bound reconstruction of the surviving MBFC Axis A analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "controlled_ideo_embeddings_dense.pt"
DEFAULT_CORPUS = ROOT / "data" / "controlled_N_dense_corpus.jsonl"
DEFAULT_MBFC = ROOT / "data" / "mbfc_continuous_scores.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "mbfc_axis_a_verification.json"
HISTORICAL_SCRIPT = ROOT / "scripts" / "run_mbfc_continuous_validation.py"

EXPECTED_ARTICLES = 320
EXPECTED_OBSERVERS = 8
EXPECTED_HIDDEN_DIM = 1536
EXPECTED_ARTICLES_PER_PUBLISHER = 20
EXPECTED_PUBLISHER_ORDER = (
    "haaretz.com",
    "presstv.ir",
    "breitbart.com",
    "townhall.com",
    "bbc.com",
    "theguardian.com",
    "aljazeera.com",
    "foxnews.com",
    "middleeasteye.net",
    "middleeastmonitor.com",
    "dailymail.co.uk",
    "wnd.com",
    "nypost.com",
    "express.co.uk",
    "cnn.com",
    "bbc.co.uk",
)

SCHEMA_NAME = "belief_transformer.mbfc_axis_a_reconstruction"
SCHEMA_VERSION = "1.0.0"


class VerificationInputError(ValueError):
    """Raised when an input cannot support the exact historical reconstruction."""


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise VerificationInputError(f"{label} is not a file: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise VerificationInputError(f"{label} is not a tensor") from exc
    tensor = value.detach().cpu()
    if not tensor.is_floating_point():
        raise VerificationInputError(f"{label} must have a floating-point dtype")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
        raise VerificationInputError(f"{label} must be nonempty and finite")
    return tensor


def load_cls_per_bot(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise VerificationInputError(f"could not load input artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise VerificationInputError("input artifact must be a mapping")
    pairs = payload.get("nli_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise VerificationInputError("input artifact must contain an nli_pairs sequence")
    if len(pairs) != EXPECTED_ARTICLES:
        raise VerificationInputError(
            f"expected {EXPECTED_ARTICLES} nli_pairs rows, found {len(pairs)}"
        )

    rows: list[torch.Tensor] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or "cls_per_bot" not in pair:
            raise VerificationInputError(f"nli_pairs[{index}] lacks cls_per_bot")
        row = _finite_tensor(pair["cls_per_bot"], f"nli_pairs[{index}].cls_per_bot")
        expected_shape = (EXPECTED_OBSERVERS, EXPECTED_HIDDEN_DIM)
        if tuple(row.shape) != expected_shape:
            raise VerificationInputError(
                f"nli_pairs[{index}].cls_per_bot must have shape {expected_shape}, "
                f"got {tuple(row.shape)}"
            )
        rows.append(row)

    declared_count = payload.get("n_articles")
    if declared_count is not None and (
        isinstance(declared_count, bool) or int(declared_count) != EXPECTED_ARTICLES
    ):
        raise VerificationInputError("artifact n_articles disagrees with the reconstruction contract")
    embeddings = torch.stack(rows, dim=0)
    expected_shape = (EXPECTED_ARTICLES, EXPECTED_OBSERVERS, EXPECTED_HIDDEN_DIM)
    if tuple(embeddings.shape) != expected_shape:
        raise VerificationInputError(
            f"stacked cls_per_bot must have shape {expected_shape}, got {tuple(embeddings.shape)}"
        )
    if float(embeddings.var()) <= 0.0:
        raise VerificationInputError("cls_per_bot has zero variance")
    return embeddings


def load_corpus_publishers(path: Path) -> list[str]:
    publishers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise VerificationInputError(f"corpus line {line_number} is blank")
            try:
                article = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationInputError(f"corpus line {line_number} is invalid JSON") from exc
            if not isinstance(article, Mapping):
                raise VerificationInputError(f"corpus line {line_number} is not an object")
            publisher = article.get("publication") or article.get("source")
            if not isinstance(publisher, str) or not publisher.strip():
                raise VerificationInputError(f"corpus line {line_number} lacks publication/source")
            publishers.append(publisher.strip())

    if len(publishers) != EXPECTED_ARTICLES:
        raise VerificationInputError(
            f"expected {EXPECTED_ARTICLES} corpus rows, found {len(publishers)}"
        )
    first_occurrence_order = tuple(dict.fromkeys(publishers))
    if first_occurrence_order != EXPECTED_PUBLISHER_ORDER:
        raise VerificationInputError(
            "publisher first-occurrence order differs from the historical reconstruction contract"
        )
    counts = Counter(publishers)
    expected_counts = {
        publisher: EXPECTED_ARTICLES_PER_PUBLISHER for publisher in EXPECTED_PUBLISHER_ORDER
    }
    if counts != expected_counts:
        raise VerificationInputError(
            "publisher membership/counts differ from the historical reconstruction contract"
        )
    return publishers


def load_mbfc_scores(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationInputError("MBFC mapping is invalid JSON") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise VerificationInputError("MBFC mapping must be a nonempty object")
    scores: dict[str, float] = {}
    for publisher, raw_score in payload.items():
        if not isinstance(publisher, str) or not publisher.strip() or isinstance(raw_score, bool):
            raise VerificationInputError("MBFC mapping contains an invalid publisher or score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise VerificationInputError(f"MBFC score for {publisher!r} is not numeric") from exc
        if not math.isfinite(score):
            raise VerificationInputError(f"MBFC score for {publisher!r} is not finite")
        scores[publisher.strip()] = score
    missing = [publisher for publisher in EXPECTED_PUBLISHER_ORDER if publisher not in scores]
    if missing:
        raise VerificationInputError("MBFC mapping lacks publishers: " + ", ".join(missing))
    return scores


def project_out(vectors: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Historical projection helper, preserved without changing its arithmetic."""

    direction = direction / torch.norm(direction)
    if vectors.dim() == 3:
        projection = (vectors * direction.view(1, 1, -1)).sum(dim=2, keepdim=True)
        return vectors - projection * direction.view(1, 1, -1)
    projection = (vectors * direction).sum(dim=1, keepdim=True)
    return vectors - projection * direction


def _require_unique_first_component(singular_values: torch.Tensor, matrix: torch.Tensor, label: str) -> None:
    if singular_values.numel() == 0 or float(singular_values[0]) <= 0.0:
        raise VerificationInputError(f"{label} has no positive first singular value")
    if singular_values.numel() > 1:
        scale = max(1.0, float(singular_values[0]))
        tolerance = torch.finfo(singular_values.dtype).eps * max(matrix.shape) * scale
        if abs(float(singular_values[0] - singular_values[1])) <= tolerance:
            raise VerificationInputError(f"{label} first component is not uniquely ordered")


def apply_cnn_fox_sign(
    article_scores: np.ndarray, publishers: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(article_scores)
    if scores.ndim != 1 or scores.size != len(publishers) or not np.isfinite(scores).all():
        raise VerificationInputError("article scores and publisher rows must be finite and aligned")
    cnn_indices = np.asarray([i for i, publisher in enumerate(publishers) if publisher == "cnn.com"])
    fox_indices = np.asarray([i for i, publisher in enumerate(publishers) if publisher == "foxnews.com"])
    if cnn_indices.size == 0 or fox_indices.size == 0:
        raise VerificationInputError("CNN/Fox sign convention requires both publishers")
    cnn_before = float(scores[cnn_indices].mean())
    fox_before = float(scores[fox_indices].mean())
    flipped = cnn_before > fox_before
    signed = -scores if flipped else scores.copy()
    cnn_after = float(signed[cnn_indices].mean())
    fox_after = float(signed[fox_indices].mean())
    if cnn_after > fox_after:
        raise VerificationInputError("CNN/Fox sign convention failed")
    return signed, {
        "rule": "multiply all article scores by -1 iff mean(CNN) > mean(Fox)",
        "flipped": bool(flipped),
        "cnn_mean_before": cnn_before,
        "fox_mean_before": fox_before,
        "cnn_mean_after": cnn_after,
        "fox_mean_after": fox_after,
    }


def compute_fixed_axis(
    embeddings: torch.Tensor, publishers: Sequence[str]
) -> dict[str, Any]:
    expected_shape = (EXPECTED_ARTICLES, EXPECTED_OBSERVERS, EXPECTED_HIDDEN_DIM)
    if tuple(embeddings.shape) != expected_shape:
        raise VerificationInputError(
            f"embeddings must have shape {expected_shape}, got {tuple(embeddings.shape)}"
        )
    if len(publishers) != EXPECTED_ARTICLES:
        raise VerificationInputError("publisher rows do not align with embeddings")
    if not torch.isfinite(embeddings).all().item():
        raise VerificationInputError("embeddings contain nonfinite values")

    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    global_center = flat_embeddings.mean(dim=0)
    flat_centered = flat_embeddings - global_center
    _, global_singular_values, global_v = torch.svd(flat_centered)
    _require_unique_first_component(global_singular_values, flat_centered, "global semantic PCA")
    semantic_pc1 = global_v[:, 0]

    residual_embeddings = project_out(embeddings, semantic_pc1)
    article_embeddings = residual_embeddings.mean(dim=1)
    article_center = article_embeddings.mean(dim=0)
    article_centered = article_embeddings - article_center
    _, article_singular_values, article_v = torch.svd(article_centered)
    _require_unique_first_component(article_singular_values, article_centered, "article PCA")
    article_pc1 = article_v[:, 0]

    unsigned_scores = (article_embeddings * article_pc1).sum(dim=1).numpy()
    article_scores, sign = apply_cnn_fox_sign(unsigned_scores, publishers)
    return {
        "semantic_pc1": semantic_pc1,
        "article_pc1": article_pc1,
        "article_scores": article_scores,
        "sign_convention": sign,
        "global_singular_values": global_singular_values,
        "article_singular_values": article_singular_values,
    }


def _correlations(x: np.ndarray, y: np.ndarray, label: str) -> dict[str, dict[str, float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise VerificationInputError(f"{label} requires aligned one-dimensional vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise VerificationInputError(f"{label} contains nonfinite values")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise VerificationInputError(f"{label} requires nonconstant values")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    values = (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )
    if not all(math.isfinite(value) for value in values):
        raise VerificationInputError(f"{label} produced a nonfinite correlation")
    return {
        "pearson": {"r": values[0], "p_value": values[1]},
        "spearman": {"rho": values[2], "p_value": values[3]},
    }


def aggregate_publishers(
    article_scores: np.ndarray,
    publishers: Sequence[str],
    mbfc_scores: Mapping[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    scores = np.asarray(article_scores)
    if scores.ndim != 1 or scores.size != len(publishers) or not np.isfinite(scores).all():
        raise VerificationInputError("article scores cannot be aggregated safely")
    order = list(dict.fromkeys(publishers))
    if tuple(order) != EXPECTED_PUBLISHER_ORDER:
        raise VerificationInputError("publisher aggregation order differs from the historical order")
    geometric = np.asarray(
        [scores[np.asarray(publishers) == publisher].mean() for publisher in order],
        dtype=np.float64,
    )
    external = np.asarray([mbfc_scores[publisher] for publisher in order], dtype=np.float64)
    rows = [
        {
            "publisher": publisher,
            "n_articles": int(sum(value == publisher for value in publishers)),
            "geometric_score": float(geometric[index]),
            "mbfc_score": float(external[index]),
        }
        for index, publisher in enumerate(order)
    ]
    return order, geometric, external, rows


def fixed_axis_leave_one_publisher_out(
    publisher_order: Sequence[str], geometric_scores: np.ndarray, mbfc_scores: np.ndarray
) -> list[dict[str, Any]]:
    geometric_scores = np.asarray(geometric_scores, dtype=np.float64)
    mbfc_scores = np.asarray(mbfc_scores, dtype=np.float64)
    if geometric_scores.shape != mbfc_scores.shape or geometric_scores.size != len(publisher_order):
        raise VerificationInputError("LOO publisher vectors are not aligned")
    records: list[dict[str, Any]] = []
    for held_out in range(len(publisher_order)):
        keep = np.arange(len(publisher_order)) != held_out
        records.append(
            {
                "held_out_publisher": str(publisher_order[held_out]),
                "n_publishers": int(keep.sum()),
                **_correlations(
                    geometric_scores[keep],
                    mbfc_scores[keep],
                    f"LOO excluding {publisher_order[held_out]}",
                ),
            }
        )
    return records


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def build_verification(input_path: Path, corpus_path: Path, mbfc_path: Path) -> dict[str, Any]:
    source_runner = _require_file(Path(__file__), "source runner")
    historical_script = _require_file(HISTORICAL_SCRIPT, "historical Axis A script")
    input_path = _require_file(input_path, "input artifact")
    corpus_path = _require_file(corpus_path, "corpus")
    mbfc_path = _require_file(mbfc_path, "MBFC mapping")
    paths = {
        "input_artifact": input_path,
        "corpus": corpus_path,
        "source_runner": source_runner,
        "historical_script": historical_script,
        "mbfc_mapping": mbfc_path,
    }
    hashes_before = _input_hashes(paths)

    embeddings = load_cls_per_bot(input_path)
    publishers = load_corpus_publishers(corpus_path)
    mbfc = load_mbfc_scores(mbfc_path)
    axis = compute_fixed_axis(embeddings, publishers)
    publisher_order, geometric, external, publisher_rows = aggregate_publishers(
        axis["article_scores"], publishers, mbfc
    )
    full_sample = _correlations(geometric, external, "full-sample publisher correlation")
    loo = fixed_axis_leave_one_publisher_out(publisher_order, geometric, external)
    hashes_after = _input_hashes(paths)
    changed = sorted(name for name in hashes_before if hashes_before[name] != hashes_after[name])
    if changed:
        raise VerificationInputError("inputs changed during verification: " + ", ".join(changed))

    press_row = next(row for row in publisher_rows if row["publisher"] == "presstv.ir")
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "provenance": {
            "files": {
                name: {"path": str(path), "sha256": hashes_after[name]}
                for name, path in paths.items()
            },
            "derived_array_hashes": {
                "semantic_pc1": sha256_array(axis["semantic_pc1"]),
                "article_pc1": sha256_array(axis["article_pc1"]),
                "article_scores_after_sign_convention": sha256_array(axis["article_scores"]),
                "publisher_scores_in_historical_order": sha256_array(geometric),
            },
            "alignment": {
                "mode": "row_order_only",
                "limitation": (
                    "The embedding artifact has no authenticated article identifiers. Equal row counts, "
                    "shapes, publisher order, and publisher counts are checked, but semantic article "
                    "alignment remains dependent on historical row order."
                ),
            },
            "rng": {"used": False, "seed": None},
        },
        "definition": {
            "name": "MBFC Axis A (surviving corpus-proxy reconstruction)",
            "representation": "cls_per_bot mean-pooled hidden-state A-minus-B differences",
            "steps": [
                "reshape 320x8x1536 to 2560x1536",
                "column-center and remove global semantic PC1 using torch.svd",
                "average the eight residual observer vectors per article",
                "column-center article vectors and fit article PC1 using torch.svd",
                "project uncentered residual article means onto article PC1",
                "multiply all article scores by -1 iff mean(CNN) > mean(Fox)",
                "take unweighted means of 20 article scores per publisher in first-occurrence order",
            ],
            "sign_convention": axis["sign_convention"],
            "loo_convention": {
                "name": "fixed_full_sample_axis_remove_one_publisher",
                "semantic_pc1_refit_per_exclusion": False,
                "article_pc1_refit_per_exclusion": False,
                "sign_refit_per_exclusion": False,
                "description": (
                    "Each sensitivity row removes one already-aggregated publisher from the correlation. "
                    "The full-sample semantic PC1, article PC1, sign, and article scores remain fixed."
                ),
            },
            "formulas": {
                "projection_out": "x_resid = x - <x,v/||v||> (v/||v||)",
                "publisher_score": "mean of the fixed signed article scores for that publisher",
                "pearson": "scipy.stats.pearsonr(publisher_score, mbfc_score)",
                "spearman": "scipy.stats.spearmanr(publisher_score, mbfc_score)",
            },
            "caveats": [
                "The definition and historical result entered the repository in the same commit; the axis was not frozen before the result.",
                "The article-to-embedding relationship is authenticated only by row order.",
                "The MBFC values are publisher-level observational targets, not article-level ground truth.",
                "LOO rows are sensitivity correlations, not out-of-sample predictive estimates.",
            ],
        },
        "results": {
            "n_articles": EXPECTED_ARTICLES,
            "n_observers": EXPECTED_OBSERVERS,
            "hidden_dim": EXPECTED_HIDDEN_DIM,
            "n_publishers": len(publisher_order),
            "publisher_order": publisher_order,
            "publisher_rows": publisher_rows,
            "full_sample": full_sample,
            "press_tv": press_row,
            "leave_one_publisher_out": loo,
        },
    }


def write_json_exclusive(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def _resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--mbfc", type=Path, default=DEFAULT_MBFC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = _resolve_cli_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    payload = build_verification(
        _resolve_cli_path(args.input),
        _resolve_cli_path(args.corpus),
        _resolve_cli_path(args.mbfc),
    )
    write_json_exclusive(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
