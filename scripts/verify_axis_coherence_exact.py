from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


TENSOR_KEYS = ("ideo_emb", "ctrl_emb", "rand_emb")
DEFAULT_INPUT = Path("data/controlled_ideo_embeddings_xlarge.pt")


class AxisCoherenceVerificationError(ValueError):
    """Raised when the deterministic verification contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding_cache(
    path: Path,
    *,
    expected_n: int = 60,
    expected_axes: int = 8,
    expected_dim: int = 1536,
) -> dict[str, torch.Tensor]:
    """Load only finite floating tensors satisfying the declared cache contract."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise AxisCoherenceVerificationError(f"could not safely load input cache: {path}") from exc
    if not isinstance(payload, Mapping):
        raise AxisCoherenceVerificationError("input cache must contain a mapping")

    expected_shape = (expected_n, expected_axes, expected_dim)
    tensors: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise AxisCoherenceVerificationError(f"input cache lacks tensor {key!r}")
        if tuple(value.shape) != expected_shape:
            raise AxisCoherenceVerificationError(
                f"{key} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if not value.dtype.is_floating_point:
            raise AxisCoherenceVerificationError(f"{key} must have a floating dtype")
        value = value.detach().contiguous().cpu()
        if not bool(torch.isfinite(value).all()):
            raise AxisCoherenceVerificationError(f"{key} contains non-finite values")
        tensors[key] = value

    dtypes = {tensor.dtype for tensor in tensors.values()}
    if len(dtypes) != 1:
        raise AxisCoherenceVerificationError("all three tensors must use the same dtype")
    return tensors


def remove_global_pc1(
    groups: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], float]:
    """Reproduce the original global centering, SVD, and PC1 subtraction."""

    if len(groups) != 3 or any(group.ndim != 3 for group in groups):
        raise AxisCoherenceVerificationError("expected exactly three rank-3 tensor groups")
    shape = tuple(groups[0].shape)
    if any(tuple(group.shape) != shape for group in groups[1:]):
        raise AxisCoherenceVerificationError("all tensor groups must have identical shapes")

    flattened = torch.cat([group.reshape(-1, shape[-1]) for group in groups], dim=0)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.svd(centered)
    total_variance = (singular_values**2).sum()
    if not bool(torch.isfinite(total_variance)) or float(total_variance) <= 0.0:
        raise AxisCoherenceVerificationError("pooled tensors have no finite positive variance")

    pc1 = right_vectors[:, 0]
    residuals: list[torch.Tensor] = []
    for group in groups:
        flat = group.reshape(-1, shape[-1])
        projection = torch.matmul(flat, pc1).unsqueeze(1) * pc1.unsqueeze(0)
        residuals.append((flat - projection).reshape(group.shape))
    pc1_evr = float((singular_values[0] ** 2) / total_variance)
    return residuals, pc1_evr


def axis_coherence_matrix(tensor: torch.Tensor) -> np.ndarray:
    """Return pairwise axis cosines averaged over articles using the original reduction."""

    if tensor.ndim != 3 or tensor.shape[1] < 2:
        raise AxisCoherenceVerificationError("coherence input must have shape [N, axes>=2, D]")
    normalized = torch.nn.functional.normalize(tensor, p=2, dim=-1)
    axis_count = tensor.shape[1]
    matrix = np.zeros((axis_count, axis_count), dtype=np.float64)
    for left in range(axis_count):
        for right in range(left + 1, axis_count):
            value = torch.sum(
                normalized[:, left, :] * normalized[:, right, :], dim=-1
            ).mean().item()
            matrix[left, right] = matrix[right, left] = value
    return matrix


def subset_coherence(matrix: np.ndarray, indices: Sequence[int]) -> float:
    index = np.asarray(tuple(indices), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AxisCoherenceVerificationError("coherence matrix must be square")
    if index.ndim != 1 or index.size < 2 or np.unique(index).size != index.size:
        raise AxisCoherenceVerificationError("subset must contain at least two unique indices")
    if index.min() < 0 or index.max() >= matrix.shape[0]:
        raise AxisCoherenceVerificationError("subset index is out of range")
    left, right = np.triu_indices(index.size, k=1)
    return float(np.mean(matrix[index[left], index[right]], dtype=np.float64))


def exhaustive_upper_tail(
    matrix: np.ndarray,
    *,
    subset_size: int,
    observed: float,
    batch_size: int = 10_000,
) -> dict[str, int | float]:
    """Enumerate every equal-size axis subset and count statistics >= observed."""

    axis_count = matrix.shape[0] if matrix.ndim == 2 else 0
    if matrix.shape != (axis_count, axis_count) or not np.isfinite(matrix).all():
        raise AxisCoherenceVerificationError("coherence matrix must be finite and square")
    if not 2 <= subset_size <= axis_count:
        raise AxisCoherenceVerificationError("subset size must be between 2 and axis count")
    if not math.isfinite(observed):
        raise AxisCoherenceVerificationError("observed statistic must be finite")
    if batch_size < 1:
        raise AxisCoherenceVerificationError("batch size must be positive")

    pair_left, pair_right = np.triu_indices(subset_size, k=1)
    combinations = iter(itertools.combinations(range(axis_count), subset_size))
    total = math.comb(axis_count, subset_size)
    tail_count = 0
    null_sum = 0.0
    null_min = math.inf
    null_max = -math.inf
    generated = 0

    while True:
        chunk = list(itertools.islice(combinations, batch_size))
        if not chunk:
            break
        indices = np.asarray(chunk, dtype=np.int64)
        statistics = matrix[indices[:, pair_left], indices[:, pair_right]].mean(
            axis=1, dtype=np.float64
        )
        tail_count += int(np.count_nonzero(statistics >= observed))
        null_sum += float(statistics.sum(dtype=np.float64))
        null_min = min(null_min, float(statistics.min()))
        null_max = max(null_max, float(statistics.max()))
        generated += statistics.size

    if generated != total:
        raise AxisCoherenceVerificationError(
            f"enumeration produced {generated} subsets, expected {total}"
        )
    return {
        "total_subsets": total,
        "upper_tail_count_ge_observed": tail_count,
        "exact_upper_tail_probability": tail_count / total,
        "null_mean": null_sum / total,
        "null_min": null_min,
        "null_max": null_max,
    }


def build_verification(
    input_path: Path,
    *,
    expected_n: int = 60,
    expected_axes: int = 8,
    expected_dim: int = 1536,
) -> dict[str, Any]:
    resolved_input = input_path.resolve(strict=True)
    tensors = load_embedding_cache(
        resolved_input,
        expected_n=expected_n,
        expected_axes=expected_axes,
        expected_dim=expected_dim,
    )
    residuals, pc1_evr = remove_global_pc1([tensors[key] for key in TENSOR_KEYS])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))
    ideology = subset_coherence(pooled_matrix, range(expected_axes))
    structure = subset_coherence(
        pooled_matrix, range(expected_axes, 2 * expected_axes)
    )
    random = subset_coherence(
        pooled_matrix, range(2 * expected_axes, 3 * expected_axes)
    )
    exact_null = exhaustive_upper_tail(
        pooled_matrix, subset_size=expected_axes, observed=ideology
    )

    runner = Path(__file__).resolve(strict=True)
    return {
        "schema_version": 1,
        "status": "new_verification_conditional_on_surviving_unmanifested_cache",
        "claim_boundary": {
            "historical_run_reproduced": False,
            "statement": (
                "This is a new deterministic verification conditional on the surviving "
                "unmanifested cache; it is not provenance for, or a reproduction of, the "
                "historical unseeded Monte Carlo run."
            ),
        },
        "provenance": {
            "input_path": str(resolved_input),
            "input_sha256": sha256_file(resolved_input),
            "runner_path": str(runner),
            "runner_sha256": sha256_file(runner),
            "tensor_keys": list(TENSOR_KEYS),
            "expected_tensor_shape": [expected_n, expected_axes, expected_dim],
            "tensor_dtype": str(tensors[TENSOR_KEYS[0]].dtype),
        },
        "method": {
            "pc1": "global center pooled [N, 3*axes, D] vectors; torch.svd; subtract PC1 projection",
            "coherence": "mean over all within-group axis pairs of article-mean cosine similarity",
            "null": "complete enumeration of every equally likely axes-sized subset from 3*axes",
            "tail": "upper tail, statistic >= observed ideology coherence",
        },
        "results": {
            "pc1_explained_variance_ratio": pc1_evr,
            "observed_coherence": {
                "ideology": ideology,
                "structure": structure,
                "random": random,
            },
            "exact_null": exact_null,
        },
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically reconstruct the N=60 axis-coherence random-grouping test."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-n", type=positive_int, default=60)
    parser.add_argument("--expected-axes", type=positive_int, default=8)
    parser.add_argument("--expected-dim", type=positive_int, default=1536)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_verification(
        args.input,
        expected_n=args.expected_n,
        expected_axes=args.expected_axes,
        expected_dim=args.expected_dim,
    )
    write_json_exclusive(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
