"""Fail-closed full-SPD tensor geometry gate.

This module is deliberately standalone from Track 4 and the FMM solver.  It
keeps production rank-one tensors in gradient form, g_i = I + v_i v_i^T, and
only materializes dense tensors for low-dimensional analytic fixtures.
"""

from __future__ import annotations

import heapq
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # Torch is only required for potential training.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal runtimes.
    torch = None  # type: ignore[assignment]


ALLOWED_TARGET_SOURCE_KINDS = frozenset({"observer_derived", "synthetic"})
FULL_SPD_FORMULAS = frozenset(
    {
        "g_i = I + v_i v_i^T",
        "g_i = I + v_i v_i^T; edge full trapezoid",
    }
)
REQUIRED_EMPIRICAL_PROVENANCE_FIELDS = (
    "tensor_family",
    "formula",
    "target_source_kind",
    "ordered_ids_hash",
    "node_gradient_hash",
    "support_hash",
    "support_weight_hash",
    "model_state_hash",
    "loss_trace_hash",
    "train_id_hash",
    "heldout_id_hash",
    "target_hash",
    "q_bank_hash",
    "seed",
    "git_commit",
)
_DENSE_FIXTURE_MAX_DIM = 64
NEAR_TIE_TOLERANCE_RULE = (
    "1e-8 * max(median(abs(eccentricities)), np.finfo(np.float64).tiny)"
)


class FullSPDGeometryError(ValueError):
    """Base fail-closed exception for this module."""


class DisconnectedSupportError(FullSPDGeometryError):
    """Raised when the kNN support is disconnected."""


class RankingCollapseError(FullSPDGeometryError):
    """Raised when an inferential ranking has no usable variation."""


class ProvenanceError(FullSPDGeometryError):
    """Raised when tensor provenance is missing, malformed, or tampered."""


@dataclass(frozen=True)
class SupportGraph:
    """Undirected Euclidean support used by all tensor arms."""

    ordered_ids: tuple[str, ...]
    k: int
    edges: np.ndarray
    euclidean_lengths: np.ndarray
    duplicates: tuple[tuple[str, str], ...]
    support_hash: str

    @property
    def n_nodes(self) -> int:
        return len(self.ordered_ids)


@dataclass(frozen=True)
class PotentialTrainingResult:
    """Train/held-out predictions, held-out gradients, and strict provenance.

    Training predictions retain the original ``train_points`` order; held-out
    predictions and gradients retain the original ``heldout_points`` order.
    """

    predictions: np.ndarray
    train_predictions: np.ndarray
    node_gradients: np.ndarray
    z_predictions: np.ndarray
    train_z_predictions: np.ndarray
    target_mean: float
    target_std: float
    loss_trace: tuple[float, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ArticleRanking:
    """Inferential midranks plus a deterministic, non-inferential serialization."""

    ranks: np.ndarray
    order: tuple[str, ...]
    tied_values: tuple[tuple[float, tuple[str, ...]], ...]


@dataclass(frozen=True)
class PotentialGeneralization:
    """Train/held-out potential-fit diagnostics with train-mean baselines."""

    g_hold: float
    g_train: float
    g_gap: float


def _as_float64_matrix(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise FullSPDGeometryError(f"{name} must be a 2D array")
    if not np.isfinite(arr).all():
        raise FullSPDGeometryError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(arr)


def _as_float64_vector(name: str, value: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise FullSPDGeometryError(f"{name} must be a 1D array")
    if n is not None and arr.shape[0] != n:
        raise FullSPDGeometryError(f"{name} length mismatch")
    if not np.isfinite(arr).all():
        raise FullSPDGeometryError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(arr)


def _canonical_ids(ordered_ids: Sequence[Any], n: int | None = None) -> tuple[str, ...]:
    ids = tuple(str(x) for x in ordered_ids)
    if n is not None and len(ids) != n:
        raise FullSPDGeometryError("ordered_ids length mismatch")
    if len(set(ids)) != len(ids):
        raise FullSPDGeometryError("ordered_ids must be unique")
    return ids


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def canonical_json_sha256(payload: Any) -> str:
    """SHA256 over canonical JSON."""

    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def canonical_array_sha256(array: Any) -> str:
    """SHA256 over shape, dtype, endianness, and raw C-order bytes."""

    arr = np.asarray(array)
    if arr.dtype == object:
        raise FullSPDGeometryError("object arrays are not canonical artifacts")
    contiguous = np.ascontiguousarray(arr)
    byteorder = contiguous.dtype.byteorder
    if byteorder == "=":
        byteorder = "<" if sys.byteorder == "little" else ">"
    header = {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "endianness": byteorder,
    }
    digest = hashlib.sha256()
    digest.update(_json_bytes(header))
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def ordered_ids_hash(ordered_ids: Sequence[Any]) -> str:
    return canonical_json_sha256({"ordered_ids": list(_canonical_ids(ordered_ids))})


def derive_deterministic_seed(
    base_seed: int, control_family: str, q_bank: str, fold: str | int
) -> int:
    """Derive a local 32-bit seed from the experimental cell identity."""

    payload = {
        "base_seed": int(base_seed),
        "control_family": str(control_family),
        "q_bank": str(q_bank),
        "fold": str(fold),
    }
    digest = hashlib.sha256(_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def tensor_metric_low_rank(gradients: Any) -> dict[str, Any]:
    """Validate production rank-one tensor storage without densifying."""

    grad = _as_float64_matrix("gradients", gradients)
    return {
        "tensor_family": "full_spd",
        "formula": "g_i = I + v_i v_i^T",
        "node_gradients": grad,
        "node_gradient_hash": canonical_array_sha256(grad),
    }


def dense_metric_fixture_from_gradients(gradients: Any) -> np.ndarray:
    """Materialize g_i = I + v_i v_i^T for low-dimensional fixture tests only."""

    grad = _as_float64_matrix("gradients", gradients)
    n, dim = grad.shape
    if dim > _DENSE_FIXTURE_MAX_DIM:
        raise FullSPDGeometryError("dense tensors are forbidden for high-D production metrics")
    eye = np.eye(dim, dtype=np.float64)
    dense = eye[None, :, :] + grad[:, :, None] * grad[:, None, :]
    return validate_dense_spd_fixtures(dense)


def validate_dense_spd_fixtures(
    tensors: Any, *, min_eigenvalue: float = 1e-8, max_condition: float = 1e10
) -> np.ndarray:
    """Validate dense low-D SPD fixture tensors."""

    arr = np.asarray(tensors, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise FullSPDGeometryError("dense fixture tensors must have shape [n,d,d]")
    if arr.shape[1] > _DENSE_FIXTURE_MAX_DIM:
        raise FullSPDGeometryError("dense fixture dimension exceeds low-D fixture limit")
    if not np.isfinite(arr).all():
        raise FullSPDGeometryError("dense fixture contains NaN or Inf")
    if not np.allclose(arr, np.swapaxes(arr, 1, 2), atol=1e-10, rtol=1e-10):
        raise FullSPDGeometryError("dense fixture tensors must be symmetric")
    evals = np.linalg.eigvalsh(arr)
    if np.min(evals) <= min_eigenvalue:
        raise FullSPDGeometryError("dense fixture tensor is non-SPD or near-singular")
    cond = np.max(evals, axis=1) / np.min(evals, axis=1)
    if np.max(cond) > max_condition:
        raise FullSPDGeometryError("dense fixture tensor condition number too large")
    return np.ascontiguousarray(arr)


def determinant_lemma_rank_one(gradients: Any) -> np.ndarray:
    """det(I + vv^T) = 1 + ||v||^2."""

    grad = _as_float64_matrix("gradients", gradients)
    return 1.0 + np.einsum("ij,ij->i", grad, grad)


def quadratic_form_rank_one(displacements: Any, gradients: Any) -> np.ndarray:
    """d^T (I + vv^T) d = ||d||^2 + (d.v)^2."""

    d = _as_float64_matrix("displacements", displacements)
    grad = _as_float64_matrix("gradients", gradients)
    if d.shape != grad.shape:
        raise FullSPDGeometryError("displacements and gradients shape mismatch")
    return np.einsum("ij,ij->i", d, d) + np.einsum("ij,ij->i", d, grad) ** 2


def build_euclidean_union_knn_support(
    points: Any, ordered_ids: Sequence[Any], *, k: int
) -> SupportGraph:
    """Build deterministic union-symmetrized Euclidean kNN support."""

    x = _as_float64_matrix("points", points)
    ids = _canonical_ids(ordered_ids, x.shape[0])
    n = x.shape[0]
    if not isinstance(k, int) or k <= 0:
        raise FullSPDGeometryError("k must be a positive integer")
    if k >= n:
        raise FullSPDGeometryError("k must be smaller than the number of points")

    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    duplicates: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] == 0.0:
                duplicates.append((ids[i], ids[j]))

    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        candidates = [
            (float(distances[i, j]), ids[j], j)
            for j in range(n)
            if j != i
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, _, j in candidates[:k]:
            edge_set.add((min(i, j), max(i, j)))

    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    lengths = np.asarray([distances[i, j] for i, j in edges], dtype=np.float64)
    graph = SupportGraph(
        ordered_ids=ids,
        k=k,
        edges=np.ascontiguousarray(edges),
        euclidean_lengths=np.ascontiguousarray(lengths),
        duplicates=tuple(duplicates),
        support_hash=canonical_json_sha256(
            {
                "ordered_ids_hash": ordered_ids_hash(ids),
                "k": k,
                "edges": edges.tolist(),
                "euclidean_lengths_hash": canonical_array_sha256(lengths),
                "duplicates": duplicates,
            }
        ),
    )
    _assert_connected(graph)
    return graph


def build_synthetic_complete_graph_support(
    points: Any, ordered_ids: Sequence[Any], *, max_n: int = 100
) -> SupportGraph:
    """Build complete support for synthetic-only fixtures; rejects N > 100."""

    x = _as_float64_matrix("points", points)
    ids = _canonical_ids(ordered_ids, x.shape[0])
    n = x.shape[0]
    hard_limit = min(int(max_n), 100)
    if n < 2:
        raise FullSPDGeometryError("complete graph requires at least two points")
    if n > hard_limit:
        raise FullSPDGeometryError("synthetic complete graph rejects N > 100")
    edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = np.asarray(edges_list, dtype=np.int64)
    lengths = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1).astype(np.float64)
    duplicates = tuple(
        (ids[i], ids[j])
        for i in range(n)
        for j in range(i + 1, n)
        if lengths[edges_list.index((i, j))] == 0.0
    )
    return SupportGraph(
        ordered_ids=ids,
        k=n - 1,
        edges=np.ascontiguousarray(edges),
        euclidean_lengths=np.ascontiguousarray(lengths),
        duplicates=duplicates,
        support_hash=canonical_json_sha256(
            {
                "support_type": "synthetic_complete",
                "ordered_ids_hash": ordered_ids_hash(ids),
                "edges": edges.tolist(),
                "euclidean_lengths_hash": canonical_array_sha256(lengths),
                "duplicates": duplicates,
            }
        ),
    )


def _assert_connected(graph: SupportGraph) -> None:
    n = graph.n_nodes
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, j in graph.edges:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in neighbors[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != n:
        missing = [graph.ordered_ids[i] for i in range(n) if i not in seen]
        raise DisconnectedSupportError(f"Euclidean kNN support disconnected; unreachable: {missing}")


def edge_costs(
    points: Any,
    gradients: Any,
    graph: SupportGraph,
    *,
    arm: str,
    yhat: Any | None = None,
) -> np.ndarray:
    """Compute square-root edge costs for a metric arm on fixed support."""

    x = _as_float64_matrix("points", points)
    grad = _as_float64_matrix("gradients", gradients)
    if x.shape != grad.shape:
        raise FullSPDGeometryError("points and gradients shape mismatch")
    if x.shape[0] != graph.n_nodes:
        raise FullSPDGeometryError("graph node count mismatch")
    if graph.edges.size == 0:
        raise FullSPDGeometryError("graph has no edges")

    i = graph.edges[:, 0]
    j = graph.edges[:, 1]
    d = x[j] - x[i]
    base2 = np.einsum("ij,ij->i", d, d)
    if arm == "identity":
        cost2 = base2
    elif arm == "full":
        vi_dot = np.einsum("ij,ij->i", d, grad[i])
        vj_dot = np.einsum("ij,ij->i", d, grad[j])
        cost2 = base2 + 0.5 * (vi_dot**2 + vj_dot**2)
    elif arm == "scalar_trace":
        dim = x.shape[1]
        trace_scale = 1.0 + 0.5 * (
            np.einsum("ij,ij->i", grad[i], grad[i]) / dim
            + np.einsum("ij,ij->i", grad[j], grad[j]) / dim
        )
        cost2 = base2 * trace_scale
    elif arm == "diagonal":
        diag_scale = 1.0 + 0.5 * (grad[i] ** 2 + grad[j] ** 2)
        cost2 = np.einsum("ij,ij->i", d**2, diag_scale)
    elif arm == "oof_conformal":
        pred = _as_float64_vector("yhat", yhat, graph.n_nodes)
        rho = 1.0 / (1.0 + np.abs(pred))
        cost2 = base2 / (rho[i] * rho[j])
    else:
        raise FullSPDGeometryError(f"unknown metric arm: {arm}")

    if not np.isfinite(cost2).all() or np.any(cost2 < -1e-12):
        raise FullSPDGeometryError("edge costs are nonfinite or negative")
    return np.sqrt(np.maximum(cost2, 0.0))


def all_pairs_dijkstra(graph: SupportGraph, weights: Any) -> np.ndarray:
    """All-pairs Dijkstra distances over the fixed support graph."""

    w = _as_float64_vector("weights", weights, graph.edges.shape[0])
    if np.any(w < 0):
        raise FullSPDGeometryError("Dijkstra weights must be nonnegative")
    n = graph.n_nodes
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i_raw, j_raw), weight in zip(graph.edges, w):
        i = int(i_raw)
        j = int(j_raw)
        adjacency[i].append((j, float(weight)))
        adjacency[j].append((i, float(weight)))

    out = np.full((n, n), np.inf, dtype=np.float64)
    for source in range(n):
        dist = out[source]
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            current, node = heapq.heappop(heap)
            if current > dist[node]:
                continue
            for nxt, weight in adjacency[node]:
                candidate = current + weight
                if candidate < dist[nxt]:
                    dist[nxt] = candidate
                    heapq.heappush(heap, (candidate, nxt))
        if not np.isfinite(dist).all():
            raise DisconnectedSupportError("Dijkstra found unreachable nodes")
    return out


def article_eccentricity(distances: Any) -> np.ndarray:
    """Mean finite distance from each article to all other fold articles."""

    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise FullSPDGeometryError("distances must be a square matrix")
    n = d.shape[0]
    if n < 2:
        raise FullSPDGeometryError("eccentricity requires at least two articles")
    if not np.allclose(np.diag(d), 0.0, atol=1e-10):
        raise FullSPDGeometryError("distance diagonal must be zero")
    mask = ~np.eye(n, dtype=bool)
    if not np.isfinite(d[mask]).all():
        raise DisconnectedSupportError("eccentricity requires finite off-diagonal distances")
    return d[mask].reshape(n, n - 1).mean(axis=1)


def rank_articles_by_value(values: Any, ordered_ids: Sequence[Any]) -> ArticleRanking:
    """Return true 1-based midranks; IDs only serialize tied articles."""

    vals = _as_float64_vector("values", values)
    ids = _canonical_ids(ordered_ids, vals.shape[0])
    ordered = sorted(range(len(ids)), key=lambda idx: (float(vals[idx]), ids[idx]))
    ranks = np.empty(len(ids), dtype=np.float64)
    tied: list[tuple[float, tuple[str, ...]]] = []
    start = 0
    while start < len(ordered):
        value = float(vals[ordered[start]])
        stop = start + 1
        while stop < len(ordered) and float(vals[ordered[stop]]) == value:
            stop += 1
        # Ranks are one-based: positions start+1 through stop inclusive.
        ranks[ordered[start:stop]] = (start + 1 + stop) / 2.0
        if stop - start > 1:
            tied.append((value, tuple(sorted(ids[idx] for idx in ordered[start:stop]))))
        start = stop
    return ArticleRanking(ranks=ranks, order=tuple(ids[idx] for idx in ordered), tied_values=tuple(tied))


def _validate_midranks(ranks: np.ndarray) -> None:
    """Reject arbitrary scores: Kendall W must receive proper per-row midranks."""

    _, n = ranks.shape
    for row in ranks:
        if np.ptp(row) == 0.0:
            raise RankingCollapseError("Kendall W ranking has zero rank variance")
        positions = 0
        for value in np.unique(row):
            count = int(np.count_nonzero(row == value))
            expected = (positions + 1 + positions + count) / 2.0
            if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
                raise FullSPDGeometryError("rank_matrix rows must contain true midranks")
            positions += count
        if positions != n:
            raise FullSPDGeometryError("rank_matrix row has invalid item count")


def kendalls_w(rank_matrix: Any, item_weights: Any | None = None) -> float:
    """Tie-corrected Kendall W for m midrank rankings over n common items."""

    ranks = _as_float64_matrix("rank_matrix", rank_matrix)
    m, n = ranks.shape
    if m < 2 or n < 2:
        raise FullSPDGeometryError("Kendall W requires at least two rankings and two articles")
    if item_weights is not None:
        weights = _as_float64_vector("item_weights", item_weights, n)
        if not np.allclose(weights, 1.0, atol=0.0, rtol=0.0):
            raise FullSPDGeometryError("tie-corrected Kendall W does not support item weights")
    _validate_midranks(ranks)
    rank_sums = np.sum(ranks, axis=0)
    mean = float(np.mean(rank_sums))
    sum_squares = float(np.sum((rank_sums - mean) ** 2))
    tie_correction = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = (m**2) * (n**3 - n) - m * tie_correction
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RankingCollapseError("Kendall W denominator is undefined after tie correction")
    result = 12.0 * sum_squares / denominator
    if not math.isfinite(result) or result < -1e-12 or result > 1.0 + 1e-12:
        raise FullSPDGeometryError("tie-corrected Kendall W is outside [0, 1]")
    return float(min(1.0, max(0.0, result)))


def observer_target_from_nli_logits(
    logits: Any, class_names: Sequence[Any] | Mapping[Any, Any]
) -> np.ndarray:
    """Return population SD of P(entail)-P(contradict) across eight observers."""

    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (8, 3):
        raise FullSPDGeometryError("logits must have shape [N, 8, 3]")
    if raw.shape[0] == 0 or not np.isfinite(raw).all():
        raise FullSPDGeometryError("logits must be nonempty and finite")
    if isinstance(class_names, Mapping):
        normalized = {str(label).strip().lower(): index for label, index in class_names.items()}
        required = {"entail", "neutral", "contradict"}
        if set(normalized) != required:
            raise FullSPDGeometryError("class_names mapping must unambiguously name entail, neutral, contradict")
        try:
            indices = {label: int(index) for label, index in normalized.items()}
        except (TypeError, ValueError) as exc:
            raise FullSPDGeometryError("class_names mapping indices must be integers") from exc
        if set(indices.values()) != {0, 1, 2}:
            raise FullSPDGeometryError("class_names mapping indices must be exactly 0, 1, 2")
    else:
        labels = tuple(str(label).strip().lower() for label in class_names)
        if len(labels) != 3 or set(labels) != {"entail", "neutral", "contradict"}:
            raise FullSPDGeometryError("class_names must unambiguously name entail, neutral, contradict")
        indices = {label: labels.index(label) for label in labels}
    shifted = raw - np.max(raw, axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    signed = probabilities[:, :, indices["entail"]] - probabilities[:, :, indices["contradict"]]
    target = np.std(signed, axis=1, ddof=0)
    if not np.isfinite(target).all():
        raise FullSPDGeometryError("observer target variance is nonfinite")
    return np.ascontiguousarray(target, dtype=np.float64)


def eccentricity_dispersion_cv(eccentricities: Any) -> float:
    """Population coefficient of variation for positive article eccentricities."""

    values = _as_float64_vector("eccentricities", eccentricities)
    if values.size < 2:
        raise FullSPDGeometryError("eccentricity dispersion requires at least two articles")
    baseline = float(np.mean(values))
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise FullSPDGeometryError("eccentricity dispersion has a nonpositive baseline")
    return float(np.std(values, ddof=0) / baseline)


def near_tie_fraction(values: Any, *, tolerance: float) -> float:
    """Fraction of unordered item pairs whose absolute difference is <= tolerance."""

    vector = _as_float64_vector("values", values)
    if vector.size < 2:
        raise FullSPDGeometryError("near-tie fraction requires at least two values")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise FullSPDGeometryError("near-tie tolerance must be finite and nonnegative")
    differences = np.abs(vector[:, None] - vector[None, :])
    pairs = differences[np.triu_indices(vector.size, k=1)]
    return float(np.mean(pairs <= tolerance))


def near_tie_tolerance(eccentricities: Any) -> float:
    """Return the frozen scale-relative near-tie tolerance for eccentricities."""

    values = _as_float64_vector("eccentricities", eccentricities)
    if values.size == 0:
        raise FullSPDGeometryError("near-tie tolerance requires at least one eccentricity")
    return float(
        1e-8
        * max(
            float(np.median(np.abs(values))),
            np.finfo(np.float64).tiny,
        )
    )


def potential_generalization_diagnostics(
    train_targets: Any,
    train_predictions: Any,
    heldout_targets: Any,
    heldout_predictions: Any,
) -> PotentialGeneralization:
    """Compute G_train, G_hold, and their gap against the train-mean baseline."""

    train_y = _as_float64_vector("train_targets", train_targets)
    train_pred = _as_float64_vector("train_predictions", train_predictions, train_y.size)
    heldout_y = _as_float64_vector("heldout_targets", heldout_targets)
    heldout_pred = _as_float64_vector("heldout_predictions", heldout_predictions, heldout_y.size)
    if train_y.size == 0 or heldout_y.size == 0:
        raise FullSPDGeometryError("potential generalization requires nonempty train and held-out targets")
    train_mean = float(np.mean(train_y))
    train_baseline = float(np.mean((train_y - train_mean) ** 2))
    heldout_baseline = float(np.mean((heldout_y - train_mean) ** 2))
    if train_baseline <= 0.0 or heldout_baseline <= 0.0:
        raise FullSPDGeometryError("potential generalization has a zero baseline")
    g_train = 1.0 - float(np.mean((train_pred - train_y) ** 2)) / train_baseline
    g_hold = 1.0 - float(np.mean((heldout_pred - heldout_y) ** 2)) / heldout_baseline
    g_gap = g_train - g_hold
    if not all(math.isfinite(value) for value in (g_hold, g_train, g_gap)):
        raise FullSPDGeometryError("potential generalization is nonfinite")
    return PotentialGeneralization(g_hold=g_hold, g_train=g_train, g_gap=g_gap)


def edge_amplification_q95(full_edge_lengths: Any, identity_edge_lengths: Any) -> float:
    """95th percentile of full-to-identity edge-length amplification ratios."""

    full = _as_float64_vector("full_edge_lengths", full_edge_lengths)
    identity = _as_float64_vector("identity_edge_lengths", identity_edge_lengths, full.size)
    if full.size == 0:
        raise FullSPDGeometryError("edge amplification requires at least one edge")
    if np.any(full <= 0.0) or np.any(identity <= 0.0):
        raise FullSPDGeometryError("edge amplification requires strictly positive edge baselines")
    result = float(np.quantile(full / identity, 0.95))
    if not math.isfinite(result):
        raise FullSPDGeometryError("edge amplification is nonfinite")
    return result


def weighted_fold_kendalls_w(
    fold_rank_matrices: Sequence[Any], fold_weights: Sequence[float] | None = None
) -> float:
    """Fold-weighted mean Kendall W; folds remain the dependence units."""

    if not fold_rank_matrices:
        raise FullSPDGeometryError("at least one fold rank matrix is required")
    values = np.asarray([kendalls_w(ranks) for ranks in fold_rank_matrices], dtype=np.float64)
    if fold_weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    else:
        weights = _as_float64_vector("fold_weights", fold_weights, len(values))
        if np.any(weights <= 0):
            raise FullSPDGeometryError("fold_weights must be positive")
    return float(np.average(values, weights=weights))


def article_shuffle_gradients_within_group(
    gradients: Any, groups: Sequence[Any], *, seed: int
) -> np.ndarray:
    """Article-shuffled tensor control: permute node gradients within group."""

    grad = _as_float64_matrix("gradients", gradients)
    if len(groups) != grad.shape[0]:
        raise FullSPDGeometryError("groups length mismatch")
    rng = np.random.default_rng(seed)
    out = grad.copy()
    groups_arr = np.asarray([str(g) for g in groups])
    for group in sorted(set(groups_arr)):
        idx = np.flatnonzero(groups_arr == group)
        if idx.size > 1:
            out[idx] = grad[rng.permutation(idx)]
    return np.ascontiguousarray(out)


def spectrum_matched_random_rank_one_directions(gradients: Any, *, seed: int) -> np.ndarray:
    """Random rank-one directions preserving each node's nontrivial spectrum."""

    grad = _as_float64_matrix("gradients", gradients)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=grad.shape)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise FullSPDGeometryError("random generator produced a zero direction")
    target_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    return np.ascontiguousarray(directions / norms * target_norms)


def observer_shuffled_target_helper(
    targets_or_observer_values: Any,
    *,
    seed: int,
    groups: Sequence[Any] | None = None,
    target_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle scalar targets or whole observer rows within publisher.

    This deliberately does not independently shuffle observer columns.  If a
    matrix is supplied, whole observer rows are shuffled within group and the
    scalar target is then derived from the shuffled rows.
    """

    values = np.asarray(targets_or_observer_values, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise FullSPDGeometryError("targets_or_observer_values must be 1D or 2D")
    if not np.isfinite(values).all():
        raise FullSPDGeometryError("targets_or_observer_values contains NaN or Inf")
    values = np.ascontiguousarray(values)
    n = values.shape[0]
    if groups is None:
        group_labels = np.asarray(["__all__"] * n)
    else:
        if len(groups) != n:
            raise FullSPDGeometryError("groups length mismatch")
        group_labels = np.asarray([str(g) for g in groups])
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    for group in sorted(set(group_labels)):
        idx = np.flatnonzero(group_labels == group)
        if idx.size > 1:
            shuffled[idx] = values[rng.permutation(idx)]
    if values.ndim == 1:
        target = shuffled
    elif target_fn is None:
        target = shuffled.std(axis=1, ddof=0)
    else:
        target = np.asarray(target_fn(shuffled), dtype=np.float64)
    target = _as_float64_vector("target", target, n)
    return np.ascontiguousarray(target), np.ascontiguousarray(shuffled)


def model_state_sha256(model: Any) -> str:
    if torch is None:
        raise FullSPDGeometryError("torch is required for model state hashing")
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        tensor = state[key].detach().cpu().numpy()
        digest.update(canonical_array_sha256(tensor).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def make_empirical_provenance(
    *,
    target_source_kind: str,
    ordered_ids: Sequence[Any],
    node_gradients: Any,
    support_graph: SupportGraph,
    support_weights: Any,
    model_state_hash: str,
    loss_trace: Sequence[float],
    train_ids: Sequence[Any],
    heldout_ids: Sequence[Any],
    targets: Any,
    q_bank_hash: str,
    seed: int,
    git_commit: str,
    tensor_family: str = "full_spd",
    formula: str = "g_i = I + v_i v_i^T; edge full trapezoid",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create strict empirical provenance for tensor artifacts."""

    if target_source_kind not in ALLOWED_TARGET_SOURCE_KINDS:
        raise ProvenanceError("target_source_kind must be observer_derived or synthetic")
    if tensor_family != "full_spd":
        raise ProvenanceError("unknown or mislabeled tensor_family")
    if formula not in FULL_SPD_FORMULAS:
        raise ProvenanceError("formula inconsistent with full_spd tensor_family")
    ids = _canonical_ids(ordered_ids)
    grad = _as_float64_matrix("node_gradients", node_gradients)
    weights = _as_float64_vector("support_weights", support_weights, support_graph.edges.shape[0])
    train_id_tuple = _canonical_ids(train_ids)
    heldout_id_tuple = _canonical_ids(heldout_ids, grad.shape[0])
    if set(train_id_tuple).intersection(heldout_id_tuple):
        raise ProvenanceError("train_ids and heldout_ids must be disjoint")
    if ids != heldout_id_tuple:
        raise ProvenanceError("ordered_ids must exactly equal heldout_ids")
    payload: dict[str, Any] = {
        "schema": "full_spd_geometry.v1",
        "tensor_family": tensor_family,
        "formula": formula,
        "target_source_kind": target_source_kind,
        "ordered_ids_hash": ordered_ids_hash(ids),
        "node_gradient_hash": canonical_array_sha256(grad),
        "support_hash": support_graph.support_hash,
        "support_weight_hash": canonical_array_sha256(weights),
        "model_state_hash": str(model_state_hash),
        "loss_trace_hash": canonical_array_sha256(np.asarray(loss_trace, dtype=np.float64)),
        "train_id_hash": ordered_ids_hash(train_id_tuple),
        "heldout_id_hash": ordered_ids_hash(heldout_id_tuple),
        "target_hash": canonical_array_sha256(targets),
        "q_bank_hash": str(q_bank_hash),
        "seed": int(seed),
        "git_commit": str(git_commit),
    }
    if extra:
        payload.update(dict(extra))
    payload["provenance_hash"] = canonical_json_sha256(payload)
    return payload


def validate_empirical_provenance(
    provenance: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate required fields and optional expected hashes/values."""

    if not isinstance(provenance, Mapping):
        raise ProvenanceError("provenance must be a mapping")
    missing = [
        field
        for field in REQUIRED_EMPIRICAL_PROVENANCE_FIELDS
        if field not in provenance or provenance[field] in (None, "")
    ]
    if missing:
        raise ProvenanceError(f"missing required empirical provenance fields: {missing}")
    if provenance["target_source_kind"] not in ALLOWED_TARGET_SOURCE_KINDS:
        raise ProvenanceError("forbidden target_source_kind")
    if provenance["tensor_family"] != "full_spd":
        raise ProvenanceError("unknown or mislabeled tensor_family")
    if provenance["formula"] not in FULL_SPD_FORMULAS:
        raise ProvenanceError("formula inconsistent with full_spd tensor_family")
    copied = dict(provenance)
    recorded_hash = copied.pop("provenance_hash", None)
    if recorded_hash is None:
        raise ProvenanceError("missing provenance_hash")
    actual_hash = canonical_json_sha256(copied)
    if actual_hash != recorded_hash:
        raise ProvenanceError("provenance hash mismatch")
    copied["provenance_hash"] = recorded_hash
    if expected is not None:
        for key, value in expected.items():
            if copied.get(key) != value:
                raise ProvenanceError(f"{key} mismatch")
    return copied


def train_oof_potential(
    train_points: Any,
    train_targets: Any,
    heldout_points: Any,
    *,
    train_ids: Sequence[Any],
    heldout_ids: Sequence[Any],
    q_bank_hash: str,
    seed: int,
    git_commit: str,
    target_source_kind: str,
    device: Any = "cpu",
) -> PotentialTrainingResult:
    """Train IdeologicalPotential and return held-out predictions/gradients.

    The target z-score is fit on training targets only.  Training is full-batch
    AdamW with lr=1e-3, wd=1e-4, exactly 300 epochs, MSE plus a 1e-4 gradient
    penalty.  ``device`` must select CPU or an available CUDA device.  Global
    CPU/CUDA RNG state and deterministic-algorithm settings are restored.
    """

    if torch is None:
        raise FullSPDGeometryError("torch is required for potential training")
    if target_source_kind not in ALLOWED_TARGET_SOURCE_KINDS:
        raise ProvenanceError("target_source_kind must be observer_derived or synthetic")
    try:
        requested_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise FullSPDGeometryError(f"invalid torch device: {device!r}") from exc
    if requested_device.type not in {"cpu", "cuda"}:
        raise FullSPDGeometryError("potential training device must be cpu or cuda")
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise FullSPDGeometryError("CUDA was requested but is unavailable")
        device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise FullSPDGeometryError(f"CUDA device index {device_index} is unavailable")
        actual_device = torch.device("cuda", device_index)
    else:
        actual_device = torch.device("cpu")

    x_train_np = _as_float64_matrix("train_points", train_points)
    x_held_np = _as_float64_matrix("heldout_points", heldout_points)
    y_train_np = _as_float64_vector("train_targets", train_targets, x_train_np.shape[0])
    if x_train_np.shape[1] != x_held_np.shape[1]:
        raise FullSPDGeometryError("train and heldout dimension mismatch")
    _canonical_ids(train_ids, x_train_np.shape[0])
    _canonical_ids(heldout_ids, x_held_np.shape[0])
    if set(str(x) for x in train_ids).intersection(str(x) for x in heldout_ids):
        raise ProvenanceError("train_ids and heldout_ids must be disjoint")

    mean = float(np.mean(y_train_np))
    std = float(np.std(y_train_np, ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        raise FullSPDGeometryError("training targets must have positive finite variance")
    y_z_np = (y_train_np - mean) / std

    from core.riemannian_geometry import IdeologicalPotential, RiemannianConfig

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices, device_type="cuda"):
        torch.manual_seed(int(seed))
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            n_ideologies = 8
            model = IdeologicalPotential(
                hidden_dim=x_train_np.shape[1],
                n_ideologies=n_ideologies,
                config=RiemannianConfig(potential_scale=1.0),
            ).to(actual_device)
            model.train()
            x_train = torch.tensor(x_train_np, dtype=torch.float32, device=actual_device)
            y_z = torch.tensor(y_z_np, dtype=torch.float32, device=actual_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_trace: list[float] = []
            for _ in range(300):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_train, ideology_idx=None)
                mse = torch.mean((prediction - y_z) ** 2)
                grad = model.gradient(x_train, ideology_idx=None)
                penalty = torch.mean(grad**2)
                loss = mse + 1e-4 * penalty
                loss.backward()
                optimizer.step()
                loss_trace.append(float(loss.detach().cpu()))

            model.eval()
            x_held = torch.tensor(x_held_np, dtype=torch.float32, device=actual_device)
            with torch.no_grad():
                train_z_pred_t = model(x_train, ideology_idx=None)
            with torch.enable_grad():
                z_pred_t = model(x_held, ideology_idx=None)
                grad_t = model.gradient(x_held, ideology_idx=None)
            train_z_pred = train_z_pred_t.detach().cpu().numpy().astype(np.float64)
            train_predictions = train_z_pred * std + mean
            z_pred = z_pred_t.detach().cpu().numpy().astype(np.float64)
            predictions = z_pred * std + mean
            gradients = grad_t.detach().cpu().numpy().astype(np.float64)
            state_hash = model_state_sha256(model)
        finally:
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.use_deterministic_algorithms(
                previous_determinism, warn_only=previous_warn_only
            )

    dummy_support = SupportGraph(
        ordered_ids=_canonical_ids(heldout_ids),
        k=1,
        edges=np.zeros((0, 2), dtype=np.int64),
        euclidean_lengths=np.zeros(0, dtype=np.float64),
        duplicates=(),
        support_hash=canonical_json_sha256({"placeholder": "potential_training_only"}),
    )
    provenance = make_empirical_provenance(
        target_source_kind=target_source_kind,
        ordered_ids=heldout_ids,
        node_gradients=gradients,
        support_graph=dummy_support,
        support_weights=np.zeros(0, dtype=np.float64),
        model_state_hash=state_hash,
        loss_trace=loss_trace,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        targets=y_train_np,
        q_bank_hash=q_bank_hash,
        seed=seed,
        git_commit=git_commit,
        extra={
            "training": {
                "model": "IdeologicalPotential",
                "config": {
                    "hidden_dim": x_train_np.shape[1],
                    "n_ideologies": 8,
                    "potential_scale": 1.0,
                    "ideology_idx": None,
                },
                "optimizer": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "loss": "MSE + 1e-4 gradient penalty",
                "device": str(actual_device),
                "prediction_hashes": {
                    "train": canonical_array_sha256(train_predictions),
                    "train_z": canonical_array_sha256(train_z_pred),
                    "heldout": canonical_array_sha256(predictions),
                    "heldout_z": canonical_array_sha256(z_pred),
                },
                "target_mean_train_only": mean,
                "target_std_train_only": std,
            }
        },
    )
    return PotentialTrainingResult(
        predictions=np.ascontiguousarray(predictions),
        train_predictions=np.ascontiguousarray(train_predictions),
        node_gradients=np.ascontiguousarray(gradients),
        z_predictions=np.ascontiguousarray(z_pred),
        train_z_predictions=np.ascontiguousarray(train_z_pred),
        target_mean=mean,
        target_std=std,
        loss_trace=tuple(loss_trace),
        provenance=provenance,
    )
