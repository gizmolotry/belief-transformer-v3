"""Closed-loop Dirichlet observer tuning for high-conflict regions.

The convention in this module is deliberate: a larger crack density produces a
smaller Dirichlet concentration ``alpha``.  For a symmetric Dirichlet this makes
samples sparser (pressure concentrates on fewer observer views) while retaining
the same expected uniform barycenter.

This module is independent of :mod:`core.dirichlet_fusion` so it can be used by
the experiment runner without changing the legacy global-alpha path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:  # Torch is a project dependency, but NumPy-only use remains supported.
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_interval(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robustly map a vector to [0, 1], returning zero for a constant field."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    low, high = np.quantile(values, [0.05, 0.95])
    if high - low <= eps:
        low, high = float(values.min()), float(values.max())
    if high - low <= eps:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def estimate_crack_density(
    embeddings: Any,
    metric_stress: Optional[Any] = None,
    *,
    k_neighbors: int = 8,
    stress_weight: float = 0.5,
) -> np.ndarray:
    """Estimate per-article crack density from local semantic discontinuity.

    Local discontinuity is the mean distance to each point's nearest semantic
    neighbors.  When Track-3 disagreement/stress is supplied, its scalar
    reduction is blended with that discontinuity.  The result is finite and in
    ``[0, 1]``; singleton and constant inputs deterministically return zeros.
    """
    points = _as_numpy(embeddings, name="embeddings")
    if points.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    n_items = points.shape[0]
    if n_items <= 1:
        return np.zeros(n_items, dtype=np.float64)

    k = min(max(int(k_neighbors), 1), n_items - 1)
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(squared, np.inf)
    nearest = np.partition(squared, kth=k - 1, axis=1)[:, :k]
    local_discontinuity = np.mean(np.sqrt(np.maximum(nearest, 0.0)), axis=1)
    local_unit = _unit_interval(local_discontinuity)

    if metric_stress is None:
        return local_unit
    stress = _as_numpy(metric_stress, name="metric_stress")
    if stress.shape[0] != n_items:
        raise ValueError("metric_stress first dimension must match embeddings")
    if stress.ndim > 1:
        stress = np.sqrt(np.mean(np.square(stress.reshape(n_items, -1)), axis=1))
    stress_unit = _unit_interval(stress)
    weight = float(np.clip(stress_weight, 0.0, 1.0))
    return np.clip((1.0 - weight) * local_unit + weight * stress_unit, 0.0, 1.0)


def scale_dirichlet_alpha(
    crack_density: Any,
    base_alpha: float = 1.0,
    *,
    sensitivity: float = 2.0,
    min_alpha: float = 0.05,
    max_alpha: float = 20.0,
) -> np.ndarray:
    """Map crack density to bounded per-article symmetric Dirichlet alpha.

    The monotone rule ``base_alpha / (1 + sensitivity * crack)`` means higher
    crack density gives lower alpha and therefore sparser, more focused mixture
    samples. Inputs are clipped to ``[0, 1]`` rather than re-ranked, so values
    remain comparable across closed-loop iterations.
    """
    cracks = _as_numpy(crack_density, name="crack_density")
    if cracks.ndim != 1:
        raise ValueError("crack_density must have shape [N]")
    if base_alpha <= 0 or min_alpha <= 0 or max_alpha < min_alpha:
        raise ValueError("alpha bounds must satisfy 0 < min_alpha <= max_alpha")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    cracks = np.clip(cracks, 0.0, 1.0)
    alpha = float(base_alpha) / (1.0 + float(sensitivity) * cracks)
    return np.clip(alpha, float(min_alpha), float(max_alpha))


def sample_dynamic_dirichlet_weights(
    article_alphas: Any,
    *,
    n_bots: int,
    n_observers: int,
    seed: int = 12345,
    dtype: Any = None,
    device: Any = None,
    as_tensor: Optional[bool] = None,
) -> Any:
    """Sample deterministic article-local mixtures with shape ``[N, K, B]``.

    NumPy's explicitly-owned generator avoids mutating global NumPy or Torch RNG
    state.  A Torch tensor is returned when the alpha input is a tensor (or when
    ``as_tensor=True``); otherwise the result is a NumPy array.
    """
    input_is_tensor = bool(torch is not None and torch.is_tensor(article_alphas))
    alphas = _as_numpy(article_alphas, name="article_alphas")
    if alphas.ndim != 1 or np.any(alphas <= 0):
        raise ValueError("article_alphas must be a positive vector with shape [N]")
    if int(n_bots) < 2 or int(n_observers) < 1:
        raise ValueError("n_bots must be >= 2 and n_observers must be >= 1")
    rng = np.random.default_rng(int(seed))
    concentration = np.broadcast_to(
        alphas[:, None, None], (alphas.size, int(n_observers), int(n_bots))
    )
    gamma = rng.gamma(shape=concentration, scale=1.0)
    weights = gamma / np.maximum(gamma.sum(axis=-1, keepdims=True), np.finfo(np.float64).tiny)
    wants_tensor = input_is_tensor if as_tensor is None else bool(as_tensor)
    if not wants_tensor:
        return weights
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch output requested but torch is unavailable")
    target_dtype = dtype or (article_alphas.dtype if input_is_tensor else torch.float32)
    target_device = device or (article_alphas.device if input_is_tensor else None)
    return torch.as_tensor(weights, dtype=target_dtype, device=target_device)


@dataclass(frozen=True)
class DynamicTuningConfig:
    base_alpha: float = 1.0
    sensitivity: float = 2.0
    min_alpha: float = 0.05
    max_alpha: float = 20.0
    damping: float = 0.5
    tolerance: float = 1e-3
    max_iterations: int = 5


@dataclass
class DynamicTuningState:
    alphas: np.ndarray
    iteration: int = 0
    converged: bool = False
    max_delta: float = float("inf")
    history: List[Dict[str, float]] = field(default_factory=list)


class DynamicDirichletController:
    """Stateful feedback controller for bounded, damped alpha updates."""

    def __init__(self, n_articles: int, config: Optional[DynamicTuningConfig] = None):
        self.config = config or DynamicTuningConfig()
        if n_articles < 0:
            raise ValueError("n_articles must be non-negative")
        initial = np.full(int(n_articles), self.config.base_alpha, dtype=np.float64)
        initial = np.clip(initial, self.config.min_alpha, self.config.max_alpha)
        self.state = DynamicTuningState(alphas=initial)

    def update(self, crack_density: Any) -> DynamicTuningState:
        """Advance one feedback step and return the mutable controller state."""
        cracks = _as_numpy(crack_density, name="crack_density")
        if cracks.shape != self.state.alphas.shape:
            raise ValueError("crack_density shape must match controller article count")
        if self.state.converged or self.state.iteration >= self.config.max_iterations:
            return self.state
        target = scale_dirichlet_alpha(
            cracks,
            self.config.base_alpha,
            sensitivity=self.config.sensitivity,
            min_alpha=self.config.min_alpha,
            max_alpha=self.config.max_alpha,
        )
        damping = float(np.clip(self.config.damping, 0.0, 1.0))
        updated = (1.0 - damping) * self.state.alphas + damping * target
        delta = float(np.max(np.abs(updated - self.state.alphas))) if updated.size else 0.0
        self.state.alphas = np.clip(updated, self.config.min_alpha, self.config.max_alpha)
        self.state.iteration += 1
        self.state.max_delta = delta
        self.state.converged = delta <= self.config.tolerance
        self.state.history.append(
            {
                "iteration": float(self.state.iteration),
                "max_delta": delta,
                "mean_alpha": float(self.state.alphas.mean()) if updated.size else 0.0,
                "mean_crack_density": float(cracks.mean()) if cracks.size else 0.0,
            }
        )
        return self.state


__all__ = [
    "DynamicDirichletController",
    "DynamicTuningConfig",
    "DynamicTuningState",
    "estimate_crack_density",
    "sample_dynamic_dirichlet_weights",
    "scale_dirichlet_alpha",
]
