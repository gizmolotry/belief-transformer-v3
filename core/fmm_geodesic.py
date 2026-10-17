"""Continuous-grid geodesic distances via the isotropic Eikonal equation."""
from __future__ import annotations

import warnings
import numpy as np


def _effective_slowness(metric: np.ndarray) -> np.ndarray:
    if metric.ndim < 2 or metric.shape[-1] != metric.shape[-2]:
        raise ValueError("metric must end in square matrix dimensions")
    if not np.allclose(metric, np.swapaxes(metric, -1, -2), atol=1e-7):
        raise ValueError("metric tensors must be symmetric")
    eigenvalues = np.linalg.eigvalsh(metric)
    if np.any(eigenvalues <= 0):
        raise ValueError("metric tensors must be positive definite")
    # skfmm is isotropic.  RMS directional cost is a documented scalar
    # approximation to a fully anisotropic metric.
    return np.sqrt(np.trace(metric, axis1=-2, axis2=-1) / metric.shape[-1])


def _fast_sweeping(slowness: np.ndarray, source: tuple[int, ...], spacing: tuple[float, ...],
                   tolerance: float = 1e-8, max_iterations: int = 1000) -> np.ndarray:
    """Monotone Gauss-Seidel solver for |grad(T)|=slowness (1D/2D)."""
    ndim = slowness.ndim
    if ndim not in (1, 2):
        raise ValueError("fallback fast sweeping currently supports one- or two-dimensional charts")
    t = np.full(slowness.shape, np.inf, dtype=float)
    t[source] = 0.0
    orders = [(range(slowness.shape[0]),)] if ndim == 1 else [
        (range(slowness.shape[0]), range(slowness.shape[1])),
        (range(slowness.shape[0] - 1, -1, -1), range(slowness.shape[1])),
        (range(slowness.shape[0]), range(slowness.shape[1] - 1, -1, -1)),
        (range(slowness.shape[0] - 1, -1, -1), range(slowness.shape[1] - 1, -1, -1)),
    ]
    for _ in range(max_iterations):
        old = t.copy()
        for order in orders:
            iterator = ((i,) for i in order[0]) if ndim == 1 else ((i, j) for i in order[0] for j in order[1])
            for idx in iterator:
                if idx == source:
                    continue
                if ndim == 1:
                    i = idx[0]
                    a = min(t[i - 1] if i else np.inf, t[i + 1] if i + 1 < t.shape[0] else np.inf)
                    candidate = a + spacing[0] * slowness[idx]
                else:
                    i, j = idx
                    a = min(t[i - 1, j] if i else np.inf, t[i + 1, j] if i + 1 < t.shape[0] else np.inf)
                    b = min(t[i, j - 1] if j else np.inf, t[i, j + 1] if j + 1 < t.shape[1] else np.inf)
                    hx, hy, s = spacing[0], spacing[1], slowness[idx]
                    candidate = min(a + hx*s, b + hy*s)
                    if np.isfinite(a) and np.isfinite(b):
                        A = 1 / hx**2 + 1 / hy**2
                        B = -2 * (a / hx**2 + b / hy**2)
                        C = a*a / hx**2 + b*b / hy**2 - s*s
                        disc = B*B - 4*A*C
                    else:
                        disc = -1.0
                    if disc >= 0:
                        root = (-B + np.sqrt(disc)) / (2*A)
                        if root >= max(a, b):
                            candidate = min(candidate, root)
                t[idx] = min(t[idx], candidate)
        finite = np.isfinite(old)
        delta = np.max(np.abs(t[finite] - old[finite])) if finite.any() else np.inf
        if np.isfinite(t).all() and delta < tolerance:
            break
    return t


def fmm_geodesic_distance(metric, source, *, spacing=1.0, use_skfmm=True,
                           tolerance=1e-8, max_iterations=1000):
    """Solve chart-grid travel time from ``source``.

    The tensor field must have shape ``grid_shape + (chart_dim, chart_dim)``.
    General unstructured point clouds require an explicit chart/grid resampling
    step and are intentionally not silently treated as a continuous domain.
    """
    metric = np.asarray(metric, dtype=float)
    slowness = _effective_slowness(metric)
    source = (int(source),) if np.isscalar(source) else tuple(map(int, source))
    if len(source) != slowness.ndim or any(i < 0 or i >= n for i, n in zip(source, slowness.shape)):
        raise ValueError("source must index the metric grid")
    spacing = (float(spacing),) * slowness.ndim if np.isscalar(spacing) else tuple(map(float, spacing))
    if len(spacing) != slowness.ndim or any(x <= 0 for x in spacing):
        raise ValueError("spacing must contain one positive value per grid dimension")
    if use_skfmm:
        try:
            import skfmm
            phi = np.ones(slowness.shape)
            phi[source] = -1.0
            # speed is reciprocal metric cost.
            result = np.asarray(skfmm.travel_time(phi, 1.0 / slowness, dx=spacing), dtype=float)
            result -= result[source]
            result[source] = 0.0
            return result
        except ImportError:
            pass
    return _fast_sweeping(slowness, source, spacing, tolerance, max_iterations)


def solve_fmm_geodesics(metric, start_idx=0, *, grid_shape=None, spacing=1.0, use_skfmm=True):
    """Compatibility wrapper returning flattened distances for flattened input.

    ``grid_shape`` is mandatory for flattened multidimensional tensor fields.
    Without it, the input is honestly interpreted as a one-dimensional chart.
    """
    array = np.asarray(metric, dtype=float)
    was_flat = array.ndim == 3
    if was_flat:
        shape = tuple(grid_shape) if grid_shape is not None else (array.shape[0],)
        if np.prod(shape) != array.shape[0]:
            raise ValueError("grid_shape product must equal the number of metric tensors")
        array = array.reshape(shape + array.shape[-2:])
        source = np.unravel_index(int(start_idx), shape)
    else:
        shape = array.shape[:-2]
        source = np.unravel_index(int(start_idx), shape) if np.isscalar(start_idx) else start_idx
    result = fmm_geodesic_distance(array, source, spacing=spacing, use_skfmm=use_skfmm)
    return result.reshape(-1) if was_flat else result
