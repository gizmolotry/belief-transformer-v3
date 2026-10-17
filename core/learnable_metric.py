"""Learnable local metric for semantic chart coordinates.

The model operates on a *chart* of a manifold.  It does not infer chart
coordinates from an arbitrary point cloud; callers retain responsibility for
constructing those coordinates (for example with PCA or UMAP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LocalMetricTensor(nn.Module):
    """A shallow MLP producing a coordinate warp and an SPD metric tensor.

    Each observation is described by scalar local density and a semantic
    gradient.  Positive diagonal Cholesky factors guarantee that the returned
    metric is symmetric positive definite, including before training.
    """

    def __init__(self, semantic_dim: int, chart_dim: Optional[int] = None,
                 hidden_dim: int = 32, min_diagonal: float = 1e-3) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.chart_dim = int(chart_dim or semantic_dim)
        self.min_diagonal = float(min_diagonal)
        n_tri = self.chart_dim * (self.chart_dim + 1) // 2
        self.network = nn.Sequential(
            nn.Linear(self.semantic_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.chart_dim + n_tri),
        )

    def forward(self, density: Tensor, semantic_gradient: Tensor) -> tuple[Tensor, Tensor]:
        if semantic_gradient.ndim != 2 or semantic_gradient.shape[1] != self.semantic_dim:
            raise ValueError("semantic_gradient must have shape (n, semantic_dim)")
        density = density.reshape(-1, 1).to(dtype=semantic_gradient.dtype,
                                            device=semantic_gradient.device)
        if density.shape[0] != semantic_gradient.shape[0]:
            raise ValueError("density and semantic_gradient must contain the same number of points")
        output = self.network(torch.cat((density, semantic_gradient), dim=1))
        warp, raw_tri = output[:, :self.chart_dim], output[:, self.chart_dim:]
        n = output.shape[0]
        chol = output.new_zeros((n, self.chart_dim, self.chart_dim))
        rows, cols = torch.tril_indices(self.chart_dim, self.chart_dim, device=output.device)
        chol[:, rows, cols] = raw_tri
        diag = torch.arange(self.chart_dim, device=output.device)
        chol[:, diag, diag] = F.softplus(chol[:, diag, diag]) + self.min_diagonal
        metric = chol @ chol.transpose(-1, -2)
        return warp, metric

    def warp_coordinates(self, coordinates: Tensor, density: Tensor,
                         semantic_gradient: Tensor, scale: float = 1.0) -> tuple[Tensor, Tensor]:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.chart_dim:
            raise ValueError("coordinates must be explicit chart coordinates of shape (n, chart_dim)")
        displacement, metric = self(density, semantic_gradient)
        return coordinates + float(scale) * displacement, metric


def metric_contrastive_loss(
    warped: Tensor,
    neighbor_pairs: Tensor,
    distant_pairs: Tensor,
    *,
    original_coordinates: Optional[Tensor] = None,
    margin: float = 1.0,
    neighborhood_weight: float = 1.0,
    distant_weight: float = 1.0,
) -> Tensor:
    """Preserve neighbor distances while separating topologically distant pairs."""
    def distances(pairs: Tensor, values: Tensor) -> Tensor:
        pairs = pairs.to(device=values.device, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pair tensors must have shape (m, 2)")
        return torch.linalg.vector_norm(values[pairs[:, 0]] - values[pairs[:, 1]], dim=1)

    near = distances(neighbor_pairs, warped)
    if original_coordinates is None:
        local_loss = near.square().mean() if near.numel() else warped.sum() * 0
    else:
        target = distances(neighbor_pairs, original_coordinates.to(warped))
        local_loss = (near - target).square().mean() if near.numel() else warped.sum() * 0
    far = distances(distant_pairs, warped)
    far_loss = F.relu(float(margin) - far).square().mean() if far.numel() else warped.sum() * 0
    return float(neighborhood_weight) * local_loss + float(distant_weight) * far_loss


@dataclass
class MetricTrainingResult:
    losses: list[float]


def train_local_metric(model: LocalMetricTensor, coordinates: Tensor, density: Tensor,
                       semantic_gradient: Tensor, neighbor_pairs: Tensor,
                       distant_pairs: Tensor, *, epochs: int = 100, lr: float = 1e-3,
                       margin: float = 1.0, warp_scale: float = 1.0) -> MetricTrainingResult:
    """Train the local warp with the self-supervised contrastive objective."""
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        warped, metric = model.warp_coordinates(coordinates, density, semantic_gradient, warp_scale)
        warp_loss = metric_contrastive_loss(warped, neighbor_pairs, distant_pairs,
                                            original_coordinates=coordinates, margin=margin)
        # Train the metric head itself, rather than relying only on gradients
        # through the shared warp trunk.  Pair costs use the midpoint tensor as
        # a local quadrature approximation to Riemannian arc length.
        def metric_pair_distance(pairs: Tensor) -> Tensor:
            pairs = pairs.to(device=coordinates.device, dtype=torch.long).reshape(-1, 2)
            if pairs.numel() == 0:
                return coordinates.new_empty((0,))
            delta = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
            midpoint_metric = 0.5 * (metric[pairs[:, 0]] + metric[pairs[:, 1]])
            squared = torch.einsum("ni,nij,nj->n", delta, midpoint_metric, delta)
            return torch.sqrt(squared.clamp(min=1e-12))

        near_metric = metric_pair_distance(neighbor_pairs)
        near_target = torch.linalg.vector_norm(
            coordinates[neighbor_pairs.reshape(-1, 2)[:, 1]]
            - coordinates[neighbor_pairs.reshape(-1, 2)[:, 0]], dim=1
        ) if neighbor_pairs.numel() else coordinates.new_empty((0,))
        metric_local = (near_metric - near_target).square().mean() if near_metric.numel() else coordinates.sum() * 0
        far_metric = metric_pair_distance(distant_pairs)
        metric_far = F.relu(float(margin) - far_metric).square().mean() if far_metric.numel() else coordinates.sum() * 0
        loss = warp_loss + metric_local + metric_far
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return MetricTrainingResult(losses)


def compute_learnable_metric_tensor(density, semantic_gradient, model: Optional[LocalMetricTensor] = None):
    """NumPy-friendly compatibility helper returning an untrained SPD metric."""
    gradient = torch.as_tensor(semantic_gradient, dtype=torch.float32)
    density_t = torch.as_tensor(density, dtype=torch.float32)
    model = model or LocalMetricTensor(gradient.shape[1])
    with torch.no_grad():
        _, metric = model(density_t, gradient)
    return metric.cpu().numpy()
