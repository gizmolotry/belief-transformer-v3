import numpy as np
import torch

from core.fmm_geodesic import fmm_geodesic_distance, solve_fmm_geodesics
from core.learnable_metric import LocalMetricTensor, metric_contrastive_loss, train_local_metric

def test_learnable_metric_tensor_is_spd_and_trainable():
    torch.manual_seed(4)
    model = LocalMetricTensor(semantic_dim=2, chart_dim=2, hidden_dim=12)
    density = torch.linspace(0.1, 1.0, 8)
    gradient = torch.randn(8, 2)
    coordinates = torch.stack((torch.arange(8, dtype=torch.float32), torch.zeros(8)), 1)
    warp, tensor = model(density, gradient)
    assert warp.shape == (8, 2)
    assert tensor.shape == (8, 2, 2)
    assert torch.all(torch.linalg.eigvalsh(tensor) > 0)
    near = torch.tensor([[i, i + 1] for i in range(7)])
    far = torch.tensor([[0, 7], [1, 6]])
    before = metric_contrastive_loss(coordinates + warp, near, far, original_coordinates=coordinates)
    result = train_local_metric(model, coordinates, density, gradient, near, far, epochs=5, lr=2e-3)
    assert len(result.losses) == 5
    assert np.isfinite(result.losses).all()
    assert any(p.grad is not None for p in model.parameters())


def test_fmm_constant_metric_matches_euclidean_axis_distance():
    metric = np.broadcast_to(np.eye(2), (15, 15, 2, 2)).copy()
    distance = fmm_geodesic_distance(metric, (7, 7), use_skfmm=False)
    assert distance[7, 7] == 0.0
    np.testing.assert_allclose(distance[7, 12], 5.0, atol=1e-8)
    assert np.all(distance >= 0)


def test_compatibility_wrapper_flattened_grid():
    metric = np.broadcast_to(np.eye(2), (25, 2, 2)).copy()
    distance = solve_fmm_geodesics(metric, start_idx=12, grid_shape=(5, 5), use_skfmm=False)
    assert distance.shape == (25,)
    assert distance[12] == 0.0
