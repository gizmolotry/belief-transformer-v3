import numpy as np
import pytest
import torch

from core.dirichlet_fusion import DirichletFusion, DirichletFusionConfig


def _fusion() -> DirichletFusion:
    fusion = DirichletFusion(
        DirichletFusionConfig(
            n_bots=2,
            hidden_dim=2,
            rks_dim=4,
            n_observers=5,
            alpha=1.0,
            crn_enabled=True,
            crn_seed=19,
            basis_seed=3,
            use_annealing=False,
        )
    )
    for basis in fusion.bases.values():
        basis.set_sigma(1.0)
    return fusion


def _views() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.1, 0.9]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )


def test_expanded_legacy_weights_match_unmodified_global_path():
    fusion = _fusion()
    views = _views()
    legacy = fusion(views, compute_curvature=False)
    expanded = fusion._locked_weights.unsqueeze(0).expand(views.shape[0], -1, -1).clone()
    dynamic = fusion(views, observer_weights=expanded, compute_curvature=False)
    torch.testing.assert_close(dynamic["fused"], legacy["fused"])
    torch.testing.assert_close(dynamic["fused_std"], legacy["fused_std"])
    assert "dynamic_weighting" not in legacy
    assert "article_observer_weights" not in legacy


def test_article_alphas_emit_deterministic_weights_and_provenance():
    fusion = _fusion()
    views = _views()
    alphas = torch.tensor([1.0, 0.2, 2.0])
    first = fusion(views, article_alphas=alphas, compute_curvature=False)
    second = fusion(views, article_alphas=alphas, compute_curvature=False)
    weights = first["article_observer_weights"]
    assert weights.shape == (3, 5, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, 5))
    torch.testing.assert_close(weights, second["article_observer_weights"])
    assert first["dynamic_weighting"]["mode"] == "article_alphas"
    assert first["provenance"]["dynamic_weighting"]["crn_seed"] == 19
    torch.testing.assert_close(first["article_alphas"], alphas)


def test_provided_weights_are_normalized_per_article_and_observer():
    fusion = _fusion()
    raw = torch.tensor(
        [
            [[9.0, 1.0], [1.0, 9.0]],
            [[2.0, 2.0], [8.0, 2.0]],
            [[1.0, 3.0], [3.0, 1.0]],
        ]
    )
    result = fusion(_views(), observer_weights=raw, compute_curvature=False)
    assert result["all_samples"].shape[:2] == (3, 2)
    torch.testing.assert_close(result["article_observer_weights"].sum(dim=-1), torch.ones(3, 2))
    assert result["dynamic_weighting"]["mode"] == "provided_weights"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"article_alphas": torch.tensor([1.0, 1.0])},
        {"article_alphas": torch.tensor([1.0, -1.0, 1.0])},
        {"observer_weights": torch.ones(3, 2, 3)},
        {"observer_weights": torch.zeros(3, 2, 2)},
    ],
)
def test_dynamic_weight_validation(kwargs):
    with pytest.raises(ValueError):
        _fusion()(_views(), compute_curvature=False, **kwargs)


def test_alpha_and_explicit_weights_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _fusion()(
            _views(),
            article_alphas=torch.ones(3),
            observer_weights=torch.ones(3, 2, 2),
            compute_curvature=False,
        )
