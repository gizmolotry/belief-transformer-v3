import torch

from core.complete_pipeline import _run_dynamic_dirichlet_feedback


class MockFusion:
    def __init__(self):
        self.calls = []

    def __call__(self, cls_per_bot, *, article_alphas=None, compute_curvature=True):
        assert article_alphas is not None
        self.calls.append(article_alphas.detach().clone())
        # Make the returned geometry depend on the tuned pressure while retaining
        # a stable high-conflict final article.
        base = cls_per_bot.mean(dim=1)
        fused = base * article_alphas.unsqueeze(-1)
        stress = torch.zeros_like(fused)
        stress[-1] = 2.0
        return {
            "fused": fused,
            "fused_std": stress,
            "curvature": {
                "participation_ratio": torch.ones(cls_per_bot.shape[0]),
                "effective_rank_90": torch.ones(cls_per_bot.shape[0]),
                "lambda1_lambda2": torch.ones(cls_per_bot.shape[0]),
            },
            "provenance": {"mock": True},
        }


def _inputs():
    views = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[0.2, 0.0], [0.3, 0.0]],
            [[8.0, 0.0], [9.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    initial = {
        "fused": views.mean(dim=1),
        "fused_std": torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]]),
        "provenance": {"initial": True},
    }
    return views, initial


def test_feedback_replaces_initial_fusion_and_focuses_cracked_article():
    fusion = MockFusion()
    views, initial = _inputs()
    tuned, diagnostics = _run_dynamic_dirichlet_feedback(
        fusion,
        views,
        initial,
        {"base_alpha": 1.0, "max_iterations": 3, "damping": 1.0, "tolerance": 0.0},
    )
    assert len(fusion.calls) == diagnostics["iterations"]
    assert diagnostics["iterations"] == 3
    assert diagnostics["article_alphas"][-1] < diagnostics["article_alphas"][0]
    assert diagnostics["crack_density"][-1] > diagnostics["crack_density"][0]
    assert diagnostics["pressure_direction"] == "higher_crack_lower_alpha_sparser_mixture"
    assert tuned["dynamic_observer_tuning"] == diagnostics
    assert tuned["provenance"]["dynamic_observer_tuning"] == diagnostics
    assert not torch.equal(tuned["fused"], initial["fused"])


def test_feedback_is_deterministic_with_lightweight_mock():
    views, initial = _inputs()
    config = {"base_alpha": 1.0, "max_iterations": 2, "damping": 0.5}
    first, first_diag = _run_dynamic_dirichlet_feedback(MockFusion(), views, initial, config)
    second, second_diag = _run_dynamic_dirichlet_feedback(MockFusion(), views, initial, config)
    torch.testing.assert_close(first["fused"], second["fused"])
    assert first_diag == second_diag


def test_zero_iterations_preserves_initial_result_values():
    views, initial = _inputs()
    fusion = MockFusion()
    result, diagnostics = _run_dynamic_dirichlet_feedback(
        fusion, views, initial, {"max_iterations": 0}
    )
    assert fusion.calls == []
    torch.testing.assert_close(result["fused"], initial["fused"])
    assert diagnostics["iterations"] == 0
    assert diagnostics["crack_density"] == []


def test_feedback_prefers_dirichlet_fracture_matrix_over_proxy():
    views, initial = _inputs()
    initial["crack_matrix"] = torch.tensor(
        [[0.0, 0.1, 0.9], [0.1, 0.0, 0.2], [0.9, 0.2, 0.0]]
    )
    _, diagnostics = _run_dynamic_dirichlet_feedback(
        MockFusion(), views, initial, {"max_iterations": 2, "damping": 1.0}
    )
    assert diagnostics["crack_signal_source"] == "dirichlet_fracture_crack_matrix"
    assert diagnostics["crack_signal_source_history"] == [
        "dirichlet_fracture_crack_matrix",
        "dirichlet_fracture_crack_matrix",
    ]
    assert diagnostics["article_alphas"][2] < diagnostics["article_alphas"][1]
