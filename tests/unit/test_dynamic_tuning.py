import numpy as np
import pytest

from core.dynamic_observer_tuning import (
    DynamicDirichletController,
    DynamicTuningConfig,
    estimate_crack_density,
    sample_dynamic_dirichlet_weights,
    scale_dirichlet_alpha,
)

def test_dynamic_dirichlet_scaling():
    """
    Mock test for R3 Dynamic Observer Tuning.
    """
    mock_crack_density = np.array([0.1, 0.9, 0.5])
    alphas = scale_dirichlet_alpha(mock_crack_density, base_alpha=1.0)
    assert alphas[1] < alphas[2] < alphas[0]


def test_scaling_is_bounded_and_does_not_mutate_input():
    cracks = np.array([-1.0, 0.0, 1.0, 4.0])
    original = cracks.copy()
    alphas = scale_dirichlet_alpha(
        cracks, base_alpha=2.0, sensitivity=9.0, min_alpha=0.25, max_alpha=3.0
    )
    np.testing.assert_array_equal(cracks, original)
    assert np.all((alphas >= 0.25) & (alphas <= 3.0))
    assert np.all(np.diff(alphas) <= 0.0)


def test_dynamic_mixture_sampling_is_seeded_and_normalized():
    alphas = np.array([1.0, 0.2, 2.0])
    first = sample_dynamic_dirichlet_weights(alphas, n_bots=4, n_observers=200, seed=7)
    second = sample_dynamic_dirichlet_weights(alphas, n_bots=4, n_observers=200, seed=7)
    other = sample_dynamic_dirichlet_weights(alphas, n_bots=4, n_observers=200, seed=8)
    assert first.shape == (3, 200, 4)
    np.testing.assert_allclose(first.sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    # Lower alpha focuses mass more strongly on individual observer views.
    assert first[1].max(axis=-1).mean() > first[0].max(axis=-1).mean()


def test_feedback_controller_damps_and_converges():
    cfg = DynamicTuningConfig(damping=0.5, tolerance=0.04, max_iterations=20)
    controller = DynamicDirichletController(3, cfg)
    cracks = np.array([0.0, 0.5, 1.0])
    previous = controller.state.alphas.copy()
    for _ in range(20):
        state = controller.update(cracks)
        assert state.alphas[2] <= state.alphas[1] <= state.alphas[0]
        assert np.max(np.abs(state.alphas - previous)) <= 0.5 + 1e-12
        previous = state.alphas.copy()
        if state.converged:
            break
    assert state.converged
    assert state.iteration <= cfg.max_iterations
    assert len(state.history) == state.iteration


def test_crack_density_combines_local_geometry_and_stress():
    embeddings = np.array([[0.0], [0.1], [0.2], [10.0]])
    stress = np.array([0.0, 0.0, 0.2, 1.0])
    cracks = estimate_crack_density(embeddings, stress, k_neighbors=1)
    assert cracks.shape == (4,)
    assert np.all((cracks >= 0.0) & (cracks <= 1.0))
    assert cracks[-1] > cracks[0]


@pytest.mark.parametrize("bad", [np.array([[0.1]]), np.array([0.1, np.nan])])
def test_scaling_rejects_malformed_crack_density(bad):
    with pytest.raises(ValueError):
        scale_dirichlet_alpha(bad)
