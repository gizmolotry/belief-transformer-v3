import importlib
import math

import numpy as np
import pytest

from core.full_spd_geometry import (
    DisconnectedSupportError,
    FullSPDGeometryError,
    ProvenanceError,
    RankingCollapseError,
    all_pairs_dijkstra,
    article_eccentricity,
    article_shuffle_gradients_within_group,
    build_euclidean_union_knn_support,
    build_synthetic_complete_graph_support,
    canonical_array_sha256,
    canonical_json_sha256,
    dense_metric_fixture_from_gradients,
    derive_deterministic_seed,
    determinant_lemma_rank_one,
    eccentricity_dispersion_cv,
    edge_costs,
    edge_amplification_q95,
    kendalls_w,
    make_empirical_provenance,
    observer_shuffled_target_helper,
    observer_target_from_nli_logits,
    ordered_ids_hash,
    quadratic_form_rank_one,
    rank_articles_by_value,
    near_tie_fraction,
    near_tie_tolerance,
    potential_generalization_diagnostics,
    spectrum_matched_random_rank_one_directions,
    tensor_metric_low_rank,
    train_oof_potential,
    validate_dense_spd_fixtures,
    validate_empirical_provenance,
    weighted_fold_kendalls_w,
)


def _square_points():
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )


def _square_ids():
    return ["a", "b", "c", "d"]


def test_rank_one_tensor_shape_symmetry_spd_determinant_and_quadratic_positivity():
    gradients = np.array([[3.0, 4.0], [0.5, -0.25]], dtype=np.float64)
    metric = tensor_metric_low_rank(gradients)
    assert metric["tensor_family"] == "full_spd"
    assert metric["node_gradients"].shape == (2, 2)

    dense = dense_metric_fixture_from_gradients(gradients)
    assert dense.shape == (2, 2, 2)
    assert np.allclose(dense, np.swapaxes(dense, 1, 2))
    assert np.all(np.linalg.eigvalsh(dense) > 0.0)
    assert np.allclose(np.linalg.det(dense), determinant_lemma_rank_one(gradients))

    displacements = np.array([[1.0, 2.0], [-2.0, 1.0]], dtype=np.float64)
    q = quadratic_form_rank_one(displacements, gradients)
    assert np.all(q > 0.0)


def test_dense_low_dim_fixtures_reject_nonfinite_nonsymmetric_and_near_singular():
    with pytest.raises(FullSPDGeometryError, match="NaN|Inf"):
        validate_dense_spd_fixtures(np.array([[[1.0, np.nan], [0.0, 1.0]]]))
    with pytest.raises(FullSPDGeometryError, match="symmetric"):
        validate_dense_spd_fixtures(np.array([[[1.0, 2.0], [0.0, 1.0]]]))
    with pytest.raises(FullSPDGeometryError, match="near-singular"):
        validate_dense_spd_fixtures(np.array([[[1e-12, 0.0], [0.0, 1.0]]]))
    with pytest.raises(FullSPDGeometryError, match="forbidden"):
        dense_metric_fixture_from_gradients(np.zeros((2, 65), dtype=np.float64))


def test_identity_edge_costs_match_euclidean_graph_and_dijkstra():
    points = _square_points()
    graph = build_euclidean_union_knn_support(points, _square_ids(), k=2)
    gradients = np.zeros_like(points)
    weights = edge_costs(points, gradients, graph, arm="identity")
    assert np.allclose(weights, graph.euclidean_lengths)
    distances = all_pairs_dijkstra(graph, weights)
    assert distances.shape == (4, 4)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert article_eccentricity(distances).shape == (4,)


def test_full_diagonal_scalar_and_conformal_edge_formulas_include_square_root():
    points = np.array([[0.0, 0.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
    graph = build_euclidean_union_knn_support(points, ["a", "b", "c"], k=1)
    gradients = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    weights_full = edge_costs(points, gradients, graph, arm="full")
    weights_diag = edge_costs(points, gradients, graph, arm="diagonal")
    weights_trace = edge_costs(points, gradients, graph, arm="scalar_trace")
    weights_conf = edge_costs(points, gradients, graph, arm="oof_conformal", yhat=[0.0, 1.0, -3.0])

    for edge_index, (i, j) in enumerate(graph.edges):
        d = points[j] - points[i]
        expected_full = math.sqrt(np.dot(d, d) + 0.5 * ((d @ gradients[i]) ** 2 + (d @ gradients[j]) ** 2))
        expected_diag = math.sqrt(np.sum(d**2 * (1.0 + 0.5 * (gradients[i] ** 2 + gradients[j] ** 2))))
        expected_trace = math.sqrt(
            np.dot(d, d)
            * (1.0 + 0.5 * (np.dot(gradients[i], gradients[i]) / 2 + np.dot(gradients[j], gradients[j]) / 2))
        )
        rho = 1.0 / (1.0 + np.abs(np.array([0.0, 1.0, -3.0])))
        expected_conf = math.sqrt(np.dot(d, d) / (rho[i] * rho[j]))
        assert weights_full[edge_index] == pytest.approx(expected_full)
        assert weights_diag[edge_index] == pytest.approx(expected_diag)
        assert weights_trace[edge_index] == pytest.approx(expected_trace)
        assert weights_conf[edge_index] == pytest.approx(expected_conf)


def test_full_metric_is_coordinate_equivariant_under_orthogonal_rotation():
    points = _square_points()
    gradients = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    graph = build_euclidean_union_knn_support(points, _square_ids(), k=2)
    base = edge_costs(points, gradients, graph, arm="full")
    q = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    rotated = edge_costs(points @ q, gradients @ q, graph, arm="full")
    assert np.allclose(base, rotated)


def test_anisotropic_full_differs_from_scalar_trace_when_direction_matters():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    graph = build_euclidean_union_knn_support(points, ["a", "b", "c"], k=1)
    gradients = np.array([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    full = edge_costs(points, gradients, graph, arm="full")
    trace = edge_costs(points, gradients, graph, arm="scalar_trace")
    assert not np.allclose(full, trace)


def test_deterministic_tie_break_uses_canonical_id_and_records_duplicates():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    graph = build_euclidean_union_knn_support(points, ["center", "b", "a", "dup"], k=1)
    assert ("center", "dup") in graph.duplicates
    center_edges = {tuple(edge) for edge in graph.edges.tolist() if 0 in edge}
    assert (0, 3) in center_edges
    assert graph.support_hash == build_euclidean_union_knn_support(points, ["center", "b", "a", "dup"], k=1).support_hash


def test_disconnected_support_fails_closed():
    points = np.array([[0.0, 0.0], [0.0, 1.0], [100.0, 0.0], [100.0, 1.0]], dtype=np.float64)
    with pytest.raises(DisconnectedSupportError, match="disconnected"):
        build_euclidean_union_knn_support(points, ["a", "b", "c", "d"], k=1)


def test_spectrum_matched_random_directions_preserve_rank_one_eigenvalues_and_rng():
    gradients = np.array([[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]], dtype=np.float64)
    before = np.random.get_state()
    random_grad = spectrum_matched_random_rank_one_directions(gradients, seed=123)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert np.allclose(np.linalg.norm(random_grad, axis=1), np.linalg.norm(gradients, axis=1))
    assert np.allclose(determinant_lemma_rank_one(random_grad), determinant_lemma_rank_one(gradients))
    assert np.allclose(random_grad, spectrum_matched_random_rank_one_directions(gradients, seed=123))


def test_article_and_observer_shuffle_are_local_deterministic_and_group_respecting():
    gradients = np.arange(24, dtype=np.float64).reshape(6, 4)
    groups = ["x", "x", "x", "y", "y", "y"]
    shuffled = article_shuffle_gradients_within_group(gradients, groups, seed=7)
    assert sorted(map(tuple, shuffled[:3])) == sorted(map(tuple, gradients[:3]))
    assert sorted(map(tuple, shuffled[3:])) == sorted(map(tuple, gradients[3:]))
    assert np.allclose(shuffled, article_shuffle_gradients_within_group(gradients, groups, seed=7))

    scalar_target = np.arange(6, dtype=np.float64)
    shuffled_target, scalar_payload = observer_shuffled_target_helper(scalar_target, seed=99, groups=groups)
    assert shuffled_target.shape == (6,)
    assert scalar_payload.shape == (6,)
    assert sorted(shuffled_target[:3]) == sorted(scalar_target[:3])
    assert sorted(shuffled_target[3:]) == sorted(scalar_target[3:])

    observer_values = np.arange(18, dtype=np.float64).reshape(6, 3)
    target, shuffled_observers = observer_shuffled_target_helper(
        observer_values, seed=99, groups=groups, target_fn=lambda rows: rows[:, 0] - rows[:, 1]
    )
    assert target.shape == (6,)
    assert shuffled_observers.shape == observer_values.shape
    assert sorted(map(tuple, shuffled_observers[:3])) == sorted(map(tuple, observer_values[:3]))
    assert sorted(map(tuple, shuffled_observers[3:])) == sorted(map(tuple, observer_values[3:]))


def test_canonical_hashes_are_deterministic_and_include_shape_dtype_and_ordered_ids():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    same = np.array([[1.0, 2.0]], dtype=np.float32)
    reshaped = np.array([1.0, 2.0], dtype=np.float32)
    cast = np.array([[1.0, 2.0]], dtype=np.float64)
    assert canonical_array_sha256(arr) == canonical_array_sha256(same)
    assert canonical_array_sha256(arr) != canonical_array_sha256(reshaped)
    assert canonical_array_sha256(arr) != canonical_array_sha256(cast)
    assert ordered_ids_hash(["a", "b"]) != ordered_ids_hash(["b", "a"])


def test_seed_derivation_is_keyed_by_control_q_bank_and_fold():
    first = derive_deterministic_seed(42, "article_shuffle", "q0", 0)
    assert first == derive_deterministic_seed(42, "article_shuffle", "q0", 0)
    variants = {
        derive_deterministic_seed(42, "article_shuffle", "q0", 1),
        derive_deterministic_seed(42, "article_shuffle", "q1", 0),
        derive_deterministic_seed(42, "spectrum_random", "q0", 0),
    }
    assert first not in variants
    assert len(variants) == 3


def test_synthetic_complete_graph_support_is_bounded_and_complete():
    points = _square_points()
    graph = build_synthetic_complete_graph_support(points, _square_ids())
    assert graph.edges.shape[0] == 6
    assert graph.k == 3
    with pytest.raises(FullSPDGeometryError, match="N > 100"):
        build_synthetic_complete_graph_support(np.zeros((101, 2)), [f"p{i}" for i in range(101)])
    with pytest.raises(FullSPDGeometryError, match="N > 100"):
        build_synthetic_complete_graph_support(
            np.zeros((101, 2)), [f"p{i}" for i in range(101)], max_n=101
        )
    with pytest.raises(FullSPDGeometryError, match="N > 100"):
        build_synthetic_complete_graph_support(
            np.zeros((101, 2)), [f"p{i}" for i in range(101)], max_n=1000
        )


def test_article_ranking_uses_midranks_and_only_uses_canonical_id_to_serialize_ties():
    ranking = rank_articles_by_value([2.0, 1.0, 1.0, 3.0], ["z", "b", "a", "q"])
    assert ranking.order == ("a", "b", "z", "q")
    assert ranking.ranks.tolist() == [3.0, 1.5, 1.5, 4.0]
    assert ranking.tied_values == ((1.0, ("a", "b")),)


def test_provenance_validation_rejects_missing_tampered_forbidden_and_mislabeled_sources():
    points = _square_points()
    graph = build_euclidean_union_knn_support(points, _square_ids(), k=2)
    gradients = np.ones_like(points)
    weights = edge_costs(points, gradients, graph, arm="full")
    provenance = make_empirical_provenance(
        target_source_kind="synthetic",
        ordered_ids=_square_ids(),
        node_gradients=gradients,
        support_graph=graph,
        support_weights=weights,
        model_state_hash="model",
        loss_trace=[1.0, 0.5],
        train_ids=["t1", "t2"],
        heldout_ids=_square_ids(),
        targets=[0.0, 1.0],
        q_bank_hash="q",
        seed=5,
        git_commit="abc",
    )
    assert validate_empirical_provenance(provenance)["seed"] == 5
    missing = dict(provenance)
    missing.pop("node_gradient_hash")
    with pytest.raises(ProvenanceError, match="missing"):
        validate_empirical_provenance(missing)
    tampered = dict(provenance)
    tampered["seed"] = 6
    with pytest.raises(ProvenanceError, match="hash mismatch"):
        validate_empirical_provenance(tampered)
    mislabeled = dict(provenance)
    mislabeled["tensor_family"] = "rank_one_spd"
    no_hash = dict(mislabeled)
    no_hash.pop("provenance_hash")
    mislabeled["provenance_hash"] = canonical_json_sha256(no_hash)
    with pytest.raises(ProvenanceError, match="tensor_family"):
        validate_empirical_provenance(mislabeled)
    with pytest.raises(ProvenanceError, match="target_source_kind"):
        make_empirical_provenance(
            target_source_kind="mbfc",
            ordered_ids=_square_ids(),
            node_gradients=gradients,
            support_graph=graph,
            support_weights=weights,
            model_state_hash="model",
            loss_trace=[1.0],
            train_ids=["t1"],
            heldout_ids=_square_ids(),
            targets=[0.0],
            q_bank_hash="q",
            seed=5,
            git_commit="abc",
        )
    with pytest.raises(ProvenanceError, match="ordered_ids"):
        make_empirical_provenance(
            target_source_kind="synthetic",
            ordered_ids=["d", "c", "b", "a"],
            node_gradients=gradients,
            support_graph=graph,
            support_weights=weights,
            model_state_hash="model",
            loss_trace=[1.0],
            train_ids=["t1"],
            heldout_ids=_square_ids(),
            targets=[0.0],
            q_bank_hash="q",
            seed=5,
            git_commit="abc",
        )
    with pytest.raises(ProvenanceError, match="disjoint"):
        make_empirical_provenance(
            target_source_kind="synthetic",
            ordered_ids=_square_ids(),
            node_gradients=gradients,
            support_graph=graph,
            support_weights=weights,
            model_state_hash="model",
            loss_trace=[1.0],
            train_ids=["a"],
            heldout_ids=_square_ids(),
            targets=[0.0],
            q_bank_hash="q",
            seed=5,
            git_commit="abc",
        )


def test_kendall_w_is_tie_corrected_and_constants_fail_closed():
    ranks = np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3]], dtype=np.float64)
    assert 0.0 <= kendalls_w(ranks) <= 1.0
    assert weighted_fold_kendalls_w([ranks, ranks], [2.0, 1.0]) == pytest.approx(kendalls_w(ranks))
    with pytest.raises(RankingCollapseError, match="zero rank variance"):
        kendalls_w(np.full((3, 4), 2.5))

    tied_ranks = np.array(
        [[1.5, 1.5, 3.0, 4.0, 5.0], [1.0, 2.5, 2.5, 4.0, 5.0], [1.0, 2.0, 3.5, 3.5, 5.0]],
        dtype=np.float64,
    )
    rank_sums = tied_ranks.sum(axis=0)
    sum_squares = np.sum((rank_sums - rank_sums.mean()) ** 2)
    tie_correction = (2**3 - 2) * 3
    expected = 12.0 * sum_squares / (3**2 * (5**3 - 5) - 3 * tie_correction)
    assert kendalls_w(tied_ranks) == pytest.approx(expected)


def test_observer_target_honors_label_order_and_uses_population_std_of_signed_scores():
    logits = np.array(
        [[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 1.0],
          [1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    target = observer_target_from_nli_logits(logits, ["neutral", "contradict", "entail"])
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=2, keepdims=True)
    expected = np.std(probabilities[:, :, 2] - probabilities[:, :, 1], axis=1, ddof=0)
    assert np.allclose(target, expected)
    reordered = logits[:, :, [2, 0, 1]]
    assert np.allclose(
        target,
        observer_target_from_nli_logits(reordered, {"entail": 0, "neutral": 1, "contradict": 2}),
    )
    with pytest.raises(FullSPDGeometryError, match="unambiguously"):
        observer_target_from_nli_logits(logits, ["neutral", "entail", "entail"])


def test_collapse_and_inflation_gate_diagnostics_fail_closed_on_bad_baselines():
    assert eccentricity_dispersion_cv([1.0, 2.0, 3.0]) == pytest.approx(np.std([1.0, 2.0, 3.0]) / 2.0)
    assert near_tie_fraction([0.0, 0.05, 0.2], tolerance=0.051) == pytest.approx(1.0 / 3.0)
    diagnostics = potential_generalization_diagnostics(
        train_targets=[0.0, 2.0], train_predictions=[0.0, 2.0],
        heldout_targets=[0.0, 4.0], heldout_predictions=[1.0, 3.0],
    )
    assert diagnostics.g_train == pytest.approx(1.0)
    assert diagnostics.g_hold == pytest.approx(0.8)
    assert diagnostics.g_gap == pytest.approx(0.2)
    assert edge_amplification_q95([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == pytest.approx(2.9)
    with pytest.raises(FullSPDGeometryError, match="nonpositive baseline"):
        eccentricity_dispersion_cv([0.0, 0.0])
    with pytest.raises(FullSPDGeometryError, match="tolerance"):
        near_tie_fraction([1.0, 2.0], tolerance=-0.1)
    with pytest.raises(FullSPDGeometryError, match="zero baseline"):
        potential_generalization_diagnostics([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(FullSPDGeometryError, match="strictly positive"):
        edge_amplification_q95([1.0, 2.0], [1.0, 0.0])


def test_near_tie_tolerance_uses_exact_frozen_scale_relative_formula():
    for values in (
        np.array([1.0, -2.0, 9.0], dtype=np.float64),
        np.array([1e-200, -2e-200, 9e-200], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    ):
        expected = 1e-8 * max(
            float(np.median(np.abs(values))), np.finfo(np.float64).tiny
        )
        actual = near_tie_tolerance(values)
        assert np.asarray(actual, dtype=np.float64).tobytes() == np.asarray(
            expected, dtype=np.float64
        ).tobytes()
    with pytest.raises(FullSPDGeometryError, match="at least one"):
        near_tie_tolerance([])


def test_kendall_w_and_eccentricity_stat_helpers():
    ranks = np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3]], dtype=np.float64)
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert np.allclose(article_eccentricity(distances), [1.5, 2.0, 2.5])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_train_oof_potential_uses_train_only_normalization_and_preserves_torch_rng():
    import torch

    torch.manual_seed(2026)
    before = torch.get_rng_state().clone()
    train_x = np.array(
        [[-2.0, 0.0], [-1.0, 0.5], [0.0, 0.0], [1.0, -0.5], [2.0, 0.0]],
        dtype=np.float64,
    )
    train_y = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64)
    heldout_x = np.array([[10.0, 10.0], [-10.0, -10.0]], dtype=np.float64)
    result_a = train_oof_potential(
        train_x,
        train_y,
        heldout_x,
        train_ids=[f"tr{i}" for i in range(5)],
        heldout_ids=["h1", "h2"],
        q_bank_hash="qbank",
        seed=11,
        git_commit="abc123",
        target_source_kind="synthetic",
        device="cpu",
    )
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert result_a.target_mean == pytest.approx(0.0)
    assert result_a.target_std == pytest.approx(np.std(train_y))
    assert result_a.predictions.shape == (2,)
    assert result_a.train_predictions.shape == (5,)
    assert result_a.train_z_predictions.shape == (5,)
    assert result_a.node_gradients.shape == heldout_x.shape
    assert np.allclose(
        result_a.train_predictions,
        result_a.train_z_predictions * result_a.target_std + result_a.target_mean,
    )
    assert len(result_a.loss_trace) == 300
    assert result_a.provenance["training"]["epochs"] == 300
    assert result_a.provenance["training"]["config"]["hidden_dim"] == 2
    assert result_a.provenance["training"]["config"]["n_ideologies"] == 8
    assert result_a.provenance["training"]["device"] == "cpu"
    prediction_hashes = result_a.provenance["training"]["prediction_hashes"]
    assert prediction_hashes["train"] == canonical_array_sha256(result_a.train_predictions)
    assert prediction_hashes["train_z"] == canonical_array_sha256(result_a.train_z_predictions)
    assert prediction_hashes["heldout"] == canonical_array_sha256(result_a.predictions)
    assert prediction_hashes["heldout_z"] == canonical_array_sha256(result_a.z_predictions)
    assert result_a.provenance["tensor_family"] == "full_spd"
    validate_empirical_provenance(result_a.provenance)

    result_b = train_oof_potential(
        train_x,
        train_y,
        heldout_x + 1000.0,
        train_ids=[f"tr{i}" for i in range(5)],
        heldout_ids=["h1", "h2"],
        q_bank_hash="qbank",
        seed=11,
        git_commit="abc123",
        target_source_kind="synthetic",
    )
    assert result_b.target_mean == pytest.approx(result_a.target_mean)
    assert result_b.target_std == pytest.approx(result_a.target_std)

    result_c = train_oof_potential(
        train_x,
        train_y,
        heldout_x,
        train_ids=[f"tr{i}" for i in range(5)],
        heldout_ids=["h1", "h2"],
        q_bank_hash="qbank",
        seed=11,
        git_commit="abc123",
        target_source_kind="synthetic",
    )
    assert np.allclose(result_c.node_gradients, result_a.node_gradients)
    assert np.allclose(result_c.train_predictions, result_a.train_predictions)
    assert np.allclose(result_c.train_z_predictions, result_a.train_z_predictions)
    assert result_c.provenance["model_state_hash"] == result_a.provenance["model_state_hash"]

    generalization = potential_generalization_diagnostics(
        train_y,
        result_a.train_predictions,
        np.array([20.0, -20.0]),
        result_a.predictions,
    )
    expected_g_train = 1.0 - np.mean((result_a.train_predictions - train_y) ** 2) / np.mean(
        (train_y - np.mean(train_y)) ** 2
    )
    expected_g_hold = 1.0 - np.mean((result_a.predictions - [20.0, -20.0]) ** 2) / np.mean(
        (np.array([20.0, -20.0]) - np.mean(train_y)) ** 2
    )
    assert generalization.g_train == pytest.approx(expected_g_train)
    assert generalization.g_hold == pytest.approx(expected_g_hold)
    assert generalization.g_gap == pytest.approx(expected_g_train - expected_g_hold)

    result_d = train_oof_potential(
        train_x,
        train_y,
        heldout_x,
        train_ids=[f"tr{i}" for i in range(5)],
        heldout_ids=["h1", "h2"],
        q_bank_hash="qbank",
        seed=12,
        git_commit="abc123",
        target_source_kind="synthetic",
    )
    assert result_d.provenance["model_state_hash"] != result_a.provenance["model_state_hash"]

    with pytest.raises(ProvenanceError, match="disjoint"):
        train_oof_potential(
            train_x,
            train_y,
            heldout_x,
            train_ids=["h1", "tr1", "tr2", "tr3", "tr4"],
            heldout_ids=["h1", "h2"],
            q_bank_hash="qbank",
            seed=11,
            git_commit="abc123",
            target_source_kind="synthetic",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_train_oof_potential_cpu_preserves_cuda_rng_and_deterministic_flags():
    import torch

    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(1201)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )

        result = train_oof_potential(
            [[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]],
            [-1.0, 0.0, 1.0],
            [[0.25, -0.25]],
            train_ids=["tr0", "tr1", "tr2"],
            heldout_ids=["h0"],
            q_bank_hash="qbank",
            seed=17,
            git_commit="abc123",
            target_source_kind="synthetic",
            device=torch.device("cpu"),
        )

        assert result.provenance["training"]["device"] == "cpu"
        assert result.train_predictions.shape == (3,)
        assert result.train_z_predictions.shape == (3,)
        assert torch.equal(cpu_before, torch.get_rng_state())
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_before, torch.cuda.get_rng_state_all())
        )
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.use_deterministic_algorithms(
            original_enabled, warn_only=original_warn_only
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_oof_potential_rejects_non_cpu_cuda_device():
    with pytest.raises(FullSPDGeometryError, match="must be cpu or cuda"):
        train_oof_potential(
            [[-1.0], [0.0], [1.0]],
            [-1.0, 0.0, 1.0],
            [[0.5]],
            train_ids=["tr0", "tr1", "tr2"],
            heldout_ids=["h0"],
            q_bank_hash="qbank",
            seed=17,
            git_commit="abc123",
            target_source_kind="synthetic",
            device="meta",
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_train_oof_potential_cuda_uses_gpu_and_preserves_global_state():
    import torch

    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(5150)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

        result = train_oof_potential(
            [[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]],
            [-1.0, 0.0, 1.0],
            [[0.25, -0.25]],
            train_ids=["tr0", "tr1", "tr2"],
            heldout_ids=["h0"],
            q_bank_hash="qbank",
            seed=23,
            git_commit="abc123",
            target_source_kind="synthetic",
            device="cuda",
        )

        assert result.predictions.shape == (1,)
        assert result.train_predictions.shape == (3,)
        assert result.train_z_predictions.shape == (3,)
        assert result.node_gradients.shape == (1, 2)
        assert result.provenance["training"]["device"] == f"cuda:{torch.cuda.current_device()}"
        assert torch.equal(cpu_before, torch.get_rng_state())
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_before, torch.cuda.get_rng_state_all())
        )
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.use_deterministic_algorithms(
            original_enabled, warn_only=original_warn_only
        )


def test_edge_support_is_identical_across_metric_arms_and_only_weights_change():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    gradients = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    graph = build_euclidean_union_knn_support(points, _square_ids(), k=2)
    support_before = graph.support_hash
    edges_before = graph.edges.copy()
    arms = {
        "identity": edge_costs(points, gradients, graph, arm="identity"),
        "full": edge_costs(points, gradients, graph, arm="full"),
        "scalar_trace": edge_costs(points, gradients, graph, arm="scalar_trace"),
        "diagonal": edge_costs(points, gradients, graph, arm="diagonal"),
        "oof_conformal": edge_costs(points, gradients, graph, arm="oof_conformal", yhat=[0.0, 1.0, 2.0, 3.0]),
    }
    assert graph.support_hash == support_before
    assert np.array_equal(graph.edges, edges_before)
    assert len({weights.shape for weights in arms.values()}) == 1
    assert not np.allclose(arms["identity"], arms["full"])
    assert not np.allclose(arms["identity"], arms["oof_conformal"])


def test_curved_spatial_fixture_refinement_moves_toward_known_arc_length():
    # Quarter circle sampled increasingly densely; identity graph length converges
    # to the analytic arc length pi/2 from endpoint to endpoint.
    errors = []
    for n in (8, 16, 32):
        theta = np.linspace(0.0, np.pi / 2.0, n)
        points = np.column_stack([np.cos(theta), np.sin(theta)])
        graph = build_euclidean_union_knn_support(points, [f"p{i:02d}" for i in range(n)], k=2)
        weights = edge_costs(points, np.zeros_like(points), graph, arm="identity")
        dist = all_pairs_dijkstra(graph, weights)
        errors.append(abs(dist[0, -1] - np.pi / 2.0))
    assert errors[2] < errors[1] < errors[0]
