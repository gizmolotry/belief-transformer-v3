import numpy as np
import torch

from core.track4_action_graph import ActionGraphConfig, run_action_graph


def _grid_inputs():
    coordinates = torch.tensor(
        [[float(i), float(j)] for i in range(3) for j in range(3)],
        dtype=torch.float32,
    )
    return coordinates, torch.ones(9), torch.zeros(9)


def test_track4_fmm_is_opt_in_and_reports_approximation():
    embeddings, density, stress = _grid_inputs()
    cfg = ActionGraphConfig(
        k_neighbors=4,
        max_paths=1,
        traversal_solver="fmm",
        fmm_grid_shape=(3, 3),
        fmm_use_skfmm=False,
        stress_weight=0.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )
    result = run_action_graph(
        embeddings,
        track3_density=density,
        metric_stress=stress,
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [8]},
        localized_metric_tensor=np.broadcast_to(np.eye(2), (9, 2, 2)).copy(),
    )
    record = result["records"][0]
    assert record["reached"] is True
    assert record["path_indices"][0] == 0
    assert record["path_indices"][-1] == 8
    assert record["traversal_solver_used"] == "fmm_eikonal"
    assert record["geodesic_distance"] > 0.0
    assert record["geodesic_approximation"]["metric_source"] == "caller_supplied_spd_tensor"
    assert result["summary"]["traversal_fallback_count"] == 0


def test_track4_fmm_without_chart_falls_back_safely():
    embeddings, density, stress = _grid_inputs()
    cfg = ActionGraphConfig(
        k_neighbors=8,
        max_paths=1,
        traversal_solver="fmm",
        fmm_allow_graph_fallback=True,
        fmm_grid_shape=(3,),  # invalid for the automatically derived two-dimensional PCA chart
    )
    result = run_action_graph(
        embeddings,
        track3_density=density,
        metric_stress=stress,
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [8]},
    )
    record = result["records"][0]
    assert record["reached"] is True
    assert record["traversal_solver_used"] == "graph_dijkstra"
    assert "fmm_grid_shape must contain 2 dimensions" in record["traversal_fallback_reason"]
    assert result["summary"]["traversal_fallback_count"] == 1


def test_track4_explicit_legacy_mode_remains_graph_dijkstra():
    embeddings, density, stress = _grid_inputs()
    result = run_action_graph(
        embeddings,
        track3_density=density,
        metric_stress=stress,
        config=ActionGraphConfig(traversal_solver="graph_dijkstra"),
        anchors=[0],
        targets_by_anchor={0: [8]},
    )
    assert result["records"][0]["traversal_solver_used"] == "graph_dijkstra"
    assert result["summary"]["traversal_solver_requested"] == "graph_dijkstra"


def test_track4_default_trains_metric_and_builds_pca_chart():
    embeddings, density, stress = _grid_inputs()
    result = run_action_graph(
        embeddings,
        track3_density=density,
        metric_stress=stress,
        config=ActionGraphConfig(fmm_use_skfmm=False, fmm_metric_training_epochs=2),
        anchors=[0],
        targets_by_anchor={0: [8]},
    )
    metadata = result["records"][0]["geodesic_approximation"]
    assert result["records"][0]["traversal_solver_used"] == "fmm_eikonal"
    assert metadata["chart_construction"].startswith("centered_pca")
    assert metadata["metric_source"] == "production_self_supervised_local_metric"
    assert metadata["metric_training"]["trained"] is True
    assert metadata["metric_training"]["epochs"] == 2
