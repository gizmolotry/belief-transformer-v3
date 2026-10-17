import argparse
import json

import numpy as np

from core.ablation_master import (
    run_auto_observer_ablation,
    run_dynamic_tuning_ablation,
    run_metric_tensor_ablation,
)
import run_full_experiment_suite as suite


ARTICLES = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]


def fake_pipeline(articles, *, auto_observers_enabled, dynamic_observer_tuning, seed, pipeline_kwargs):
    del seed, pipeline_kwargs
    offset = 2.0 if auto_observers_enabled else (1.0 if dynamic_observer_tuning else 0.0)
    fused = np.arange(len(articles) * 2).reshape(len(articles), 2) + offset
    result = {
        "features": fused,
        "dirichlet_fused": fused,
        "dirichlet_fused_std": np.full_like(fused, 0.1, dtype=float),
    }
    if auto_observers_enabled:
        result["auto_observers"] = {"pairs": [{}, {}]}
    if dynamic_observer_tuning:
        result.update(
            dynamic_observer_tuning={"article_alphas": [0.2, 0.4, 0.6]},
            topological_crack_density=np.array([0.1, 0.5, 0.9]),
        )
    else:
        result["track3_density_rho"] = np.array([0.2, 0.2, 0.2])
    return result


def test_auto_observer_summary_is_paired_and_json_friendly():
    summary = run_auto_observer_ablation(ARTICLES, pipeline_runner=fake_pipeline)
    assert summary["design"] == "paired_common_seed"
    assert summary["arms"]["endogenous"]["observer_pair_count"] == 2
    assert summary["comparison"]["mean_paired_l2_divergence"] > 0


def test_dynamic_tuning_summary_reports_cracks_and_sparsity():
    summary = run_dynamic_tuning_ablation(ARTICLES, pipeline_runner=fake_pipeline)
    assert summary["arms"]["closed_loop"]["crack_density_method"].startswith("estimate_crack_density")
    assert summary["arms"]["closed_loop"]["observer_weight_sparsity"] is not None
    assert "reconstruction" in summary["arms"]["static"]["observer_weight_sparsity_method"]
    assert summary["comparison"]["sparsity_delta_closed_loop_minus_static"] is not None


def test_metric_arms_share_landmarks_and_read_action_field():
    calls = []

    def fake_action(points, *, config, anchors, targets_by_anchor):
        calls.append((list(anchors), dict(targets_by_anchor), config.traversal_solver))
        action = 4.0 if config.traversal_solver == "fmm" else 1.0
        return {
            "records": [{"action": action, "reached": True, "target_zone": "elastic"}],
            "summary": {"traversal_solvers_used": [config.traversal_solver]},
        }

    summary = run_metric_tensor_ablation(np.eye(3), action_graph_runner=fake_action)
    assert calls[0][:2] == calls[1][:2]
    assert summary["arms"]["riemannian_fmm"]["work_integral"]["values"] == [4.0]
    assert summary["comparison"]["mean_work_delta"] == 3.0


def test_metric_invalid_fraction_uses_unreached_or_nonfinite_union():
    def fake_action(points, *, config, anchors, targets_by_anchor):
        del points, config, anchors, targets_by_anchor
        return {"records": [
            {"action": float("inf"), "reached": False},
            {"action": 1.0, "reached": True},
        ], "summary": {}}

    summary = run_metric_tensor_ablation(np.eye(3), seed=19, action_graph_runner=fake_action)
    assert summary["arms"]["euclidean"]["invalid_or_trapped_fraction"] == 0.5
    assert summary["arms"]["euclidean"]["invalid_or_trapped_definition"] == "path_unreached_or_nonfinite_action"


def test_suite_writer_uses_exact_upgrade_summary_filename(tmp_path, monkeypatch):
    args = argparse.Namespace(
        seeds=[7], limit=3, synthetic=False,
        dynamic_tuning_max_iterations=1, dynamic_tuning_sensitivity=2.0,
        dynamic_tuning_damping=0.5, dynamic_tuning_tolerance=1e-3,
        dynamic_tuning_min_alpha=0.05, dynamic_tuning_max_alpha=20.0,
    )
    monkeypatch.setattr(suite, "_load_upgrade_ablation_articles", lambda _args: ARTICLES)
    expected = {"schema_version": "test", "value": 1.25}
    monkeypatch.setattr("core.ablation_master.run_upgrade_ablations", lambda *a, **k: expected)
    assert suite._run_and_save_upgrade_ablations(args, tmp_path) == expected
    output = tmp_path / "upgrade_ablation_summary.json"
    assert json.loads(output.read_text(encoding="utf-8")) == expected
