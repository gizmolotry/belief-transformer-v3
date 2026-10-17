from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import core.full_spd_simulation as simulation
import core.full_spd_cell_builder as cell_builder
from core.full_spd_experiment import CalibrationError
from core.full_spd_geometry import NEAR_TIE_TOLERANCE_RULE


def test_frozen_contract_has_required_grid_and_at_least_forty_records_per_kind():
    config = simulation.frozen_configuration()
    assert config["banks"] == ["Q0", "Q1", "Q2"]
    assert config["seeds"] == [42, 420, 4200]
    assert config["folds"] == 5
    assert config["default_records_per_kind"] >= 40
    assert config["near_tie_tolerance_rule"] == NEAR_TIE_TOLERANCE_RULE
    assert set(config["scenario_parameters"]) == {
        "no_effect",
        "planted",
        "collapse",
        "inflation",
    }
    assert config["scenario_parameters"] == {
        "no_effect": {
            "stable_scale": 1.0,
            "nuisance_scale": 1.45,
            "shared_gradient": 1e-6,
            "nuisance_gradient": 0.0,
            "radial_jitter": 0.0,
        },
        "planted": {
            "stable_scale": 1.0,
            "nuisance_scale": 1.45,
            "shared_gradient": 6.0,
            "nuisance_gradient": 0.20,
            "radial_jitter": 0.0,
        },
        "collapse": {
            "stable_scale": 1.0,
            "nuisance_scale": 0.0,
            "shared_gradient": 1e-4,
            "nuisance_gradient": 0.0,
            "radial_jitter": 1e-4,
        },
        "inflation": {
            "stable_scale": 1.0,
            "nuisance_scale": 1.45,
            "shared_gradient": 80.0,
            "nuisance_gradient": 0.20,
            "radial_jitter": 0.0,
        },
    }


def test_simulator_and_builder_tolerances_are_byte_identical_across_scales():
    for scale in (1e-200, 1e-12, 1.0, 1e150):
        eccentricities = np.asarray([0.5, 1.0, 3.0, 8.0], dtype=np.float64) * scale
        simulator_value = simulation.near_tie_tolerance(eccentricities)
        _, builder_value = cell_builder._metrics(
            eccentricities,
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.25, 0.75]),
            np.asarray([0.25, 0.75]),
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 1.0]),
        )
        assert np.asarray(simulator_value, dtype=np.float64).tobytes() == np.asarray(
            builder_value, dtype=np.float64
        ).tobytes()


def test_simulation_calls_production_geometry_and_arm_statistic(monkeypatch):
    calls = {name: 0 for name in ("support", "edges", "dijkstra", "eccentricity", "statistics")}
    originals = {
        "support": simulation.build_synthetic_complete_graph_support,
        "edges": simulation.edge_costs,
        "dijkstra": simulation.all_pairs_dijkstra,
        "eccentricity": simulation.article_eccentricity,
        "statistics": simulation.arm_statistics,
    }

    def wrap(label):
        def wrapped(*args, **kwargs):
            calls[label] += 1
            return originals[label](*args, **kwargs)

        return wrapped

    monkeypatch.setattr(simulation, "build_synthetic_complete_graph_support", wrap("support"))
    monkeypatch.setattr(simulation, "edge_costs", wrap("edges"))
    monkeypatch.setattr(simulation, "all_pairs_dijkstra", wrap("dijkstra"))
    monkeypatch.setattr(simulation, "article_eccentricity", wrap("eccentricity"))
    monkeypatch.setattr(simulation, "arm_statistics", wrap("statistics"))

    record = simulation._simulate_one("planted", 0)
    assert record.kind == "planted"
    assert calls["support"] == 5 * 3 * 3
    assert calls["edges"] == 5 * 3 * 3 * 7
    assert calls["dijkstra"] == 5 * 3 * 3 * 5
    assert calls["eccentricity"] == calls["dijkstra"]
    assert calls["statistics"] == 1


def test_reduced_frozen_records_are_deterministic_and_gates_are_worst_cell_summaries():
    first = simulation.generate_simulation_records(records_per_kind=1)
    second = simulation.generate_simulation_records(records_per_kind=1)
    assert first == second
    by_kind = {record.kind: record for record in first}
    assert by_kind["no_effect"].delta == pytest.approx(0.0, abs=1e-12)
    assert by_kind["planted"].delta > by_kind["no_effect"].delta
    assert by_kind["collapse"].dispersion < by_kind["planted"].dispersion
    assert 0.0 <= by_kind["collapse"].tie_fraction <= 1.0
    assert 0.0 <= by_kind["planted"].tie_fraction <= 1.0
    assert by_kind["inflation"].amplification > by_kind["planted"].amplification


def test_frozen_reduced_calibration_fails_closed_when_margin_has_no_power():
    records = simulation.generate_simulation_records(records_per_kind=2)
    with pytest.raises(CalibrationError, match="practical margin"):
        simulation.calibrate_synthetic_records(records)

    # A caller cannot rescue the frozen failure by relabelling one record.
    malformed = list(records)
    malformed[0] = replace(malformed[0], kind="unknown")
    with pytest.raises(CalibrationError):
        simulation.calibrate_synthetic_records(malformed)


def test_bounded_learned_potential_recovery_passes_frozen_thresholds():
    result = simulation.run_learned_potential_recovery(device="cpu")
    assert result.accepted
    assert result.g_hold >= simulation.RECOVERY_G_HOLD_MIN
    assert result.gradient_cosine_mean >= simulation.RECOVERY_GRADIENT_COSINE_MIN
    assert result.gradient_cosine_min > 0.0
