"""Deterministic synthetic calibration for the full-SPD necessity gate.

This module is intentionally isolated from real corpus and model artifacts.  It
constructs a small, frozen synthetic cohort and obtains every eccentricity from
the production support, edge-cost, Dijkstra, and eccentricity functions before
passing the resulting container to :func:`core.full_spd_experiment.arm_statistics`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.full_spd_experiment import (
    BANKS,
    PRIMARY_COMPARATORS,
    CalibrationError,
    CalibrationResult,
    EccentricityContainer,
    FoldAssignment,
    SimulationSummary,
    arm_statistics,
    calibrate_thresholds,
    deterministic_publisher_folds,
)
from core.full_spd_geometry import (
    all_pairs_dijkstra,
    article_eccentricity,
    build_synthetic_complete_graph_support,
    edge_amplification_q95,
    edge_costs,
    eccentricity_dispersion_cv,
    near_tie_fraction,
    near_tie_tolerance,
    NEAR_TIE_TOLERANCE_RULE,
    potential_generalization_diagnostics,
    train_oof_potential,
)


EXPERIMENT_SEEDS = (42, 420, 4200)
SCENARIO_KINDS = ("no_effect", "planted", "collapse", "inflation")
DEFAULT_RECORDS_PER_KIND = 40
N_PUBLISHERS = 10
ARTICLES_PER_PUBLISHER = 5
DIMENSION = 4

# Frozen before any calibration execution.  They are deliberately public so a
# manifest can bind the exact data-generating process.
SCENARIO_PARAMETERS: Mapping[str, Mapping[str, float]] = {
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

# One bounded, independently interpretable learned-potential recovery gate.
RECOVERY_SEED = 20260823
RECOVERY_TRAIN_N = 96
RECOVERY_HELDOUT_N = 32
RECOVERY_G_HOLD_MIN = 0.50
RECOVERY_GRADIENT_COSINE_MIN = 0.80


class FullSPDSimulationError(ValueError):
    """Fail-closed error for malformed or unsuccessful synthetic calibration."""


@dataclass(frozen=True)
class LearnedPotentialRecovery:
    g_train: float
    g_hold: float
    g_gap: float
    gradient_cosine_mean: float
    gradient_cosine_min: float
    accepted: bool
    thresholds: Mapping[str, float]


def frozen_configuration() -> Mapping[str, Any]:
    """Return a JSON-compatible copy of the prespecified simulation contract."""

    return {
        "banks": list(BANKS),
        "seeds": list(EXPERIMENT_SEEDS),
        "folds": 5,
        "publishers": N_PUBLISHERS,
        "articles_per_publisher": ARTICLES_PER_PUBLISHER,
        "dimension": DIMENSION,
        "support": "synthetic_complete",
        "near_tie_tolerance_rule": NEAR_TIE_TOLERANCE_RULE,
        "default_records_per_kind": DEFAULT_RECORDS_PER_KIND,
        "scenario_parameters": {
            kind: dict(SCENARIO_PARAMETERS[kind]) for kind in SCENARIO_KINDS
        },
        "recovery": {
            "seed": RECOVERY_SEED,
            "train_n": RECOVERY_TRAIN_N,
            "heldout_n": RECOVERY_HELDOUT_N,
            "g_hold_min": RECOVERY_G_HOLD_MIN,
            "gradient_cosine_min": RECOVERY_GRADIENT_COSINE_MIN,
        },
    }


def _cohort() -> tuple[tuple[str, ...], dict[str, str], FoldAssignment]:
    ids: list[str] = []
    publisher_by_id: dict[str, str] = {}
    for publisher_index in range(N_PUBLISHERS):
        publisher = f"publisher-{publisher_index:02d}"
        for article_index in range(ARTICLES_PER_PUBLISHER):
            article_id = f"{publisher}-article-{article_index:02d}"
            ids.append(article_id)
            publisher_by_id[article_id] = publisher
    ordered_ids = tuple(ids)
    folds = deterministic_publisher_folds(ordered_ids, publisher_by_id, n_folds=5)
    if any(len(folds.heldout_ids[fold]) != 10 for fold in range(5)):
        raise FullSPDSimulationError("frozen cohort must yield ten articles per fold")
    return ordered_ids, publisher_by_id, folds


def _basis() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    shared = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    nuisance = (
        np.asarray([0.5, 0.5, -0.5, -0.5], dtype=np.float64),
        np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float64),
        np.asarray([0.5, -0.5, -0.5, 0.5], dtype=np.float64),
    )
    return shared, nuisance


def _article_coordinates(article_id: str) -> tuple[int, int]:
    parts = article_id.split("-")
    return int(parts[1]), int(parts[-1])


def _regular_fold_points(ids: Sequence[str], replicate_seed: int) -> np.ndarray:
    """Near-regular polygon used to manufacture a detectable collapse."""

    params = SCENARIO_PARAMETERS["collapse"]
    phase = (replicate_seed % 997) * 1e-7
    points = np.zeros((len(ids), DIMENSION), dtype=np.float64)
    for local, article_id in enumerate(ids):
        publisher, article = _article_coordinates(article_id)
        angle = 2.0 * math.pi * local / len(ids) + phase
        deterministic_jitter = params["radial_jitter"] * (
            0.35 * math.sin((replicate_seed + 1) * (local + 1))
            + 0.15 * (publisher - article)
        )
        radius = 1.0 + deterministic_jitter
        points[local] = [
            radius * math.cos(angle),
            radius * math.sin(angle),
            1e-5 * math.sin(3.0 * angle),
            1e-5 * math.cos(2.0 * angle),
        ]
    return points


def _bank_points(
    ids: Sequence[str], bank: str, kind: str, replicate_seed: int
) -> np.ndarray:
    if bank not in BANKS or kind not in SCENARIO_KINDS:
        raise FullSPDSimulationError("unknown bank or simulation kind")
    if kind == "collapse":
        return _regular_fold_points(ids, replicate_seed + 37 * BANKS.index(bank))

    params = SCENARIO_PARAMETERS[kind]
    shared, nuisances = _basis()
    nuisance = nuisances[BANKS.index(bank)]
    auxiliary = nuisances[(BANKS.index(bank) + 1) % len(nuisances)]
    phase = (replicate_seed % 1009) / 1009.0
    out = np.zeros((len(ids), DIMENSION), dtype=np.float64)
    for row, article_id in enumerate(ids):
        publisher, article = _article_coordinates(article_id)
        stable = (article - 2.0) / 1.35 + 0.09 * (publisher - 4.5)
        if bank == "Q0":
            bank_noise = math.sin(1.35 * stable + 0.43 * publisher + phase)
        elif bank == "Q1":
            bank_noise = math.cos(1.70 * stable - 0.31 * publisher + 1.7 * phase)
        else:
            bank_noise = math.sin(2.15 * stable + 0.19 * publisher - 1.3 * phase)
        tiny = 0.025 * math.sin((row + 1) * (replicate_seed % 17 + 1))
        out[row] = (
            params["stable_scale"] * stable * shared
            + params["nuisance_scale"] * bank_noise * nuisance
            + (0.11 * stable**2 + tiny) * auxiliary
        )
    return out


def _potential_and_gradient(
    points: np.ndarray, bank: str, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    params = SCENARIO_PARAMETERS[kind]
    shared, nuisances = _basis()
    nuisance = nuisances[BANKS.index(bank)]
    shared_projection = points @ shared
    nuisance_projection = points @ nuisance
    a = params["shared_gradient"]
    b = params["nuisance_gradient"]
    potential = a * shared_projection + 0.5 * b * nuisance_projection**2
    gradients = a * shared[None, :] + b * nuisance_projection[:, None] * nuisance[None, :]
    return np.ascontiguousarray(potential), np.ascontiguousarray(gradients)


def _prediction_error(
    values: np.ndarray, *, seed: int, heldout: bool
) -> np.ndarray:
    scale = max(float(np.std(values)), 1e-12)
    positions = np.arange(values.size, dtype=np.float64) + 1.0
    phase = 0.017 * seed + (0.7 if heldout else 0.0)
    return 0.025 * scale * np.sin(positions * 1.73 + phase)


def _simulate_one(kind: str, replicate: int) -> SimulationSummary:
    if kind not in SCENARIO_KINDS:
        raise FullSPDSimulationError(f"unknown scenario: {kind}")
    ordered_ids, publisher_by_id, folds = _cohort()
    replicate_seed = 100_003 + 7_919 * SCENARIO_KINDS.index(kind) + replicate
    values: dict[tuple[int, int, str, str], np.ndarray] = {}
    dispersions: list[float] = []
    tie_fractions: list[float] = []
    generalizations: list[float] = []
    gaps: list[float] = []
    amplifications: list[float] = []

    for fold in range(5):
        heldout_ids = folds.heldout_ids[fold]
        heldout_set = set(heldout_ids)
        train_ids = tuple(article_id for article_id in ordered_ids if article_id not in heldout_set)
        for seed in EXPERIMENT_SEEDS:
            cell_seed = replicate_seed + seed + 101 * fold
            for bank in BANKS:
                x_hold = _bank_points(heldout_ids, bank, kind, cell_seed)
                x_train = _bank_points(train_ids, bank, kind, cell_seed)
                y_hold, gradients = _potential_and_gradient(x_hold, bank, kind)
                y_train, _ = _potential_and_gradient(x_train, bank, kind)
                train_predictions = y_train + _prediction_error(y_train, seed=cell_seed, heldout=False)
                heldout_predictions = y_hold + _prediction_error(y_hold, seed=cell_seed, heldout=True)
                diagnostics = potential_generalization_diagnostics(
                    y_train, train_predictions, y_hold, heldout_predictions
                )
                generalizations.append(diagnostics.g_hold)
                gaps.append(diagnostics.g_gap)

                graph = build_synthetic_complete_graph_support(x_hold, heldout_ids)
                identity_weights = edge_costs(x_hold, gradients, graph, arm="identity")
                full_weights = edge_costs(x_hold, gradients, graph, arm="full")
                amplifications.append(edge_amplification_q95(full_weights, identity_weights))
                for arm in ("full", *PRIMARY_COMPARATORS):
                    weights = edge_costs(
                        x_hold,
                        gradients,
                        graph,
                        arm=arm,
                        yhat=heldout_predictions if arm == "oof_conformal" else None,
                    )
                    eccentricity = article_eccentricity(all_pairs_dijkstra(graph, weights))
                    values[(fold, seed, bank, arm)] = eccentricity
                    dispersions.append(eccentricity_dispersion_cv(eccentricity))
                    tie_fractions.append(
                        near_tie_fraction(
                            eccentricity,
                            tolerance=near_tie_tolerance(eccentricity),
                        )
                    )

    container = EccentricityContainer(
        ordered_ids=ordered_ids,
        publisher_by_id=publisher_by_id,
        folds=folds,
        seeds=EXPERIMENT_SEEDS,
        arms=("full", *PRIMARY_COMPARATORS),
        values=values,
    )
    delta = arm_statistics(container).delta
    summary = SimulationSummary(
        kind=kind,
        dispersion=float(min(dispersions)),
        tie_fraction=float(max(tie_fractions)),
        generalization=float(min(generalizations)),
        generalization_gap=float(max(gaps)),
        amplification=float(max(amplifications)),
        delta=float(delta),
    )
    if not all(math.isfinite(float(value)) for value in asdict(summary).values() if not isinstance(value, str)):
        raise FullSPDSimulationError("simulation produced a nonfinite summary")
    return summary


def generate_simulation_records(
    *, records_per_kind: int = DEFAULT_RECORDS_PER_KIND
) -> tuple[SimulationSummary, ...]:
    """Generate the complete frozen scenario family without reading real data."""

    if not isinstance(records_per_kind, int) or records_per_kind <= 0:
        raise FullSPDSimulationError("records_per_kind must be a positive integer")
    return tuple(
        _simulate_one(kind, replicate)
        for kind in SCENARIO_KINDS
        for replicate in range(records_per_kind)
    )


def calibrate_synthetic_records(
    records: Sequence[SimulationSummary],
) -> CalibrationResult:
    """Calibrate with the engine and independently enforce the frozen rates."""

    result = calibrate_thresholds(records)
    if result.catch_rate < 0.95 or result.planted_retention < 0.95:
        raise CalibrationError("synthetic gates miss the 95% catch/retention contract")
    if result.type_i_error > 0.05 or result.power < 0.80:
        raise CalibrationError("synthetic margin misses the type-I/power contract")
    return result


def _row_cosines(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", actual, expected)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def run_learned_potential_recovery(*, device: Any = "cpu") -> LearnedPotentialRecovery:
    """Run one bounded recovery check against a known smooth linear potential."""

    rng = np.random.default_rng(RECOVERY_SEED)
    train_points = rng.uniform(-1.0, 1.0, size=(RECOVERY_TRAIN_N, DIMENSION))
    heldout_points = rng.uniform(-0.9, 0.9, size=(RECOVERY_HELDOUT_N, DIMENSION))
    direction = np.asarray([0.55, -0.35, 0.65, 0.40], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    train_targets = train_points @ direction
    heldout_targets = heldout_points @ direction
    q_hash = hashlib.sha256(b"full-spd-synthetic-recovery-q0").hexdigest()
    learned = train_oof_potential(
        train_points,
        train_targets,
        heldout_points,
        train_ids=[f"recovery-train-{index:03d}" for index in range(RECOVERY_TRAIN_N)],
        heldout_ids=[f"recovery-heldout-{index:03d}" for index in range(RECOVERY_HELDOUT_N)],
        q_bank_hash=q_hash,
        seed=RECOVERY_SEED,
        git_commit="synthetic-calibration",
        target_source_kind="synthetic",
        device=device,
    )
    diagnostics = potential_generalization_diagnostics(
        train_targets, learned.train_predictions, heldout_targets, learned.predictions
    )
    expected_gradient = np.broadcast_to(
        direction / learned.target_std, learned.node_gradients.shape
    )
    cosines = _row_cosines(learned.node_gradients, expected_gradient)
    cosine_mean = float(np.mean(cosines))
    cosine_min = float(np.min(cosines))
    accepted = bool(
        diagnostics.g_hold >= RECOVERY_G_HOLD_MIN
        and cosine_mean >= RECOVERY_GRADIENT_COSINE_MIN
    )
    result = LearnedPotentialRecovery(
        g_train=float(diagnostics.g_train),
        g_hold=float(diagnostics.g_hold),
        g_gap=float(diagnostics.g_gap),
        gradient_cosine_mean=cosine_mean,
        gradient_cosine_min=cosine_min,
        accepted=accepted,
        thresholds={
            "g_hold_min": RECOVERY_G_HOLD_MIN,
            "gradient_cosine_mean_min": RECOVERY_GRADIENT_COSINE_MIN,
        },
    )
    if not result.accepted:
        raise CalibrationError(
            "learned-potential recovery failed frozen held-out or gradient-cosine threshold"
        )
    return result
