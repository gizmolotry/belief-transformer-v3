"""Pure statistical engine for the preregistered full-SPD necessity test.

This module deliberately consumes cached arrays only.  It does not perform
model inference, load a corpus, or expose a command-line interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

BANKS = ("Q0", "Q1", "Q2")
PRIMARY_COMPARATORS = ("identity", "oof_conformal", "scalar_trace", "diagonal")
NEGATIVE_CONTROLS = ("article_shuffled", "observer_shuffled", "spectrum_matched_random")
DEFAULT_REPLICATES = 10_000


class FullSPDExperimentError(ValueError):
    """Base fail-closed error for the pure experiment engine."""


class ManifestValidationError(FullSPDExperimentError):
    pass


class ContainerValidationError(FullSPDExperimentError):
    pass


class RankingCollapseError(FullSPDExperimentError):
    pass


class CalibrationError(FullSPDExperimentError):
    pass


class Decision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


@dataclass(frozen=True)
class BankArtifact:
    bank: str
    prompt_hash: str
    raw_array_hash: str
    ordered_ids: tuple[str, ...]


@dataclass(frozen=True)
class BankManifest:
    artifacts: Mapping[str, BankArtifact]
    publisher_by_id: Mapping[str, str]


@dataclass(frozen=True)
class FoldAssignment:
    fold_for_publisher: Mapping[str, int]
    heldout_ids: Mapping[int, tuple[str, ...]]


@dataclass(frozen=True)
class EccentricityContainer:
    """Cached eccentricities keyed by ``(fold, seed, bank, arm)``."""

    ordered_ids: tuple[str, ...]
    publisher_by_id: Mapping[str, str]
    folds: FoldAssignment
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    values: Mapping[tuple[int, int, str, str], np.ndarray]


@dataclass(frozen=True)
class ArmStatistics:
    fold_w: Mapping[tuple[int, str], float]
    t_by_arm: Mapping[str, float]
    strongest_comparator: str
    delta: float


@dataclass(frozen=True)
class PermutationResult:
    observed: ArmStatistics
    null_deltas: np.ndarray
    p_value: float
    mappings: tuple[Mapping[tuple[int, str], np.ndarray], ...]


@dataclass(frozen=True)
class BootstrapResult:
    observed: ArmStatistics
    deltas: np.ndarray
    lower_bound: float
    sampled_indices: tuple[Mapping[int, np.ndarray], ...]


@dataclass(frozen=True)
class ContractInference:
    """All seven contractual comparisons from one shared null stream."""

    observed: Mapping[str, ArmStatistics]
    permutation_deltas: Mapping[str, np.ndarray]
    bootstrap_deltas: Mapping[str, np.ndarray]
    permutation_seed: int
    bootstrap_seed: int
    n_replicates: int


@dataclass(frozen=True)
class GateThresholds:
    tau_disp: float
    tau_tie: float
    tau_gen: float
    tau_gap: float
    tau_amp: float
    delta_min: float


@dataclass(frozen=True)
class GateMetrics:
    dispersion: float
    tie_fraction: float
    generalization: float
    generalization_gap: float
    amplification: float


@dataclass(frozen=True)
class GateEvaluation:
    valid: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class LabeledGateMetrics:
    """One gate record with enough labels to audit its experimental cell."""

    fold: int
    seed: int
    bank: str
    arm: str
    metrics: GateMetrics


@dataclass(frozen=True)
class NegativeControlResult:
    name: str
    raw_p: float
    bootstrap_lower: float
    candidate_decision: Decision


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SimulationSummary:
    kind: str  # no_effect, planted, collapse, or inflation
    dispersion: float
    tie_fraction: float
    generalization: float
    generalization_gap: float
    amplification: float
    delta: float


@dataclass(frozen=True)
class CalibrationResult:
    thresholds: GateThresholds
    catch_rate: float
    planted_retention: float
    type_i_error: float
    power: float


def _sha(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def validate_bank_manifest(manifest: BankManifest) -> tuple[str, ...]:
    """Fail closed on anything other than three bound, distinct Q artifacts."""
    if set(manifest.artifacts) != set(BANKS):
        raise ManifestValidationError("manifest must contain exactly Q0, Q1, and Q2")
    first: tuple[str, ...] | None = None
    raw_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    for bank in BANKS:
        artifact = manifest.artifacts[bank]
        if artifact.bank != bank or not _sha(artifact.prompt_hash) or not _sha(artifact.raw_array_hash):
            raise ManifestValidationError("bank artifact has invalid identity or SHA-256")
        ids = tuple(artifact.ordered_ids)
        if len(ids) != 320 or len(set(ids)) != 320:
            raise ManifestValidationError("real gate requires exactly 320 unique ordered IDs")
        if first is None:
            first = ids
        elif ids != first:
            raise ManifestValidationError("all banks must have the same ordered cohort IDs")
        raw_hashes.add(artifact.raw_array_hash)
        prompt_hashes.add(artifact.prompt_hash)
    if len(raw_hashes) != 3:
        raise ManifestValidationError("Q0/Q1/Q2 must have distinct raw-array hashes")
    if len(prompt_hashes) != 3:
        raise ManifestValidationError("Q0/Q1/Q2 must have distinct prompt hashes")
    assert first is not None
    publishers = [manifest.publisher_by_id.get(article_id) for article_id in first]
    if any(not publisher for publisher in publishers):
        raise ManifestValidationError("every cohort article needs a publisher")
    counts: dict[str, int] = {}
    for publisher in publishers:
        counts[str(publisher)] = counts.get(str(publisher), 0) + 1
    if len(counts) != 16 or set(counts.values()) != {20}:
        raise ManifestValidationError("real gate requires exactly 16 publishers x 20 articles")
    return first


def deterministic_publisher_folds(
    ordered_ids: Sequence[str], publisher_by_id: Mapping[str, str], n_folds: int = 5
) -> FoldAssignment:
    if n_folds != 5:
        raise ManifestValidationError("the preregistered design has exactly five folds")
    ids = tuple(ordered_ids)
    publishers = sorted({publisher_by_id.get(i, "") for i in ids})
    if "" in publishers or len(publishers) < n_folds:
        raise ManifestValidationError("publisher map is incomplete")
    # Stable hash ordering prevents filesystem/locale ordering from influencing folds.
    publishers.sort(key=lambda p: (hashlib.sha256(p.encode("utf-8")).hexdigest(), p))
    fold_for = {publisher: index % n_folds for index, publisher in enumerate(publishers)}
    held = {fold: tuple(i for i in ids if fold_for[publisher_by_id[i]] == fold) for fold in range(n_folds)}
    if any(not held[fold] for fold in held):
        raise ManifestValidationError("every deterministic fold must contain articles")
    return FoldAssignment(fold_for_publisher=fold_for, heldout_ids=held)


def validate_fold_assignment(
    assignment: FoldAssignment, ordered_ids: Sequence[str], publisher_by_id: Mapping[str, str]
) -> None:
    ids = tuple(ordered_ids)
    seen: list[str] = []
    for fold in range(5):
        if fold not in assignment.heldout_ids:
            raise ContainerValidationError("missing fold")
        fold_ids = tuple(assignment.heldout_ids[fold])
        if not fold_ids:
            raise ContainerValidationError("empty held-out fold")
        for article_id in fold_ids:
            if article_id not in publisher_by_id:
                raise ContainerValidationError("unknown article in fold")
            publisher = publisher_by_id[article_id]
            if assignment.fold_for_publisher.get(publisher) != fold:
                raise ContainerValidationError("publisher is split across folds")
        seen.extend(fold_ids)
    if tuple(sorted(seen)) != tuple(sorted(ids)) or len(set(seen)) != len(ids):
        raise ContainerValidationError("folds must partition the ordered cohort exactly once")


def validate_eccentricity_container(container: EccentricityContainer) -> None:
    ids = tuple(container.ordered_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ContainerValidationError("container ordered IDs must be unique and nonempty")
    if not container.seeds or len(set(container.seeds)) != len(container.seeds):
        raise ContainerValidationError("container must have unique seeds")
    if "full" not in container.arms or any(a not in container.arms for a in PRIMARY_COMPARATORS):
        raise ContainerValidationError("container lacks full or a required comparator")
    validate_fold_assignment(container.folds, ids, container.publisher_by_id)
    for fold, seed, bank, arm in product(range(5), container.seeds, BANKS, container.arms):
        key = (fold, seed, bank, arm)
        if key not in container.values:
            raise ContainerValidationError(f"missing eccentricity cell {key}")
        value = np.asarray(container.values[key], dtype=float)
        if value.ndim != 1 or value.size != len(container.folds.heldout_ids[fold]):
            raise ContainerValidationError(f"eccentricity cell shape mismatch {key}")
        if not np.isfinite(value).all():
            raise ContainerValidationError(f"non-finite eccentricities {key}")


def midranks(values: Sequence[float]) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ContainerValidationError("midranks require a finite nonempty vector")
    order = np.argsort(value, kind="mergesort")
    result = np.empty(value.size, dtype=float)
    start = 0
    while start < value.size:
        end = start + 1
        while end < value.size and value[order[end]] == value[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    if np.ptp(result) == 0:
        raise RankingCollapseError("complete ranking collapse")
    return result


def kendalls_w_tie_corrected(rank_matrix: np.ndarray) -> float:
    ranks = np.asarray(rank_matrix, dtype=float)
    if ranks.ndim != 2 or ranks.shape[0] != 3 or ranks.shape[1] < 2:
        raise ContainerValidationError("Kendall W requires Q0/Q1/Q2 rankings")
    if any(np.ptp(row) == 0 for row in ranks):
        raise RankingCollapseError("complete ranking collapse")
    m, n = ranks.shape
    correction = sum(np.sum(counts**3 - counts) for row in ranks for _, counts in [np.unique(row, return_counts=True)])
    sums = ranks.sum(axis=0)
    denominator = m * m * (n**3 - n) - m * correction
    if denominator <= 0:
        raise RankingCollapseError("undefined tie-corrected Kendall W")
    value = 12.0 * float(np.sum((sums - sums.mean()) ** 2)) / float(denominator)
    if not -1e-12 <= value <= 1 + 1e-12:
        raise ContainerValidationError("Kendall W outside [0, 1]")
    return float(np.clip(value, 0.0, 1.0))


def _cell(container: EccentricityContainer, fold: int, seed: int, bank: str, arm: str, indices: np.ndarray | None, mapping: Mapping[tuple[int, str], np.ndarray] | None) -> np.ndarray:
    value = np.asarray(container.values[(fold, seed, bank, arm)], dtype=float)
    if mapping is not None and bank != "Q0":
        value = value[np.asarray(mapping[(fold, bank)], dtype=int)]
    return value if indices is None else value[np.asarray(indices, dtype=int)]


def _arm_statistics_unchecked(
    container: EccentricityContainer,
    *,
    candidate_arm: str = "full",
    comparator_arms: Sequence[str] = PRIMARY_COMPARATORS,
    indices_by_fold: Mapping[int, np.ndarray] | None = None,
    mapping: Mapping[tuple[int, str], np.ndarray] | None = None,
) -> ArmStatistics:
    arms = (candidate_arm, *tuple(comparator_arms))
    if len(set(arms)) != len(arms) or any(arm not in container.arms for arm in arms):
        raise ContainerValidationError("candidate/comparator arm contract is malformed")
    fold_w: dict[tuple[int, str], float] = {}
    weights: dict[int, int] = {}
    for fold in range(5):
        index = None if indices_by_fold is None else np.asarray(indices_by_fold[fold], dtype=int)
        n = len(container.folds.heldout_ids[fold]) if index is None else len(index)
        if n < 2:
            raise RankingCollapseError("bootstrap fold contains fewer than two articles")
        weights[fold] = n
        for arm in arms:
            rows = []
            for bank in BANKS:
                averaged = np.mean([_cell(container, fold, seed, bank, arm, index, mapping) for seed in container.seeds], axis=0)
                rows.append(midranks(averaged))
            fold_w[(fold, arm)] = kendalls_w_tie_corrected(np.asarray(rows))
    total = sum(weights.values())
    t_by_arm = {arm: sum(weights[f] * fold_w[(f, arm)] for f in range(5)) / total for arm in arms}
    strongest = max(comparator_arms, key=lambda arm: (t_by_arm[arm], arm))
    return ArmStatistics(fold_w=fold_w, t_by_arm=t_by_arm, strongest_comparator=strongest, delta=t_by_arm[candidate_arm] - t_by_arm[strongest])


def arm_statistics(
    container: EccentricityContainer,
    *,
    candidate_arm: str = "full",
    comparator_arms: Sequence[str] = PRIMARY_COMPARATORS,
    indices_by_fold: Mapping[int, np.ndarray] | None = None,
    mapping: Mapping[tuple[int, str], np.ndarray] | None = None,
) -> ArmStatistics:
    """Validated public one-off statistic helper."""
    validate_eccentricity_container(container)
    return _arm_statistics_unchecked(
        container, candidate_arm=candidate_arm, comparator_arms=comparator_arms,
        indices_by_fold=indices_by_fold, mapping=mapping,
    )


def _q_within_publisher_permutation_unchecked(
    container: EccentricityContainer, rng: np.random.Generator
) -> Mapping[tuple[int, str], np.ndarray]:
    """Return one Q0-fixed map and independent within-publisher Q1/Q2 maps."""
    result: dict[tuple[int, str], np.ndarray] = {}
    for fold in range(5):
        ids = container.folds.heldout_ids[fold]
        groups: dict[str, list[int]] = {}
        for i, article_id in enumerate(ids):
            groups.setdefault(container.publisher_by_id[article_id], []).append(i)
        result[(fold, "Q0")] = np.arange(len(ids), dtype=int)
        for bank in ("Q1", "Q2"):
            mapping = np.arange(len(ids), dtype=int)
            for positions in groups.values():
                mapping[positions] = rng.permutation(positions)
            result[(fold, bank)] = mapping
    return result


def q_within_publisher_permutation(
    container: EccentricityContainer, rng: np.random.Generator
) -> Mapping[tuple[int, str], np.ndarray]:
    validate_eccentricity_container(container)
    return _q_within_publisher_permutation_unchecked(container, rng)


def permutation_inference(
    container: EccentricityContainer, *, n_replicates: int = DEFAULT_REPLICATES, seed: int = 0,
    candidate_arm: str = "full", comparator_arms: Sequence[str] = PRIMARY_COMPARATORS,
) -> PermutationResult:
    if n_replicates <= 0:
        raise ContainerValidationError("n_replicates must be positive")
    validate_eccentricity_container(container)
    observed = _arm_statistics_unchecked(container, candidate_arm=candidate_arm, comparator_arms=comparator_arms)
    rng = np.random.default_rng(seed)
    null = np.empty(n_replicates, dtype=float)
    maps = []
    for i in range(n_replicates):
        mapping = _q_within_publisher_permutation_unchecked(container, rng)
        maps.append(mapping)
        null[i] = _arm_statistics_unchecked(container, candidate_arm=candidate_arm, comparator_arms=comparator_arms, mapping=mapping).delta
    p = (1.0 + float(np.count_nonzero(null >= observed.delta))) / (n_replicates + 1.0)
    return PermutationResult(observed, null, p, tuple(maps))


def publisher_cluster_bootstrap_indices(container: EccentricityContainer, rng: np.random.Generator) -> Mapping[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for fold in range(5):
        ids = container.folds.heldout_ids[fold]
        groups: dict[str, list[int]] = {}
        for i, article_id in enumerate(ids):
            groups.setdefault(container.publisher_by_id[article_id], []).append(i)
        publishers = tuple(groups)
        selected = rng.choice(publishers, size=len(publishers), replace=True)
        result[fold] = np.asarray([position for p in selected for position in groups[p]], dtype=int)
    return result


def publisher_cluster_bootstrap(
    container: EccentricityContainer, *, n_replicates: int = DEFAULT_REPLICATES, seed: int = 0,
    candidate_arm: str = "full", comparator_arms: Sequence[str] = PRIMARY_COMPARATORS,
) -> BootstrapResult:
    if n_replicates <= 0:
        raise ContainerValidationError("n_replicates must be positive")
    validate_eccentricity_container(container)
    observed = _arm_statistics_unchecked(container, candidate_arm=candidate_arm, comparator_arms=comparator_arms)
    rng = np.random.default_rng(seed)
    samples, deltas = [], np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        indices = publisher_cluster_bootstrap_indices(container, rng)
        samples.append(indices)
        deltas[i] = _arm_statistics_unchecked(container, candidate_arm=candidate_arm, comparator_arms=comparator_arms, indices_by_fold=indices).delta
    return BootstrapResult(observed, deltas, float(np.quantile(deltas, 0.025)), tuple(samples))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = midranks(a), midranks(b)
    value = float(np.corrcoef(ra, rb)[0, 1])
    if not math.isfinite(value):
        raise RankingCollapseError("undefined Spearman stability")
    return value


def pairwise_spearman_deltas(
    container: EccentricityContainer, *, candidate_arm: str = "full", comparator: str | None = None
) -> Mapping[tuple[int, str, str], float]:
    stats = arm_statistics(container, candidate_arm=candidate_arm) if comparator is None else None
    baseline = comparator or stats.strongest_comparator  # type: ignore[union-attr]
    out: dict[tuple[int, str, str], float] = {}
    for seed in container.seeds:
        for left, right in (("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2")):
            full_values, base_values, total = 0.0, 0.0, 0
            for fold in range(5):
                n = len(container.folds.heldout_ids[fold])
                full_values += n * _spearman(_cell(container, fold, seed, left, candidate_arm, None, None), _cell(container, fold, seed, right, candidate_arm, None, None))
                base_values += n * _spearman(_cell(container, fold, seed, left, baseline, None, None), _cell(container, fold, seed, right, baseline, None, None))
                total += n
            out[(seed, left, right)] = full_values / total - base_values / total
    return out


def robustness_fraction(deltas: Mapping[tuple[int, str, str], float]) -> float:
    if not deltas:
        raise ContainerValidationError("no seed x Q-pair robustness cells")
    return float(sum(value > 0.0 for value in deltas.values()) / len(deltas))


def holm_adjust_three(p_values: Mapping[str, float]) -> Mapping[str, float]:
    if set(p_values) != set(NEGATIVE_CONTROLS):
        raise ContainerValidationError("Holm family must contain exactly the three negative controls")
    if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in p_values.values()):
        raise ContainerValidationError("p-values must be finite in [0,1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, (3 - rank) * float(p))
        adjusted[name] = min(1.0, running)
    return adjusted


def negative_candidate_substitution(container: EccentricityContainer, control: str) -> ArmStatistics:
    if control not in NEGATIVE_CONTROLS:
        raise ContainerValidationError("unknown negative control")
    return arm_statistics(container, candidate_arm=control)


def evaluate_gates(metrics: GateMetrics, thresholds: GateThresholds) -> GateEvaluation:
    values = (*metrics.__dict__.values(), *thresholds.__dict__.values())
    if any(not math.isfinite(float(x)) for x in values):
        return GateEvaluation(False, ("non_numeric_gate",))
    failures = []
    if metrics.dispersion < thresholds.tau_disp: failures.append("dispersion")
    if metrics.tie_fraction > thresholds.tau_tie: failures.append("ties")
    if metrics.generalization < thresholds.tau_gen: failures.append("generalization")
    if metrics.generalization_gap > thresholds.tau_gap: failures.append("generalization_gap")
    if metrics.amplification > thresholds.tau_amp: failures.append("amplification")
    return GateEvaluation(not failures, tuple(failures))


def decide(
    *, construction_valid: bool, gate: GateEvaluation, statistics: ArmStatistics, p_value: float,
    bootstrap_lower: float, thresholds: GateThresholds, pairwise_deltas: Mapping[tuple[int, str, str], float],
    negative_controls: Sequence[NegativeControlResult], corpus_control_passed: bool = False,
) -> DecisionResult:
    if not construction_valid or corpus_control_passed:
        return DecisionResult(Decision.INVALID, ("construction_invalid" if not construction_valid else "corpus_control_passed",))
    if not gate.valid:
        return DecisionResult(Decision.INCONCLUSIVE, gate.failures)
    if len(negative_controls) != 3 or {n.name for n in negative_controls} != set(NEGATIVE_CONTROLS):
        return DecisionResult(Decision.INVALID, ("negative_control_family_malformed",))
    adjusted = holm_adjust_three({n.name: n.raw_p for n in negative_controls})
    if any(n.candidate_decision == Decision.PASS for n in negative_controls):
        return DecisionResult(Decision.INVALID, ("negative_control_passed_primary",))
    negative_ok = all(adjusted[n.name] < .05 and n.bootstrap_lower > 0 for n in negative_controls)
    pair_means = _seed_averaged_pairwise_deltas(pairwise_deltas)
    direct = pairwise_deltas.values()
    checks = {
        "strict_permutation": p_value < .05,
        "bootstrap": bootstrap_lower > 0,
        "margin": statistics.delta >= thresholds.delta_min,
        "trace": statistics.t_by_arm["full"] > statistics.t_by_arm["scalar_trace"],
        "diagonal": statistics.t_by_arm["full"] > statistics.t_by_arm["diagonal"],
        "pairwise": set(pair_means) == {("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2")} and all(value > 0 for value in pair_means.values()),
        "robustness": robustness_fraction(pairwise_deltas) >= .80 if direct else False,
        "negative_controls": negative_ok,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return DecisionResult(Decision.PASS if not failures else Decision.FAIL, failures)


def _seed_averaged_pairwise_deltas(
    deltas: Mapping[tuple[int, str, str], float]
) -> Mapping[tuple[str, str], float]:
    """Return the three frozen Q-pair means; individual cells are robustness only."""
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for key, value in deltas.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ContainerValidationError("pairwise delta keys must be (seed, Qleft, Qright)")
        seed, left, right = key
        if not isinstance(seed, int):
            raise ContainerValidationError("pairwise delta seed must be an integer")
        if (left, right) not in {("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2")}:
            raise ContainerValidationError("unexpected Q-pair delta")
        if not math.isfinite(float(value)):
            raise ContainerValidationError("pairwise delta must be finite")
        pair = (left, right)
        if seed in grouped.setdefault(pair, {}):
            raise ContainerValidationError("duplicate pairwise seed")
        grouped[pair][seed] = float(value)
    expected_pairs = {("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2")}
    if set(grouped) != expected_pairs:
        raise ContainerValidationError("pairwise deltas must contain exactly all three Q pairs")
    seed_sets = [set(values) for values in grouped.values()]
    if not seed_sets[0] or any(values != seed_sets[0] for values in seed_sets[1:]):
        raise ContainerValidationError("all Q pairs must contain the same nonempty seed set")
    return {pair: float(np.mean(list(values.values()))) for pair, values in grouped.items()}


def _candidates(values: Sequence[float], direction: str) -> tuple[float, ...]:
    unique = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not unique:
        raise CalibrationError("no finite calibration values")
    # Exact observed values are deterministic and sufficient for strict threshold comparisons.
    return tuple(unique)


def calibrate_thresholds(records: Sequence[SimulationSummary]) -> CalibrationResult:
    no_effect = [r for r in records if r.kind == "no_effect"]
    planted = [r for r in records if r.kind == "planted"]
    bad = [r for r in records if r.kind in {"collapse", "inflation"}]
    if not no_effect or not planted or not bad:
        raise CalibrationError("need no_effect, planted, and collapse/inflation simulations")
    metrics = ("dispersion", "tie_fraction", "generalization", "generalization_gap", "amplification")
    pools = {name: _candidates([getattr(r, name) for r in records], name) for name in metrics}
    best: tuple[float, GateThresholds, float, float] | None = None
    # The pool is deliberately capped to keep calibration deterministic and bounded.
    pools = {name: values if len(values) <= 12 else tuple(np.quantile(values, np.linspace(0, 1, 12))) for name, values in pools.items()}
    for disp, tie, gen, gap, amp in product(pools["dispersion"], pools["tie_fraction"], pools["generalization"], pools["generalization_gap"], pools["amplification"]):
        provisional = GateThresholds(float(disp), float(tie), float(gen), float(gap), float(amp), 0.0)
        catch = float(np.mean([not evaluate_gates(GateMetrics(r.dispersion, r.tie_fraction, r.generalization, r.generalization_gap, r.amplification), provisional).valid for r in bad]))
        retain = float(np.mean([evaluate_gates(GateMetrics(r.dispersion, r.tie_fraction, r.generalization, r.generalization_gap, r.amplification), provisional).valid for r in planted]))
        if catch >= .95 and retain >= .95:
            score = catch + retain
            if best is None or score > best[0]: best = (score, provisional, catch, retain)
    if best is None:
        raise CalibrationError("no gate thresholds meet 95% catch and planted retention")
    maximum = max(0.0, max(r.delta for r in records))
    for margin in np.round(np.arange(0.0, math.ceil(maximum * 100) / 100 + .001, .01), 2):
        type_i = float(np.mean([r.delta >= margin for r in no_effect]))
        power = float(np.mean([r.delta >= margin for r in planted]))
        if type_i <= .05 and power >= .80:
            p = best[1]
            thresholds = GateThresholds(p.tau_disp, p.tau_tie, p.tau_gen, p.tau_gap, p.tau_amp, float(margin))
            return CalibrationResult(thresholds, best[2], best[3], type_i, power)
    raise CalibrationError("no 0.01 practical margin meets type-I/power contract")


# High-level cached-array adapter -------------------------------------------------
#
# A cell is a JSON-compatible object with these required fields:
# ordered_ids, publisher_by_id, fold_for_publisher, fold, seed, bank, arm,
# eccentricities, and gate_metrics.  ``provenance`` is additionally required by
# run_real.  Keeping this schema at the engine boundary prevents the CLI from
# quietly reimplementing statistics or changing the inferential unit.


def _numeric_mapping(value: Any, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ContainerValidationError(f"{label} must be an object")
    expected = {"dispersion", "tie_fraction", "generalization", "generalization_gap", "amplification"}
    if set(value) != expected:
        raise ContainerValidationError(f"{label} must contain exactly the calibrated gate metrics")
    try:
        converted = {key: float(value[key]) for key in expected}
    except (TypeError, ValueError) as exc:
        raise ContainerValidationError(f"{label} values must be numeric") from exc
    if any(not math.isfinite(number) for number in converted.values()):
        raise ContainerValidationError(f"{label} values must be finite")
    return converted


def _thresholds_from_mapping(value: Any) -> GateThresholds:
    """Parse only fully calibrated numeric thresholds; no defaults exist."""
    if not isinstance(value, Mapping):
        raise ContainerValidationError("calibrated thresholds must be an object")
    source = value.get("calibration", value.get("thresholds", value))
    if not isinstance(source, Mapping):
        raise ContainerValidationError("calibrated thresholds must be an object")
    expected = {"tau_disp", "tau_tie", "tau_gen", "tau_gap", "tau_amp", "delta_min"}
    if set(source) != expected:
        raise ContainerValidationError("calibrated thresholds must contain exactly the six frozen keys")
    try:
        thresholds = GateThresholds(**{key: float(source[key]) for key in expected})
    except (TypeError, ValueError) as exc:
        raise ContainerValidationError("calibrated thresholds must be numeric") from exc
    if any(not math.isfinite(float(number)) for number in thresholds.__dict__.values()):
        raise ContainerValidationError("calibrated thresholds must be finite")
    return thresholds


def _jsonable_statistics(statistics: ArmStatistics) -> Mapping[str, Any]:
    return {
        "fold_w": {f"{fold}:{arm}": float(value) for (fold, arm), value in statistics.fold_w.items()},
        "t_by_arm": {arm: float(value) for arm, value in statistics.t_by_arm.items()},
        "strongest_comparator": statistics.strongest_comparator,
        "delta": float(statistics.delta),
    }


def _build_cached_container(
    cells: Sequence[Mapping[str, Any]], *, require_real_seeds: bool
) -> tuple[EccentricityContainer, list[LabeledGateMetrics], bool, bool, tuple[str, ...]]:
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)) or not cells:
        raise ContainerValidationError("cached evaluation needs a nonempty sequence of cells")
    first = cells[0]
    if not isinstance(first, Mapping):
        raise ContainerValidationError("each cached cell must be an object")
    try:
        ordered_ids = tuple(str(value) for value in first["ordered_ids"])
        publisher_by_id = {str(key): str(value) for key, value in first["publisher_by_id"].items()}
        fold_for_publisher = {str(key): int(value) for key, value in first["fold_for_publisher"].items()}
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ContainerValidationError("cached cells need ordered_ids, publisher_by_id, and fold_for_publisher") from exc
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ContainerValidationError("cached ordered IDs must be unique")
    if set(publisher_by_id) != set(ordered_ids):
        raise ContainerValidationError("publisher_by_id must cover exactly the ordered cohort")
    if set(fold_for_publisher.values()) - set(range(5)):
        raise ContainerValidationError("fold_for_publisher must use only folds 0 through 4")
    heldout_ids = {
        fold: tuple(article_id for article_id in ordered_ids if fold_for_publisher[publisher_by_id[article_id]] == fold)
        for fold in range(5)
    }
    assignment = FoldAssignment(fold_for_publisher=fold_for_publisher, heldout_ids=heldout_ids)
    values: dict[tuple[int, int, str, str], np.ndarray] = {}
    metric_cells: list[LabeledGateMetrics] = []
    seeds: set[int] = set()
    cell_keys: set[tuple[int, int, str]] = set()
    construction_valid = True
    corpus_control_passed = False
    connectivity_failures: set[str] = set()
    foundation = (ordered_ids, publisher_by_id, fold_for_publisher)
    required_gate_arms = {"full", *PRIMARY_COMPARATORS}
    required_arms = {"full", *PRIMARY_COMPARATORS, *NEGATIVE_CONTROLS}
    for raw in cells:
        if not isinstance(raw, Mapping):
            raise ContainerValidationError("each cached cell must be an object")
        try:
            candidate_foundation = (
                tuple(str(value) for value in raw["ordered_ids"]),
                {str(key): str(value) for key, value in raw["publisher_by_id"].items()},
                {str(key): int(value) for key, value in raw["fold_for_publisher"].items()},
            )
            fold, seed, bank = int(raw["fold"]), int(raw["seed"]), str(raw["bank"])
            eccentricities = raw["eccentricities"]
            gate_mapping = raw["gate_metrics"]
            connected = raw["connected"]
            cell_construction_valid = raw["construction_valid"]
            provenance_valid = raw["provenance_valid"]
            cell_corpus_control_passed = raw["corpus_control_passed"]
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ContainerValidationError("cached cell is missing a required field") from exc
        if candidate_foundation != foundation:
            raise ContainerValidationError("all cached cells must bind the same cohort and fold assignment")
        if fold not in range(5) or bank not in BANKS:
            raise ContainerValidationError("cached cell has an invalid fold or Q bank")
        cell_key = (fold, seed, bank)
        if cell_key in cell_keys:
            raise ContainerValidationError(f"duplicate cached cell {cell_key}")
        if not isinstance(eccentricities, Mapping) or set(eccentricities) != required_arms:
            raise ContainerValidationError("each cell eccentricities must contain exactly the eight frozen arms")
        if not isinstance(gate_mapping, Mapping) or set(gate_mapping) != required_gate_arms:
            raise ContainerValidationError("gate_metrics must contain exactly full and four required comparators")
        if not isinstance(connected, Mapping) or set(connected) != required_gate_arms or not all(isinstance(value, bool) for value in connected.values()):
            raise ContainerValidationError("connected must contain boolean entries for exactly full and four required comparators")
        if not all(isinstance(value, bool) for value in (cell_construction_valid, provenance_valid, cell_corpus_control_passed)):
            raise ContainerValidationError("construction_valid, provenance_valid, and corpus_control_passed must be booleans")
        for arm, raw_eccentricity in eccentricities.items():
            eccentricity = np.asarray(raw_eccentricity, dtype=float)
            key = (fold, seed, bank, arm)
            if eccentricity.ndim != 1 or eccentricity.size != len(heldout_ids[fold]) or not np.isfinite(eccentricity).all():
                raise ContainerValidationError(f"cached eccentricity shape/value mismatch {key}")
            values[key] = eccentricity
        for arm in sorted(required_gate_arms):
            metric = _numeric_mapping(gate_mapping[arm], f"gate_metrics.{arm}")
            metric_cells.append(LabeledGateMetrics(fold, seed, bank, arm, GateMetrics(**metric)))
            if not connected[arm]:
                connectivity_failures.add(f"{arm}:connectivity")
        construction_valid = construction_valid and cell_construction_valid and provenance_valid
        corpus_control_passed = corpus_control_passed or cell_corpus_control_passed
        cell_keys.add(cell_key)
        seeds.add(seed)
    if require_real_seeds and tuple(sorted(seeds)) != (42, 420, 4200):
        raise ContainerValidationError("real endpoint requires exactly seeds 42, 420, and 4200")
    expected_cells = {(fold, seed, bank) for fold in range(5) for seed in seeds for bank in BANKS}
    if cell_keys != expected_cells:
        raise ContainerValidationError("cached cells must form a complete 5 x Q0/Q1/Q2 x seed grid")
    if require_real_seeds and (len(cells) != 45 or len(cell_keys) != 45):
        raise ContainerValidationError("real endpoint requires exactly 45 bank x fold x seed cells")
    container = EccentricityContainer(ordered_ids, publisher_by_id, assignment, tuple(sorted(seeds)), tuple(sorted(required_arms)), values)
    validate_eccentricity_container(container)
    return container, metric_cells, construction_valid, corpus_control_passed, tuple(sorted(connectivity_failures))


def _combined_gate(
    metrics: Sequence[LabeledGateMetrics],
    thresholds: GateThresholds,
    connectivity_failures: Sequence[str] = (),
) -> GateEvaluation:
    """Evaluate arm-wide rank safeguards and full-only potential safeguards.

    Dispersion and tie thresholds protect every required ranking arm.  The
    generalization, train/holdout-gap, and amplification thresholds diagnose the
    single shared learned potential and therefore apply only to ``full``.
    """
    if not metrics:
        return GateEvaluation(False, ("missing_gate_metrics",))
    failures: set[str] = set(connectivity_failures)
    threshold_values = tuple(float(value) for value in thresholds.__dict__.values())
    if any(not math.isfinite(value) for value in threshold_values):
        return GateEvaluation(False, ("non_numeric_gate",))
    for record in metrics:
        arm, metric = record.arm, record.metrics
        if arm not in {"full", *PRIMARY_COMPARATORS}:
            raise ContainerValidationError("only labelled required arms may enter primary gate evaluation")
        if metric.dispersion < thresholds.tau_disp:
            failures.add(f"{arm}:dispersion")
        if metric.tie_fraction > thresholds.tau_tie:
            failures.add(f"{arm}:ties")
        if arm == "full":
            if metric.generalization < thresholds.tau_gen:
                failures.add("full:generalization")
            if metric.generalization_gap > thresholds.tau_gap:
                failures.add("full:generalization_gap")
            if metric.amplification > thresholds.tau_amp:
                failures.add("full:amplification")
    return GateEvaluation(not failures, tuple(sorted(failures)))


def _jsonable_gate_records(metrics: Sequence[LabeledGateMetrics]) -> list[Mapping[str, Any]]:
    """Keep every labelled input metric visible in the evaluation artifact."""
    return [
        {
            "fold": record.fold,
            "seed": record.seed,
            "bank": record.bank,
            "arm": record.arm,
            **{name: float(value) for name, value in record.metrics.__dict__.items()},
        }
        for record in metrics
    ]


def _comparison_specs() -> Mapping[str, tuple[str, tuple[str, ...]]]:
    """The seven prespecified comparisons; no post-hoc arm selection is possible."""
    specs: dict[str, tuple[str, tuple[str, ...]]] = {"primary": ("full", PRIMARY_COMPARATORS)}
    for control in NEGATIVE_CONTROLS:
        specs[f"direct:{control}"] = ("full", (control,))
        specs[f"candidate:{control}"] = (control, PRIMARY_COMPARATORS)
    return specs


def _contract_inference(
    container: EccentricityContainer, *, n_replicates: int, permutation_seed: int, bootstrap_seed: int
) -> ContractInference:
    """Compute all comparisons with one mapping/sample per replicate.

    The maps and bootstrap samples are intentionally discarded immediately.  The
    returned statistic arrays plus seeds/count are sufficient to reproduce them.
    """
    if not isinstance(n_replicates, int) or n_replicates <= 0:
        raise ContainerValidationError("n_replicates must be a positive integer")
    validate_eccentricity_container(container)
    specs = _comparison_specs()
    observed = {
        name: _arm_statistics_unchecked(container, candidate_arm=candidate, comparator_arms=comparators)
        for name, (candidate, comparators) in specs.items()
    }
    permutation_deltas = {name: np.empty(n_replicates, dtype=float) for name in specs}
    bootstrap_deltas = {name: np.empty(n_replicates, dtype=float) for name in specs}
    permutation_rng = np.random.default_rng(permutation_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    for replicate in range(n_replicates):
        mapping = _q_within_publisher_permutation_unchecked(container, permutation_rng)
        for name, (candidate, comparators) in specs.items():
            permutation_deltas[name][replicate] = _arm_statistics_unchecked(
                container, candidate_arm=candidate, comparator_arms=comparators, mapping=mapping
            ).delta
        indices = publisher_cluster_bootstrap_indices(container, bootstrap_rng)
        for name, (candidate, comparators) in specs.items():
            bootstrap_deltas[name][replicate] = _arm_statistics_unchecked(
                container, candidate_arm=candidate, comparator_arms=comparators, indices_by_fold=indices
            ).delta
    return ContractInference(observed, permutation_deltas, bootstrap_deltas, permutation_seed, bootstrap_seed, n_replicates)


def _one_sided_p(null: np.ndarray, observed: float) -> float:
    return (1.0 + float(np.count_nonzero(null >= observed))) / (len(null) + 1.0)


def _candidate_core_decision(
    inference: ContractInference, candidate: str, thresholds: GateThresholds, gate: GateEvaluation,
    container: EccentricityContainer,
) -> tuple[Decision, Mapping[tuple[int, str, str], float]]:
    """Evaluate the substituted candidate against primary comparators only."""
    if not gate.valid:
        return Decision.INCONCLUSIVE, {}
    name = f"candidate:{candidate}"
    statistics = inference.observed[name]
    p_value = _one_sided_p(inference.permutation_deltas[name], statistics.delta)
    lower = float(np.quantile(inference.bootstrap_deltas[name], .025))
    pairwise = pairwise_spearman_deltas(container, candidate_arm=candidate, comparator=statistics.strongest_comparator)
    means = _seed_averaged_pairwise_deltas(pairwise)
    passes = (
        p_value < .05 and lower > 0 and statistics.delta >= thresholds.delta_min
        and statistics.t_by_arm[candidate] > statistics.t_by_arm["scalar_trace"]
        and statistics.t_by_arm[candidate] > statistics.t_by_arm["diagonal"]
        and set(means) == {("Q0", "Q1"), ("Q0", "Q2"), ("Q1", "Q2")}
        and all(value > 0 for value in means.values()) and robustness_fraction(pairwise) >= .80
    )
    return (Decision.PASS if passes else Decision.FAIL), pairwise


def _evaluate_container(
    container: EccentricityContainer,
    metrics: Sequence[LabeledGateMetrics],
    thresholds: GateThresholds,
    *,
    n_replicates: int,
    permutation_seed: int,
    bootstrap_seed: int,
    construction_valid: bool,
    corpus_control_passed: bool,
    connectivity_failures: Sequence[str],
) -> Mapping[str, Any]:
    gate = _combined_gate(metrics, thresholds, connectivity_failures)
    inference = _contract_inference(
        container, n_replicates=n_replicates, permutation_seed=permutation_seed, bootstrap_seed=bootstrap_seed
    )
    primary = inference.observed["primary"]
    pairwise = pairwise_spearman_deltas(container, comparator=primary.strongest_comparator)
    controls: list[NegativeControlResult] = []
    control_payload: dict[str, Any] = {}
    for control in NEGATIVE_CONTROLS:
        direct_name, candidate_name = f"direct:{control}", f"candidate:{control}"
        direct = inference.observed[direct_name]
        direct_p = _one_sided_p(inference.permutation_deltas[direct_name], direct.delta)
        direct_lower = float(np.quantile(inference.bootstrap_deltas[direct_name], .025))
        candidate_decision, candidate_pairwise = _candidate_core_decision(inference, control, thresholds, gate, container)
        candidate = inference.observed[candidate_name]
        candidate_p = _one_sided_p(inference.permutation_deltas[candidate_name], candidate.delta)
        candidate_lower = float(np.quantile(inference.bootstrap_deltas[candidate_name], .025))
        controls.append(NegativeControlResult(control, direct_p, direct_lower, candidate_decision))
        control_payload[control] = {
            "direct_delta": float(direct.delta),
            "raw_p": float(direct_p),
            "bootstrap_lower": float(direct_lower),
            "candidate_decision": candidate_decision.value,
            "direct_permutation_deltas": inference.permutation_deltas[direct_name].tolist(),
            "direct_bootstrap_deltas": inference.bootstrap_deltas[direct_name].tolist(),
            "candidate": {
                "statistics": _jsonable_statistics(candidate),
                "permutation_p": float(candidate_p),
                "bootstrap_lower": float(candidate_lower),
                "permutation_deltas": inference.permutation_deltas[candidate_name].tolist(),
                "bootstrap_deltas": inference.bootstrap_deltas[candidate_name].tolist(),
                "pairwise_spearman_deltas": {f"{seed}:{left}:{right}": float(value) for (seed, left, right), value in candidate_pairwise.items()},
            },
        }
    adjusted = holm_adjust_three({item.name: item.raw_p for item in controls})
    for control in NEGATIVE_CONTROLS:
        control_payload[control]["holm_adjusted_p"] = float(adjusted[control])
    decision = decide(
        construction_valid=construction_valid,
        gate=gate,
        statistics=primary,
        p_value=_one_sided_p(inference.permutation_deltas["primary"], primary.delta),
        bootstrap_lower=float(np.quantile(inference.bootstrap_deltas["primary"], .025)),
        thresholds=thresholds,
        pairwise_deltas=pairwise,
        negative_controls=controls,
        corpus_control_passed=corpus_control_passed,
    )
    return {
        "decision": decision.decision.value,
        "reasons": list(decision.reasons),
        "replicates": {"permutations": n_replicates, "bootstraps": n_replicates, "permutation_seed": permutation_seed, "bootstrap_seed": bootstrap_seed, "shared_mapping_per_replicate": True, "shared_bootstrap_sample_per_replicate": True},
        "gates": {
            "valid": gate.valid,
            "failures": list(gate.failures),
            "scope": {
                "every_required_arm": ["dispersion", "tie_fraction"],
                "full_only": ["generalization", "generalization_gap", "amplification"],
            },
            "metrics": _jsonable_gate_records(metrics),
        },
        "primary": {
            "statistics": _jsonable_statistics(primary),
            "permutation_p": float(_one_sided_p(inference.permutation_deltas["primary"], primary.delta)),
            "bootstrap_lower": float(np.quantile(inference.bootstrap_deltas["primary"], .025)),
            "permutation_deltas": inference.permutation_deltas["primary"].tolist(),
            "bootstrap_deltas": inference.bootstrap_deltas["primary"].tolist(),
        },
        "pairwise_spearman_deltas": {f"{seed}:{left}:{right}": float(value) for (seed, left, right), value in pairwise.items()},
        "seed_averaged_pairwise_deltas": {f"{left}:{right}": float(value) for (left, right), value in _seed_averaged_pairwise_deltas(pairwise).items()},
        "robustness_fraction": float(robustness_fraction(pairwise)),
        "inference": {
            "permutation_deltas": {name: values.tolist() for name, values in inference.permutation_deltas.items()},
            "bootstrap_deltas": {name: values.tolist() for name, values in inference.bootstrap_deltas.items()},
            "permutation_seed": inference.permutation_seed,
            "bootstrap_seed": inference.bootstrap_seed,
            "n_replicates": inference.n_replicates,
        },
        "negative_controls": control_payload,
    }


def evaluate_cached(
    *,
    cells: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
    n_replicates: int = DEFAULT_REPLICATES,
    permutation_seed: int = 0,
    bootstrap_seed: int = 1,
) -> Mapping[str, Any]:
    """Evaluate synthetic/development cached arrays deterministically.

    ``n_replicates`` may be reduced only here for tests/synthetic development;
    run_real always enforces the contractual 10,000 for both inferential loops.
    """
    if thresholds is None:
        # Repeating calibrated thresholds in every cell makes a single JSON cell
        # artifact self-contained without inventing a default decision threshold.
        thresholds = cells[0].get("thresholds") if cells and isinstance(cells[0], Mapping) else None
    parsed_thresholds = _thresholds_from_mapping(thresholds)
    container, metrics, construction_valid, corpus_control_passed, connectivity = _build_cached_container(cells, require_real_seeds=False)
    return _evaluate_container(
        container, metrics, parsed_thresholds,
        n_replicates=n_replicates, permutation_seed=int(permutation_seed), bootstrap_seed=int(bootstrap_seed),
        construction_valid=construction_valid, corpus_control_passed=corpus_control_passed, connectivity_failures=connectivity,
    )


def _real_contract_thresholds(contract: Mapping[str, Any], calibration: Mapping[str, Any]) -> GateThresholds:
    if not isinstance(contract, Mapping) or not isinstance(calibration, Mapping):
        raise ManifestValidationError("real endpoint requires contract and calibration objects")
    config = contract.get("experiment", contract)
    if not isinstance(config, Mapping):
        raise ManifestValidationError("contract experiment must be an object")
    if tuple(config.get("seeds", ())) != (42, 420, 4200):
        raise ManifestValidationError("real contract must freeze exactly seeds 42, 420, and 4200")
    if config.get("k") != 10 or int(config.get("permutations", -1)) != DEFAULT_REPLICATES or int(config.get("bootstraps", -1)) != DEFAULT_REPLICATES:
        raise ManifestValidationError("real contract must require k=10 and exactly 10000 permutations/bootstraps")
    arms = set(config.get("arms", ()))
    required = {"full", *PRIMARY_COMPARATORS, *NEGATIVE_CONTROLS}
    if arms != required:
        raise ManifestValidationError("real contract arms must equal the eight frozen arms")
    calibrated = _thresholds_from_mapping(calibration)
    contractual = config.get("thresholds", contract.get("thresholds"))
    if contractual is None:
        raise ManifestValidationError("real contract lacks committed thresholds")
    if _thresholds_from_mapping(contractual) != calibrated:
        raise ManifestValidationError("contract and calibration thresholds do not match")
    return calibrated


def run_real(
    *,
    cells: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    calibration: Mapping[str, Any],
    preflight: Mapping[str, Any],
    n_replicates: int = DEFAULT_REPLICATES,
    permutation_seed: int = 0,
    bootstrap_seed: int = 1,
) -> Mapping[str, Any]:
    """Run the one-shot real endpoint; rejects any weakened contract or provenance."""
    if n_replicates != DEFAULT_REPLICATES:
        raise ManifestValidationError("run_real refuses non-contractual replicate counts")
    thresholds = _real_contract_thresholds(contract, calibration)
    if not isinstance(preflight, Mapping) or not isinstance(preflight.get("banks"), Mapping):
        raise ManifestValidationError("real endpoint requires a prompt-hash-bound preflight")
    if preflight.get("git_dirty") is not False or not isinstance(preflight.get("git_commit"), str) or not preflight["git_commit"]:
        raise ManifestValidationError("real preflight must be clean and bind a nonempty git commit")
    banks = preflight["banks"]
    container, metrics, construction_valid, corpus_control_passed, connectivity = _build_cached_container(cells, require_real_seeds=True)
    if len(container.ordered_ids) != 320:
        raise ManifestValidationError("real endpoint requires the frozen 320-article cohort")
    counts: dict[str, int] = {}
    for article_id in container.ordered_ids:
        publisher = container.publisher_by_id[article_id]
        counts[publisher] = counts.get(publisher, 0) + 1
    if len(counts) != 16 or set(counts.values()) != {20}:
        raise ManifestValidationError("real endpoint requires exactly 16 publishers x 20 articles")
    for cell in cells:
        bank = str(cell.get("bank"))
        provenance = cell.get("provenance")
        expected = banks.get(bank, {}).get("prompt_hash") if isinstance(banks.get(bank), Mapping) else None
        if not isinstance(provenance, Mapping) or provenance.get("q_bank_hash") != expected:
            raise ManifestValidationError("real cell has mismatched prompt-bank provenance")
        if provenance.get("target_source_kind") != "observer_derived":
            raise ManifestValidationError("real cells must use observer-derived targets")
        if provenance.get("git_commit") != preflight["git_commit"]:
            raise ManifestValidationError("real cell git commit does not match clean preflight")
    result = _evaluate_container(
        container, metrics, thresholds,
        n_replicates=DEFAULT_REPLICATES, permutation_seed=int(permutation_seed), bootstrap_seed=int(bootstrap_seed),
        construction_valid=construction_valid, corpus_control_passed=corpus_control_passed, connectivity_failures=connectivity,
    )
    return {**result, "mode": "real", "contractual_replicates_enforced": True}
