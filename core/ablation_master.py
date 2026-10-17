"""
ablation_master.py — Legacy Code Archive for Ablation Studies

This module contains the OLD implementations that have been superseded by
proper ASTER v3.2 components. Preserved here for ablation comparisons.

OLD APPROACH (Before ASTER v3.2):
- Track 1.5 Antagonism: Simple bot-variance (directionless)
- Track 4 Walker: No u_axis awareness, just MCMC exploration

NEW APPROACH (ASTER v3.2):
- Track 1.5 Antagonism: SpectralPolarity (PCA via SVD, gauge-fixed u_axis)
- Track 4 Walker: u_axis-aware energy, work integral W, state classification

Usage:
    from core.ablation_master import compute_legacy_antagonism
    legacy_antag = compute_legacy_antagonism(cls_per_bot_list, rks_basis)
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LegacyAntagonismResult:
    """Result from legacy bot-variance antagonism computation."""
    # [N, D] — antagonism vectors (directionless, based on variance)
    antagonism: torch.Tensor
    # [N, H] — raw bot variance in embedding space
    bot_variance: torch.Tensor
    # Method name for logging
    method: str = "legacy_bot_variance"


def compute_legacy_antagonism(
    cls_per_bot_list: List[torch.Tensor],
    rks_basis,
    device: str = "cuda",
) -> LegacyAntagonismResult:
    """
    LEGACY: Compute antagonism using simple bot-variance.

    This was the original Track 1.5 implementation before spectral polarity.
    It computes variance across the 8 bots per article, which captures
    "disagreement" but loses all directional information.

    Args:
        cls_per_bot_list: List of [8, H] tensors (per-bot embeddings)
        rks_basis: SharedRKSBasis for projection to Track 5 space
        device: Compute device

    Returns:
        LegacyAntagonismResult with bot_variance and projected antagonism
    """
    if not cls_per_bot_list:
        return None

    # Stack: [N, 8, H]
    cls_stacked = torch.stack(cls_per_bot_list, dim=0).to(device)

    # LEGACY: Simple variance across bots (dimension 1)
    # This gives [N, H] but is DIRECTIONLESS - just magnitude of disagreement
    bot_variance = cls_stacked.var(dim=1)  # [N, H]

    # Project through RKS basis
    antagonism = rks_basis(bot_variance)  # [N, D]

    return LegacyAntagonismResult(
        antagonism=antagonism,
        bot_variance=bot_variance,
        method="legacy_bot_variance",
    )


def compute_legacy_walker(
    cls_per_bot: torch.Tensor,
    rks_basis,
    temperature: float = 0.5,
    n_walkers: int = 20,
    n_steps: int = 10,
) -> Dict[str, Any]:
    """
    LEGACY: Run MCMC walker without u_axis awareness.

    This was the original Track 4 implementation. The walker explores
    the bot-weight simplex using gradient-based energy (norm of fused gradient),
    but without any compass direction for orientation.

    Args:
        cls_per_bot: [8, H] per-bot embeddings
        rks_basis: SharedRKSBasis for projection
        temperature: Walker plasticity
        n_walkers: Number of MCMC walkers
        n_steps: Steps per walker

    Returns:
        Dict with walker_output (no work_integral or state classification)
    """
    from .physarum_walk import SemanticWalker

    n_bots = cls_per_bot.shape[0]
    bot_grads = cls_per_bot - cls_per_bot.mean(dim=0, keepdim=True)

    # Create explorer WITHOUT u_axis (legacy mode)
    explorer = SemanticWalker(
        embeddings=cls_per_bot,
        gradients=bot_grads,
        rks_basis=rks_basis,
        temperature=temperature,
        u_axis=None,  # KEY: No compass direction
    )

    # Run exploration
    trajectory = explorer.run_swarm(
        n_walkers=n_walkers,
        n_steps=n_steps,
        start_from_poles=False,  # Can't start from poles without u_axis
    )

    # Project final position
    projected = explorer.project_trajectory(trajectory)
    endpoint = projected[-1].mean(dim=0)  # [D]

    return {
        "walker_output": endpoint,
        "method": "legacy_no_u_axis",
        # No work_integral or state - those require u_axis
    }


def run_ablation_comparison(
    cls_per_bot_list: List[torch.Tensor],
    rks_basis,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run both legacy and new methods for direct comparison.

    Returns dict with:
        - legacy_antagonism: [N, D] from bot-variance
        - legacy_walker_outputs: List of [D] from non-u_axis walker
        - spectral_antagonism: [N, D] from SpectralPolarity
        - spectral_evr: [N] explained variance ratios
        - spectral_dipole_valid: [N] validity flags
        - walker_work_integrals: [N] work integral values
        - walker_states: List of state strings
    """
    from .spectral_polarity import SpectralPolarity, SpectralPolarityConfig
    from .physarum_walk import compute_walker_resistance

    results = {}

    # 1. Legacy antagonism
    legacy_antag = compute_legacy_antagonism(cls_per_bot_list, rks_basis, device)
    if legacy_antag:
        results["legacy_antagonism"] = legacy_antag.antagonism.detach().cpu().numpy()
        results["legacy_bot_variance"] = legacy_antag.bot_variance.detach().cpu().numpy()

    # 2. New spectral polarity
    sp = SpectralPolarity(SpectralPolarityConfig(multi_scale_enabled=True))
    spectral_results = sp.compute_batch(cls_per_bot_list)
    spectral_antagonism = rks_basis(spectral_results.antagonism)
    results["spectral_antagonism"] = spectral_antagonism.detach().cpu().numpy()
    results["spectral_evr"] = spectral_results.evr.detach().cpu().numpy()
    results["spectral_dipole_valid"] = spectral_results.dipole_valid.detach().cpu().numpy()
    results["spectral_u_axis"] = spectral_results.u_axis.detach().cpu().numpy()

    # 3. Legacy walker (no u_axis)
    legacy_walker_outputs = []
    for i, bot_embs in enumerate(cls_per_bot_list[:50]):  # Cap for efficiency
        legacy_result = compute_legacy_walker(bot_embs, rks_basis)
        legacy_walker_outputs.append(legacy_result["walker_output"].detach().cpu().numpy())
    if legacy_walker_outputs:
        results["legacy_walker_outputs"] = np.stack(legacy_walker_outputs)

    # 4. New walker (with u_axis)
    walker_work_integrals = []
    walker_states = []
    walker_outputs = []
    cls_stacked = torch.stack(cls_per_bot_list, dim=0)
    for i in range(min(len(cls_per_bot_list), 50)):
        bot_embs = cls_stacked[i]
        u_axis_i = spectral_results.u_axis[i]
        result = compute_walker_resistance(
            cls_per_bot=bot_embs,
            rks_basis=rks_basis,
            u_axis=u_axis_i,
            n_walkers=20,
            n_steps=10,
        )
        walker_outputs.append(result["walker_output"].detach().cpu().numpy())
        walker_work_integrals.append(result["work_integral"])
        walker_states.append(result["state"])

    if walker_outputs:
        results["new_walker_outputs"] = np.stack(walker_outputs)
        results["walker_work_integrals"] = np.array(walker_work_integrals)
        results["walker_states"] = walker_states

    return results


def compute_ablation_metrics(comparison: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute comparison metrics between legacy and new approaches.

    Returns metrics like:
        - antagonism_cosine_sim: Mean cosine similarity between legacy and new
        - antagonism_l2_diff: Mean L2 distance
        - walker_cosine_sim: Mean cosine similarity of walker outputs
        - evr_mean: Mean explained variance ratio (new only)
        - dipole_valid_frac: Fraction of valid dipoles (new only)
    """
    metrics = {}

    # Antagonism comparison
    if "legacy_antagonism" in comparison and "spectral_antagonism" in comparison:
        leg = comparison["legacy_antagonism"]
        spec = comparison["spectral_antagonism"]

        # Normalize for cosine sim
        leg_norm = leg / (np.linalg.norm(leg, axis=1, keepdims=True) + 1e-9)
        spec_norm = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-9)

        cosine_sims = (leg_norm * spec_norm).sum(axis=1)
        metrics["antagonism_cosine_sim_mean"] = float(cosine_sims.mean())
        metrics["antagonism_cosine_sim_std"] = float(cosine_sims.std())

        l2_diffs = np.linalg.norm(leg - spec, axis=1)
        metrics["antagonism_l2_diff_mean"] = float(l2_diffs.mean())
        metrics["antagonism_l2_diff_std"] = float(l2_diffs.std())

    # Walker comparison
    if "legacy_walker_outputs" in comparison and "new_walker_outputs" in comparison:
        leg_w = comparison["legacy_walker_outputs"]
        new_w = comparison["new_walker_outputs"]

        n = min(len(leg_w), len(new_w))
        leg_w, new_w = leg_w[:n], new_w[:n]

        leg_norm = leg_w / (np.linalg.norm(leg_w, axis=1, keepdims=True) + 1e-9)
        new_norm = new_w / (np.linalg.norm(new_w, axis=1, keepdims=True) + 1e-9)

        cosine_sims = (leg_norm * new_norm).sum(axis=1)
        metrics["walker_cosine_sim_mean"] = float(cosine_sims.mean())
        metrics["walker_cosine_sim_std"] = float(cosine_sims.std())

    # New-only metrics
    if "spectral_evr" in comparison:
        metrics["evr_mean"] = float(comparison["spectral_evr"].mean())
        metrics["evr_std"] = float(comparison["spectral_evr"].std())
    if "spectral_dipole_valid" in comparison:
        metrics["dipole_valid_frac"] = float(comparison["spectral_dipole_valid"].mean())
    if "walker_work_integrals" in comparison:
        w = comparison["walker_work_integrals"]
        metrics["work_integral_mean"] = float(w.mean())
        metrics["work_integral_std"] = float(w.std())
    if "walker_states" in comparison:
        states = comparison["walker_states"]
        from collections import Counter
        counts = Counter(states)
        total = max(len(states), 1)
        metrics["elastic_frac"] = counts.get("elastic", 0) / total
        metrics["trapped_frac"] = counts.get("trapped", 0) / total
        metrics["broken_frac"] = counts.get("broken", 0) / total

    return metrics


__all__ = [
    'LegacyAntagonismResult',
    'compute_legacy_antagonism',
    'compute_legacy_walker',
    'run_ablation_comparison',
    'compute_ablation_metrics',
    'run_auto_observer_ablation',
    'run_metric_tensor_ablation',
    'run_dynamic_tuning_ablation',
    'run_upgrade_ablations',
]


# ---------------------------------------------------------------------------
# Upgrade ablations (ASTER v3.2+)
# ---------------------------------------------------------------------------

def _json_number(value: Any) -> Optional[float]:
    """Return a finite builtin float, suitable for strict JSON encoders."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_matrix(result: Dict[str, Any]) -> np.ndarray:
    value = result.get("integrated_vectors")
    if value is None:
        value = result.get("features")
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("pipeline arm did not return a two-dimensional feature matrix")
    return matrix


def _matrix_summary(matrix: np.ndarray) -> Dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return {
        "n_articles": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "logit_or_feature_variance": _json_number(np.mean(np.var(matrix, axis=0))),
        "mean_pairwise_divergence": _json_number(
            np.mean(np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=-1))
        ),
    }


def _default_pipeline_arm(
    articles: List[Dict[str, Any]], *, auto_observers_enabled: bool, dynamic_observer_tuning: bool,
    seed: int, pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one real, deterministic complete-pipeline arm."""
    from .complete_pipeline import BeliefTransformerPipeline, initialize_full_pipeline

    kwargs = dict(pipeline_kwargs or {})
    kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    kwargs.setdefault("use_cls_tokens", True)
    kwargs.setdefault("use_dirichlet_fusion", True)
    kwargs.setdefault("dirichlet_n_observers", 16)
    kwargs.setdefault("use_gru", False)
    kwargs.setdefault("use_attention", False)
    components = initialize_full_pipeline(
        random_seed=int(seed),
        auto_observers_enabled=bool(auto_observers_enabled),
        dynamic_observer_tuning=bool(dynamic_observer_tuning),
        **kwargs,
    )
    pipeline = BeliefTransformerPipeline(components=components, random_seed=int(seed))
    return pipeline.process_month(list(articles), month_name="upgrade_ablation")


def _run_arm(runner: Any, articles: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    return runner(list(articles), **kwargs)


def run_auto_observer_ablation(
    articles: List[Dict[str, Any]], *, seed: int = 42, pipeline_runner: Any = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Paired endogenous-observer versus generic/static-observer NLI ablation."""
    runner = pipeline_runner or _default_pipeline_arm
    arms = {}
    matrices = {}
    for name, enabled in (("endogenous", True), ("static", False)):
        result = _run_arm(
            runner, articles, auto_observers_enabled=enabled,
            dynamic_observer_tuning=False, seed=seed, pipeline_kwargs=pipeline_kwargs,
        )
        matrices[name] = _as_matrix(result)
        arms[name] = _matrix_summary(matrices[name])
        arms[name]["auto_observers_enabled"] = enabled
        arms[name]["observer_pair_count"] = int(len(result.get("auto_observers", {}).get("pairs", [])))
    n = min(len(matrices["endogenous"]), len(matrices["static"]))
    paired = matrices["endogenous"][:n] - matrices["static"][:n]
    return {
        "design": "paired_common_seed",
        "seed": int(seed),
        "arms": arms,
        "comparison": {
            "mean_paired_l2_divergence": _json_number(np.mean(np.linalg.norm(paired, axis=1))),
            "variance_delta_endogenous_minus_static": _json_number(
                arms["endogenous"]["logit_or_feature_variance"] - arms["static"]["logit_or_feature_variance"]
            ),
        },
        "_endogenous_features": matrices["endogenous"],
    }


def _action_arm_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    records = list(result.get("records", []))
    work = [float(row.get("action")) for row in records if _json_number(row.get("action")) is not None]
    invalid_rows = [
        row for row in records
        if not bool(row.get("reached", False)) or _json_number(row.get("action")) is None
    ]
    invalid = len(invalid_rows)
    denom = max(len(records), 1)
    return {
        "path_count": int(len(records)),
        "invalid_or_trapped_fraction": float(invalid / denom),
        "invalid_fraction": float(invalid / denom),
        "trapped_fraction": float(invalid / denom),
        "invalid_or_trapped_definition": "path_unreached_or_nonfinite_action",
        "work_integral": {
            "mean": _json_number(np.mean(work)) if work else None,
            "std": _json_number(np.std(work)) if work else None,
            "median": _json_number(np.median(work)) if work else None,
            "values": work,
        },
        "solvers_used": list(result.get("summary", {}).get("traversal_solvers_used", [])),
    }


def run_metric_tensor_ablation(
    embeddings: Any, *, seed: int = 42, action_graph_runner: Any = None,
    action_graph_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Paired LocalMetricTensor/FMM versus Euclidean graph traversal ablation."""
    from .track4_action_graph import ActionGraphConfig, run_action_graph

    points = torch.as_tensor(embeddings, dtype=torch.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("metric tensor ablation requires at least two embedding rows")
    runner = action_graph_runner or run_action_graph
    common = dict(action_graph_kwargs or {})
    torch.manual_seed(int(seed))
    fmm_cfg = ActionGraphConfig(
        k_neighbors=min(8, int(points.shape[0]) - 1), traversal_solver="fmm",
        transport_metric_type="riemannian", fmm_grid_shape=(10, 10),
        fmm_use_skfmm=False, fmm_allow_graph_fallback=True,
    )
    euclidean_cfg = ActionGraphConfig(
        k_neighbors=min(8, int(points.shape[0]) - 1), traversal_solver="graph",
        transport_metric_type="euclidean",
    )
    # Landmark selection must not depend on either arm's learned metric.
    anchors = list(common.pop("anchors", [0]))
    targets = common.pop("targets_by_anchor", None)
    if targets is None:
        target_idx = int(points.shape[0]) - 1
        targets = {int(anchor): [target_idx if int(anchor) != target_idx else 0] for anchor in anchors}
    riemannian = runner(points, config=fmm_cfg, anchors=anchors, targets_by_anchor=targets, **common)
    euclidean = runner(points, config=euclidean_cfg, anchors=anchors, targets_by_anchor=targets, **common)
    arms = {"riemannian_fmm": _action_arm_summary(riemannian), "euclidean": _action_arm_summary(euclidean)}
    return {
        "design": "paired_identical_embeddings_and_landmarks",
        "arms": arms,
        "comparison": {
            "invalid_or_trapped_fraction_delta": _json_number(
                arms["riemannian_fmm"]["invalid_or_trapped_fraction"] - arms["euclidean"]["invalid_or_trapped_fraction"]
            ),
            "mean_work_delta": _json_number(
                (arms["riemannian_fmm"]["work_integral"]["mean"] or 0.0)
                - (arms["euclidean"]["work_integral"]["mean"] or 0.0)
            ),
        },
    }


def _weight_sparsity(
    result: Dict[str, Any], *, enabled: bool, seed: int, pipeline_kwargs: Dict[str, Any], n_articles: int,
) -> tuple[Optional[float], str]:
    from .dynamic_observer_tuning import sample_dynamic_dirichlet_weights

    weights = result.get("article_observer_weights")
    if weights is None:
        weights = result.get("observer_weights")
    if weights is not None:
        array = np.asarray(weights, dtype=np.float64)
        return _json_number(np.mean(np.sum(array * array, axis=-1))), "pipeline_article_observer_weights"
    diagnostics = result.get("dynamic_observer_tuning", {}) or {}
    alphas = np.asarray(diagnostics.get("article_alphas", []), dtype=np.float64)
    if not alphas.size:
        alphas = np.full(n_articles, float(pipeline_kwargs.get("dirichlet_alpha", 1.0)))
    if alphas.size != n_articles:
        return None, "unavailable_alpha_shape_mismatch"
    reconstructed = sample_dynamic_dirichlet_weights(
        alphas, n_bots=int(pipeline_kwargs.get("n_bots", 8)),
        n_observers=int(pipeline_kwargs.get("dirichlet_n_observers", 16)),
        seed=int(pipeline_kwargs.get("dirichlet_crn_seed", seed)), as_tensor=False,
    )
    method = "deterministic_dynamic_alpha_reconstruction" if enabled else "deterministic_static_alpha_reconstruction"
    return _json_number(np.mean(np.sum(reconstructed * reconstructed, axis=-1))), method


def run_dynamic_tuning_ablation(
    articles: List[Dict[str, Any]], *, seed: int = 42, pipeline_runner: Any = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Paired closed-loop reactive-alpha versus static-alpha Track 3/5 ablation."""
    runner = pipeline_runner or _default_pipeline_arm
    from .dynamic_observer_tuning import estimate_crack_density

    resolved_kwargs = dict(pipeline_kwargs or {})
    arms = {}
    for name, enabled in (("closed_loop", True), ("static", False)):
        result = _run_arm(
            runner, articles, auto_observers_enabled=False,
            dynamic_observer_tuning=enabled, seed=seed, pipeline_kwargs=pipeline_kwargs,
        )
        fused = result.get("dirichlet_fused")
        stress = result.get("dirichlet_fused_std")
        cracks = (
            estimate_crack_density(fused, stress)
            if fused is not None else np.asarray([], dtype=np.float64)
        )
        sparsity, sparsity_method = _weight_sparsity(
            result, enabled=enabled, seed=seed, pipeline_kwargs=resolved_kwargs,
            n_articles=len(articles),
        )
        arms[name] = {
            "dynamic_observer_tuning": enabled,
            "track3_crack_density_mean": _json_number(np.mean(cracks)) if cracks.size else None,
            "track3_crack_density_std": _json_number(np.std(cracks)) if cracks.size else None,
            "crack_density_method": "estimate_crack_density(dirichlet_fused,dirichlet_fused_std)",
            "observer_weight_sparsity": sparsity,
            "observer_weight_sparsity_method": sparsity_method,
        }
    return {
        "design": "paired_common_seed",
        "seed": int(seed),
        "arms": arms,
        "comparison": {
            "crack_density_delta_closed_loop_minus_static": _json_number(
                (arms["closed_loop"]["track3_crack_density_mean"] or 0.0)
                - (arms["static"]["track3_crack_density_mean"] or 0.0)
            ),
            "sparsity_delta_closed_loop_minus_static": (
                _json_number(arms["closed_loop"]["observer_weight_sparsity"] - arms["static"]["observer_weight_sparsity"])
                if arms["closed_loop"]["observer_weight_sparsity"] is not None
                and arms["static"]["observer_weight_sparsity"] is not None else None
            ),
        },
    }


def run_upgrade_ablations(
    articles: List[Dict[str, Any]], *, seed: int = 42, max_articles: int = 24,
    pipeline_runner: Any = None, action_graph_runner: Any = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all three architecture-upgrade ablations on one deterministic subset."""
    subset = list(articles)[:max(2, int(max_articles))]
    if len(subset) < 2:
        raise ValueError("upgrade ablations require at least two articles")
    auto = run_auto_observer_ablation(
        subset, seed=seed, pipeline_runner=pipeline_runner, pipeline_kwargs=pipeline_kwargs,
    )
    features = auto.pop("_endogenous_features")
    return {
        "schema_version": "1.0",
        "n_articles": int(len(subset)),
        "seed": int(seed),
        "auto_observer": auto,
        "metric_tensor": run_metric_tensor_ablation(
            features, seed=seed, action_graph_runner=action_graph_runner,
        ),
        "dynamic_tuning": run_dynamic_tuning_ablation(
            subset, seed=seed, pipeline_runner=pipeline_runner, pipeline_kwargs=pipeline_kwargs,
        ),
    }
