"""Experiment B: production component-necessity ablations with provenance."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dirichlet_fusion import DirichletFusion, DirichletFusionConfig
from core.hadamard_fusion import ConformalMetric, HadamardFusion
from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig


SEED = 42
REAL_EMBEDDINGS_PATH = ROOT / "data" / "real_pilot_embeddings.pt"
LABEL_PATH = ROOT / "gemini_blind_ratings.csv"
MANIFEST_PATH = ROOT / "results" / "exp_B_manifest.json"


def load_embeddings(path: Path = REAL_EMBEDDINGS_PATH) -> torch.Tensor:
    """Load the required production embeddings and reject malformed inputs."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required real embeddings file not found: {path}\n"
            "FAIL-CLOSED POLICY: run the embedding extraction pipeline first."
        )
    x = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("Embeddings must be a tensor with shape [N, B, H]")
    if any(size <= 0 for size in x.shape):
        raise ValueError(f"Embedding dimensions must be non-empty, got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Embeddings contain non-finite values")
    return x.float()


def load_matching_labels(
    n_articles: int,
    path: Path = LABEL_PATH,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """Load ratings only when they provide a one-to-one article match."""
    if not path.exists():
        return None, {
            "status": "skipped",
            "reason": "ratings_file_missing",
            "path": str(path),
            "expected_rows": n_articles,
            "observed_rows": 0,
        }

    table = pd.read_csv(path)
    if "gemini_score" not in table.columns:
        raise ValueError(f"Ratings CSV lacks required gemini_score column: {path}")

    observed_rows = len(table)
    if observed_rows != n_articles:
        return None, {
            "status": "skipped",
            "reason": "row_count_mismatch",
            "path": str(path),
            "expected_rows": n_articles,
            "observed_rows": observed_rows,
        }

    labels = pd.to_numeric(table["gemini_score"], errors="coerce").to_numpy(float)
    if not np.isfinite(labels).all():
        raise ValueError("gemini_score must contain one finite value per article")
    return labels, {
        "status": "matched",
        "path": str(path),
        "expected_rows": n_articles,
        "observed_rows": observed_rows,
    }


def rbf_gram(features: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return torch.exp(-torch.cdist(features, features).square() / (2 * sigma**2))


def stable_rbf_gram(features: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Float64 RBF used to verify exact kernel identities."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    features = features.double()
    differences = features[:, None, :] - features[None, :, :]
    squared_distances = differences.square().sum(dim=-1)
    return torch.exp(-squared_distances / (2 * sigma**2))


def centered(kernel: torch.Tensor) -> torch.Tensor:
    return (
        kernel
        - kernel.mean(0, keepdim=True)
        - kernel.mean(1, keepdim=True)
        + kernel.mean()
    )


def geometric_energy(kernel: torch.Tensor) -> np.ndarray:
    """One scalar per article: diag(HKH), whose sum is trace(HKH)."""
    return torch.diagonal(centered(kernel)).detach().cpu().numpy()


def evaluate_out_of_fold(
    predictions: np.ndarray,
    labels: np.ndarray,
    max_splits: int = 5,
) -> tuple[float, float]:
    """Assemble fixed unsupervised predictions on held-out fold positions."""
    p = np.asarray(predictions, float).reshape(-1)
    y = np.asarray(labels, float).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("predictions/labels mismatch")
    n_splits = min(max_splits, len(y))
    if n_splits < 2:
        raise ValueError("At least two matched labels are required")

    oof = np.empty_like(y)
    for _, test in KFold(n_splits, shuffle=True, random_state=SEED).split(p):
        oof[test] = p[test]  # No label-dependent parameters are fitted.
    result = spearmanr(oof, y)
    return float(result.statistic), float(result.pvalue)


def report(
    name: str,
    predictions: np.ndarray,
    labels: Optional[np.ndarray],
    label_status: dict[str, Any],
) -> dict[str, Any]:
    if labels is None:
        reason = str(label_status["reason"])
        print(f"  {name:<30} skipped ({reason})")
        return {"status": "skipped", "reason": reason}

    correlation, probability = evaluate_out_of_fold(predictions, labels)
    print(f"  {name:<30} rho={correlation: .6f}, p={probability:.6g}")
    return {
        "status": "evaluated",
        "spearman_rho": correlation if np.isfinite(correlation) else None,
        "p_value": probability if np.isfinite(probability) else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _distance_scale(distance_matrix: torch.Tensor) -> float:
    """Return a finite positive median off-diagonal distance."""
    if distance_matrix.shape[0] <= 1:
        return 1.0
    indices = torch.triu_indices(
        distance_matrix.shape[0], distance_matrix.shape[1], offset=1
    )
    distances = distance_matrix[indices[0], indices[1]]
    positive = distances[torch.isfinite(distances) & (distances > 0)]
    if positive.numel() == 0:
        return 1.0
    return max(float(positive.median()), 1e-6)


def _standardize_pairwise_scale(features: torch.Tensor) -> torch.Tensor:
    scale = _distance_scale(torch.cdist(features, features))
    return features / scale


def _kernel_from_distances(distance_matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
    sigma = max(_distance_scale(distance_matrix) / np.sqrt(2.0), 0.1)
    kernel = torch.exp(-distance_matrix.square() / (2 * sigma**2))
    return kernel, sigma


def main() -> None:
    print("Running Experiment B: Component Necessity")
    x = load_embeddings()
    n_articles, n_bots, hidden_dim = x.shape
    labels, label_status = load_matching_labels(n_articles)
    if labels is None:
        print(
            "Ratings evaluation disabled: "
            f"{label_status['reason']} "
            f"(expected {label_status['expected_rows']}, "
            f"observed {label_status['observed_rows']})."
        )

    spectral_config = SpectralPolarityConfig()
    dirichlet_config = DirichletFusionConfig(
        n_bots=n_bots,
        hidden_dim=hidden_dim,
        rks_dim=2048,
        n_observers=50,
        alpha=1.0,
        basis_seed=SEED,
        crn_enabled=True,
        crn_seed=12345,
    )

    with torch.no_grad():
        spectral_result = SpectralPolarity(spectral_config).compute_batch(
            list(x.unbind(dim=0))
        )
        dirichlet_module = DirichletFusion(dirichlet_config)
        dirichlet_result = dirichlet_module(x, compute_curvature=False)

    u_axis = spectral_result.u_axis.detach().cpu()
    fused = dirichlet_result["fused"].detach().cpu()
    fused_std = dirichlet_result["fused_std"].detach().cpu()
    bot_rkhs = dirichlet_result["bot_rkhs"].detach().cpu()
    if tuple(u_axis.shape) != (n_articles, hidden_dim):
        raise RuntimeError(f"Unexpected u_axis shape: {tuple(u_axis.shape)}")
    if fused.shape[0] != n_articles or fused_std.shape != fused.shape:
        raise RuntimeError("DirichletFusion returned incompatible fused tensors")
    if tuple(bot_rkhs.shape[:2]) != (n_articles, n_bots):
        raise RuntimeError("DirichletFusion returned incompatible RKHS views")

    observer_weights = getattr(dirichlet_module, "_locked_weights", None)
    if observer_weights is None or tuple(observer_weights.shape) != (
        dirichlet_config.n_observers,
        n_bots,
    ):
        raise RuntimeError("DirichletFusion did not expose its configured CRN weights")
    observer_weights = observer_weights.detach().cpu()

    conformal_metric = ConformalMetric()
    rho_density = conformal_metric.compute_density(fused_std)
    if tuple(rho_density.shape) != (n_articles,) or not torch.isfinite(rho_density).all():
        raise RuntimeError("Conformal density must be a finite [N] tensor")

    hadamard_module = HadamardFusion()
    semantic_kernel, spectral_kernel = hadamard_module.compute_kernel_matrices(
        fused, u_axis
    )
    hadamard_kernel, n_rescued, isolated_indices = hadamard_module.hadamard_product(
        semantic_kernel, spectral_kernel
    )
    hadamard_distance = torch.sqrt((2.0 - 2.0 * hadamard_kernel).clamp(min=0.0))
    conformal_distance = conformal_metric.apply_conformal_scaling(
        hadamard_distance, rho_density
    )
    if tuple(conformal_distance.shape) != (n_articles, n_articles):
        raise RuntimeError("Conformal distance must have shape [N, N]")

    evaluations: dict[str, dict[str, Any]] = {}

    print("\n1. Exact RBF vs production RKS")
    raw = x.mean(dim=1)
    with torch.no_grad():
        production_rks = dirichlet_module.basis(raw).detach().cpu()
    production_sigma = float(dirichlet_module.basis._sigma)
    exact_kernel = rbf_gram(raw, production_sigma)
    rks_kernel = production_rks @ production_rks.T
    evaluations["exact_rbf"] = report(
        "Exact RBF", geometric_energy(exact_kernel), labels, label_status
    )
    evaluations["production_rks"] = report(
        "Production RKS", geometric_energy(rks_kernel), labels, label_status
    )

    print("\n2. Uniform vs production Dirichlet")
    uniform_rks = bot_rkhs.mean(dim=1)
    uniform_kernel = uniform_rks @ uniform_rks.T
    dirichlet_kernel = fused @ fused.T
    evaluations["uniform_average"] = report(
        "Uniform average", geometric_energy(uniform_kernel), labels, label_status
    )
    evaluations["dirichlet_average"] = report(
        "Dirichlet average", geometric_energy(dirichlet_kernel), labels, label_status
    )

    print("\n3. Spectral track off vs on")
    evaluations["spectral_off"] = report(
        "Spectral off", geometric_energy(semantic_kernel), labels, label_status
    )
    evaluations["spectral_on"] = report(
        "Spectral on (Hadamard)",
        geometric_energy(hadamard_kernel),
        labels,
        label_status,
    )

    print("\n4. Product kernel vs scaled-concat equivalent")
    semantic_features = _standardize_pairwise_scale(fused)
    spectral_features = _standardize_pairwise_scale(u_axis)
    product = stable_rbf_gram(semantic_features) * stable_rbf_gram(spectral_features)
    scaled_concat_features = torch.cat((semantic_features, spectral_features), dim=1)
    equivalent = stable_rbf_gram(scaled_concat_features)
    maximum_gram_difference = float((product - equivalent).abs().max())
    print(f"  maximum Gram difference: {maximum_gram_difference:.3g}")
    evaluations["product_kernel"] = report(
        "Product kernel",
        np.round(geometric_energy(product), 12),
        labels,
        label_status,
    )
    evaluations["scaled_representation"] = report(
        "Scaled representation",
        np.round(geometric_energy(equivalent), 12),
        labels,
        label_status,
    )

    print("\n5. Simple RBF vs full conformal Track 5")
    conformal_kernel, conformal_sigma = _kernel_from_distances(conformal_distance)
    evaluations["simple_rbf_energy"] = report(
        "Simple RBF energy", geometric_energy(semantic_kernel), labels, label_status
    )
    evaluations["full_hadamard_track5"] = report(
        "Full Hadamard Track 5",
        geometric_energy(conformal_kernel),
        labels,
        label_status,
    )

    effective_config = {
        "seed": SEED,
        "n_articles": n_articles,
        "n_bots": n_bots,
        "hidden_dim": hidden_dim,
        "spectral_polarity": asdict(spectral_config),
        "dirichlet_fusion": asdict(dirichlet_config),
        "conformal_metric": {"temperature_scale": conformal_metric.temperature_scale},
        "hadamard_fusion": asdict(hadamard_module.config),
        "evaluation_max_splits": 5,
    }
    manifest = {
        "provenance": {
            "git_commit": _git_commit(),
            "embeddings_sha256": _sha256(REAL_EMBEDDINGS_PATH),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": effective_config,
        },
        "results": {
            "embeddings_shape": list(x.shape),
            "labels": label_status,
            "evaluations": evaluations,
            "spectral": {
                "u_axis_shape": list(u_axis.shape),
                "u_axis_sha256": _tensor_sha256(u_axis),
                "evr_mean": float(spectral_result.evr.mean()),
                "dipole_valid_count": int(spectral_result.dipole_valid.sum()),
            },
            "dirichlet": {
                "rho_shape": list(rho_density.shape),
                "rho_min": float(rho_density.min()),
                "rho_max": float(rho_density.max()),
                "rho_mean": float(rho_density.mean()),
                "rho_sha256": _tensor_sha256(rho_density),
                "observer_weights_shape": list(observer_weights.shape),
                "observer_weights_mean": observer_weights.mean(dim=0).tolist(),
                "observer_weights_sha256": _tensor_sha256(observer_weights),
                "module_provenance": dirichlet_result["provenance"],
            },
            "hadamard": {
                "d_conformal_shape": list(conformal_distance.shape),
                "d_conformal_min": float(conformal_distance.min()),
                "d_conformal_max": float(conformal_distance.max()),
                "d_conformal_mean": float(conformal_distance.mean()),
                "d_conformal_sha256": _tensor_sha256(conformal_distance),
                "conformal_kernel_sigma": conformal_sigma,
                "maximum_product_concat_gram_difference": maximum_gram_difference,
                "n_isolated_rescued": int(n_rescued),
                "isolated_indices": (
                    isolated_indices.tolist() if isolated_indices is not None else []
                ),
            },
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nProvenance manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
