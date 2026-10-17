"""
waterfall_checkpoints.py — Data Lineage & Checkpoint System

Implements the "Dam at Every Step" pattern to capture immutable artifacts
at each stage of the pipeline, enabling precise debugging of score anomalies.

Checkpoint Hierarchy:
    Phase 1 - Raw Materials:
        T0: Logit Substrate (raw probabilities)

    Phase 2 - Geometry:
        T1: Semantic Embeddings (raw CLS/mean-pooled)
        T1.5: Spectral State (u_axis, EVR, rotation matrix)

    Phase 3 - Topology:
        T2: Kernel Projections (RBF, Matern, Laplacian)
        T3: Social Topology (bonds, cracks, ruptures)

    Phase 4 - Visualization:
        T4: Viz Coordinates (UMAP/t-SNE projections)

Usage:
    checkpoint = WaterfallCheckpoint(output_dir)
    checkpoint.save_t0_substrate(logit_probs)
    checkpoint.save_t1_embeddings(cls_embeddings)
    # ... etc

    # Later debugging:
    artifacts = checkpoint.load_all()

Author: Belief Transformer Project (ASTER v3.2)
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib

try:
    from .kernel_library import (
        normalize_kernel_type,
        validate_kernel_provenance,
        validate_positive_sigma,
        validate_shape_params,
    )
except ImportError:  # pragma: no cover - script-style fallback
    from kernel_library import (
        normalize_kernel_type,
        validate_kernel_provenance,
        validate_positive_sigma,
        validate_shape_params,
    )


# =============================================================================
# ARTIFACT SPECIFICATIONS
# =============================================================================

@dataclass
class T0_LogitSubstrate:
    """Checkpoint 0: Raw logit probabilities from NLI model."""
    # Shape: (N_articles, 8_bot_pairs, 3_classes)
    probabilities: np.ndarray
    # Metadata
    n_articles: int
    n_bot_pairs: int = 8
    class_names: Tuple[str, ...] = ("entail", "neutral", "contradict")
    model_name: str = "deberta-v3-large"
    timestamp: str = ""
    checksum: str = ""

    def compute_checksum(self) -> str:
        return hashlib.sha256(self.probabilities.tobytes()).hexdigest()[:16]


@dataclass
class T1_SemanticEmbeddings:
    """Checkpoint 1: Raw semantic embeddings before kernel projection."""
    # Shape: (N_articles, hidden_dim)
    embeddings: np.ndarray
    # Per-bot embeddings for spectral analysis
    # Shape: (N_articles, 8, hidden_dim)
    embeddings_per_bot: Optional[np.ndarray] = None
    # Metadata
    hidden_dim: int = 1536
    extraction_method: str = "cls_token"  # or "mean_pool"
    timestamp: str = ""
    checksum: str = ""


@dataclass
class T15_SpectralState:
    """Checkpoint 1.5: Spectral compass alignment state."""
    # The eigenvector (compass direction)
    # Shape: (hidden_dim,)
    u_axis: np.ndarray
    # Explained variance ratio (signal strength)
    evr: float
    # Full eigenspectrum
    # Shape: (min(8, n_components),)
    singular_values: np.ndarray
    # Probe magnitudes per article
    # Shape: (N_articles, 8)
    probe_magnitudes: np.ndarray
    # Dipole validity flags
    # Shape: (N_articles,)
    dipole_valid: np.ndarray
    # Multi-scale persistence
    n_persistent_scales: int = 0
    sigma_values_tested: Tuple[float, ...] = ()
    # Metadata
    reference_probe_idx: int = 0
    evr_threshold: float = 0.5
    timestamp: str = ""


@dataclass
class T2_KernelProjections:
    """Checkpoint 2: Kernel-projected embeddings (multi-universe)."""
    # RBF universe - Shape: (N_articles, rks_dim)
    z_rbf: np.ndarray
    # Matern universe (optional)
    z_matern: Optional[np.ndarray] = None
    # Laplacian universe (optional)
    z_laplacian: Optional[np.ndarray] = None
    # IMQ universe (optional)
    z_imq: Optional[np.ndarray] = None
    # RQ universe (optional)
    z_rq: Optional[np.ndarray] = None
    # Kernel parameters
    rbf_sigma: float = 1.0
    matern_nu: float = 1.5
    laplacian_sigma: float = 1.0
    kernel_type: str = "rbf"
    requested_kernel_type: str = "rbf"
    kernel_provenance: Optional[Dict[str, Any]] = None
    # RKS parameters
    rks_dim: int = 2048
    random_seed: int = 42
    # Metadata
    timestamp: str = ""
    checksum: str = ""


@dataclass
class T3_SocialTopology:
    """Checkpoint 3: Social structure (bonds, cracks, ruptures)."""
    # Bond adjacency matrix - Shape: (N_articles, N_articles)
    bond_matrix: np.ndarray
    # Crack intensity matrix - Shape: (N_articles, N_articles)
    crack_matrix: np.ndarray
    # Rupture pairs (indices where W → ∞)
    rupture_pairs: List[Tuple[int, int]]
    # Dirichlet fusion output
    # Shape: (N_articles, fused_dim)
    dirichlet_fused: Optional[np.ndarray] = None
    # Walker work integrals - Shape: (N_articles,)
    walker_work: Optional[np.ndarray] = None
    # Validation scores
    nmi_score: float = 0.0
    ari_score: float = 0.0
    # Metadata
    n_observers: int = 10
    alpha: float = 1.0
    timestamp: str = ""


@dataclass
class T4_VizCoordinates:
    """Checkpoint 4: 2D visualization coordinates."""
    # UMAP coordinates - Shape: (N_articles, 2)
    x_umap: np.ndarray
    # t-SNE coordinates (optional) - Shape: (N_articles, 2)
    x_tsne: Optional[np.ndarray] = None
    # Metric UMAP (from kernel distance) - Shape: (N_articles, 2)
    x_metric_umap: Optional[np.ndarray] = None
    # Which kernel was used for metric UMAP
    metric_kernel: str = "rbf"
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    # Metadata
    timestamp: str = ""


# =============================================================================
# WATERFALL CHECKPOINT MANAGER
# =============================================================================

class WaterfallCheckpoint:
    """
    Manager for saving and loading pipeline checkpoints.

    Implements the "Dam at Every Step" pattern for data lineage tracking.
    """

    def __init__(self, output_dir: Path, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            run_id: Optional unique identifier for this run
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track what's been saved
        self.manifest = {
            "run_id": self.run_id,
            "created": datetime.now().isoformat(),
            "checkpoints": {},
        }

    def _now(self) -> str:
        return datetime.now().isoformat()

    # =========================================================================
    # PHASE 1: RAW MATERIALS
    # =========================================================================

    def save_t0_substrate(
        self,
        probabilities: np.ndarray,
        model_name: str = "deberta-v3-large",
    ) -> Path:
        """
        Save T0: Logit substrate (raw NLI probabilities).

        Args:
            probabilities: (N, 8, 3) array of [entail, neutral, contradict] probs
            model_name: Name of the NLI model used

        Returns:
            Path to saved file
        """
        checkpoint = T0_LogitSubstrate(
            probabilities=probabilities,
            n_articles=probabilities.shape[0],
            model_name=model_name,
            timestamp=self._now(),
        )
        checkpoint.checksum = checkpoint.compute_checksum()

        # Save numpy array
        path = self.checkpoint_dir / "T0_substrate.npy"
        np.save(path, probabilities)

        # Save metadata
        meta_path = self.checkpoint_dir / "T0_substrate_meta.json"
        meta = {
            "n_articles": checkpoint.n_articles,
            "n_bot_pairs": checkpoint.n_bot_pairs,
            "class_names": checkpoint.class_names,
            "model_name": checkpoint.model_name,
            "timestamp": checkpoint.timestamp,
            "checksum": checkpoint.checksum,
            "shape": list(probabilities.shape),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        self.manifest["checkpoints"]["T0"] = str(path)
        print(f"[Checkpoint T0] Saved logit substrate: {path}")
        return path

    # =========================================================================
    # PHASE 2: GEOMETRY
    # =========================================================================

    def save_t1_embeddings(
        self,
        embeddings: np.ndarray,
        embeddings_per_bot: Optional[np.ndarray] = None,
        extraction_method: str = "cls_token",
    ) -> Path:
        """
        Save T1: Raw semantic embeddings.

        Args:
            embeddings: (N, hidden_dim) array
            embeddings_per_bot: Optional (N, 8, hidden_dim) per-bot embeddings
            extraction_method: "cls_token" or "mean_pool"

        Returns:
            Path to saved file
        """
        path = self.checkpoint_dir / "T1_embeddings_raw.npy"
        np.save(path, embeddings)

        if embeddings_per_bot is not None:
            bot_path = self.checkpoint_dir / "T1_embeddings_per_bot.npy"
            np.save(bot_path, embeddings_per_bot)

        meta = {
            "hidden_dim": embeddings.shape[1],
            "n_articles": embeddings.shape[0],
            "extraction_method": extraction_method,
            "timestamp": self._now(),
            "checksum": hashlib.sha256(embeddings.tobytes()).hexdigest()[:16],
        }
        with open(self.checkpoint_dir / "T1_embeddings_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.manifest["checkpoints"]["T1"] = str(path)
        print(f"[Checkpoint T1] Saved semantic embeddings: {path}")
        return path

    def save_t15_spectral(
        self,
        u_axis: np.ndarray,
        evr: float,
        singular_values: np.ndarray,
        probe_magnitudes: np.ndarray,
        dipole_valid: np.ndarray,
        n_persistent_scales: int = 0,
        sigma_values_tested: Tuple[float, ...] = (),
    ) -> Path:
        """
        Save T1.5: Spectral compass state.
        """
        # Save as NPZ archive
        path = self.checkpoint_dir / "T1.5_spectral_state.npz"
        np.savez(
            path,
            u_axis=u_axis,
            singular_values=singular_values,
            probe_magnitudes=probe_magnitudes,
            dipole_valid=dipole_valid,
        )

        # Save scalars as JSON
        meta = {
            "evr": float(evr),
            "n_persistent_scales": n_persistent_scales,
            "sigma_values_tested": list(sigma_values_tested),
            "evr_threshold": 0.5,
            "timestamp": self._now(),
        }
        with open(self.checkpoint_dir / "T1.5_spectral_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.manifest["checkpoints"]["T1.5"] = str(path)
        print(f"[Checkpoint T1.5] Saved spectral state: EVR={evr:.3f}")
        return path

    # =========================================================================
    # PHASE 3: TOPOLOGY
    # =========================================================================

    def save_t2_kernel_projections(
        self,
        z_rbf: Optional[np.ndarray] = None,
        z_matern: Optional[np.ndarray] = None,
        z_laplacian: Optional[np.ndarray] = None,
        z_imq: Optional[np.ndarray] = None,
        z_rq: Optional[np.ndarray] = None,
        z_kernel: Optional[np.ndarray] = None,
        kernel_type: str = "rbf",
        requested_kernel_type: Optional[str] = None,
        kernel_sigma: Optional[float] = None,
        kernel_shape_params: Optional[Dict[str, Any]] = None,
        kernel_provenance: Optional[Dict[str, Any]] = None,
        basis_hash: Optional[str] = None,
        code_path: Optional[str] = None,
        kernel_input_dim: Optional[int] = None,
        rbf_sigma: float = 1.0,
        rks_dim: int = 2048,
        random_seed: int = 42,
    ) -> Path:
        """
        Save T2: Kernel-projected embeddings.
        """
        path = self.checkpoint_dir / "T2_kernel_projections.npz"

        if kernel_provenance is None:
            raise ValueError("T2 writes require explicit kernel_provenance; anonymous writes are forbidden")
        if requested_kernel_type is None:
            raise ValueError("T2 writes require explicit requested_kernel_type")
        if kernel_sigma is None:
            raise ValueError("T2 writes require explicit kernel_sigma")
        if kernel_shape_params is None:
            raise ValueError("T2 writes require explicit kernel_shape_params (use {} for RBF)")
        if not str(basis_hash or "").strip():
            raise ValueError("T2 writes require a nonempty basis_hash")
        if not str(code_path or "").strip():
            raise ValueError("T2 writes require a nonempty code_path")
        if kernel_input_dim is None or int(kernel_input_dim) <= 0:
            raise ValueError("T2 writes require a positive explicit kernel_input_dim")

        actual_kernel = normalize_kernel_type(kernel_type)
        requested_kernel = normalize_kernel_type(requested_kernel_type)
        if requested_kernel != actual_kernel:
            raise ValueError(f"T2 kernel mismatch: requested={requested_kernel}, actual={actual_kernel}")

        arrays: Dict[str, np.ndarray] = {}
        explicit_arrays = {
            "rbf": z_rbf,
            "matern": z_matern,
            "laplacian": z_laplacian,
            "imq": z_imq,
            "rq": z_rq,
        }
        if z_kernel is not None:
            explicit_arrays[actual_kernel] = z_kernel
        elif z_rbf is not None and actual_kernel != "rbf":
            # Backward-compatible call shape, but never serialize a non-RBF
            # projection under an RBF label when the caller declares otherwise.
            explicit_arrays[actual_kernel] = z_rbf
            explicit_arrays["rbf"] = None

        for family, arr in explicit_arrays.items():
            if arr is not None:
                arrays[f"z_{family}"] = np.asarray(arr)

        if z_matern is not None:
            arrays["z_matern"] = np.asarray(z_matern)
        if z_laplacian is not None:
            arrays["z_laplacian"] = np.asarray(z_laplacian)
        if z_imq is not None:
            arrays["z_imq"] = np.asarray(z_imq)
        if z_rq is not None:
            arrays["z_rq"] = np.asarray(z_rq)

        active_key = f"z_{actual_kernel}"
        if active_key not in arrays:
            raise ValueError(f"No T2 projection array provided for kernel '{actual_kernel}'")

        active = arrays[active_key]
        if active.ndim < 2:
            raise ValueError("T2 projection array must have at least two dimensions")
        active_output_dim = int(active.shape[-1])
        if int(rks_dim) != active_output_dim:
            raise ValueError(
                f"T2 output_dim mismatch: array={active_output_dim}, rks_dim={rks_dim}"
            )
        sigma_value = validate_positive_sigma(kernel_sigma, "T2 kernel_sigma")
        canonical_params = validate_shape_params(
            actual_kernel,
            kernel_shape_params,
            name="T2 kernel_shape_params",
        )
        provenance = dict(kernel_provenance)
        validate_kernel_provenance(
            provenance,
            requested_kernel=requested_kernel,
            input_dim=int(kernel_input_dim),
            output_dim=active_output_dim,
            shape_params=canonical_params,
            require_dimensions=True,
            seed=int(random_seed),
            sigma=sigma_value,
            basis_hash=str(basis_hash),
            code_path=str(code_path),
        )

        np.savez(path, **arrays)

        meta = {
            "kernel_type": actual_kernel,
            "requested_kernel_type": requested_kernel,
            "actual_kernel": actual_kernel,
            "active_projection_key": active_key,
            "kernel_sigma": sigma_value,
            "kernel_shape_params": canonical_params,
            "kernel_provenance": provenance,
            "sampler_id": provenance["sampler_id"],
            "basis_hash": str(basis_hash),
            "code_path": str(code_path),
            "rbf_sigma": sigma_value if actual_kernel == "rbf" else rbf_sigma,
            "rks_dim": rks_dim,
            "kernel_input_dim": int(kernel_input_dim),
            "random_seed": random_seed,
            "n_articles": active.shape[0],
            "has_rbf": "z_rbf" in arrays,
            "has_matern": "z_matern" in arrays,
            "has_laplacian": "z_laplacian" in arrays,
            "has_imq": "z_imq" in arrays,
            "has_rq": "z_rq" in arrays,
            "timestamp": self._now(),
        }
        with open(self.checkpoint_dir / "T2_kernel_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.manifest["checkpoints"]["T2"] = str(path)
        print(f"[Checkpoint T2] Saved kernel projections: {path}")
        return path

    def save_t3_topology(
        self,
        bond_matrix: np.ndarray,
        crack_matrix: np.ndarray,
        rupture_pairs: List[Tuple[int, int]],
        dirichlet_fused: Optional[np.ndarray] = None,
        dirichlet_fused_std: Optional[np.ndarray] = None,
        walker_work: Optional[np.ndarray] = None,
        nmi_score: float = 0.0,
        ari_score: float = 0.0,
    ) -> Path:
        """
        Save T3: Social topology (bonds, cracks, ruptures).
        """
        path = self.checkpoint_dir / "T3_topology.npz"

        arrays = {
            "bond_matrix": bond_matrix,
            "crack_matrix": crack_matrix,
        }
        if dirichlet_fused is not None:
            arrays["dirichlet_fused"] = dirichlet_fused
        if dirichlet_fused_std is not None:
            arrays["dirichlet_fused_std"] = dirichlet_fused_std
        if walker_work is not None:
            arrays["walker_work"] = walker_work

        np.savez(path, **arrays)

        # Save topology metadata as JSON
        topo_json = {
            "rupture_pairs": rupture_pairs,
            "nmi_score": nmi_score,
            "ari_score": ari_score,
            "n_ruptures": len(rupture_pairs),
            "timestamp": self._now(),
        }
        with open(self.checkpoint_dir / "T3_topology.json", 'w') as f:
            json.dump(topo_json, f, indent=2)

        self.manifest["checkpoints"]["T3"] = str(path)
        print(f"[Checkpoint T3] Saved topology: NMI={nmi_score:.3f}, {len(rupture_pairs)} ruptures")
        return path

    # =========================================================================
    # PHASE 4: VISUALIZATION
    # =========================================================================

    def save_t4_viz_coords(
        self,
        x_umap: np.ndarray,
        x_tsne: Optional[np.ndarray] = None,
        x_metric_umap: Optional[np.ndarray] = None,
        metric_kernel: str = "rbf",
        terrain_states: Optional[List[str]] = None,
    ) -> Path:
        """
        Save T4: 2D visualization coordinates.
        """
        import csv

        path = self.checkpoint_dir / "T4_viz_coords.csv"

        # Build rows
        n = x_umap.shape[0]
        rows = []
        for i in range(n):
            row = {
                "article_idx": i,
                "x_umap": x_umap[i, 0],
                "y_umap": x_umap[i, 1],
            }
            if terrain_states is not None and i < len(terrain_states):
                row["terrain_state"] = str(terrain_states[i])
            if x_tsne is not None:
                row["x_tsne"] = x_tsne[i, 0]
                row["y_tsne"] = x_tsne[i, 1]
            if x_metric_umap is not None:
                row["x_metric_umap"] = x_metric_umap[i, 0]
                row["y_metric_umap"] = x_metric_umap[i, 1]
            rows.append(row)

        # Write CSV
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Also save as NPZ for convenience
        npz_path = self.checkpoint_dir / "T4_viz_coords.npz"
        arrays = {"x_umap": x_umap}
        if terrain_states is not None:
            arrays["terrain_state"] = np.asarray(terrain_states, dtype=object)
        if x_tsne is not None:
            arrays["x_tsne"] = x_tsne
        if x_metric_umap is not None:
            arrays["x_metric_umap"] = x_metric_umap
        np.savez(npz_path, **arrays)

        self.manifest["checkpoints"]["T4"] = str(path)
        print(f"[Checkpoint T4] Saved viz coordinates: {path}")
        return path

    # =========================================================================
    # MANIFEST & LOADING
    # =========================================================================

    def save_manifest(self) -> Path:
        """Save the checkpoint manifest."""
        path = self.checkpoint_dir / "manifest.json"
        self.manifest["finalized"] = self._now()
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"[Checkpoint] Manifest saved: {path}")
        return path

    def load_all(self) -> Dict[str, Any]:
        """Load all available checkpoints."""
        result = {}

        # T0
        t0_path = self.checkpoint_dir / "T0_substrate.npy"
        if t0_path.exists():
            result["T0_substrate"] = np.load(t0_path)

        # T1
        t1_path = self.checkpoint_dir / "T1_embeddings_raw.npy"
        if t1_path.exists():
            result["T1_embeddings"] = np.load(t1_path)

        # T1.5
        t15_path = self.checkpoint_dir / "T1.5_spectral_state.npz"
        if t15_path.exists():
            result["T1.5_spectral"] = dict(np.load(t15_path))

        # T2
        t2_path = self.checkpoint_dir / "T2_kernel_projections.npz"
        if t2_path.exists():
            result["T2_kernels"] = dict(np.load(t2_path))

        # T3
        t3_path = self.checkpoint_dir / "T3_topology.npz"
        if t3_path.exists():
            result["T3_topology"] = dict(np.load(t3_path))
        t3_json = self.checkpoint_dir / "T3_topology.json"
        if t3_json.exists():
            with open(t3_json) as f:
                result["T3_topology_meta"] = json.load(f)

        # T4
        t4_path = self.checkpoint_dir / "T4_viz_coords.npz"
        if t4_path.exists():
            result["T4_viz"] = dict(np.load(t4_path))

        return result

    @staticmethod
    def diagnose_score_anomaly(checkpoint_dir: Path) -> str:
        """
        Run diagnostic checks on saved checkpoints to identify score anomalies.

        Returns human-readable diagnostic report.
        """
        report = ["=" * 60, "WATERFALL DIAGNOSTIC REPORT", "=" * 60, ""]

        ckpt = WaterfallCheckpoint(checkpoint_dir.parent, checkpoint_dir.name)
        artifacts = ckpt.load_all()

        # Check T0: Model failure?
        if "T0_substrate" in artifacts:
            probs = artifacts["T0_substrate"]
            neutral_ratio = (probs[:, :, 1] > 0.9).mean()  # High neutral = model confusion
            if neutral_ratio > 0.8:
                report.append("⚠️  T0 ALERT: Model produced >80% high-confidence NEUTRAL")
                report.append(f"   Neutral ratio: {neutral_ratio:.1%}")
            else:
                report.append("✓  T0 OK: Model produced varied predictions")

        # Check T1.5: Spectral compass broken?
        if "T1.5_spectral" in artifacts:
            evr_meta_path = checkpoint_dir / "T1.5_spectral_meta.json"
            if evr_meta_path.exists():
                with open(evr_meta_path) as f:
                    meta = json.load(f)
                evr = meta.get("evr", 0)
                if evr < 0.1:
                    report.append(f"⚠️  T1.5 ALERT: EVR={evr:.3f} < 0.1 (compass failed)")
                elif evr < 0.5:
                    report.append(f"⚠️  T1.5 WARNING: EVR={evr:.3f} < 0.5 (weak dipole)")
                else:
                    report.append(f"✓  T1.5 OK: EVR={evr:.3f} (valid dipole)")

        # Check T2: Kernel collapse?
        if "T2_kernels" in artifacts:
            t2 = artifacts["T2_kernels"]
            active_key = next((key for key in ("z_rbf", "z_matern", "z_laplacian", "z_imq", "z_rq") if key in t2), None)
            if active_key is not None:
                z_active = t2.get(active_key)
                variance = z_active.var()
                family = active_key.removeprefix("z_").upper()
                if variance < 1e-6:
                    report.append(f"⚠️  T2 ALERT: {family} variance={variance:.2e} (collapsed!)")
                else:
                    report.append(f"✓  T2 OK: {family} variance={variance:.4f}")

        # Check T3: Topology scores
        if "T3_topology_meta" in artifacts:
            meta = artifacts["T3_topology_meta"]
            nmi = meta.get("nmi_score", 0)
            n_rupt = meta.get("n_ruptures", 0)
            report.append(f"   T3 Scores: NMI={nmi:.3f}, Ruptures={n_rupt}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
