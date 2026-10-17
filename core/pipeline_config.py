from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


TRACK5_ASSEMBLY_MODE_HADAMARD = "hadamard"
TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN = "strict_riemannian"
TRACK5_ASSEMBLY_MODE_CONCATENATE = "concatenate"


def normalize_track5_assembly_mode(
    mode: Any,
    *,
    default: str = TRACK5_ASSEMBLY_MODE_HADAMARD,
) -> str:
    """Map user/config variants onto the supported Track 5 assembly modes."""
    normalized_default = str(default or TRACK5_ASSEMBLY_MODE_HADAMARD).strip().lower()
    text = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": normalized_default,
        "hadamard": TRACK5_ASSEMBLY_MODE_HADAMARD,
        "hadamard_strict": TRACK5_ASSEMBLY_MODE_HADAMARD,
        "strict_riemannian": TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
        "riemannian": TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
        "riemannian_strict": TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
        "strict": TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
        "concat": TRACK5_ASSEMBLY_MODE_CONCATENATE,
        "concatenate": TRACK5_ASSEMBLY_MODE_CONCATENATE,
        "none": TRACK5_ASSEMBLY_MODE_CONCATENATE,
    }
    return aliases.get(text, normalized_default)


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    """Single source of truth for pipeline runtime knobs consumed by runners."""

    use_contrastive: bool = False
    use_pca_removal: bool = False
    use_cls_tokens: bool = False
    use_gru: bool = True
    use_multi_framing_rks: bool = True
    use_attention: bool = True
    use_dirichlet_fusion: bool = False
    normalize_features: bool = True
    normalize_before_projection: bool = True
    geometry_mode: str = "rks"
    kernel_type: str = "matern"
    track5_assembly_mode: str = TRACK5_ASSEMBLY_MODE_HADAMARD

    # Experimental Breakthrough Toggles (80/20 optimizations)
    use_nli_contrast_gate: bool = False
    align_observer_gauges: bool = False
    transport_metric_type: str = "euclidean"

    projection_dim: int = 256
    apply_pca_to_cls: bool = True
    dirichlet_rks_dim: int = 2048
    dirichlet_n_observers: int = 50
    dirichlet_alpha: float = 1.0
    dirichlet_hidden_dim: int = 1536

    @classmethod
    def from_mode_config(
        cls,
        mode_config: Mapping[str, Any],
        *,
        kernel_override: Optional[str] = None,
    ) -> "PipelineRuntimeConfig":
        if "mix_in_rkhs" in mode_config:
            raise ValueError("MigrationError: mix_in_rkhs has been removed from the architecture. Please update your config to remove this dead flag.")

        return cls(
            use_contrastive=bool(mode_config.get("use_contrastive", cls.use_contrastive)),
            use_pca_removal=bool(mode_config.get("use_pca_removal", cls.use_pca_removal)),
            use_cls_tokens=bool(mode_config.get("use_cls_tokens", cls.use_cls_tokens)),
            use_gru=bool(mode_config.get("use_gru", cls.use_gru)),
            use_multi_framing_rks=bool(mode_config.get("use_multi_framing_rks", cls.use_multi_framing_rks)),
            use_attention=bool(mode_config.get("use_attention", cls.use_attention)),
            use_dirichlet_fusion=bool(mode_config.get("use_dirichlet_fusion", cls.use_dirichlet_fusion)),
            normalize_features=bool(mode_config.get("normalize_features", cls.normalize_features)),
            normalize_before_projection=bool(
                mode_config.get("normalize_before_projection", cls.normalize_before_projection)
            ),
            geometry_mode=str(mode_config.get("geometry_mode", cls.geometry_mode)),
            kernel_type=str(kernel_override or mode_config.get("kernel_type", cls.kernel_type)),
            track5_assembly_mode=normalize_track5_assembly_mode(
                mode_config.get("track5_assembly_mode", cls.track5_assembly_mode),
                default=cls.track5_assembly_mode,
            ),
            use_nli_contrast_gate=bool(mode_config.get("use_nli_contrast_gate", cls.use_nli_contrast_gate)),
            align_observer_gauges=bool(mode_config.get("align_observer_gauges", cls.align_observer_gauges)),
            transport_metric_type=str(mode_config.get("transport_metric_type", cls.transport_metric_type)),
            projection_dim=int(mode_config.get("projection_dim", cls.projection_dim)),
            apply_pca_to_cls=bool(mode_config.get("apply_pca_to_cls", cls.apply_pca_to_cls)),
            dirichlet_rks_dim=int(mode_config.get("dirichlet_rks_dim", cls.dirichlet_rks_dim)),
            dirichlet_n_observers=int(mode_config.get("dirichlet_n_observers", cls.dirichlet_n_observers)),
            dirichlet_alpha=float(mode_config.get("dirichlet_alpha", cls.dirichlet_alpha)),
            dirichlet_hidden_dim=int(mode_config.get("dirichlet_hidden_dim", cls.dirichlet_hidden_dim)),
        )

    def to_initialize_kwargs(self) -> dict[str, Any]:
        return {
            "use_contrastive": self.use_contrastive,
            "use_cls_tokens": self.use_cls_tokens,
            "use_gru": self.use_gru,
            "use_rks": self.use_multi_framing_rks,
            "use_attention": self.use_attention,
            "use_dirichlet_fusion": self.use_dirichlet_fusion,
            "normalize_features": self.normalize_features,
            "geometry_mode": self.geometry_mode,
            "kernel_type": self.kernel_type,
            "use_nli_contrast_gate": getattr(self, "use_nli_contrast_gate", False),
            "align_observer_gauges": getattr(self, "align_observer_gauges", False),
            "transport_metric_type": getattr(self, "transport_metric_type", "euclidean"),
            "track5_assembly_mode": normalize_track5_assembly_mode(
                self.track5_assembly_mode,
                default=TRACK5_ASSEMBLY_MODE_HADAMARD,
            ),
            "pca_remove": self.use_pca_removal,
            "dirichlet_rks_dim": self.dirichlet_rks_dim,
            "dirichlet_n_observers": self.dirichlet_n_observers,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_hidden_dim": self.dirichlet_hidden_dim,
        }


DEFAULT_PIPELINE_RUNTIME_CONFIG = PipelineRuntimeConfig()
