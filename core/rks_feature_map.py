"""
rks_feature_map.py

Reusable kernel feature map utility supporting:
- Injected shared basis (for CRN/reproducibility)
- Both 2D [N, H] and 3D [N, B, H] inputs
- Legacy multi-framing mode (via MultiFramingRKS)
- Direct RKS mode (no framing structure)

USAGE MODES:
1. Legacy framing mode (default): Uses MultiFramingRKS for structured logits
2. Direct mode: Simple RKS projection for CLS embeddings
3. Injected basis mode: Accepts external (omega, b) for CRN

This file provides the kernel mapping used by:
- complete_pipeline.py (legacy path)
- dirichlet_fusion.py (new CLS path)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .kernel_library import (
    canonical_shape_params,
    compute_basis_hash,
    normalize_kernel_type,
    sample_spectral_frequencies,
    validate_kernel_provenance,
    validate_positive_sigma,
    validate_shape_params,
)
from .multi_framing_rks import MultiFramingRKS


SHARED_BASIS_GENERATED_CODE_PATH = "core.rks_feature_map.SharedBasis.generated"
SHARED_BASIS_INJECTED_CODE_PATH = "core.rks_feature_map.SharedBasis.injected"
_UNSET = object()


def _resolve_legacy_gamma_values(
    kernel_type: str,
    sigma: Optional[float],
    gamma: Optional[float],
) -> Tuple[Optional[float], None]:
    """Validate and resolve the legacy RBF gamma alias."""
    if gamma is None:
        return sigma, None
    if normalize_kernel_type(kernel_type) != "rbf":
        raise ValueError("gamma is a legacy bandwidth alias supported only for the RBF family")
    if sigma is not None:
        raise ValueError("gamma and an explicit sigma are mutually exclusive")
    try:
        gamma_f = validate_positive_sigma(gamma, "gamma")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gamma must be a positive finite value, got {gamma!r}") from exc
    return math.sqrt(1.0 / (2.0 * gamma_f)), None


@dataclass
class RKSConfig:
    input_dim: int
    output_dim: int
    n_framings: int = 8
    dims_per_framing: Optional[int] = None
    output_per_framing: Optional[int] = None

    kernel_type: str = "rbf"
    sigma: Optional[float] = None
    gamma: Optional[float] = None  # legacy bandwidth alias; never a shape parameter
    kernel_params: Optional[Dict[str, Any]] = None

    random_seed: int = 42
    device: str = "cuda"

    auto_sigma: bool = True
    verbose: bool = False
    
    # NEW: Direct mode (no framing structure)
    direct_mode: bool = False
    
    # NEW: Injected basis for CRN
    injected_omega: Optional[torch.Tensor] = None
    injected_bias: Optional[torch.Tensor] = None
    injected_provenance: Optional[Dict[str, Any]] = None
    
    # NEW: KernelContext integration
    kernel_ctx: Optional[Any] = None  # KernelContext for provenance
    
    def __post_init__(self):
        """Apply KernelContext overrides if provided."""
        if self.kernel_ctx is not None:
            try:
                self.kernel_type = self.kernel_ctx.kernel_type
                if self.kernel_ctx.bandwidth is not None:
                    self.sigma = self.kernel_ctx.bandwidth
                if self.kernel_ctx.rks_dim is not None:
                    self.output_dim = self.kernel_ctx.rks_dim
                self.random_seed = self.kernel_ctx.seed
            except AttributeError:
                pass  # kernel_ctx doesn't have expected attributes


class SharedBasis(nn.Module):
    """
    Shared RKS basis that can be saved/loaded for CRN (Common Random Numbers).
    
    This ensures reproducibility across conditions by using the same (omega, b).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        omega: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        kernel_type: str = "rbf",
        sigma: Optional[float] = None,
        kernel_params: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.requested_kernel_type = str(kernel_type)
        self.kernel_type = normalize_kernel_type(kernel_type)
        self.kernel_params = (
            canonical_shape_params(self.kernel_type)
            if kernel_params is None
            else validate_shape_params(
                self.kernel_type, dict(kernel_params), name="kernel_params"
            )
        )

        if omega is not None and bias is not None:
            # Use injected basis
            assert omega.shape == (input_dim, output_dim), f"omega shape mismatch: {omega.shape}"
            assert bias.shape == (output_dim,), f"bias shape mismatch: {bias.shape}"
            self.register_buffer('omega', omega.clone())
            self.register_buffer('b', bias.clone())
            if provenance is None:
                raise ValueError(
                    "Injected RKS tensors require kernel provenance for every family, including RBF."
                )
            injected_hash = compute_basis_hash(omega, bias)
            validate_kernel_provenance(
                provenance,
                requested_kernel=self.kernel_type,
                input_dim=input_dim,
                output_dim=output_dim,
                shape_params=self.kernel_params,
                require_dimensions=True,
                seed=seed,
                basis_hash=injected_hash,
            )
            provenance_sigma = provenance.get("sigma")
            if sigma is not None and provenance_sigma is not None:
                if not np.isclose(float(sigma), float(provenance_sigma), rtol=0.0, atol=1e-12):
                    raise ValueError("Injected basis sigma does not match kernel provenance")
            if provenance["sampler_sigma_mode"] != "runtime_forward_scaling":
                raise ValueError(
                    "Injected SharedBasis tensors require runtime_forward_scaling provenance"
                )
            self.provenance = dict(provenance)
            self.provenance["code_path"] = SHARED_BASIS_INJECTED_CODE_PATH
            self.provenance["sampler_sigma_mode"] = "runtime_forward_scaling"
        else:
            # Generate deterministic basis
            omega_out, bias_out, self.provenance = sample_spectral_frequencies(
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_type=self.kernel_type,
                sigma=1.0,
                seed=seed,
                **self.kernel_params,
            )
            self.provenance["sampler_sigma_mode"] = "runtime_forward_scaling"
            self.register_buffer('omega', omega_out.t().contiguous())
            self.register_buffer('b', bias_out)
            self.provenance["basis_hash"] = compute_basis_hash(self.omega, self.b)
            self.provenance["code_path"] = SHARED_BASIS_GENERATED_CODE_PATH
        
        self._sigma: Optional[float] = None
        if sigma is not None:
            self.set_sigma(float(sigma))
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for provenance tracking."""
        return compute_basis_hash(self.omega, self.b)
    
    @property
    def basis_hash(self) -> str:
        return self._compute_hash()

    def validate_provenance(self) -> Dict[str, Any]:
        """Authenticate current owned tensors against their provenance."""
        return validate_kernel_provenance(
            self.provenance,
            requested_kernel=self.requested_kernel_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            shape_params=self.kernel_params,
            require_dimensions=True,
            seed=self.seed,
            sigma=self._sigma,
            basis_hash=self._compute_hash(),
        )
    
    def set_sigma(self, sigma: float):
        validated = validate_positive_sigma(sigma)
        self._sigma = validated
        self.provenance["sigma"] = validated
    
    def estimate_sigma(self, X: torch.Tensor, policy: str = 'median') -> float:
        """Estimate bandwidth from data."""
        with torch.no_grad():
            flat = X.reshape(-1, self.input_dim)
            n = min(500, flat.shape[0])
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(self.seed + 104729)
            idx = torch.randperm(flat.shape[0], generator=generator, device=flat.device)[:n]
            X_sub = flat[idx]
            
            dists = torch.cdist(X_sub, X_sub)
            mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            
            if dists[mask].numel() == 0:
                return 1.0
            
            if policy == 'median':
                sigma = float(torch.median(dists[mask]).item())
            else:
                sigma = float(torch.mean(dists[mask]).item())
            
            return max(sigma, 1e-6)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project to RKHS: phi(x) = sqrt(2/D) * cos(x @ omega/sigma + b)
        
        Supports both 2D [N, H] and 3D [N, B, H] inputs.
        """
        if self._sigma is None:
            raise ValueError("Sigma not set. Call set_sigma() first.")
        self.validate_provenance()
        
        omega = self.omega.to(X.device) / self._sigma
        b = self.b.to(X.device)
        
        # Handle both 2D and 3D inputs
        original_shape = X.shape
        if X.dim() == 3:
            N, B, H = X.shape
            X = X.reshape(N * B, H)
        
        proj = X @ omega + b
        scale = np.sqrt(2.0 / self.output_dim)
        result = scale * torch.cos(proj)
        
        if len(original_shape) == 3:
            result = result.reshape(N, B, -1)
        
        return result
    
    def save(self, path: str):
        """Save basis for reuse."""
        if self._sigma is None:
            raise ValueError("Cannot persist an RKS basis without an explicit sigma/bandwidth")
        self._hash = self._compute_hash()
        self.validate_provenance()
        torch.save({
            'omega': self.omega.cpu(),
            'b': self.b.cpu(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seed': self.seed,
            'kernel_type': self.kernel_type,
            'requested_kernel_type': self.requested_kernel_type,
            'kernel_params': self.kernel_params,
            'kernel_provenance': self.provenance,
            'sigma': self._sigma,
            'hash': self._hash,
        }, path)
    
    @classmethod
    def load(cls, path: str, expected_kernel_type: Optional[str] = None) -> 'SharedBasis':
        """Load pre-generated basis."""
        data = torch.load(path, map_location='cpu')
        required = {
            'omega', 'b', 'input_dim', 'output_dim', 'seed', 'kernel_type',
            'requested_kernel_type', 'kernel_params', 'kernel_provenance', 'sigma', 'hash',
        }
        missing = sorted(field for field in required if field not in data)
        if missing:
            raise ValueError(
                f"Basis at {path} lacks required identity fields: {', '.join(missing)}; "
                "regenerate or explicitly migrate the basis."
            )
        invalid = sorted(
            field for field in required
            if field != 'kernel_params' and data[field] is None
        )
        if invalid or not isinstance(data['kernel_params'], dict):
            details = invalid or ['kernel_params']
            raise ValueError(
                f"Basis at {path} has invalid identity fields: {', '.join(details)}"
            )
        saved_kernel = data['kernel_type']
        canonical_saved = normalize_kernel_type(saved_kernel)
        saved_requested = data['requested_kernel_type']
        if normalize_kernel_type(saved_requested) != canonical_saved:
            raise ValueError(
                f"Basis at {path} requested {saved_requested}, but records actual {canonical_saved}"
            )
        if expected_kernel_type is not None and normalize_kernel_type(expected_kernel_type) != canonical_saved:
            raise ValueError(
                f"Basis at {path} is {canonical_saved}, not requested {normalize_kernel_type(expected_kernel_type)}"
            )
        provenance = data['kernel_provenance']
        if provenance is None:
            raise ValueError(
                f"Basis at {path} lacks kernel_provenance; anonymous legacy tensors cannot "
                "prove a kernel family. Regenerate or explicitly migrate the basis."
            )
        validate_kernel_provenance(
            provenance,
            requested_kernel=expected_kernel_type or saved_requested,
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            shape_params=data['kernel_params'],
            require_dimensions=True,
            seed=data['seed'],
        )
        provenance_sigma = provenance.get('sigma')
        stored_sigma = validate_positive_sigma(data['sigma'], "Persisted basis sigma")
        if not np.isclose(
            float(provenance_sigma), stored_sigma, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Basis sigma provenance mismatch at {path}")
        computed_hash = compute_basis_hash(data['omega'], data['b'])
        stored_hash = data['hash']
        if not stored_hash:
            raise ValueError(
                f"Basis at {path} lacks a tensor hash; regenerate or explicitly migrate the basis."
            )
        if stored_hash != computed_hash:
            raise ValueError(
                f"Basis tensor hash mismatch at {path}: stored={stored_hash}, computed={computed_hash}"
            )
        validate_kernel_provenance(
            provenance,
            requested_kernel=expected_kernel_type or saved_requested,
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            shape_params=data['kernel_params'],
            require_dimensions=True,
            seed=data['seed'],
            sigma=stored_sigma,
            basis_hash=computed_hash,
        )
        basis = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            seed=data['seed'],
            omega=data['omega'],
            bias=data['b'],
            kernel_type=saved_requested,
            sigma=stored_sigma,
            kernel_params=data['kernel_params'],
            provenance=provenance,
        )
        basis.provenance = dict(provenance)
        basis._sigma = stored_sigma
        basis._hash = computed_hash
        return basis


class RKSFeatureMap:
    """
    Deterministic multi-framing Random Kitchen Sinks feature map.

    MODES:
    1. Legacy framing mode (default): Uses MultiFramingRKS for structured inputs
    2. Direct mode (direct_mode=True): Simple RKS for flat/block inputs
    3. Injected basis mode: Use external basis for CRN reproducibility

    Important invariants:
    - Never mixes CLS and logits channels (that's upstream in NLI extraction).
    - Preserves framing block structure in legacy mode.
    - Direct mode supports both [N, H] and [N, B, H] inputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_type: str = "rbf",
        gamma: Optional[float] = None,
        sigma: Optional[float] = None,
        n_framings: int = 8,
        dims_per_framing: Optional[int] = None,
        output_per_framing: Optional[int] = None,
        random_seed: int = 42,
        device: str = "cuda",
        kernel_params: Optional[Dict[str, Any]] = None,
        auto_sigma: bool = True,
        verbose: bool = False,
        # NEW: Direct mode (bypass MultiFramingRKS)
        direct_mode: bool = False,
        # NEW: Injected basis for CRN
        shared_basis: Optional[SharedBasis] = None,
        injected_omega: Optional[torch.Tensor] = None,
        injected_bias: Optional[torch.Tensor] = None,
        injected_provenance: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        self.cfg = RKSConfig(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            n_framings=int(n_framings),
            dims_per_framing=(int(dims_per_framing) if dims_per_framing is not None else None),
            output_per_framing=(int(output_per_framing) if output_per_framing is not None else None),
            kernel_type=str(kernel_type),
            sigma=(float(sigma) if sigma is not None else None),
            gamma=gamma,
            kernel_params=(dict(kernel_params) if kernel_params is not None else None),
            random_seed=int(random_seed),
            device=str(device),
            auto_sigma=bool(auto_sigma),
            verbose=bool(verbose),
            direct_mode=bool(direct_mode),
            injected_omega=injected_omega,
            injected_bias=injected_bias,
            injected_provenance=(dict(injected_provenance) if injected_provenance is not None else None),
        )

        self._normalize_legacy_gamma()
        
        # Store shared basis if provided
        self._shared_basis = shared_basis
        
        # Choose mode based on config
        if direct_mode or shared_basis is not None or injected_omega is not None:
            self._mode = 'direct'
            self._build_direct()
        else:
            self._mode = 'framing'
            self._infer_dims()
            self._build()

    def _normalize_legacy_gamma(self) -> None:
        """Resolve the legacy RBF-style gamma alias without polluting shape params."""
        self.cfg.sigma, self.cfg.gamma = _resolve_legacy_gamma_values(
            self.cfg.kernel_type, self.cfg.sigma, self.cfg.gamma
        )
    
    def _build_direct(self) -> None:
        """Build direct RKS (no framing structure)."""
        if (self.cfg.injected_omega is None) != (self.cfg.injected_bias is None):
            raise ValueError("injected_omega and injected_bias must be provided together")
        if self._shared_basis is not None:
            self.basis = self._shared_basis
            if normalize_kernel_type(self.cfg.kernel_type) != normalize_kernel_type(self.basis.kernel_type):
                raise ValueError(
                    f"Shared basis kernel {self.basis.kernel_type} does not match "
                    f"requested {self.cfg.kernel_type}"
                )
        elif self.cfg.injected_omega is not None and self.cfg.injected_bias is not None:
            self.basis = SharedBasis(
                input_dim=self.cfg.input_dim,
                output_dim=self.cfg.output_dim,
                seed=self.cfg.random_seed,
                omega=self.cfg.injected_omega,
                bias=self.cfg.injected_bias,
                kernel_type=self.cfg.kernel_type,
                sigma=self.cfg.sigma,
                kernel_params=self.cfg.kernel_params,
                provenance=self.cfg.injected_provenance,
            )
        else:
            self.basis = SharedBasis(
                input_dim=self.cfg.input_dim,
                output_dim=self.cfg.output_dim,
                seed=self.cfg.random_seed,
                kernel_type=self.cfg.kernel_type,
                sigma=self.cfg.sigma,
                kernel_params=self.cfg.kernel_params,
            )
        
        if self.cfg.sigma is not None:
            self.basis.set_sigma(self.cfg.sigma)
        
        self.rks = None  # Not using MultiFramingRKS in direct mode

    def _infer_dims(self) -> None:
        # Infer dims_per_framing from input_dim if not given
        if self.cfg.dims_per_framing is None:
            if self.cfg.input_dim % self.cfg.n_framings != 0:
                raise ValueError(
                    f"input_dim={self.cfg.input_dim} must be divisible by n_framings={self.cfg.n_framings} "
                    "to infer dims_per_framing safely."
                )
            self.cfg.dims_per_framing = self.cfg.input_dim // self.cfg.n_framings

        # Allocate output evenly per framing unless explicitly overridden
        if self.cfg.output_per_framing is None:
            if self.cfg.output_dim % self.cfg.n_framings != 0:
                raise ValueError(
                    f"output_dim={self.cfg.output_dim} must be divisible by n_framings={self.cfg.n_framings} "
                    "or set output_per_framing explicitly."
                )
            self.cfg.output_per_framing = self.cfg.output_dim // self.cfg.n_framings

        # Shape parameters remain family-specific. Legacy gamma is normalized to
        # sigma before this point and must never enter this dictionary.
        self.cfg.kernel_params = (
            canonical_shape_params(self.cfg.kernel_type)
            if self.cfg.kernel_params is None
            else dict(self.cfg.kernel_params)
        )

    def _build(self) -> None:
        self.rks = MultiFramingRKS(
            n_framings=self.cfg.n_framings,
            dims_per_framing=int(self.cfg.dims_per_framing),
            output_per_framing=int(self.cfg.output_per_framing),
            kernel_types=self.cfg.kernel_type,
            sigma=self.cfg.sigma,
            kernel_params=self.cfg.kernel_params,
            seed=self.cfg.random_seed,
            device=self.cfg.device,
        )

    @property
    def kernel_type(self) -> str:
        return self.cfg.kernel_type

    @kernel_type.setter
    def kernel_type(self, value: str) -> None:
        # Do NOT allow silent mutation without rebuild; callers should call rebuild().
        self.cfg.kernel_type = str(value)

    def rebuild(
        self,
        kernel_type: Optional[str] = None,
        sigma: Any = _UNSET,
        kernel_params: Optional[Dict[str, Any]] = None,
        gamma: Any = _UNSET,
        device: Optional[str] = None,
    ) -> None:
        """
        Reinitialize the RKS kernels (needed when you change kernel_type / sigma).
        """
        next_kernel_type = self.cfg.kernel_type if kernel_type is None else str(kernel_type)
        next_sigma = self.cfg.sigma if sigma is _UNSET else sigma
        next_gamma = self.cfg.gamma if gamma is _UNSET else gamma
        next_sigma = float(next_sigma) if next_sigma is not None else None
        next_sigma, next_gamma = _resolve_legacy_gamma_values(
            next_kernel_type, next_sigma, next_gamma
        )

        self.cfg.kernel_type = next_kernel_type
        self.cfg.sigma = next_sigma
        self.cfg.gamma = next_gamma
        if kernel_params is not None:
            self.cfg.kernel_params = dict(kernel_params)
        if device is not None:
            self.cfg.device = str(device)

        self._infer_dims()
        self._build()

    def transform(self, X: torch.Tensor, rep_kind: Optional[str] = None) -> torch.Tensor:
        """
        Apply the RKS map.
        
        For direct mode: handles both [N, H] and [N, B, H] inputs.
        For framing mode: returns [N, output_dim] combined output.
        
        Args:
            X: Input tensor
            rep_kind: Optional rep_kind string for contract enforcement.
                      If 'logits_raw', raises ValueError (defense-in-depth).
        """
        # DEFENSE-IN-DEPTH: Refuse to transform logits
        if rep_kind is not None and rep_kind.lower() == 'logits_raw':
            raise ValueError(
                "[CONSTITUTIONAL CONTRACT] RKSFeatureMap.transform() refuses rep_kind='logits_raw'. "
                "Logits are verdict coordinates and must not be kernelized."
            )
        
        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self._mode == 'direct':
            return self._transform_direct(X)
        else:
            return self._transform_framing(X)
    
    def _transform_direct(self, X: torch.Tensor) -> torch.Tensor:
        """Direct mode transform using SharedBasis."""
        # Estimate sigma if needed
        if self.basis._sigma is None:
            if self.cfg.auto_sigma:
                sigma = self.basis.estimate_sigma(X)
                self.basis.set_sigma(sigma)
                if self.cfg.verbose:
                    print(f"[RKSFeatureMap] Auto-estimated sigma = {sigma:.4f}")
            else:
                raise ValueError("Sigma not set and auto_sigma=False")
        
        # Move basis to device
        if X.device != self.basis.omega.device:
            self.basis = self.basis.to(X.device)
        
        return self.basis(X)
    
    def _transform_framing(self, X: torch.Tensor) -> torch.Tensor:
        """Legacy framing mode transform using MultiFramingRKS."""
        # Ensure module is on the same device
        try:
            if next(self.rks.parameters()).device != X.device:
                self.rks = self.rks.to(X.device)
        except StopIteration:
            self.rks = self.rks.to(X.device)

        # Ensure sigma is set before forward
        if hasattr(self.rks, "estimate_and_set_sigma"):
            need_sigma = getattr(self.rks, "sigma", None) is None
            if self.cfg.auto_sigma or need_sigma:
                try:
                    _ = self.rks.estimate_and_set_sigma(X, sample_size=1000, percentile=50.0, per_framing=True)
                except Exception as e:
                    if getattr(self.rks, "sigma", None) is None:
                        raise RuntimeError(f"RKS sigma estimation failed: {e}") from e

        out = self.rks(X)
        if isinstance(out, dict):
            if "combined" in out:
                return out["combined"]
            for v in out.values():
                if torch.is_tensor(v):
                    return v
            raise RuntimeError("MultiFramingRKS returned dict without tensor outputs.")
        return out
    
    def get_basis(self) -> Optional[SharedBasis]:
        """Get the shared basis (direct mode only)."""
        if self._mode == 'direct':
            return self.basis
        return None
    
    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance metadata for artifact logging."""
        prov = {
            'mode': self._mode,
            'input_dim': self.cfg.input_dim,
            'output_dim': self.cfg.output_dim,
            'kernel_type': self.cfg.kernel_type,
            'seed': self.cfg.random_seed,
        }
        
        if self._mode == 'direct' and self.basis is not None:
            self.basis.validate_provenance()
            prov['basis_hash'] = self.basis.basis_hash
            prov['sigma'] = self.basis._sigma
            prov['requested_kernel'] = self.basis.provenance['requested_kernel']
            prov['actual_kernel'] = self.basis.provenance['actual_kernel']
            prov['sampler_id'] = self.basis.provenance['sampler_id']
            prov['kernel_provenance'] = dict(self.basis.provenance)
        elif self._mode == 'framing' and self.rks is not None:
            prov['sigma'] = getattr(self.rks, 'sigma', None)
            prov['n_framings'] = self.cfg.n_framings
            prov['framing_provenance'] = self.rks.get_provenance()['framings']
        
        return prov

    def get_full_output(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Return the full output dict (for debugging / research).
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self._mode == 'direct':
            features = self._transform_direct(X)
            return {
                'combined': features,
                'mode': 'direct',
                'basis_hash': self.basis.basis_hash,
                'sigma': self.basis._sigma,
            }

        # Legacy framing mode
        try:
            if next(self.rks.parameters()).device != X.device:
                self.rks = self.rks.to(X.device)
        except StopIteration:
            self.rks = self.rks.to(X.device)

        if hasattr(self.rks, "estimate_and_set_sigma"):
            need_sigma = getattr(self.rks, "sigma", None) is None
            if self.cfg.auto_sigma or need_sigma:
                try:
                    _ = self.rks.estimate_and_set_sigma(X, sample_size=1000, percentile=50.0, per_framing=True)
                except Exception as e:
                    if getattr(self.rks, "sigma", None) is None:
                        raise RuntimeError(f"RKS sigma estimation failed: {e}") from e

        out = self.rks(X)
        if isinstance(out, dict):
            out['mode'] = 'framing'
            return out
        return {"combined": out, "mode": "framing"}


# =============================================================================
# CONVENIENCE FACTORIES
# =============================================================================

def create_shared_basis(
    input_dim: int,
    output_dim: int,
    seed: int = 42,
    sigma: Optional[float] = None,
    kernel_type: str = "rbf",
    kernel_params: Optional[Dict[str, Any]] = None,
) -> SharedBasis:
    """Create a SharedBasis for CRN use."""
    basis = SharedBasis(
        input_dim,
        output_dim,
        seed,
        kernel_type=kernel_type,
        sigma=sigma,
        kernel_params=kernel_params,
    )
    if sigma is not None:
        basis.set_sigma(sigma)
    return basis


def create_direct_rks(
    input_dim: int,
    output_dim: int,
    seed: int = 42,
    sigma: Optional[float] = None,
    shared_basis: Optional[SharedBasis] = None,
    kernel_type: str = "rbf",
    kernel_params: Optional[Dict[str, Any]] = None,
) -> RKSFeatureMap:
    """Create an RKSFeatureMap in direct mode (for CLS embeddings)."""
    return RKSFeatureMap(
        input_dim=input_dim,
        output_dim=output_dim,
        kernel_type=kernel_type,
        random_seed=seed,
        sigma=sigma,
        kernel_params=kernel_params,
        direct_mode=True,
        shared_basis=shared_basis,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'RKSConfig',
    'RKSFeatureMap',
    'SharedBasis',
    'create_shared_basis',
    'create_direct_rks',
]
