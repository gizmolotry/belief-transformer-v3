"""
Canonical kernel and Random Kitchen Sinks utilities.

This module owns exact kernel formulas, spectral samplers, sampler provenance,
and basis hashing used by kernelized observers. Other modules may wrap these
utilities, but must not silently substitute one kernel family for another.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.special import gamma as scipy_gamma
from scipy.special import kv as scipy_kv

try:
    from .kernel_context import KernelContext
except ImportError:  # pragma: no cover - optional type/import path
    KernelContext = None


CANONICAL_KERNEL_ALIASES = {
    "rbf": "rbf",
    "gaussian": "rbf",
    "laplacian": "laplacian",
    "laplace": "laplacian",
    "l1_laplacian": "laplacian",
    "rq": "rq",
    "rational_quadratic": "rq",
    "rationalquadratic": "rq",
    "imq": "imq",
    "inverse_multiquadratic": "imq",
    "inverse_multiquadric": "imq",
    "matern": "matern",
    "matérn": "matern",
}

SAMPLER_IDS = {
    "rbf": "rbf_gaussian_v1",
    "laplacian": "laplacian_l1_cauchy_v1",
    "rq": "rq_gamma_gaussian_v1",
    "imq": "imq_rq_alpha_0_5_v1",
    "matern": "matern_multivariate_student_t_v1",
}

SUPPORTED_KERNELS = ["rbf", "laplacian", "rq", "imq", "matern"]
CANONICAL_SAMPLER_CODE_PATH = "core.kernel_library.sample_spectral_frequencies"
CANONICAL_KERNEL_CODE_PATH = "core.kernel_library.create_kernel"
KNOWN_KERNEL_CODE_PATHS = {
    CANONICAL_SAMPLER_CODE_PATH,
    CANONICAL_KERNEL_CODE_PATH,
    "core.rks_feature_map.SharedBasis.generated",
    "core.rks_feature_map.SharedBasis.injected",
    "core.dirichlet_fusion.SharedRKSBasis.generated",
}
SAMPLER_SIGMA_MODE_BY_CODE_PATH = {
    CANONICAL_SAMPLER_CODE_PATH: "scaled",
    CANONICAL_KERNEL_CODE_PATH: "scaled",
    "core.rks_feature_map.SharedBasis.generated": "runtime_forward_scaling",
    "core.rks_feature_map.SharedBasis.injected": "runtime_forward_scaling",
    "core.dirichlet_fusion.SharedRKSBasis.generated": "runtime_forward_scaling",
}
_PROVENANCE_UNSET = object()


def normalize_kernel_type(kernel_type: str) -> str:
    """Return the canonical kernel family name or raise for unsupported input."""
    key = str(kernel_type).strip().lower()
    if key not in CANONICAL_KERNEL_ALIASES:
        raise ValueError(
            f"Unsupported kernel '{kernel_type}'. Supported kernels: {SUPPORTED_KERNELS}"
        )
    return CANONICAL_KERNEL_ALIASES[key]


def validate_positive_sigma(value: Any, name: str = "sigma") -> float:
    """Return a finite positive bandwidth or raise consistently."""
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}")
    return out


def _positive_float(value: Any, name: str) -> float:
    return validate_positive_sigma(value, name)


def _resolve_legacy_gamma(
    kernel_type: str,
    sigma: Optional[float],
    raw_params: Dict[str, Any],
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """Resolve the legacy RBF gamma alias while failing closed on ambiguity."""
    kt = normalize_kernel_type(kernel_type)
    params = dict(raw_params)
    if "gamma" not in params:
        return kt, sigma, params

    gamma = params.pop("gamma")
    if kt != "rbf":
        raise ValueError("gamma is a legacy bandwidth alias supported only for the RBF family")
    if sigma is not None:
        raise ValueError("gamma and an explicit sigma are mutually exclusive")
    try:
        gamma_f = _positive_float(gamma, "gamma")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gamma must be a positive finite value, got {gamma!r}") from exc
    return kt, math.sqrt(1.0 / (2.0 * gamma_f)), params


def validate_shape_params(
    kernel_type: str,
    params: Dict[str, Any],
    *,
    name: str = "shape_params",
) -> Dict[str, float]:
    """Validate the exact raw parameter schema for a canonical family."""
    kt = normalize_kernel_type(kernel_type)
    if not isinstance(params, dict):
        raise ValueError(f"{name} must be a dict")
    expected_keys = {
        "rbf": set(),
        "laplacian": set(),
        "rq": {"alpha"},
        "imq": {"alpha"},
        "matern": {"nu"},
    }[kt]
    actual_keys = set(params)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{name} schema mismatch for {kt}: expected {sorted(expected_keys)}, "
            f"got {sorted(actual_keys)}"
        )
    if kt == "rq":
        return {"alpha": _positive_float(params["alpha"], f"{name}.alpha")}
    if kt == "imq":
        alpha = _positive_float(params["alpha"], f"{name}.alpha")
        if not math.isclose(alpha, 0.5, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{name}.alpha must be exactly 0.5 for imq")
        return {"alpha": 0.5}
    if kt == "matern":
        return {"nu": _positive_float(params["nu"], f"{name}.nu")}
    return {}


def canonical_shape_params(kernel_type: str, **kwargs: Any) -> Dict[str, float]:
    """Return defaults when omitted; otherwise enforce the exact raw schema."""
    kt = normalize_kernel_type(kernel_type)
    if not kwargs:
        defaults = {
            "rbf": {},
            "laplacian": {},
            "rq": {"alpha": 1.0},
            "imq": {"alpha": 0.5},
            "matern": {"nu": 1.5},
        }[kt]
        return dict(defaults)
    return validate_shape_params(kt, dict(kwargs), name="kernel_params")


def kernel_provenance(
    *,
    requested_kernel: str,
    actual_kernel: Optional[str] = None,
    sigma: Optional[float] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    seed: Optional[int] = None,
    shape_params: Optional[Dict[str, Any]] = None,
    sampler_sigma_mode: str = "scaled",
    basis_hash: str,
    code_path: str,
) -> Dict[str, Any]:
    """Structured provenance for persisted RKS bases and diagnostics."""
    canonical = normalize_kernel_type(actual_kernel or requested_kernel)
    requested_canonical = normalize_kernel_type(requested_kernel)
    if requested_canonical != canonical:
        raise ValueError(
            f"Requested kernel '{requested_kernel}' resolved to '{requested_canonical}', "
            f"but actual kernel is '{canonical}'"
        )
    params = (
        canonical_shape_params(canonical)
        if shape_params is None
        else validate_shape_params(canonical, dict(shape_params), name="shape_params")
    )
    if not str(basis_hash).strip():
        raise ValueError("Kernel provenance basis_hash must be nonempty")
    if not str(code_path).strip():
        raise ValueError("Kernel provenance code_path must be nonempty")
    return {
        "requested_kernel": str(requested_kernel),
        "requested_canonical_kernel": requested_canonical,
        "actual_kernel": canonical,
        "sampler_id": SAMPLER_IDS[canonical],
        "sampler_version": 1,
        "sampler_sigma_mode": sampler_sigma_mode,
        "sigma": (None if sigma is None else float(sigma)),
        "shape_params": params,
        "input_dim": (None if input_dim is None else int(input_dim)),
        "output_dim": (None if output_dim is None else int(output_dim)),
        "seed": (None if seed is None else int(seed)),
        "basis_hash": str(basis_hash),
        "code_path": str(code_path),
    }


def validate_kernel_provenance(
    provenance: Dict[str, Any],
    *,
    requested_kernel: Optional[str] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    shape_params: Optional[Dict[str, Any]] = None,
    require_dimensions: bool = False,
    seed: Optional[int] = None,
    basis_hash: Optional[str] = None,
    code_path: Optional[str] = None,
    sigma: Any = _PROVENANCE_UNSET,
) -> Dict[str, Any]:
    """Fail closed on mismatched or incomplete kernel provenance."""
    if not isinstance(provenance, dict):
        raise ValueError("Kernel provenance must be a dict")

    required_fields = {
        "requested_kernel",
        "requested_canonical_kernel",
        "actual_kernel",
        "sampler_id",
        "sampler_version",
        "sampler_sigma_mode",
        "sigma",
        "shape_params",
        "input_dim",
        "output_dim",
        "seed",
        "basis_hash",
        "code_path",
    }
    missing = sorted(field for field in required_fields if field not in provenance)
    if missing:
        raise ValueError(f"Kernel provenance is incomplete; missing: {', '.join(missing)}")
    if not str(provenance["basis_hash"]).strip():
        raise ValueError("Kernel provenance basis_hash must be nonempty")
    if not str(provenance["code_path"]).strip():
        raise ValueError("Kernel provenance code_path must be nonempty")
    if provenance["code_path"] not in KNOWN_KERNEL_CODE_PATHS:
        raise ValueError(f"Unknown kernel provenance code_path: {provenance['code_path']!r}")
    expected_sigma_mode = SAMPLER_SIGMA_MODE_BY_CODE_PATH[provenance["code_path"]]
    if provenance["sampler_sigma_mode"] != expected_sigma_mode:
        raise ValueError(
            "Kernel provenance sampler_sigma_mode mismatch for "
            f"{provenance['code_path']}: expected {expected_sigma_mode!r}, "
            f"got {provenance['sampler_sigma_mode']!r}"
        )
    validate_positive_sigma(provenance["sigma"], "Kernel provenance sigma")

    actual = normalize_kernel_type(provenance.get("actual_kernel"))
    requested = normalize_kernel_type(
        provenance.get("requested_canonical_kernel", provenance.get("requested_kernel", actual))
    )
    if normalize_kernel_type(provenance["requested_kernel"]) != requested:
        raise ValueError("Kernel provenance requested_kernel mismatch")
    if requested != actual:
        raise ValueError(f"Kernel provenance mismatch: requested={requested}, actual={actual}")

    sampler_id = provenance.get("sampler_id")
    if sampler_id != SAMPLER_IDS[actual]:
        raise ValueError(
            f"Sampler provenance mismatch for {actual}: {sampler_id!r} != {SAMPLER_IDS[actual]!r}"
        )
    if provenance.get("sampler_version") != 1:
        raise ValueError(
            f"Unsupported sampler provenance version: {provenance.get('sampler_version')!r}"
        )

    canonical_recorded_params = validate_shape_params(
        actual,
        provenance["shape_params"],
        name="Kernel provenance shape_params",
    )

    if requested_kernel is not None and normalize_kernel_type(requested_kernel) != actual:
        raise ValueError(
            f"Loaded basis is {actual}, not requested {normalize_kernel_type(requested_kernel)}"
        )
    if input_dim is not None:
        recorded_input_dim = provenance.get("input_dim")
        if recorded_input_dim != int(input_dim) and (require_dimensions or recorded_input_dim is not None):
            raise ValueError("Kernel provenance input_dim mismatch")
    if output_dim is not None:
        recorded_output_dim = provenance.get("output_dim")
        if recorded_output_dim != int(output_dim) and (require_dimensions or recorded_output_dim is not None):
            raise ValueError("Kernel provenance output_dim mismatch")
    if seed is not None and provenance.get("seed") != int(seed):
        raise ValueError("Kernel provenance seed mismatch")
    if basis_hash is not None and provenance.get("basis_hash") != str(basis_hash):
        raise ValueError("Kernel provenance basis_hash mismatch")
    if code_path is not None and provenance.get("code_path") != str(code_path):
        raise ValueError("Kernel provenance code_path mismatch")
    if sigma is not _PROVENANCE_UNSET:
        recorded_sigma = provenance.get("sigma")
        if (recorded_sigma is None) != (sigma is None):
            raise ValueError("Kernel provenance sigma mismatch")
        if recorded_sigma is not None and not math.isclose(
            float(recorded_sigma), float(sigma), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Kernel provenance sigma mismatch")

    if shape_params is not None:
        expected_params = validate_shape_params(
            actual, dict(shape_params), name="kernel_params"
        )
        if canonical_recorded_params != expected_params:
            raise ValueError(
                "Kernel provenance shape_params mismatch: "
                f"{canonical_recorded_params} != {expected_params}"
            )

    return provenance


def compute_basis_hash(omega: torch.Tensor, bias: torch.Tensor) -> str:
    """Hash exact named tensor representations, including order, shape, and dtype."""
    digest = hashlib.sha256()
    for name, tensor in (("omega", omega), ("b", bias)):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        descriptor = json.dumps(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "order": "C",
                "nbytes": int(array.nbytes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, byteorder="big", signed=False))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sample_spectral_frequencies(
    input_dim: int,
    output_dim: int,
    kernel_type: str,
    sigma: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Sample RKS frequencies from the canonical spectral measure.

    Returns omega [output_dim, input_dim], bias [output_dim], and provenance.
    """
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    sigma = validate_positive_sigma(sigma)
    kt = normalize_kernel_type(kernel_type)
    params = canonical_shape_params(kt, **kwargs)
    rng = np.random.default_rng(int(seed))

    if kt == "rbf":
        omega = rng.standard_normal(size=(output_dim, input_dim)) / sigma
    elif kt == "laplacian":
        omega = rng.standard_cauchy(size=(output_dim, input_dim)) / sigma
    elif kt == "rq":
        alpha = params["alpha"]
        tau = rng.gamma(shape=alpha, scale=1.0 / alpha, size=(output_dim, 1))
        omega = np.sqrt(tau) * rng.standard_normal(size=(output_dim, input_dim)) / sigma
    elif kt == "imq":
        tau = rng.gamma(shape=0.5, scale=2.0, size=(output_dim, 1))
        omega = np.sqrt(tau) * rng.standard_normal(size=(output_dim, input_dim)) / sigma
    elif kt == "matern":
        df = 2.0 * params["nu"]
        chi2 = rng.chisquare(df=df, size=(output_dim, 1))
        omega = rng.standard_normal(size=(output_dim, input_dim)) / np.sqrt(chi2 / df) / sigma
    else:  # pragma: no cover
        raise ValueError(f"Unsupported kernel '{kernel_type}'")

    bias = 2.0 * math.pi * rng.random(size=output_dim)
    omega_t = torch.tensor(omega, dtype=torch.float32, device=device)
    bias_t = torch.tensor(bias, dtype=torch.float32, device=device)
    basis_hash = compute_basis_hash(omega_t, bias_t)
    provenance = kernel_provenance(
        requested_kernel=kernel_type,
        actual_kernel=kt,
        sigma=sigma,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        shape_params=params,
        sampler_sigma_mode="scaled",
        basis_hash=basis_hash,
        code_path=CANONICAL_SAMPLER_CODE_PATH,
    )
    return omega_t, bias_t, provenance


def create_rbf_basis(
    input_dim: int,
    rks_dim: int,
    bandwidth: float,
    seed: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible RBF basis helper."""
    omega, bias, _ = sample_spectral_frequencies(
        input_dim=input_dim,
        output_dim=rks_dim,
        kernel_type="rbf",
        sigma=bandwidth,
        seed=seed,
        device=device,
    )
    return omega, bias


def ensure_shared_basis(
    ctx: "KernelContext",
    basis_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[Optional["SharedBasis"], str]:
    """Create or load a shared basis for a KernelContext."""
    from .rks_feature_map import SharedBasis

    kt = str(ctx.kernel_type).lower()
    if kt in ("cosine", "linear"):
        return (None, "")

    canonical = normalize_kernel_type(kt)
    if basis_path is not None and Path(basis_path).exists():
        basis = SharedBasis.load(basis_path, expected_kernel_type=canonical)
        basis_hash = compute_basis_hash(basis.omega, basis.b)
        print(f"[M-OBSERVER] Loaded basis from {basis_path}, hash={basis_hash}")
        return (basis, basis_hash)

    sigma = _positive_float(ctx.bandwidth, "ctx.bandwidth")
    basis = SharedBasis(
        input_dim=ctx.input_dim,
        output_dim=ctx.rks_dim,
        seed=ctx.seed,
        kernel_type=canonical,
        sigma=sigma,
    )
    basis_hash = compute_basis_hash(basis.omega, basis.b)
    print(
        f"[M-OBSERVER] Created {canonical} basis: D={ctx.rks_dim}, "
        f"sigma={sigma}, hash={basis_hash}"
    )
    return (basis, basis_hash)


def estimate_rbf_sigma(
    features: torch.Tensor,
    sample_size: int = 1000,
    percentile: float = 50.0,
) -> float:
    """Estimate an RBF bandwidth from pairwise Euclidean distances."""
    if torch.is_tensor(features):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.array(features)

    if len(features_np) > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(features_np), sample_size, replace=False)
        features_np = features_np[idx]

    distances = pdist(features_np, metric="euclidean")
    if len(distances) == 0:
        return 1.0
    if distances.max() < 1e-8:
        print("  [Kernel] Warning: Degenerate data (distances near 0). Using sigma=1.0")
        return 1.0
    sigma = np.percentile(distances, percentile)
    return float(max(sigma, 1e-6))


def analytic_kernel_matrix(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    kernel_type: str = "rbf",
    sigma: Optional[float] = 1.0,
    **kwargs: Any,
) -> torch.Tensor:
    """Evaluate the exact canonical kernel matrix."""
    kt, sigma, raw_params = _resolve_legacy_gamma(kernel_type, sigma, kwargs)
    params = validate_shape_params(kt, raw_params, name="kernel_params")
    sigma = _positive_float(sigma, "sigma")

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.float()
    Y = X if Y is None else torch.as_tensor(Y, dtype=X.dtype, device=X.device)

    if kt == "laplacian":
        d_l1 = torch.cdist(X, Y, p=1)
        return torch.exp(-d_l1 / sigma)

    r = torch.cdist(X, Y, p=2)
    r2 = r.square()
    if kt == "rbf":
        return torch.exp(-r2 / (2.0 * sigma**2))
    if kt == "rq":
        alpha = params["alpha"]
        return torch.pow(1.0 + r2 / (2.0 * alpha * sigma**2), -alpha)
    if kt == "imq":
        return torch.rsqrt(1.0 + r2 / (sigma**2))
    if kt == "matern":
        nu = params["nu"]
        scaled = math.sqrt(2.0 * nu) * r / sigma
        if abs(nu - 0.5) < 1e-12:
            return torch.exp(-scaled)
        if abs(nu - 1.5) < 1e-12:
            return (1.0 + scaled) * torch.exp(-scaled)
        if abs(nu - 2.5) < 1e-12:
            return (1.0 + scaled + scaled.square() / 3.0) * torch.exp(-scaled)

        scaled_np = scaled.detach().cpu().numpy()
        out = np.ones_like(scaled_np, dtype=np.float64)
        mask = scaled_np > 0
        coeff = (2.0 ** (1.0 - nu)) / scipy_gamma(nu)
        out[mask] = coeff * (scaled_np[mask] ** nu) * scipy_kv(nu, scaled_np[mask])
        return torch.as_tensor(out, dtype=X.dtype, device=X.device)

    raise ValueError(f"Unsupported kernel '{kernel_type}'")


class KernelBase:
    """Base class for RKS kernel feature maps."""

    def __init__(self, input_dim: int, output_dim: int, sigma: float, seed: int = 42):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sigma = validate_positive_sigma(sigma)
        self.seed = int(seed)
        self.kernel_type = normalize_kernel_type(getattr(self, "kernel_type", "rbf"))
        if self.kernel_type == "rq":
            raw_params = {"alpha": self.alpha}
        elif self.kernel_type == "imq":
            raw_params = {"alpha": 0.5}
        elif self.kernel_type == "matern":
            raw_params = {"nu": self.nu}
        else:
            raw_params = {}
        self.shape_params = validate_shape_params(
            self.kernel_type, raw_params, name="kernel_params"
        )
        self.W, self.b, self.provenance = sample_spectral_frequencies(
            self.input_dim,
            self.output_dim,
            self.kernel_type,
            sigma=self.sigma,
            seed=self.seed,
            **self.shape_params,
        )
        self.provenance["basis_hash"] = compute_basis_hash(self.W, self.b)
        self.provenance["code_path"] = CANONICAL_KERNEL_CODE_PATH

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        validate_kernel_provenance(
            self.provenance,
            requested_kernel=self.kernel_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            shape_params=self.shape_params,
            require_dimensions=True,
            seed=self.seed,
            sigma=self.sigma,
            basis_hash=compute_basis_hash(self.W, self.b),
            code_path=CANONICAL_KERNEL_CODE_PATH,
        )
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.float()
        W = self.W.to(X.device)
        b = self.b.to(X.device)
        projection = X @ W.T + b
        return torch.cos(projection) * math.sqrt(2.0 / self.output_dim)

    def exact_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return analytic_kernel_matrix(
            X,
            Y,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            **self.shape_params,
        )

    def approximation_diagnostic(self, X: torch.Tensor, n_samples: int = 1000) -> Dict[str, float]:
        """Compare RKS Gram approximation against the exact same kernel family."""
        n = min(len(X), n_samples)
        if n < 2:
            return {"correlation": 0.0, "mae": 0.0, "rmse": 0.0}

        gen = torch.Generator(device=X.device if torch.is_tensor(X) else "cpu")
        gen.manual_seed(self.seed + 7919)
        indices = torch.randperm(len(X), generator=gen)[:n]
        X_sample = X[indices] if torch.is_tensor(X) else torch.tensor(X[indices], dtype=torch.float32)
        X_sample = X_sample.to(self.W.device).float()

        K_true = self.exact_kernel(X_sample)
        Z = self.transform(X_sample)
        K_approx = Z @ Z.T
        diff = (K_true - K_approx).abs()
        mae = float(diff.mean().item())
        rmse = float(torch.sqrt((diff**2).mean()).item())

        mask = torch.triu(torch.ones_like(K_true), diagonal=1).bool()
        true_vals = K_true[mask].detach().cpu().numpy()
        approx_vals = K_approx[mask].detach().cpu().numpy()
        correlation = 0.0
        if len(true_vals) > 1 and np.std(true_vals) > 0 and np.std(approx_vals) > 0:
            correlation = float(np.corrcoef(true_vals, approx_vals)[0, 1])

        return {
            "correlation": correlation,
            "mae": mae,
            "rmse": rmse,
            "sigma": self.sigma,
            "kernel_type": self.kernel_type,
            "sampler_id": self.provenance["sampler_id"],
        }


class RBFKernel(KernelBase):
    """RBF kernel: exp(-||x-y||^2 / (2 sigma^2))."""

    def __init__(self, input_dim: int, output_dim: int, sigma: float, seed: int = 42):
        self.kernel_type = "rbf"
        super().__init__(input_dim, output_dim, sigma, seed)


class LaplacianKernel(KernelBase):
    """L1 Laplacian kernel: exp(-||x-y||_1 / sigma)."""

    def __init__(self, input_dim: int, output_dim: int, sigma: float, seed: int = 42):
        self.kernel_type = "laplacian"
        super().__init__(input_dim, output_dim, sigma, seed)


class RationalQuadraticKernel(KernelBase):
    """Rational quadratic kernel."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma: float,
        alpha: float = 1.0,
        seed: int = 42,
    ):
        self.kernel_type = "rq"
        self.alpha = _positive_float(alpha, "alpha")
        super().__init__(input_dim, output_dim, sigma, seed)


class IMQKernel(KernelBase):
    """Inverse multiquadratic kernel, equivalent to RQ alpha=0.5."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma: float,
        alpha: float = 0.5,
        roughness: Optional[float] = None,
        seed: int = 42,
    ):
        self.kernel_type = "imq"
        self.alpha = validate_shape_params(
            "imq", {"alpha": alpha}, name="kernel_params"
        )["alpha"]
        self.roughness = roughness
        super().__init__(input_dim, output_dim, sigma, seed)


class MaternKernel(KernelBase):
    """Isotropic Matern kernel with multivariate Student-t spectral measure."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma: float,
        nu: float = 1.5,
        seed: int = 42,
    ):
        self.kernel_type = "matern"
        self.nu = _positive_float(nu, "nu")
        super().__init__(input_dim, output_dim, sigma, seed)


def create_kernel(
    kernel_type: str,
    input_dim: int,
    output_dim: int,
    sigma: Optional[float],
    seed: int = 42,
    **kwargs: Any,
):
    """Factory for canonical RKS kernel feature maps."""
    kt, sigma, raw_params = _resolve_legacy_gamma(kernel_type, sigma, kwargs)
    params = validate_shape_params(kt, raw_params, name="kernel_params")
    sigma = validate_positive_sigma(sigma)
    kernel_map = {
        "rbf": RBFKernel,
        "laplacian": LaplacianKernel,
        "rq": RationalQuadraticKernel,
        "imq": IMQKernel,
        "matern": MaternKernel,
    }
    return kernel_map[kt](input_dim, output_dim, sigma, seed=seed, **params)


KERNEL_INFO = {
    "rbf": {
        "name": "RBF (Gaussian)",
        "formula": "exp(-||x-y||^2/(2 sigma^2))",
        "properties": "Smooth, global, standard choice",
        "params": ["sigma"],
    },
    "laplacian": {
        "name": "L1 Laplacian",
        "formula": "exp(-||x-y||_1/sigma)",
        "properties": "Sharper, coordinatewise Cauchy spectral measure",
        "params": ["sigma"],
    },
    "rq": {
        "name": "Rational Quadratic",
        "formula": "(1 + ||x-y||^2/(2 alpha sigma^2))^-alpha",
        "properties": "Scale mixture of RBF kernels",
        "params": ["sigma", "alpha"],
    },
    "imq": {
        "name": "Inverse Multiquadratic",
        "formula": "(1 + ||x-y||^2/sigma^2)^-1/2",
        "properties": "RQ with alpha=0.5",
        "params": ["sigma"],
    },
    "matern": {
        "name": "Matern",
        "formula": "2^(1-nu)/Gamma(nu) * a^nu K_nu(a)",
        "properties": "Smoothness controlled by nu",
        "params": ["sigma", "nu"],
    },
}


def print_kernel_info() -> None:
    print("=" * 70)
    print("AVAILABLE KERNELS")
    print("=" * 70)
    for ktype, info in KERNEL_INFO.items():
        print(f"\n{ktype.upper()}: {info['name']}")
        print(f"  Formula: {info['formula']}")
        print(f"  Properties: {info['properties']}")
        print(f"  Parameters: {', '.join(info['params'])}")
    print("=" * 70)


__all__ = [
    "CANONICAL_KERNEL_ALIASES",
    "SAMPLER_IDS",
    "SUPPORTED_KERNELS",
    "CANONICAL_SAMPLER_CODE_PATH",
    "CANONICAL_KERNEL_CODE_PATH",
    "KNOWN_KERNEL_CODE_PATHS",
    "SAMPLER_SIGMA_MODE_BY_CODE_PATH",
    "normalize_kernel_type",
    "canonical_shape_params",
    "validate_shape_params",
    "validate_positive_sigma",
    "kernel_provenance",
    "validate_kernel_provenance",
    "compute_basis_hash",
    "sample_spectral_frequencies",
    "create_rbf_basis",
    "ensure_shared_basis",
    "estimate_rbf_sigma",
    "analytic_kernel_matrix",
    "KernelBase",
    "RBFKernel",
    "LaplacianKernel",
    "RationalQuadraticKernel",
    "IMQKernel",
    "MaternKernel",
    "create_kernel",
    "KERNEL_INFO",
    "print_kernel_info",
]
