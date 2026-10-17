"""
test_audited_regressions.py

Named unit regression tests locking in every critical bug fix discovered during the
Belief Transformer V3 architectural audit:

1. Zero-variance constant input robust_normalize fix yielding exact zeros and 100% consensus classification (core/riemannian_geometry.py).
2. Conformal metric scaling formula D_conformal(i, j) = D_raw(i, j) / sqrt(rho_i * rho_j) (core/hadamard_fusion.py).
3. Unit L2 normalization for u_axis spectral polarity primary stress vector (core/spectral_polarity.py).
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from core.kernel_library import (
    SAMPLER_IDS,
    analytic_kernel_matrix,
    compute_basis_hash,
    create_kernel,
    validate_kernel_provenance,
    validate_shape_params,
)
from core.rks_feature_map import RKSFeatureMap, SharedBasis
from core.dirichlet_fusion import DirichletFusion, DirichletFusionConfig, SharedRKSBasis
from core.kernel_context import KernelContext
from core.complete_pipeline import (
    _compute_kernel_matrix,
    _serialize_rks_basis_state,
    nystrom_kernel_pca_embed,
)
from core.observer_local_recompute import RKSProvenanceError, _project_track2, _restore_rks_basis
from core.waterfall_checkpoints import WaterfallCheckpoint
from core.riemannian_geometry import compute_riemannian_physics
from core.hadamard_fusion import ConformalMetric
from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig


KERNEL_TOY_CASES = {
    "rbf": ({}, float(np.exp(-0.5))),
    "laplacian": ({}, float(np.exp(-1.0))),
    "rq": ({"alpha": 1.0}, 2.0 / 3.0),
    "imq": ({"alpha": 0.5}, 1.0 / float(np.sqrt(2.0))),
    "matern": ({"nu": 1.5}, float((1.0 + np.sqrt(3.0)) * np.exp(-np.sqrt(3.0)))),
}

STRICT_KERNEL_PARAMS = {
    "rbf": {},
    "laplacian": {},
    "rq": {"alpha": 1.0},
    "imq": {"alpha": 0.5},
    "matern": {"nu": 1.5},
}

INVALID_SIGMAS = [0.0, -1.0, np.nan, np.inf, -np.inf]


def _toy_points():
    return torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)


def _toy_grid():
    xs = torch.linspace(-1.0, 1.0, 5)
    return torch.cartesian_prod(xs, xs).float()


@pytest.mark.parametrize("kernel_type", list(KERNEL_TOY_CASES))
def test_canonical_kernel_exact_two_point_closed_forms(kernel_type):
    """Lock exact kernel math for all canonical families on a two-point toy."""
    params, expected_offdiag = KERNEL_TOY_CASES[kernel_type]
    K = analytic_kernel_matrix(_toy_points(), kernel_type=kernel_type, sigma=1.0, **params)

    assert torch.allclose(torch.diag(K), torch.ones(2), atol=1e-7)
    assert abs(float(K[0, 1]) - expected_offdiag) < 1e-6
    assert abs(float(K[1, 0]) - expected_offdiag) < 1e-6


PUBLIC_API_MALFORMED_PARAMS = [
    ("rbf_alpha", "rbf", {"alpha": 1.0}),
    ("laplacian_extra", "laplacian", {"extra": 1.0}),
    ("rq_missing", "rq", {}),
    ("rq_extra", "rq", {"alpha": 1.0, "extra": 1.0}),
    ("imq_missing", "imq", {}),
    ("imq_roughness", "imq", {"roughness": 3}),
    ("imq_extra", "imq", {"alpha": 0.5, "extra": 1.0}),
    ("imq_wrong_alpha", "imq", {"alpha": 1.0}),
    ("matern_missing", "matern", {}),
    ("matern_extra", "matern", {"nu": 1.5, "extra": 1.0}),
    ("gamma_plus_extra", "rbf", {"gamma": 2.0, "extra": 1.0}),
]


@pytest.mark.parametrize("api", ["create", "analytic"])
@pytest.mark.parametrize(
    ("case_name", "kernel_type", "params"),
    PUBLIC_API_MALFORMED_PARAMS,
    ids=[case[0] for case in PUBLIC_API_MALFORMED_PARAMS],
)
def test_public_kernel_apis_reject_every_malformed_raw_shape_schema(
    api, case_name, kernel_type, params
):
    del case_name
    with pytest.raises(
        ValueError,
        match="schema mismatch|exactly 0.5|gamma and an explicit sigma are mutually exclusive",
    ):
        if api == "create":
            create_kernel(
                kernel_type,
                input_dim=2,
                output_dim=16,
                sigma=1.0,
                seed=7,
                **params,
            )
        else:
            analytic_kernel_matrix(
                _toy_points(), kernel_type=kernel_type, sigma=1.0, **params
            )


@pytest.mark.parametrize("api", ["create", "analytic"])
@pytest.mark.parametrize("kernel_type", list(STRICT_KERNEL_PARAMS))
def test_public_kernel_apis_accept_only_frozen_valid_shape_schemas(api, kernel_type):
    params = STRICT_KERNEL_PARAMS[kernel_type]
    if api == "create":
        kernel = create_kernel(
            kernel_type,
            input_dim=2,
            output_dim=16,
            sigma=1.0,
            seed=7,
            **params,
        )
        assert kernel.shape_params == params
    else:
        matrix = analytic_kernel_matrix(
            _toy_points(), kernel_type=kernel_type, sigma=1.0, **params
        )
        assert matrix.shape == (2, 2)
        assert torch.allclose(torch.diag(matrix), torch.ones(2), atol=1e-7)


@pytest.mark.parametrize("api", ["create", "analytic"])
def test_public_kernel_apis_treat_gamma_only_as_bandwidth_alias(api):
    if api == "create":
        kernel = create_kernel(
            "rbf", input_dim=2, output_dim=16, sigma=None, seed=7, gamma=2.0
        )
        assert kernel.sigma == pytest.approx(0.5)
        assert kernel.shape_params == {}
    else:
        matrix = analytic_kernel_matrix(
            _toy_points(), kernel_type="rbf", sigma=None, gamma=2.0
        )
        assert float(matrix[0, 1]) == pytest.approx(float(np.exp(-2.0)), abs=1e-6)


@pytest.mark.parametrize("api", ["create", "analytic"])
@pytest.mark.parametrize(
    ("kernel_type", "sigma", "gamma", "message"),
    [
        ("rbf", 1.25, 2.0, "mutually exclusive"),
        ("laplacian", None, 2.0, "only for the RBF family"),
        ("matern", None, 2.0, "only for the RBF family"),
        ("rbf", None, 0.0, "positive finite"),
        ("rbf", None, -1.0, "positive finite"),
        ("rbf", None, float("inf"), "positive finite"),
        ("rbf", None, "invalid", "positive finite"),
    ],
)
def test_public_kernel_apis_reject_invalid_or_ambiguous_legacy_gamma(
    api, kernel_type, sigma, gamma, message
):
    params = {"nu": 1.5} if kernel_type == "matern" else {}
    with pytest.raises(ValueError, match=message):
        if api == "create":
            create_kernel(
                kernel_type,
                input_dim=2,
                output_dim=16,
                sigma=sigma,
                seed=7,
                gamma=gamma,
                **params,
            )
        else:
            analytic_kernel_matrix(
                _toy_points(),
                kernel_type=kernel_type,
                sigma=sigma,
                gamma=gamma,
                **params,
            )


@pytest.mark.parametrize("kernel_type", list(KERNEL_TOY_CASES))
def test_canonical_rks_grid_gram_meets_preregistered_correlation_gate(kernel_type):
    """D=2048 RKS meets the preregistered operational fidelity screen.

    The 0.95 correlation threshold is a prespecified implementation-quality
    check, not a theoretical guarantee about random-feature approximation.
    """
    params, _ = KERNEL_TOY_CASES[kernel_type]
    kernel = create_kernel(
        kernel_type,
        input_dim=2,
        output_dim=2048,
        sigma=1.0,
        seed=123,
        **params,
    )
    X = _toy_grid()
    Z = kernel.transform(X)
    K_rks = Z @ Z.T
    K_exact = analytic_kernel_matrix(X, kernel_type=kernel_type, sigma=1.0, **params)
    mask = torch.triu(torch.ones_like(K_exact, dtype=torch.bool), diagonal=1)
    corr = float(np.corrcoef(K_exact[mask].numpy(), K_rks[mask].detach().numpy())[0, 1])
    mae = float((K_rks - K_exact).abs().mean().item())
    rmse = float(torch.sqrt(torch.mean((K_rks - K_exact).square())).item())

    assert corr >= 0.95
    assert np.isfinite(mae)
    assert np.isfinite(rmse)
    assert kernel.provenance["actual_kernel"] == kernel_type
    assert kernel.provenance["sampler_id"] == SAMPLER_IDS[kernel_type]


@pytest.mark.parametrize("kernel_type", list(KERNEL_TOY_CASES))
def test_canonical_rks_fixed_random_grid_meets_preregistered_gate(kernel_type):
    """The D=2048 operational screen also passes on a deterministic random grid."""
    params, _ = KERNEL_TOY_CASES[kernel_type]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260821)
    X = 4.0 * torch.rand((36, 3), generator=generator) - 2.0
    kernel = create_kernel(
        kernel_type,
        input_dim=3,
        output_dim=2048,
        sigma=1.15,
        seed=2718,
        **params,
    )
    K_rks = kernel.transform(X) @ kernel.transform(X).T
    K_exact = analytic_kernel_matrix(X, kernel_type=kernel_type, sigma=1.15, **params)
    upper = torch.triu(torch.ones_like(K_exact, dtype=torch.bool), diagonal=1)
    corr = float(np.corrcoef(K_exact[upper].numpy(), K_rks[upper].numpy())[0, 1])
    mae = float((K_rks - K_exact).abs().mean())
    rmse = float(torch.sqrt(torch.mean((K_rks - K_exact).square())))

    assert corr >= 0.95
    assert np.isfinite(mae)
    assert np.isfinite(rmse)


@pytest.mark.parametrize("kernel_type", list(KERNEL_TOY_CASES))
def test_all_rks_construction_paths_report_canonical_kernel_and_sampler(kernel_type):
    """KernelBase, direct SharedBasis, and Dirichlet SharedRKSBasis expose provenance."""
    params, _ = KERNEL_TOY_CASES[kernel_type]

    kernel = create_kernel(kernel_type, input_dim=2, output_dim=64, sigma=1.0, seed=7, **params)
    direct_basis = SharedBasis(
        input_dim=2,
        output_dim=64,
        seed=7,
        kernel_type=kernel_type,
        sigma=1.0,
        kernel_params=params,
    )
    dirichlet_basis = SharedRKSBasis(
        input_dim=2,
        output_dim=64,
        seed=7,
        kernel_type=kernel_type,
        nu=float(params.get("nu", 1.5)),
    )
    dirichlet_basis.set_sigma(1.0)

    provenances = [kernel.provenance, direct_basis.provenance, dirichlet_basis.provenance]
    for provenance in provenances:
        assert provenance["requested_canonical_kernel"] == kernel_type
        assert provenance["actual_kernel"] == kernel_type
        assert provenance["sampler_id"] == SAMPLER_IDS[kernel_type]
        assert provenance["sigma"] == 1.0
        assert provenance["basis_hash"]
        assert provenance["code_path"]


@pytest.mark.parametrize("kernel_type", list(KERNEL_TOY_CASES))
def test_framing_mode_reports_complete_validated_per_framing_provenance(kernel_type):
    params, _ = KERNEL_TOY_CASES[kernel_type]
    feature_map = RKSFeatureMap(
        input_dim=4,
        output_dim=128,
        n_framings=2,
        dims_per_framing=2,
        output_per_framing=64,
        kernel_type=kernel_type,
        kernel_params=params,
        sigma=1.0,
        random_seed=73,
        device="cpu",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    output = feature_map.transform(torch.randn(6, 4, generator=generator))
    provenance = feature_map.get_provenance()

    assert output.shape == (6, 128)
    assert len(provenance["framing_provenance"]) == 2
    for idx, record in enumerate(provenance["framing_provenance"]):
        kernel = feature_map.rks._kernels[idx]
        assert record["requested_canonical_kernel"] == kernel_type
        assert record["actual_kernel"] == kernel_type
        assert record["sampler_id"] == SAMPLER_IDS[kernel_type]
        assert record["sampler_version"] == 1
        assert record["shape_params"] == ({"alpha": 0.5} if kernel_type == "imq" else params)
        assert record["input_dim"] == 2
        assert record["output_dim"] == 64
        assert record["code_path"] == "core.kernel_library.create_kernel"
        assert record["basis_hash"] == compute_basis_hash(kernel.W, kernel.b)


def test_framing_mode_rejects_unsupported_kernel():
    with pytest.raises(ValueError, match="Unsupported kernel"):
        RKSFeatureMap(
            input_dim=4,
            output_dim=64,
            n_framings=2,
            kernel_type="polynomial",
            sigma=1.0,
            device="cpu",
        )


def test_rbf_framing_legacy_gamma_is_bandwidth_only_and_shape_schema_stays_strict():
    feature_map = RKSFeatureMap(
        input_dim=4,
        output_dim=32,
        n_framings=2,
        dims_per_framing=2,
        output_per_framing=16,
        kernel_type="rbf",
        gamma=2.0,
        random_seed=31,
        device="cpu",
    )

    assert feature_map.cfg.sigma == pytest.approx(0.5)
    assert feature_map.cfg.gamma is None
    assert feature_map.cfg.kernel_params == {}
    provenance = feature_map.get_provenance()
    assert all(record["shape_params"] == {} for record in provenance["framing_provenance"])
    assert all(record["sigma"] == pytest.approx(0.5) for record in provenance["framing_provenance"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        RKSFeatureMap(
            input_dim=4,
            output_dim=32,
            n_framings=2,
            dims_per_framing=2,
            output_per_framing=16,
            kernel_type="rbf",
            gamma=99.0,
            sigma=1.25,
            random_seed=31,
            device="cpu",
        )

    with pytest.raises(ValueError, match="only for the RBF family"):
        RKSFeatureMap(
            input_dim=4,
            output_dim=32,
            n_framings=2,
            dims_per_framing=2,
            output_per_framing=16,
            kernel_type="laplacian",
            gamma=2.0,
            random_seed=31,
            device="cpu",
        )

    for invalid_gamma in (0.0, -1.0, float("nan"), float("inf"), "invalid"):
        with pytest.raises(ValueError, match="positive finite"):
            RKSFeatureMap(
                input_dim=4,
                output_dim=32,
                n_framings=2,
                dims_per_framing=2,
                output_per_framing=16,
                kernel_type="rbf",
                gamma=invalid_gamma,
                random_seed=31,
                device="cpu",
            )

    with pytest.raises(ValueError, match="schema mismatch"):
        RKSFeatureMap(
            input_dim=4,
            output_dim=32,
            n_framings=2,
            dims_per_framing=2,
            output_per_framing=16,
            kernel_type="rbf",
            gamma=2.0,
            kernel_params={"alpha": 1.0},
            random_seed=31,
            device="cpu",
        )


def test_framing_rebuild_enforces_legacy_gamma_contract():
    feature_map = RKSFeatureMap(
        input_dim=4,
        output_dim=32,
        n_framings=2,
        dims_per_framing=2,
        output_per_framing=16,
        kernel_type="rbf",
        sigma=1.0,
        random_seed=31,
        device="cpu",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        feature_map.rebuild(gamma=2.0)
    assert feature_map.cfg.kernel_type == "rbf"
    assert feature_map.cfg.sigma == pytest.approx(1.0)
    assert feature_map.cfg.gamma is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        feature_map.rebuild(sigma=1.25, gamma=2.0)
    assert feature_map.cfg.sigma == pytest.approx(1.0)
    with pytest.raises(ValueError, match="only for the RBF family"):
        feature_map.rebuild(kernel_type="laplacian", sigma=None, gamma=2.0)
    assert feature_map.cfg.kernel_type == "rbf"
    assert feature_map.cfg.sigma == pytest.approx(1.0)

    for invalid_gamma in (0.0, -1.0, float("nan"), float("inf"), "invalid"):
        with pytest.raises(ValueError, match="positive finite"):
            feature_map.rebuild(sigma=None, gamma=invalid_gamma)
        assert feature_map.cfg.sigma == pytest.approx(1.0)

    valid = RKSFeatureMap(
        input_dim=4,
        output_dim=32,
        n_framings=2,
        dims_per_framing=2,
        output_per_framing=16,
        kernel_type="rbf",
        sigma=1.0,
        random_seed=31,
        device="cpu",
    )
    valid.rebuild(sigma=None, gamma=2.0)
    assert valid.cfg.sigma == pytest.approx(0.5)
    assert valid.cfg.gamma is None


def test_kernel_rks_dirichlet_active_paths_preserve_global_rng_states():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)
    direct_x = torch.randn(32, 3, generator=generator)
    framing_x = torch.randn(16, 4, generator=generator)
    fusion_x = torch.randn(3, 2, 3, generator=generator)

    torch.manual_seed(991)
    np.random.seed(991)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()

    shared = SharedBasis(3, 64, seed=4, kernel_type="matern", kernel_params={"nu": 1.5})
    shared.set_sigma(shared.estimate_sigma(direct_x))
    _ = shared(direct_x)
    _ = shared.basis_hash

    dirichlet_basis = SharedRKSBasis(3, 64, seed=5, kernel_type="laplacian")
    dirichlet_basis.set_sigma(dirichlet_basis.estimate_sigma(direct_x))
    _ = dirichlet_basis(direct_x)

    fusion = DirichletFusion(
        DirichletFusionConfig(
            hidden_dim=3,
            rks_dim=64,
            n_bots=2,
            n_observers=4,
            crn_enabled=True,
            crn_seed=17,
            kernel_type="rbf",
        )
    )
    assert fusion._locked_weights is not None

    unlocked_fusion = DirichletFusion(
        DirichletFusionConfig(
            hidden_dim=3,
            rks_dim=32,
            n_bots=2,
            n_observers=3,
            crn_enabled=False,
            crn_seed=19,
            kernel_type="rbf",
        )
    )
    _ = unlocked_fusion(fusion_x, compute_curvature=False)

    framing = RKSFeatureMap(
        input_dim=4,
        output_dim=64,
        n_framings=2,
        kernel_type="rq",
        kernel_params={"alpha": 1.0},
        sigma=1.0,
        random_seed=8,
        device="cpu",
    )
    _ = framing.transform(framing_x)
    _ = framing.get_provenance()

    torch_after = torch.get_rng_state()
    numpy_after = np.random.get_state()
    assert torch.equal(torch_before, torch_after)
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_direct_non_rbf_feature_map_is_not_gaussian_fallback():
    """Direct RKSFeatureMap construction must honor non-RBF kernel_type."""
    lap = RKSFeatureMap(
        input_dim=2,
        output_dim=4096,
        kernel_type="laplacian",
        sigma=1.0,
        random_seed=99,
        direct_mode=True,
    )
    rbf = RKSFeatureMap(
        input_dim=2,
        output_dim=4096,
        kernel_type="rbf",
        sigma=1.0,
        random_seed=99,
        direct_mode=True,
    )

    lap_prov = lap.get_provenance()
    assert lap_prov["actual_kernel"] == "laplacian"
    assert lap_prov["sampler_id"] == SAMPLER_IDS["laplacian"]
    assert not torch.allclose(lap.get_basis().omega, rbf.get_basis().omega)

    X = _toy_points()
    lap_offdiag = float((lap.transform(X) @ lap.transform(X).T)[0, 1])
    assert abs(lap_offdiag - float(np.exp(-1.0))) < abs(lap_offdiag - float(np.exp(-0.5)))


def test_raw_rks_injection_requires_and_accepts_validated_provenance():
    """Raw omega/b injection is allowed only with matching canonical provenance."""
    source = SharedBasis(
        input_dim=2,
        output_dim=64,
        kernel_type="matern",
        sigma=1.0,
        seed=31,
        kernel_params={"nu": 1.5},
    )
    feature_map = RKSFeatureMap(
        input_dim=2,
        output_dim=64,
        kernel_type="matern",
        kernel_params={"nu": 1.5},
        sigma=1.0,
        random_seed=31,
        direct_mode=True,
        injected_omega=source.omega,
        injected_bias=source.b,
        injected_provenance=source.provenance,
    )
    assert feature_map.get_provenance()["actual_kernel"] == "matern"
    assert feature_map.get_basis().basis_hash == compute_basis_hash(source.omega, source.b)

    with pytest.raises(ValueError, match="require kernel provenance"):
        RKSFeatureMap(
            input_dim=2,
            output_dim=64,
            kernel_type="rbf",
            sigma=1.0,
            direct_mode=True,
            injected_omega=source.omega,
            injected_bias=source.b,
        )

    mismatched = dict(source.provenance)
    mismatched["actual_kernel"] = "rbf"
    mismatched["requested_canonical_kernel"] = "rbf"
    mismatched["sampler_id"] = SAMPLER_IDS["rbf"]
    with pytest.raises(ValueError, match="mismatch|not requested"):
        RKSFeatureMap(
            input_dim=2,
            output_dim=64,
            kernel_type="matern",
            kernel_params={"nu": 1.5},
            sigma=1.0,
            random_seed=31,
            direct_mode=True,
            injected_omega=source.omega,
            injected_bias=source.b,
            injected_provenance=mismatched,
        )

    rbf_source = SharedBasis(2, 64, seed=31, kernel_type="rbf", sigma=1.0)
    with pytest.raises(ValueError, match="basis_hash mismatch"):
        RKSFeatureMap(
            input_dim=2,
            output_dim=64,
            kernel_type="matern",
            kernel_params={"nu": 1.5},
            sigma=1.0,
            random_seed=31,
            direct_mode=True,
            injected_omega=rbf_source.omega,
            injected_bias=rbf_source.b,
            injected_provenance=source.provenance,
        )


@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
@pytest.mark.parametrize("tamper_kind", ["tensor", "hash", "missing_hash"])
def test_persisted_basis_load_recomputes_hash_and_rejects_tampering(basis_kind, tamper_kind):
    """Neither loader trusts a serialized hash without recomputing from omega/b."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{basis_kind}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(2, 32, seed=8, kernel_type="matern", sigma=1.0, kernel_params={"nu": 1.5})
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(2, 32, seed=8, kernel_type="matern", nu=1.5)
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        if tamper_kind == "tensor":
            payload["omega"] = payload["omega"].clone()
            payload["omega"][0, 0] += 0.25
        elif tamper_kind == "hash":
            payload["hash"] = "0" * 16
        else:
            payload.pop("hash")
        torch.save(payload, path)

        expected_error = "required identity fields: hash" if tamper_kind == "missing_hash" else "hash mismatch"
        with pytest.raises(ValueError, match=expected_error):
            loader(str(path), expected_kernel_type="matern")


@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
@pytest.mark.parametrize(
    "missing_field",
    [
        "kernel_type",
        "requested_kernel_type",
        "kernel_params",
        "kernel_provenance",
        "hash",
        "seed",
        "input_dim",
        "output_dim",
        "omega",
        "b",
        "sigma",
    ],
)
def test_persisted_loaders_require_every_identity_field(basis_kind, missing_field):
    """Every persisted identity field is mandatory, including for RBF artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"legacy_{basis_kind}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(2, 32, seed=12, kernel_type="rbf", sigma=1.0)
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(2, 32, seed=12, kernel_type="rbf")
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        payload.pop(missing_field)
        torch.save(payload, path)

        with pytest.raises(ValueError, match=f"required identity fields: {missing_field}"):
            loader(str(path), expected_kernel_type="rbf")


def test_basis_hash_binds_names_order_shape_dtype_and_exact_bytes():
    omega = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias = torch.arange(4, dtype=torch.float32)
    baseline = compute_basis_hash(omega, bias)

    assert compute_basis_hash(omega.double(), bias.double()) != baseline
    assert compute_basis_hash(bias, omega) != baseline
    assert compute_basis_hash(omega.clone(), bias.clone()) == baseline


@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
@pytest.mark.parametrize("tamper_kind", ["dtype", "tensor_order"])
def test_persisted_loaders_reject_dtype_and_tensor_order_changes(basis_kind, tamper_kind):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"exact_{basis_kind}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(2, 16, seed=22, kernel_type="rbf", sigma=1.0)
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(2, 16, seed=22, kernel_type="rbf")
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        if tamper_kind == "dtype":
            payload["omega"] = payload["omega"].double()
        else:
            payload["omega"], payload["b"] = payload["b"], payload["omega"]
        torch.save(payload, path)

        with pytest.raises(ValueError, match="hash mismatch"):
            loader(str(path), expected_kernel_type="rbf")


@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
def test_persisted_basis_hash_is_stable_across_normal_save_load(basis_kind):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"roundtrip_{basis_kind}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(2, 16, seed=24, kernel_type="matern", sigma=1.0, kernel_params={"nu": 1.5})
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(2, 16, seed=24, kernel_type="matern", nu=1.5)
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        restored = loader(str(path), expected_kernel_type="matern")

        assert restored.basis_hash == basis.basis_hash
        assert restored.provenance["basis_hash"] == basis.basis_hash


def test_observer_rks_basis_restore_non_rbf_roundtrip_and_fail_closed():
    """Observer replay authenticates complete non-RBF state and rejects ambiguity."""
    basis = SharedRKSBasis(3, 48, seed=19, kernel_type="matern", nu=1.5)
    basis.set_sigma(1.2)
    state = _serialize_rks_basis_state(basis)
    restored = _restore_rks_basis(state)

    assert restored.requested_kernel_type == "matern"
    assert restored.kernel_type == "matern"
    assert restored.kernel_params == {"nu": 1.5}
    assert restored.provenance == basis.provenance
    assert restored.basis_hash == basis.basis_hash
    assert torch.equal(restored.omega, basis.omega)
    assert torch.equal(restored.b, basis.b)

    missing = dict(state)
    missing.pop("kernel_provenance")
    with pytest.raises(RKSProvenanceError, match="missing: kernel_provenance"):
        _restore_rks_basis(missing)

    mismatched = dict(state)
    mismatched["kernel_provenance"] = dict(state["kernel_provenance"])
    mismatched["kernel_provenance"]["output_dim"] = 47
    with pytest.raises(RKSProvenanceError, match="output_dim mismatch"):
        _restore_rks_basis(mismatched)

    seed_mismatched = dict(state)
    seed_mismatched["seed"] = state["seed"] + 1
    with pytest.raises(RKSProvenanceError, match="seed mismatch"):
        _restore_rks_basis(seed_mismatched)

    coordinated_seed_mismatch = dict(state)
    coordinated_seed_mismatch["kernel_provenance"] = dict(state["kernel_provenance"])
    coordinated_seed_mismatch["seed"] = state["seed"] + 1
    coordinated_seed_mismatch["kernel_provenance"]["seed"] = state["seed"] + 1
    with pytest.raises(RKSProvenanceError, match="deterministic SharedRKSBasis regeneration"):
        _restore_rks_basis(coordinated_seed_mismatch)

    sigma_mismatched = dict(state)
    sigma_mismatched["sigma"] = state["sigma"] + 0.5
    with pytest.raises(RKSProvenanceError, match="sigma mismatch"):
        _restore_rks_basis(sigma_mismatched)

    code_path_mismatched = dict(state)
    code_path_mismatched["code_path"] = "core.kernel_library.create_kernel"
    with pytest.raises(RKSProvenanceError, match="code_path mismatch"):
        _restore_rks_basis(code_path_mismatched)

    coordinated_code_path_mismatch = dict(state)
    coordinated_code_path_mismatch["kernel_provenance"] = dict(state["kernel_provenance"])
    coordinated_code_path_mismatch["code_path"] = "core.rks_feature_map.SharedBasis.generated"
    coordinated_code_path_mismatch["kernel_provenance"]["code_path"] = (
        "core.rks_feature_map.SharedBasis.generated"
    )
    with pytest.raises(RKSProvenanceError, match="SharedRKSBasis code_path mismatch"):
        _restore_rks_basis(coordinated_code_path_mismatch)

    hash_mismatched = dict(state)
    hash_mismatched["hash"] = "0" * 64
    with pytest.raises(RKSProvenanceError, match="does not match provenance basis_hash"):
        _restore_rks_basis(hash_mismatched)

    tensor_tampered = dict(state)
    tensor_tampered["omega"] = state["omega"].clone()
    tensor_tampered["omega"][0, 0] += 1.0
    with pytest.raises(RKSProvenanceError, match="tensor hash mismatch"):
        _restore_rks_basis(tensor_tampered)

    with pytest.raises(RKSProvenanceError, match="must be a mapping"):
        _project_track2(np.zeros((2, 3), dtype=np.float32), {"rks_basis_state": "invalid"})


def test_unsupported_kernels_and_provenance_mismatches_raise():
    """Unsupported families and mislabeled persisted bases fail closed."""
    with pytest.raises(ValueError):
        create_kernel("polynomial", input_dim=2, output_dim=16, sigma=1.0)
    with pytest.raises(ValueError):
        SharedBasis(input_dim=2, output_dim=16, kernel_type="polynomial", sigma=1.0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        basis = SharedBasis(input_dim=2, output_dim=16, kernel_type="rbf", sigma=1.0, seed=3)
        path = tmp_path / "basis.pt"
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        payload["kernel_provenance"]["actual_kernel"] = "laplacian"
        torch.save(payload, path)

        with pytest.raises(ValueError):
            SharedBasis.load(str(path), expected_kernel_type="rbf")

        dirichlet_path = tmp_path / "dirichlet_basis.pt"
        dirichlet_basis = SharedRKSBasis(input_dim=2, output_dim=16, kernel_type="matern", seed=4)
        dirichlet_basis.set_sigma(1.0)
        dirichlet_basis.save(str(dirichlet_path))
        with pytest.raises(ValueError):
            SharedRKSBasis.load(str(dirichlet_path), expected_kernel_type="rbf")


def test_dirichlet_kernel_context_keeps_kernel_type_and_bases_synchronized():
    """KernelContext overrides must not leave kernel_types stuck at the old default."""
    ctx = KernelContext(
        kernel_type="matern",
        input_dim=2,
        rks_dim=32,
        seed=17,
        bandwidth=1.0,
        nu=1.5,
    )
    cfg = DirichletFusionConfig(
        hidden_dim=2,
        rks_dim=8,
        n_observers=2,
        kernel_ctx=ctx,
    )
    assert cfg.kernel_type == "matern"
    assert cfg.kernel_types == ["matern"]
    assert cfg.rks_dim == 32
    assert cfg.basis_seed == 17

    fusion = DirichletFusion(cfg)
    assert list(fusion.bases.keys()) == ["matern"]
    assert fusion.basis.kernel_type == "matern"
    assert fusion.basis.provenance["actual_kernel"] == "matern"
    assert fusion.basis.provenance["sampler_id"] == SAMPLER_IDS["matern"]


def test_complete_pipeline_kernel_matrix_matern_exact_and_no_polynomial_fallback():
    """KPCA/Nystrom helper uses canonical Matérn and rejects unsupported families."""
    X = _toy_grid()
    K_matern = _compute_kernel_matrix(X, kernel_type="matern", gamma=1.5, sigma=1.0)
    K_exact = analytic_kernel_matrix(X, kernel_type="matern", sigma=1.0, nu=1.5)
    K_rbf = analytic_kernel_matrix(X, kernel_type="rbf", sigma=1.0)

    assert torch.allclose(K_matern, K_exact, atol=1e-6)
    assert not torch.allclose(K_matern, K_rbf, atol=1e-3)
    with pytest.raises(ValueError):
        _compute_kernel_matrix(X, kernel_type="polynomial", sigma=1.0)


def test_nystrom_cross_kernel_matern_uses_matern_not_rbf(monkeypatch):
    """Nyström C=K(X, landmarks) must use the requested family, not RBF fallback."""
    import core.complete_pipeline as complete_pipeline

    calls = []
    original = complete_pipeline.analytic_kernel_matrix

    def recording_kernel_matrix(X, Y=None, kernel_type="rbf", sigma=1.0, **kwargs):
        calls.append(
            {
                "kernel_type": kernel_type,
                "has_cross_landmarks": Y is not None,
                "kwargs": dict(kwargs),
            }
        )
        return original(X, Y, kernel_type=kernel_type, sigma=sigma, **kwargs)

    monkeypatch.setattr(complete_pipeline, "analytic_kernel_matrix", recording_kernel_matrix)

    X = _toy_grid()
    out = nystrom_kernel_pca_embed(
        X,
        kernel_type="matern",
        gamma=1.5,
        sigma=1.0,
        n_components=4,
        m_landmarks=6,
        center=False,
        seed=11,
    )

    assert out.shape == (X.shape[0], 4)
    cross_calls = [call for call in calls if call["has_cross_landmarks"]]
    assert cross_calls
    assert all(call["kernel_type"] == "matern" for call in cross_calls)
    assert all(call["kwargs"] == {"nu": 1.5} for call in cross_calls)
    assert not any(call["kernel_type"] == "rbf" for call in calls)

    with pytest.raises(ValueError):
        nystrom_kernel_pca_embed(X, kernel_type="polynomial", sigma=1.0, m_landmarks=4)


def test_waterfall_t2_checkpoint_serializes_actual_kernel_family_and_provenance():
    """T2 checkpoint writes z_<family> for non-RBF and preserves sampler provenance."""
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    basis = SharedBasis(
        2,
        4,
        seed=44,
        kernel_type="matern",
        sigma=1.25,
        kernel_params={"nu": 1.5},
    )
    provenance = dict(basis.provenance)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = WaterfallCheckpoint(Path(tmp), run_id="case")
        ckpt.save_t2_kernel_projections(
            z_kernel=z,
            kernel_type="matern",
            requested_kernel_type="matern",
            kernel_sigma=1.25,
            kernel_shape_params={"nu": 1.5},
            kernel_provenance=provenance,
            basis_hash=basis.basis_hash,
            code_path=provenance["code_path"],
            kernel_input_dim=2,
            rks_dim=4,
            random_seed=44,
        )
        with np.load(ckpt.checkpoint_dir / "T2_kernel_projections.npz") as arrays:
            array_files = list(arrays.files)
            z_matern = np.asarray(arrays["z_matern"])
        meta = json.loads((ckpt.checkpoint_dir / "T2_kernel_meta.json").read_text())

        assert "z_matern" in array_files
        assert "z_rbf" not in array_files
        assert np.allclose(z_matern, z)
        assert meta["kernel_type"] == "matern"
        assert meta["actual_kernel"] == "matern"
        assert meta["active_projection_key"] == "z_matern"
        assert meta["sampler_id"] == SAMPLER_IDS["matern"]
        assert meta["kernel_provenance"]["shape_params"] == {"nu": 1.5}


def test_waterfall_t2_checkpoint_keeps_rbf_compatibility_and_rejects_unsupported():
    """Authenticated RBF writes retain legacy reader fields; anonymous writes fail."""
    z = np.ones((2, 3), dtype=np.float32)
    basis = SharedBasis(2, 3, seed=5, kernel_type="rbf", sigma=2.0)
    provenance = dict(basis.provenance)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = WaterfallCheckpoint(Path(tmp), run_id="case")
        ckpt.save_t2_kernel_projections(
            z_rbf=z,
            kernel_type="rbf",
            requested_kernel_type="rbf",
            kernel_sigma=2.0,
            kernel_shape_params={},
            kernel_provenance=provenance,
            basis_hash=basis.basis_hash,
            code_path=provenance["code_path"],
            kernel_input_dim=2,
            rbf_sigma=2.0,
            rks_dim=3,
            random_seed=5,
        )
        with np.load(ckpt.checkpoint_dir / "T2_kernel_projections.npz") as arrays:
            array_files = list(arrays.files)
        meta = json.loads((ckpt.checkpoint_dir / "T2_kernel_meta.json").read_text())
        assert "z_rbf" in array_files
        assert meta["kernel_type"] == "rbf"
        assert meta["active_projection_key"] == "z_rbf"
        assert meta["rbf_sigma"] == 2.0

        with pytest.raises(ValueError, match="explicit kernel_provenance"):
            ckpt.save_t2_kernel_projections(z_rbf=z, rbf_sigma=2.0, rks_dim=3, random_seed=5)

        with pytest.raises(ValueError, match="Unsupported kernel"):
            ckpt.save_t2_kernel_projections(
                z_kernel=z,
                kernel_type="polynomial",
                requested_kernel_type="polynomial",
                kernel_sigma=2.0,
                kernel_shape_params={},
                kernel_provenance=provenance,
                basis_hash=basis.basis_hash,
                code_path=provenance["code_path"],
                kernel_input_dim=2,
                rks_dim=3,
                random_seed=5,
            )


@pytest.mark.parametrize(
    "contradiction",
    ["output_dim", "rks_dim", "seed", "sigma", "params", "hash", "code_path"],
)
def test_waterfall_t2_rejects_every_provenance_contradiction(contradiction):
    z = np.ones((3, 4), dtype=np.float32)
    basis = SharedBasis(
        2,
        4,
        seed=44,
        kernel_type="matern",
        sigma=1.25,
        kernel_params={"nu": 1.5},
    )
    provenance = dict(basis.provenance)
    kwargs = {
        "z_kernel": z,
        "kernel_type": "matern",
        "requested_kernel_type": "matern",
        "kernel_sigma": 1.25,
        "kernel_shape_params": {"nu": 1.5},
        "kernel_provenance": provenance,
        "basis_hash": basis.basis_hash,
        "code_path": provenance["code_path"],
        "kernel_input_dim": 2,
        "rks_dim": 4,
        "random_seed": 44,
    }
    if contradiction == "output_dim":
        provenance["output_dim"] = 5
    elif contradiction == "rks_dim":
        kwargs["rks_dim"] = 5
    elif contradiction == "seed":
        provenance["seed"] = 45
    elif contradiction == "sigma":
        provenance["sigma"] = 2.0
    elif contradiction == "params":
        provenance["shape_params"] = {"nu": 2.5}
    elif contradiction == "hash":
        kwargs["basis_hash"] = "0" * 64
    else:
        kwargs["code_path"] = "core.kernel_library.create_kernel"

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = WaterfallCheckpoint(Path(tmp), run_id="case")
        with pytest.raises(ValueError, match="mismatch"):
            ckpt.save_t2_kernel_projections(**kwargs)


@pytest.mark.parametrize("invalid_sigma", INVALID_SIGMAS)
@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
def test_basis_sigma_setters_reject_every_nonpositive_or_nonfinite_value(basis_kind, invalid_sigma):
    if basis_kind == "shared":
        basis = SharedBasis(2, 8, seed=7, kernel_type="rbf")
    else:
        basis = SharedRKSBasis(2, 8, seed=7, kernel_type="rbf")
    with pytest.raises(ValueError, match="positive finite"):
        basis.set_sigma(invalid_sigma)


@pytest.mark.parametrize("invalid_sigma", INVALID_SIGMAS)
def test_kernel_provenance_rejects_every_nonpositive_or_nonfinite_sigma(invalid_sigma):
    basis = SharedBasis(2, 8, seed=7, kernel_type="rbf", sigma=1.0)
    provenance = dict(basis.provenance)
    provenance["sigma"] = invalid_sigma
    with pytest.raises(ValueError, match="positive finite"):
        validate_kernel_provenance(
            provenance,
            requested_kernel="rbf",
            input_dim=2,
            output_dim=8,
            shape_params={},
            require_dimensions=True,
            seed=7,
            basis_hash=basis.basis_hash,
        )


@pytest.mark.parametrize("invalid_sigma", INVALID_SIGMAS)
@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
def test_persisted_loaders_reject_every_nonpositive_or_nonfinite_sigma(basis_kind, invalid_sigma):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"invalid_sigma_{basis_kind}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(2, 8, seed=7, kernel_type="rbf", sigma=1.0)
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(2, 8, seed=7, kernel_type="rbf")
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        payload["sigma"] = invalid_sigma
        payload["kernel_provenance"]["sigma"] = invalid_sigma
        torch.save(payload, path)

        with pytest.raises(ValueError, match="positive finite"):
            loader(str(path), expected_kernel_type="rbf")


@pytest.mark.parametrize("invalid_sigma", INVALID_SIGMAS)
def test_waterfall_t2_rejects_every_nonpositive_or_nonfinite_sigma(invalid_sigma):
    z = np.ones((2, 4), dtype=np.float32)
    basis = SharedBasis(2, 4, seed=7, kernel_type="rbf", sigma=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = WaterfallCheckpoint(Path(tmp), run_id="sigma")
        with pytest.raises(ValueError, match="positive finite"):
            checkpoint.save_t2_kernel_projections(
                z_rbf=z,
                kernel_type="rbf",
                requested_kernel_type="rbf",
                kernel_sigma=invalid_sigma,
                kernel_shape_params={},
                kernel_provenance=dict(basis.provenance),
                basis_hash=basis.basis_hash,
                code_path=basis.provenance["code_path"],
                kernel_input_dim=2,
                rks_dim=4,
                random_seed=7,
            )


@pytest.mark.parametrize(
    ("kernel_type", "bad_params"),
    [
        ("rbf", {"alpha": 1.0}),
        ("laplacian", {"extra": 1.0}),
        ("rq", {"alpha": 1.0, "extra": 1.0}),
        ("imq", {"alpha": 0.5, "extra": 1.0}),
        ("matern", {"nu": 1.5, "extra": 1.0}),
        ("rq", {}),
        ("imq", {}),
        ("matern", {}),
    ],
)
def test_raw_kernel_shape_parameter_schemas_are_exact(kernel_type, bad_params):
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_shape_params(kernel_type, bad_params)


@pytest.mark.parametrize("kernel_type", list(STRICT_KERNEL_PARAMS))
@pytest.mark.parametrize("tamper_target", ["kernel_params", "provenance_shape_params"])
@pytest.mark.parametrize("basis_kind", ["shared", "dirichlet"])
def test_persisted_loaders_reject_additional_shape_parameter_keys(
    basis_kind, tamper_target, kernel_type
):
    params = STRICT_KERNEL_PARAMS[kernel_type]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"extra_params_{basis_kind}_{kernel_type}.pt"
        if basis_kind == "shared":
            basis = SharedBasis(
                2, 8, seed=13, kernel_type=kernel_type, sigma=1.0, kernel_params=params
            )
            loader = SharedBasis.load
        else:
            basis = SharedRKSBasis(
                2,
                8,
                seed=13,
                kernel_type=kernel_type,
                nu=float(params.get("nu", 1.5)),
            )
            basis.set_sigma(1.0)
            loader = SharedRKSBasis.load
        basis.save(str(path))
        payload = torch.load(path, map_location="cpu")
        extra_key = "alpha" if kernel_type == "rbf" else "unexpected"
        if tamper_target == "kernel_params":
            payload["kernel_params"] = {**payload["kernel_params"], extra_key: 9.0}
        else:
            payload["kernel_provenance"]["shape_params"] = {
                **payload["kernel_provenance"]["shape_params"],
                extra_key: 9.0,
            }
        torch.save(payload, path)

        with pytest.raises(ValueError, match="schema mismatch"):
            loader(str(path), expected_kernel_type=kernel_type)


@pytest.mark.parametrize("kernel_type", list(STRICT_KERNEL_PARAMS))
def test_waterfall_t2_rejects_additional_shape_parameter_keys_for_every_family(kernel_type):
    params = STRICT_KERNEL_PARAMS[kernel_type]
    basis = SharedBasis(
        2, 4, seed=17, kernel_type=kernel_type, sigma=1.0, kernel_params=params
    )
    extra_key = "alpha" if kernel_type == "rbf" else "unexpected"
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = WaterfallCheckpoint(Path(tmp), run_id="params")
        with pytest.raises(ValueError, match="schema mismatch"):
            checkpoint.save_t2_kernel_projections(
                z_kernel=np.ones((2, 4), dtype=np.float32),
                kernel_type=kernel_type,
                requested_kernel_type=kernel_type,
                kernel_sigma=1.0,
                kernel_shape_params={**params, extra_key: 9.0},
                kernel_provenance=dict(basis.provenance),
                basis_hash=basis.basis_hash,
                code_path=basis.provenance["code_path"],
                kernel_input_dim=2,
                rks_dim=4,
                random_seed=17,
            )


@pytest.mark.parametrize("kernel_type", ["rbf", "matern"])
def test_waterfall_t2_requires_exact_input_dim_and_sampler_sigma_mode(kernel_type):
    params = STRICT_KERNEL_PARAMS[kernel_type]
    basis = SharedBasis(
        2, 4, seed=23, kernel_type=kernel_type, sigma=1.0, kernel_params=params
    )
    kwargs = {
        "z_kernel": np.ones((2, 4), dtype=np.float32),
        "kernel_type": kernel_type,
        "requested_kernel_type": kernel_type,
        "kernel_sigma": 1.0,
        "kernel_shape_params": params,
        "kernel_provenance": dict(basis.provenance),
        "basis_hash": basis.basis_hash,
        "code_path": basis.provenance["code_path"],
        "kernel_input_dim": 2,
        "rks_dim": 4,
        "random_seed": 23,
    }
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = WaterfallCheckpoint(Path(tmp), run_id="identity")
        missing_dim = dict(kwargs)
        missing_dim["kernel_input_dim"] = None
        with pytest.raises(ValueError, match="positive explicit kernel_input_dim"):
            checkpoint.save_t2_kernel_projections(**missing_dim)

        wrong_dim = dict(kwargs)
        wrong_dim["kernel_input_dim"] = 999
        with pytest.raises(ValueError, match="input_dim mismatch"):
            checkpoint.save_t2_kernel_projections(**wrong_dim)

        nonsense_mode = dict(kwargs)
        nonsense_mode["kernel_provenance"] = dict(kwargs["kernel_provenance"])
        nonsense_mode["kernel_provenance"]["sampler_sigma_mode"] = "nonsense"
        with pytest.raises(ValueError, match="sampler_sigma_mode mismatch"):
            checkpoint.save_t2_kernel_projections(**nonsense_mode)

def test_riemannian_physics_zero_variance_consensus_classification():
    """Verify compute_riemannian_physics on constant zero-variance input returns exact zeros & consensus."""
    x_const = torch.ones((10, 1536), dtype=torch.float32) * 3.14159
    physics = compute_riemannian_physics(x_const)
    
    assert torch.all(physics["blinker"] == 0.0), "blinker score must be exact zeros on constant input"
    assert torch.all(physics["walker"] == 0.0), "walker score must be exact zeros on constant input"
    assert all(c == "consensus" for c in physics["singularity_types"]), "singularity types must be 100% consensus"

def test_conformal_scaling_formula_matches_theoretical_definition():
    """
    Verify apply_conformal_scaling matches D_conformal(i, j) = D_raw(i, j) / sqrt(rho_i * rho_j)
    using asymmetric non-perfect-square rho values [0.3, 0.7] across full matrix (diagonal and off-diagonal).
    """
    D_raw = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    rho = torch.tensor([0.3, 0.7], dtype=torch.float32)
    
    cm = ConformalMetric()
    D_conf = cm.apply_conformal_scaling(D_raw, rho)
    
    # Off-diagonal: D_conf[0, 1] = 2.0 / sqrt(0.3 * 0.7) = 2.0 / sqrt(0.21) ≈ 4.3643578
    expected_offdiag = 2.0 / float(np.sqrt(0.3 * 0.7))
    assert abs(float(D_conf[0, 1]) - expected_offdiag) < 1e-5
    assert abs(float(D_conf[1, 0]) - expected_offdiag) < 1e-5
    
    # Diagonal self-scaling: D_conf[0, 0] = 1.0 / sqrt(0.3 * 0.3) = 1.0 / 0.3 ≈ 3.3333333
    expected_diag0 = 1.0 / 0.3
    expected_diag1 = 3.0 / 0.7
    assert abs(float(D_conf[0, 0]) - expected_diag0) < 1e-5
    assert abs(float(D_conf[1, 1]) - expected_diag1) < 1e-5


def test_spectral_polarity_u_axis_unit_l2_norm():
    """Verify SpectralPolarity u_axis vector has strict unit L2 norm ||u_0||_2 = 1.0."""
    embeddings = [torch.randn((8, 1536)) for _ in range(5)] # 5 articles, 8 bots each
    sp = SpectralPolarity(SpectralPolarityConfig())
    res = sp.compute_batch(embeddings)
    
    u_axis = res.u_axis # [5, 1536]
    norms = torch.norm(u_axis, p=2, dim=1)
    
    for norm in norms:
        assert abs(float(norm) - 1.0) < 1e-5, f"u_axis norm must be 1.0, got {float(norm)}"

def test_canonical_feature_selection_prefers_integrated_vectors():
    """Verify that validate_against_ground_truth prefers canonical 'integrated_vectors' over legacy 'features'."""
    from run_full_experiment_suite import validate_against_ground_truth
    
    # 12 samples: features collapse all to 0, integrated_vectors separate into two 6-sample clusters
    features_legacy = np.zeros((12, 2), dtype=np.float64)
    integrated_vecs = np.vstack([np.zeros((6, 2)), np.ones((6, 2))])
    
    result = {
        "features": features_legacy,
        "integrated_vectors": integrated_vecs,
    }
    ground_truth = {i: ("class_a" if i < 6 else "class_b") for i in range(12)}
    
    val_res = validate_against_ground_truth(result, ground_truth, n_clusters=2, use_integrated=True)
    assert val_res["feature_source"] == "integrated_vectors (Track 5)"
    assert val_res["nmi"] == 1.0 # integrated_vectors separates [0, 1] perfectly
