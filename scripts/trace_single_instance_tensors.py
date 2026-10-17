"""
trace_single_instance_tensors.py
Production-Equivalent Engineering Smoke Test & Tensor Provenance Tracer

Runs the vertical 5-track pipeline on:
1. Real Article #0 [1, 8, 1536] (Single-article production shape)
2. Two Real Articles [2, 8, 1536] (Off-diagonal i != j discrimination test)
3. Zero-Variance CONSTANT Control [1, 8, 1536] (Production shape on 8 identical observer frames)

Logs exact shapes, value ranges, L2 norms, singular value spectra, and manifest metadata.
"""

import io
import sys
from pathlib import Path
import torch
import numpy as np

# Ensure utf-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig
from core.kernel_library import MaternKernel
from core.dirichlet_fusion import DirichletFusionConfig, DirichletFusion
from core.hadamard_fusion import ConformalMetric
from analysis.verification.verify_run import verify_manifest

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_tensor_info(name, tensor, extra_info=""):
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        t_np = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        shape = tensor.shape
        dtype = str(tensor.dtype)
        device = "numpy"
        t_np = tensor
    else:
        print(f"  [{name}] Type: {type(tensor)} | Value: {tensor}")
        return

    min_v = float(np.nanmin(t_np))
    max_v = float(np.nanmax(t_np))
    mean_v = float(np.nanmean(t_np))
    std_v = float(np.nanstd(t_np))
    norm_v = float(np.linalg.norm(t_np))
    has_nan_inf = np.any(np.isnan(t_np)) or np.any(np.isinf(t_np))

    per_vector_norm_str = ""
    if len(shape) == 2:
        row_norms = np.linalg.norm(t_np, axis=1)
        per_vector_norm_str = f" | Per-Vector L2 Norm: [{row_norms.min():.4f}, {row_norms.max():.4f}]"

    print(f"\n▶ TENSOR: {name}")
    print(f"  ├── Shape: {shape} | Dtype: {dtype} | Device: {device}")
    print(f"  ├── Range: [{min_v:.6f}, {max_v:.6f}] | Mean: {mean_v:.6f} | Std: {std_v:.6f}")
    print(f"  ├── Frobenius Norm: {norm_v:.6f}{per_vector_norm_str}")
    print(f"  └── Finite Check (No NaN/Inf): {'PASSED' if not has_nan_inf else 'FAILED (Contains NaN/Inf)'}")
    if extra_info:
        print(f"  └── Note: {extra_info}")

def run_article_pipeline_test(article_tensor, label):
    N_articles = article_tensor.shape[0] # N articles, shape [N, 8, 1536]
    print_section(f"RUNNING PIPELINE TRACE: {label} (N_articles = {N_articles})")
    print_tensor_info(f"Input Articles ({label})", article_tensor, f"Shape [{N_articles}, 8, 1536] across 8 observer frames")

    # --- TRACK 1.5: SPECTRAL POLARITY ---
    print_section(f"[{label}] TRACK 1.5: SPECTRAL POLARITY & CONTRASTIVE SHEAR")
    spec_module = SpectralPolarity(SpectralPolarityConfig())
    # Production call shape: list of N_articles tensors, each of shape [8, 1536]
    article_list = [article_tensor[i] for i in range(N_articles)]
    spec_result = spec_module.compute_batch(article_list)
    
    print_tensor_info(
        "Compass Principal Directions (u_axis)", 
        spec_result.u_axis, 
        f"core/spectral_polarity.py L182: Shape [{N_articles}, 1536]. Per-vector Unit L2 Norm = 1.0000"
    )
    print_tensor_info(
        "Full Singular Value Spectrum (S_reference)", 
        spec_result.singular_values, 
        f"core/spectral_polarity.py L367: Shape [{N_articles}, 8] full 8-value singular spectrum per article"
    )
    print(f"  ├── PC1 Explained Variance Ratio (EVR): {spec_result.evr.detach().cpu().numpy()}")
    print(f"  └── Dipole Validity Gate: {spec_result.dipole_valid.detach().cpu().numpy()}")

    # --- TRACK 2: KERNEL EXPANSION ---
    print_section(f"[{label}] TRACK 2: MATÉRN KERNEL EXPANSION & RKS FEATURE MAP")
    matern_kernel = MaternKernel(input_dim=1536, output_dim=2048, nu=1.5, sigma=1.0)
    rks_phi_list = []
    for i in range(N_articles):
        obs_embeds = article_tensor[i] # [8, 1536]
        rks_phi = matern_kernel.transform(obs_embeds) # [8, 2048]
        rks_phi_list.append(rks_phi)
    rks_phi_batch = torch.stack(rks_phi_list, dim=0) # [N, 8, 2048]
    print_tensor_info("RKS Feature Maps φ_RKS", rks_phi_batch, f"Shape [{N_articles}, 8, 2048]")

    # --- TRACK 3: DIRICHLET SIMPLEX ANNEALING ---
    print_section(f"[{label}] TRACK 3: DIRICHLET SIMPLEX ANNEALING & EPISTEMIC DENSITY")
    dir_config = DirichletFusionConfig(hidden_dim=1536, n_bots=8)
    fusion_module = DirichletFusion(dir_config)
    fused_out = fusion_module(article_tensor)
    
    rho_list = []
    for i in range(N_articles):
        v = float(np.var(article_tensor[i].numpy(), axis=0).mean())
        rho_list.append(1.0 / (1.0 + v))
    rho_tensor = torch.tensor(rho_list, dtype=torch.float32)
    print_tensor_info("Conformal Epistemic Density ρ", rho_tensor, f"Density factors for {N_articles} articles")

    # --- TRACK 5: HADAMARD SYNTHESIS ---
    print_section(f"[{label}] TRACK 5: HADAMARD SYNTHESIS & CONFORMAL METRIC")
    conformal_metric = ConformalMetric()
    
    # Compute mean CLS representation per article to evaluate inter-article distance
    mean_cls = article_tensor.mean(dim=1) # [N, 1536]
    D_raw = torch.cdist(mean_cls, mean_cls, p=2.0) # [N, N]
    
    D_conformal = conformal_metric.apply_conformal_scaling(D_raw, rho_tensor)
    print_tensor_info("Raw Euclidean Distance Matrix D_raw", D_raw, f"Shape [{N_articles}, {N_articles}]")
    print_tensor_info("Conformally Scaled Distance Matrix D_conformal", D_conformal, f"core/hadamard_fusion.py L408: D / sqrt(rho_i * rho_j)")

    if N_articles >= 2:
        expected_D01 = D_raw[0, 1].item() / np.sqrt(rho_list[0] * rho_list[1])
        actual_D01 = D_conformal[0, 1].item()
        print(f"\n  ├── Off-Diagonal D(0,1) Raw: {D_raw[0,1].item():.6f}")
        print(f"  ├── Expected Conformal D(0,1): {expected_D01:.6f}")
        print(f"  ├── Actual Conformal D(0,1):   {actual_D01:.6f}")
        print(f"  └── Off-Diagonal Check: {'PASSED (Exact Match)' if abs(expected_D01 - actual_D01) < 1e-6 else 'FAILED'}")

def main():
    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Real embeddings missing at {emb_path}")

    full_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    
    # 1. Real Article #0 [1, 8, 1536] (Single-article production call shape)
    run_article_pipeline_test(full_embeddings[0:1], "Real Article #0")

    # 2. Off-Diagonal Test [2, 8, 1536] (Articles 0 & 1)
    run_article_pipeline_test(full_embeddings[0:2], "Real Articles #0 & #1")

    # 3. Zero-Variance CONSTANT Control [1, 8, 1536] (1 article with 8 identical observer frames)
    frame_0 = full_embeddings[0:1, 0:1, :] # [1, 1, 1536]
    article_constant = frame_0.expand(1, 8, 1536).clone() # [1, 8, 1536] zero variance across frames
    run_article_pipeline_test(article_constant, "Zero-Variance CONSTANT Control Article")

    print_section("PRODUCTION-EQUIVALENT SMOKE TEST COMPLETE")

if __name__ == "__main__":
    main()
