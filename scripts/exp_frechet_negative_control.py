"""
exp_frechet_negative_control.py
Fréchet Double-Collapse Negative-Control Experiment

Evaluates whether the 94.04% variance loss (Double Collapse) is unique to the 8 adversarial
observer frames or if any 8 arbitrary/random contrastive directions in embedding space produce
the same variance loss due to transformer anisotropy (Zipf's law / cone clustering).
"""

import io
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure utf-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def compute_frechet_collapse_metrics(embeddings_tensor):
    """
    embeddings_tensor: [N, B, H] where N=articles, B=observers/directions, H=hidden_dim
    """
    N, B, H = embeddings_tensor.shape
    
    # 1. Raw total representation variance across all N*B vectors
    raw_flat = embeddings_tensor.reshape(N * B, H).numpy()
    total_raw_var = float(np.var(raw_flat, axis=0).sum())

    # 2. Consensus (Fréchet Mean) Variance: mean across B observers per article -> shape [N, H]
    frechet_means = embeddings_tensor.mean(dim=1).numpy()
    frechet_mean_var = float(np.var(frechet_means, axis=0).sum())

    # 3. Inter-Observer Dispersion Variance
    observer_dispersion_var = total_raw_var - frechet_mean_var
    observer_collapse_pct = (1.0 - (frechet_mean_var / total_raw_var)) * 100.0

    # 4. 1D PC1 Projection Variance of Fréchet Means
    means_centered = frechet_means - frechet_means.mean(axis=0, keepdims=True)
    _, S_means, _ = np.linalg.svd(means_centered, full_matrices=False)
    pc1_var = float((S_means[0] ** 2) / (N - 1)) if N > 1 else 0.0
    dim_collapse_pct = (1.0 - (pc1_var / frechet_mean_var)) * 100.0 if frechet_mean_var > 0 else 0.0

    # Cumulative Double Collapse
    cumulative_loss_pct = (1.0 - (pc1_var / total_raw_var)) * 100.0
    retained_pct = 100.0 - cumulative_loss_pct

    return {
        "total_raw_var": total_raw_var,
        "frechet_mean_var": frechet_mean_var,
        "observer_dispersion_var": observer_dispersion_var,
        "observer_collapse_pct": observer_collapse_pct,
        "pc1_var": pc1_var,
        "dim_collapse_pct": dim_collapse_pct,
        "cumulative_loss_pct": cumulative_loss_pct,
        "retained_pct": retained_pct,
    }

def main():
    print("=" * 70)
    print(" FRÉCHET DOUBLE-COLLAPSE NEGATIVE-CONTROL EXPERIMENT")
    print("=" * 70)

    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Real embeddings missing at {emb_path}")

    real_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False) # [60, 8, 1536]
    N, B, H = real_embeddings.shape
    print(f"▶ Input Dataset: {N} real articles, {B} observers, {H} dimensions")

    # 1. Real Adversarial Observer Frames [60, 8, 1536]
    real_metrics = compute_frechet_collapse_metrics(real_embeddings)

    # 2. Negative Control: 8 Random Orthogonal Direction Warps of the mean article vector
    # We take the mean article vector [60, 1, 1536] and add 8 random Gaussian direction warps
    # matching the average per-observer variance magnitude of the real frames.
    mean_articles = real_embeddings.mean(dim=1, keepdim=True) # [60, 1, 1536]
    real_dispersion_std = float(np.std((real_embeddings - mean_articles).numpy()))

    rng = torch.Generator().manual_seed(42)
    random_warps = torch.randn((N, B, H), generator=rng) * real_dispersion_std
    negative_control_embeddings = mean_articles + random_warps

    control_metrics = compute_frechet_collapse_metrics(negative_control_embeddings)

    print("\n▶ REAL ADVERSARIAL FRAMING METRICS:")
    print(f"  ├── Total Raw Representation Variance:   {real_metrics['total_raw_var']:.4f}")
    print(f"  ├── Fréchet Mean (Consensus) Variance:   {real_metrics['frechet_mean_var']:.4f}")
    print(f"  ├── Observer Dispersion Variance:        {real_metrics['observer_dispersion_var']:.4f}")
    print(f"  ├── Observer Collapse (Variance Lost):   {real_metrics['observer_collapse_pct']:.2f}%")
    print(f"  └── Cumulative Double Collapse Loss:     {real_metrics['cumulative_loss_pct']:.2f}%")

    print("\n▶ NEGATIVE CONTROL (8 Random Direction Warps):")
    print(f"  ├── Total Raw Representation Variance:   {control_metrics['total_raw_var']:.4f}")
    print(f"  ├── Fréchet Mean (Consensus) Variance:   {control_metrics['frechet_mean_var']:.4f}")
    print(f"  ├── Observer Dispersion Variance:        {control_metrics['observer_dispersion_var']:.4f}")
    print(f"  ├── Observer Collapse (Variance Lost):   {control_metrics['observer_collapse_pct']:.2f}%")
    print(f"  └── Cumulative Double Collapse Loss:     {control_metrics['cumulative_loss_pct']:.2f}%")

    print("\n▶ SCIENTIFIC DIAGNOSTIC COMPARISON:")
    diff_collapse = real_metrics['observer_collapse_pct'] - control_metrics['observer_collapse_pct']
    print(f"  ├── Observer Variance Loss (Real vs Control): {real_metrics['observer_collapse_pct']:.2f}% vs {control_metrics['observer_collapse_pct']:.2f}%")
    if abs(diff_collapse) < 5.0:
        print("  └── SCIENTIFIC CONCLUSION: The ~92-94% variance loss is an inherent geometric property of")
        print("      high-dimensional representation spaces (anisotropy / cone clustering) when combining")
        print("      8 directional frames, NOT unique to adversarial ideological prompt framing.")
    else:
        print("  └── SCIENTIFIC CONCLUSION: Adversarial prompt framing produces significantly greater observer dispersion")
        print("      than random directional warps.")

if __name__ == "__main__":
    main()
