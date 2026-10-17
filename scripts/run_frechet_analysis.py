"""
Calculate empirical Fréchet mean and double variance collapse metrics
on the Qwen synthetic corpus representations (from observer_global.pt).
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
sys.path.insert(0, str(ROOT_DIR))
import torch

TARGET_FILE = ROOT_DIR / "outputs" / "experiments" / "runs" / "experiments_20260710_130624" / "synthetic" / "rbf_seed42" / "observer_global.pt"

def main():
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {TARGET_FILE}")
        
    data = torch.load(TARGET_FILE, map_location="cpu", weights_only=False)
    
    # Extract the high-dimensional CLS manifolds: Shape [N, B, H]
    # N = 60 articles, B = 8 observers, H = 1536 dimensions
    cls_per_bot = data["cls_per_bot"]
    N, B, H = cls_per_bot.shape
    
    print("\n" + "="*70)
    # 1. Compute the Fréchet mean (consensus center) for each article
    # In Euclidean space, this is the mean vector across the 8 observers
    frechet_means = cls_per_bot.mean(axis=1) # Shape [N, H]
    
    # 2. Verify and compute the inter-observer dispersion
    # Variance of observer representations about their respective Fréchet means
    dispersions = []
    for n in range(N):
        article_obs = cls_per_bot[n] # Shape [B, H]
        mean_vec = frechet_means[n]   # Shape [H]
        # Mean squared Euclidean distance to Fréchet mean
        sq_dist = np.sum((article_obs - mean_vec) ** 2, axis=1)
        dispersions.append(sq_dist.mean())
        
    mean_dispersion = np.mean(dispersions)
    
    # 3. Calculate First Collapse (Across Observers)
    # Global mean of raw observer representations: Shape [H]
    global_raw_mean = cls_per_bot.mean(axis=(0, 1))
    
    # Total variance of raw observer representations
    total_var_raw = np.sum((cls_per_bot - global_raw_mean) ** 2) / (N * B)
    
    # Global mean of consensus representations: Shape [H]
    global_consensus_mean = frechet_means.mean(axis=0)
    
    # Total variance of consensus representations
    total_var_consensus = np.sum((frechet_means - global_consensus_mean) ** 2) / N
    
    # Percentage lost in observer collapse
    observer_collapse_pct = (1.0 - (total_var_consensus / total_var_raw)) * 100.0
    
    # 4. Calculate Second Collapse (Across Dimensions - 1D PC1 Projection)
    # Fit PCA on consensus centers to find the variance along the first principal component
    pca = PCA(n_components=1)
    pca.fit(frechet_means)
    var_1d = pca.explained_variance_[0]
    
    # Percentage lost in 1D projection collapse
    projection_collapse_pct = (1.0 - (var_1d / total_var_consensus)) * 100.0
    
    # 5. Cumulative Double Collapse
    cumulative_retained_pct = (var_1d / total_var_raw) * 100.0
    cumulative_collapsed_pct = 100.0 - cumulative_retained_pct
    
    print("QWEN SYNTHETIC CORPUS: DOUBLE VARIANCE COLLAPSE STATS")
    print("="*70)
    print(f"Number of articles (N):              {N}")
    print(f"Number of observers (B):             {B}")
    print(f"Embedding dimension (H):             {H}")
    print("-" * 70)
    print(f"Mean Inter-Observer Dispersion:      {mean_dispersion:.6f}")
    print(f"Total Raw Representation Variance:   {total_var_raw:.6f}")
    print(f"Consensus (Fréchet Mean) Variance:   {total_var_consensus:.6f}")
    print(f"  -> Observer Collapse (Averaging):  {observer_collapse_pct:.2f}% variance lost")
    print("-" * 70)
    print(f"1D Projection (PC1) Variance:        {var_1d:.6f}")
    print(f"  -> Dimension Collapse (1D PC1):    {projection_collapse_pct:.2f}% variance lost")
    print("-" * 70)
    print(f"Cumulative Double Collapse:          {cumulative_collapsed_pct:.2f}% variance lost")
    print(f"Cumulative Variance Retained:        {cumulative_retained_pct:.2f}%")
    print("="*70)
    
    # Save metrics
    metrics = {
        "n_articles": N,
        "n_observers": B,
        "embedding_dim": H,
        "mean_inter_observer_dispersion": float(mean_dispersion),
        "total_var_raw": float(total_var_raw),
        "total_var_consensus": float(total_var_consensus),
        "observer_collapse_pct": float(observer_collapse_pct),
        "var_1d": float(var_1d),
        "projection_collapse_pct": float(projection_collapse_pct),
        "cumulative_collapsed_pct": float(cumulative_collapsed_pct),
        "cumulative_retained_pct": float(cumulative_retained_pct)
    }
    
    output_file = DATA_DIR / "frechet_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Fréchet variance metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
