"""
Run Experiment Baseline Comparison.
Compares the multidimensional Semantic Interferometer manifold (T5 RKS)
against a standard 1D horizontal bias baseline (e.g. Ground News/AllSides style
publisher bias ratings) on the unstructured non-circular corpus.
Computes distance correlation, stress, and k-NN Jaccard neighbor overlap.
"""

import json
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import jaccard_score

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RUN_DIR = ROOT_DIR / "outputs" / "non_circular_suite" / "exp1_unstructured"

# Define standard 1D horizontal bias ratings (Ground News/AllSides style)
PUBLISHER_BIAS_MAP = {
    "Al Jazeera": -1.0,      # Left / Pan-Arab
    "Wafa News Agency": -0.8,# Left-Center
    "Haaretz": -0.4,         # Left-Center
    "Ynet (Security Desk)": 0.5, # Right-Center
    "CBN News": 1.0          # Right
}

def distcorr(X, Y):
    """Compute distance correlation between two distance matrices or coordinate sets."""
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    
    n = X.shape[0]
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    
    if dcov2_xx * dcov2_yy == 0:
        return 0.0
    return np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy))

def compute_knn_overlap(D1, D2, k=5):
    """Compute mean Jaccard overlap of k-NN neighbor sets between two distance matrices."""
    n = D1.shape[0]
    overlaps = []
    for idx in range(n):
        # Find indices of k nearest neighbors (excluding self)
        knn1 = set(np.argsort(D1[idx])[1:k+1])
        knn2 = set(np.argsort(D2[idx])[1:k+1])
        intersection = len(knn1.intersection(knn2))
        union = len(knn1.union(knn2))
        overlaps.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(overlaps))

def main():
    observer_file = RUN_DIR / "observer_42.pt"
    if not observer_file.exists():
        raise FileNotFoundError(f"Run Experiment 1 first. File not found: {observer_file}")
        
    data = torch.load(observer_file, map_location="cpu", weights_only=False)
    features = data["features"]
    articles = data["article_metadata"]
    
    # 1. Map articles to 1D baseline coordinates
    baseline_coords = []
    for art in articles:
        pub = art["publication"]
        bias = PUBLISHER_BIAS_MAP.get(pub, 0.0)
        baseline_coords.append(bias)
    baseline_coords = np.array(baseline_coords, dtype=np.float64)
    
    # 2. Compute distance matrices
    D_manifold = squareform(pdist(features))
    D_baseline = squareform(pdist(baseline_coords[:, None]))
    
    # 3. Compute metrics
    # Distance Correlation
    d_corr = distcorr(features, baseline_coords)
    
    # Stress: Normalized sum of squared differences between distances
    # Measures how much geometric distortion is introduced by the 1D baseline
    num = np.sum((D_manifold - D_baseline) ** 2)
    den = np.sum(D_manifold ** 2)
    stress = np.sqrt(num / den) if den > 0 else 1.0
    
    # k-NN Neighborhood Overlap (local topology preservation)
    knn_overlap_5 = compute_knn_overlap(D_manifold, D_baseline, k=5)
    knn_overlap_10 = compute_knn_overlap(D_manifold, D_baseline, k=10)
    
    metrics = {
        "distance_correlation": float(d_corr),
        "geometric_stress": float(stress),
        "knn_overlap_k5": float(knn_overlap_5),
        "knn_overlap_k10": float(knn_overlap_10)
    }
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON (MULTIDIMENSIONAL VS 1D HORIZONTAL)")
    print("="*70)
    print(f"Distance Correlation: {d_corr:.4f}")
    print(f"Geometric Stress (Distortion): {stress:.4f}")
    print(f"k-NN Overlap (k=5): {knn_overlap_5:.4f}")
    print(f"k-NN Overlap (k=10): {knn_overlap_10:.4f}")
    print("="*70)
    
    # Write to target path
    output_file = DATA_DIR / "baseline_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Baseline comparison metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
