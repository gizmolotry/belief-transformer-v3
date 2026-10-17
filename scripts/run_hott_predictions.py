"""
Verify HoTT Topological Predictions.
Performs Dirichlet annealing over the M-observer representations,
identifies topological bonds and cracks (ruptures),
and correlates them with observer distance variance (perspectival friction).
"""

import json
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_ind

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OBSERVER_FILE = ROOT_DIR / "outputs" / "observer_43.pt"

def main():
    if not OBSERVER_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {OBSERVER_FILE}")
        
    data = torch.load(OBSERVER_FILE, map_location="cpu", weights_only=False)
    bot_rkhs = data["bot_rkhs"].numpy() # Shape [50, 8, 2048]
    N, B, D = bot_rkhs.shape
    
    # 1. Sweep alpha to find topological bonds and cracks
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_samples = 50
    
    # Track mean similarity and peak variance
    mean_similarities = []
    max_variances = np.zeros((N, N))
    
    np.random.seed(42)
    
    # For each alpha, sample blended observers and compute Gram matrices
    for alpha in alphas:
        # Dirichlet weights: shape [n_samples, B]
        weights = np.random.dirichlet([alpha] * B, size=n_samples)
        
        # Fused representations: shape [N, n_samples, D]
        fused = np.einsum('nbd,kb->nkd', bot_rkhs, weights)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(fused, axis=-1, keepdims=True)
        fused_norm = fused / (norms + 1e-10)
        
        # Compute Gram matrices: shape [n_samples, N, N]
        grams = np.zeros((n_samples, N, N))
        for k in range(n_samples):
            grams[k] = fused_norm[:, k, :] @ fused_norm[:, k, :].T
            
        mean_gram = grams.mean(axis=0)
        var_gram = grams.var(axis=0)
        
        mean_similarities.append(mean_gram)
        max_variances = np.maximum(max_variances, var_gram)
        
    # Global mean similarity
    global_mean_sim = np.mean(mean_similarities, axis=0)
    
    # Define bonds and cracks based on thresholds
    bond_threshold = 0.45
    crack_threshold = 0.003
    
    bond_matrix = (global_mean_sim > bond_threshold).astype(int)
    np.fill_diagonal(bond_matrix, 0) # Remove self-bonds
    
    crack_matrix = (max_variances > crack_threshold).astype(int)
    np.fill_diagonal(crack_matrix, 0) # Remove self-cracks
    
    # 2. Compute pairwise observer distance variance (Perspectival Friction)
    # D_obs[o, i, j] = distance between i and j under observer o
    D_obs = np.zeros((B, N, N))
    for o in range(B):
        # Cosine distance under observer o
        feats = bot_rkhs[:, o, :]
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        feats_norm = feats / (norms + 1e-10)
        D_obs[o] = 1.0 - (feats_norm @ feats_norm.T)
        
    # Variance of pairwise distances across the 8 observers
    dist_variance = D_obs.var(axis=0)
    np.fill_diagonal(dist_variance, 0.0)
    
    # 3. Correlate cracks and bonds with distance variance (Friction)
    # Flatten upper triangles to avoid self-relations and double-counting
    tri_idx = np.triu_indices(N, k=1)
    flat_cracks = crack_matrix[tri_idx]
    flat_bonds = bond_matrix[tri_idx]
    flat_friction = dist_variance[tri_idx]
    flat_sim = global_mean_sim[tri_idx]
    
    # Correlations
    r_crack_friction, p_crack_friction = pearsonr(flat_cracks, flat_friction)
    rho_crack_friction, p_rho_crack_friction = spearmanr(flat_cracks, flat_friction)
    
    r_bond_friction, p_bond_friction = pearsonr(flat_bonds, flat_friction)
    
    # T-test: Do cracked pairs have higher friction (distance variance) than non-cracked pairs?
    cracked_friction = flat_friction[flat_cracks == 1]
    non_cracked_friction = flat_friction[flat_cracks == 0]
    t_stat, t_p_value = ttest_ind(cracked_friction, non_cracked_friction, equal_var=False)
    
    total_pairs = len(flat_cracks)
    n_bonds = int(flat_bonds.sum())
    n_cracks = int(flat_cracks.sum())
    
    metrics = {
        "n_articles": N,
        "total_pairs": total_pairs,
        "n_bonds": n_bonds,
        "bond_ratio": float(n_bonds / total_pairs),
        "n_cracks": n_cracks,
        "crack_ratio": float(n_cracks / total_pairs),
        "correlation_crack_friction": {
            "pearson_r": float(r_crack_friction),
            "pearson_p": float(p_crack_friction),
            "spearman_rho": float(rho_crack_friction),
            "spearman_p": float(p_rho_crack_friction)
        },
        "correlation_bond_friction": {
            "pearson_r": float(r_bond_friction),
            "pearson_p": float(p_bond_friction)
        },
        "t_test_cracked_vs_non_cracked": {
            "t_statistic": float(t_stat),
            "p_value": float(t_p_value),
            "cracked_mean_friction": float(np.mean(cracked_friction)),
            "non_cracked_mean_friction": float(np.mean(non_cracked_friction))
        }
    }
    
    print("\n" + "="*70)
    print("HoTT TOPOLOGICAL VERIFICATION: PATH-CONNECTIVITY & OBSTRUCTION")
    print("="*70)
    print(f"Total Pairs:              {total_pairs}")
    print(f"Bonds (Path-Connected):   {n_bonds} ({n_bonds/total_pairs*100:.2f}%)")
    print(f"Cracks (Obstructed):      {n_cracks} ({n_cracks/total_pairs*100:.2f}%)")
    print("-" * 70)
    print("Correlation: Cracks vs Perspectival Friction (Distance Var)")
    print(f"  Pearson r:    {r_crack_friction:.4f} (p = {p_crack_friction:.2e})")
    print(f"  Spearman rho: {rho_crack_friction:.4f} (p = {p_rho_crack_friction:.2e})")
    print("-" * 70)
    print("Difference in Perspectival Friction (T-Test)")
    print(f"  Cracked Pairs Mean Friction:     {np.mean(cracked_friction):.6f}")
    print(f"  Non-Cracked Pairs Mean Friction: {np.mean(non_cracked_friction):.6f}")
    print(f"  T-statistic:                     {t_stat:.4f} (p = {t_p_value:.2e})")
    print("="*70)
    
    # Save to file
    output_file = DATA_DIR / "hott_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"HoTT verification metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
