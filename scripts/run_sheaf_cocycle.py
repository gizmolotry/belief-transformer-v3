"""
Run Sheaf-Theoretic Cocycle Consistency Test on Procrustes Transport.
Computes the pairwise Procrustes rotation matrices between the 8 observer representations,
tests the cocycle consistency condition R_{i->k} ≈ R_{j->k} @ R_{i->j} for all triples,
and reports the normalized Frobenius norm residuals.
"""

import json
import torch
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RUN_DIR = ROOT_DIR / "outputs" / "non_circular_suite" / "exp1_unstructured"

def compute_procrustes_rotation(X, Y):
    """Compute the orthogonal Procrustes rotation matrix R that aligns X to Y."""
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Compute SVD of the cross-covariance matrix
    A = X_centered.T @ Y_centered
    U, _, Vt = np.linalg.svd(A)
    
    # Orthogonal rotation matrix
    R = U @ Vt
    return R

def main():
    observer_file = RUN_DIR / "observer_42.pt"
    if not observer_file.exists():
        raise FileNotFoundError(f"Run Experiment 1 first. File not found: {observer_file}")
        
    data = torch.load(observer_file, map_location="cpu", weights_only=False)
    t1_logits = data["T1_embeddings"].numpy() # Shape [80, 24]
    
    N = t1_logits.shape[0]
    n_observers = 8
    
    # 1. Extract the 3D logit manifolds for each of the 8 observers
    X = []
    for o in range(n_observers):
        # Extract the 3 logits (Entailment, Neutral, Contradiction)
        X_o = t1_logits[:, 3 * o : 3 * o + 3]
        X.append(X_o)
        
    # 2. Compute all pairwise Procrustes rotation matrices
    R = {}
    for i in range(n_observers):
        for j in range(n_observers):
            R[(i, j)] = compute_procrustes_rotation(X[i], X[j])
            
    # 3. Test cocycle consistency on all triples (i, j, k) where they are distinct
    residuals = []
    triples = []
    for i in range(n_observers):
        for j in range(n_observers):
            for k in range(n_observers):
                if i == j or j == k or i == k:
                    continue
                # Expected: R_{i->k} ≈ R_{j->k} @ R_{i->j}
                R_direct = R[(i, k)]
                R_composed = R[(j, k)] @ R[(i, j)]
                
                # Compute Frobenius norm of the difference
                frob_diff = np.linalg.norm(R_direct - R_composed, ord='fro')
                
                # Normalize by the Frobenius norm of an orthogonal 3x3 matrix (sqrt(3))
                norm_res = frob_diff / np.sqrt(3.0)
                residuals.append(float(norm_res))
                triples.append((i, j, k))
                
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    min_res = np.min(residuals)
    max_res = np.max(residuals)
    
    metrics = {
        "mean_cocycle_residual": mean_res,
        "std_cocycle_residual": std_res,
        "min_cocycle_residual": min_res,
        "max_cocycle_residual": max_res,
        "is_sheaf_like": bool(mean_res < 0.15), # Threshold for sheaf-like transport
        "n_triples_evaluated": len(residuals)
    }
    
    print("\n" + "="*70)
    print("SHEAF COCYCLE CONSISTENCY TEST ON OBSERVER TRANSPORT")
    print("="*70)
    print(f"Number of triples evaluated: {len(residuals)}")
    print(f"Mean Cocycle Residual:      {mean_res:.6f}")
    print(f"Std Cocycle Residual:       {std_res:.6f}")
    print(f"Min Cocycle Residual:       {min_res:.6f}")
    print(f"Max Cocycle Residual:       {max_res:.6f}")
    print(f"Transport Nature:           {'Sheaf-like (Gluing Holds)' if mean_res < 0.15 else 'Path-Dependent (Non-Sheaf)'}")
    print("="*70)
    
    # Save results
    output_file = DATA_DIR / "sheaf_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Sheaf transport metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
