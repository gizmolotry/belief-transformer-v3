"""
verify_riemannian_fix.py
Direct Verification Script for core/riemannian_geometry.py Zero-Variance Fix

Tests the ablation module's robust_normalize function and compute_riemannian_physics
on a zero-variance constant control array to verify that it returns exact zeros
and classifies as consensus (0% residual), rather than generating np.random.uniform noise.
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

from core.riemannian_geometry import compute_riemannian_physics

def test_robust_normalize_zero_variance():
    print("=" * 70)
    print(" DIRECT ABLATION TEST: core/riemannian_geometry.py robust_normalize")
    print("=" * 70)

    # 1. Create a zero-variance constant array (10 items, all equal to 0.42)
    N = 10
    D = 1536
    # 10 identical embeddings -> zero-variance across instances
    track2_const = torch.ones(N, D)

    # Pass zero-variance embeddings to compute_riemannian_physics
    physics_result = compute_riemannian_physics(track2_const)
    
    blinker_norm = physics_result['blinker'] # Normalized blinker score
    walker_norm = physics_result['walker'] # Normalized walker score
    singularities = physics_result['singularity_types']

    print("\n▶ ABLATION COMPUTATION RESULTS:")
    print(f"  ├── Normalized Blinker Scores: {blinker_norm.numpy()}")
    print(f"  ├── Normalized Walker Scores:  {walker_norm.numpy()}")
    print(f"  ├── Singularity Classifications: {singularities}")

    # 3. Assertions
    is_zeros_blinker = torch.all(blinker_norm == 0.0).item()
    is_zeros_walker = torch.all(walker_norm == 0.0).item()
    all_consensus = all(s == 'consensus' for s in singularities)

    print("\n▶ VERIFICATION VERDICT:")
    print(f"  ├── Blinker Is Exact Zero (No Random Noise): {'PASSED' if is_zeros_blinker else 'FAILED'}")
    print(f"  ├── Walker Is Exact Zero (No Random Noise):  {'PASSED' if is_zeros_walker else 'FAILED'}")
    print(f"  └── Classification Is 100% Consensus:        {'PASSED' if all_consensus else 'FAILED'}")

    assert is_zeros_blinker, "FAILED: blinker score produced non-zero or random fallback!"
    assert is_zeros_walker, "FAILED: walker score produced non-zero or random fallback!"
    assert all_consensus, "FAILED: zero-variance input produced non-consensus singularity!"
    print("\n▶ RESULT: core/riemannian_geometry.py zero-variance fix is 100% EMPIRICALLY VERIFIED!")

if __name__ == "__main__":
    test_robust_normalize_zero_variance()
