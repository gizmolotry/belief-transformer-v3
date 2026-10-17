from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_loao_for_matrix(
    polarity_matrix: np.ndarray,
    axis_labels: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    N, K = polarity_matrix.shape
    assert K == 8, f"Expected 8 axes, got {K}"

    # Baseline: Full 8-axis PCA
    pca_full = PCA().fit(polarity_matrix)
    evr_full = float(pca_full.explained_variance_ratio_[0])
    evr2_full = float(pca_full.explained_variance_ratio_[1])
    z_full = pca_full.transform(polarity_matrix)[:, 0]
    ranks_full = np.argsort(np.argsort(z_full))

    # 1. Leave-One-Axis-Out (LOAO)
    loao_results = []
    r_loao_list = []
    rho_loao_list = []
    max_rank_shifts = []

    for j in range(K):
        mask = [idx for idx in range(K) if idx != j]
        sub_matrix = polarity_matrix[:, mask]
        pca_sub = PCA().fit(sub_matrix)
        evr_sub = float(pca_sub.explained_variance_ratio_[0])
        z_sub = pca_sub.transform(sub_matrix)[:, 0]

        # Fix sign
        if np.corrcoef(z_sub, z_full)[0, 1] < 0:
            z_sub = -z_sub

        r_val, r_p = pearsonr(z_sub, z_full)
        rho_val, rho_p = spearmanr(z_sub, z_full)
        ranks_sub = np.argsort(np.argsort(z_sub))
        max_shift = int(np.max(np.abs(ranks_sub - ranks_full)))

        # Subspace angle in R^8
        v_full = pca_full.components_[0]  # [8]
        v_sub_8d = np.zeros(K)
        v_sub_8d[mask] = pca_sub.components_[0]
        v_sub_8d /= np.linalg.norm(v_sub_8d)
        angle_rad = subspace_angles(v_full[:, None], v_sub_8d[:, None])[0]
        angle_deg = float(np.degrees(angle_rad))

        r_loao_list.append(float(r_val))
        rho_loao_list.append(float(rho_val))
        max_rank_shifts.append(max_shift)

        loao_results.append({
            "dropped_axis_index": j,
            "dropped_axis_label": axis_labels[j],
            "pc1_evr_without_axis": evr_sub,
            "delta_pc1_evr": float(evr_sub - evr_full),
            "pearson_r_with_full": float(r_val),
            "pearson_p": float(r_p),
            "spearman_rho_with_full": float(rho_val),
            "spearman_p": float(rho_p),
            "max_rank_shift": max_shift,
            "subspace_angle_deg": angle_deg,
        })

    # 2. Leave-Two-Axes-Out (LTAO)
    ltao_results = []
    r_ltao_list = []
    for pair in itertools.combinations(range(K), 2):
        mask = [idx for idx in range(K) if idx not in pair]
        sub_matrix = polarity_matrix[:, mask]
        pca_sub = PCA().fit(sub_matrix)
        z_sub = pca_sub.transform(sub_matrix)[:, 0]
        if np.corrcoef(z_sub, z_full)[0, 1] < 0:
            z_sub = -z_sub
        r_val, _ = pearsonr(z_sub, z_full)
        r_ltao_list.append(float(r_val))
        ltao_results.append({
            "dropped_axes": [axis_labels[pair[0]], axis_labels[pair[1]]],
            "pearson_r": float(r_val),
        })

    # 3. Half-Split Stability (4 vs 4)
    r_half_list = []
    for quad in itertools.combinations(range(K), 4):
        sub_matrix = polarity_matrix[:, quad]
        pca_sub = PCA().fit(sub_matrix)
        z_sub = pca_sub.transform(sub_matrix)[:, 0]
        if np.corrcoef(z_sub, z_full)[0, 1] < 0:
            z_sub = -z_sub
        r_val, _ = pearsonr(z_sub, z_full)
        r_half_list.append(float(r_val))

    print(f"\n===========================================================================")
    print(f"LEAVE-ONE-AXIS-OUT ANALYSIS: {dataset_name} (N={N})")
    print(f"===========================================================================")
    print(f"Full 8-Axis Baseline PC1 EVR : {evr_full*100:.2f}% (PC2: {evr2_full*100:.2f}%)")
    print(f"LOAO Min Pearson r           : {np.min(r_loao_list):.4f} (Axis {axis_labels[np.argmin(r_loao_list)]})")
    print(f"LOAO Mean Pearson r          : {np.mean(r_loao_list):.4f} +/- {np.std(r_loao_list):.4f}")
    print(f"LOAO Min Spearman rho        : {np.min(rho_loao_list):.4f}")
    print(f"LOAO Max Rank Shift          : {np.max(max_rank_shifts)} / {N} articles")
    print(f"LTAO (Leave-2) Min r         : {np.min(r_ltao_list):.4f} (Mean: {np.mean(r_ltao_list):.4f})")
    print(f"Half-Split (4-of-8) Min r    : {np.min(r_half_list):.4f} (Mean: {np.mean(r_half_list):.4f})")

    construct_type = "BANK-LEVEL LATENT CONSTRUCT" if np.min(r_loao_list) >= 0.95 else "ITEM-DRIVEN CONSTRUCT"
    print(f"Diagnostic Conclusion        : {construct_type}")
    print(f"===========================================================================")

    return {
        "dataset": dataset_name,
        "n_samples": N,
        "full_8_axis_pc1_evr": evr_full,
        "full_8_axis_pc2_evr": evr2_full,
        "diagnostic_conclusion": construct_type,
        "loao_summary": {
            "min_r": float(np.min(r_loao_list)),
            "mean_r": float(np.mean(r_loao_list)),
            "std_r": float(np.std(r_loao_list)),
            "min_rho": float(np.min(rho_loao_list)),
            "mean_rho": float(np.mean(rho_loao_list)),
            "max_rank_shift": int(np.max(max_rank_shifts)),
        },
        "ltao_summary": {
            "min_r": float(np.min(r_ltao_list)),
            "mean_r": float(np.mean(r_ltao_list)),
            "std_r": float(np.std(r_ltao_list)),
        },
        "half_split_summary": {
            "min_r": float(np.min(r_half_list)),
            "mean_r": float(np.mean(r_half_list)),
            "std_r": float(np.std(r_half_list)),
        },
        "loao_per_axis": loao_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Axis-Out Robustness Analysis.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "leave_one_axis_out_scaling_analysis.json")
    args = parser.parse_args()

    axis_labels = [
        "1_Israeli_Actions",
        "2_Palestinian_Framing",
        "3_Proportionality",
        "4_Resistance_vs_Terrorism",
        "5_Occupation",
        "6_Civilian_Casualties",
        "7_Historical_Context",
        "8_International_Law",
    ]

    all_results = {}

    # 1. Fresh N=120 Corpus (from cached extraction)
    cache_fresh = PROJECT_ROOT / "data" / "cache_fresh_replication" / "fresh_120_q0_ideology_embeddings.pt"
    if cache_fresh.exists():
        d = torch.load(cache_fresh, map_location="cpu", weights_only=False)
        probs = torch.softmax(d["logits_raw"].float(), dim=-1).numpy()
        pol_fresh = probs[..., 2] - probs[..., 0]
        res_fresh = run_loao_for_matrix(pol_fresh, axis_labels, "Fresh_Unseen_N120")
        all_results["fresh_unseen_N120"] = res_fresh

    # 2. Canonical N=60 Corpus (from cached extraction)
    cache_n60 = PROJECT_ROOT / "data" / "cache_cross_model" / "cross_model_microsoft_deberta-v2-xlarge-mnli_N60.pt"
    if cache_n60.exists():
        d = torch.load(cache_n60, map_location="cpu", weights_only=False)
        probs = torch.softmax(d["ideo_logits"].float(), dim=-1).numpy()
        pol_n60 = probs[..., 2] - probs[..., 0]
        res_n60 = run_loao_for_matrix(pol_n60, axis_labels, "Controlled_Canonical_N60")
        all_results["controlled_canonical_N60"] = res_n60

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SUCCESS] Saved LOAO robustness report to {args.output}")


if __name__ == "__main__":
    main()
