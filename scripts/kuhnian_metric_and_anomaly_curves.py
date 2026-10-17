#!/usr/bin/env python3
"""Metric-Independent Neighborhoods and Anomaly Agreement Hierarchy.

Fulfills Points 12 and 13 of the Kuhnian Clarification Battery:
1. Point 12: Neighborhood sensitivity sweep k in [3, 50] across 6 distance metrics:
   - Euclidean, Cosine, Correlation, Standardized Euclidean, Whitened (ZCA), Rank Distance.
   - Evaluated across: Same FW / Diff Realization, Paraphrase, Conceptual Shift,
     Operator Shift, and Absurd Control.
   - Tests metric-invariance of neighborhood divergence.
2. Point 13: Anomaly hierarchy:
   r(same fw, diff bank) > r(diff fw, same op) > r(diff op)
   across Reconstruction Error, Mahalanobis Distance, and Local Outlier Factor.

Outputs:
- paper/kuhnian_metric_anomaly_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Distance Metrics
# -----------------------------------------------------------------------------

def compute_distance_matrix(X: np.ndarray, metric_name: str) -> np.ndarray:
    N, D = X.shape
    if metric_name == "euclidean":
        return cdist(X, X, metric="euclidean")
    elif metric_name == "cosine":
        return cdist(X, X, metric="cosine")
    elif metric_name == "correlation":
        return cdist(X, X, metric="correlation")
    elif metric_name == "standardized":
        std = np.std(X, axis=0, keepdims=True) + 1e-6
        X_std = X / std
        return cdist(X_std, X_std, metric="euclidean")
    elif metric_name == "whitened":
        # ZCA Whitening
        X_c = X - np.mean(X, axis=0, keepdims=True)
        cov = (X_c.T @ X_c) / (N - 1)
        U, S, Vt = np.linalg.svd(cov)
        S_inv = np.diag(1.0 / np.sqrt(S + 1e-5))
        W_zca = U @ S_inv @ U.T
        X_zca = X_c @ W_zca
        return cdist(X_zca, X_zca, metric="euclidean")
    elif metric_name == "rank":
        # Spearman / rank distance
        X_ranked = np.zeros_like(X)
        for i in range(N):
            X_ranked[i] = rankdata(X[i])
        return cdist(X_ranked, X_ranked, metric="correlation")
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def compute_neighborhood_overlap_curve(
    dist_A: np.ndarray, dist_B: np.ndarray, k_values: List[int]
) -> Dict[int, float]:
    N = len(dist_A)
    dA = dist_A.copy()
    dB = dist_B.copy()
    np.fill_diagonal(dA, np.inf)
    np.fill_diagonal(dB, np.inf)

    curve = {}
    for k in k_values:
        j_list = []
        for i in range(N):
            nn_A = set(np.argsort(dA[i])[:k])
            nn_B = set(np.argsort(dB[i])[:k])
            inter = len(nn_A & nn_B)
            union = len(nn_A | nn_B)
            j_list.append(inter / union if union > 0 else 0.0)
        curve[k] = float(np.mean(j_list))
    return curve


# -----------------------------------------------------------------------------
# Anomaly Scoring Methods
# -----------------------------------------------------------------------------

def compute_anomaly_scores(X: np.ndarray) -> Dict[str, np.ndarray]:
    N, D = X.shape
    # 1. Reconstruction error from PC1
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Z)
    rec = pc1 @ pca.components_
    anom_recon = np.mean((Z - rec) ** 2, axis=1)

    # 2. Regularized Mahalanobis distance to centroid
    cov = (Z.T @ Z) / (N - 1) + 0.05 * np.eye(D)
    cov_inv = np.linalg.inv(cov)
    anom_mahal = np.sum((Z @ cov_inv) * Z, axis=1)

    # 3. Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=15, novelty=False)
    lof.fit(Z)
    anom_lof = -lof.negative_outlier_factor_  # Higher = more anomalous

    return {
        "reconstruction_error": anom_recon,
        "mahalanobis_distance": anom_mahal,
        "local_outlier_factor": anom_lof,
    }


def run_all():
    print("=" * 80)
    print("METRIC-INDEPENDENT NEIGHBORHOODS & ANOMALY HIERARCHY (POINTS 12, 13)")
    print("=" * 80, flush=True)

    # 1. Load data
    cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    g_cache_path = ROOT / "data" / "cache_replicated_g_theory_logits.pt"
    g_cache = torch.load(g_cache_path, map_location="cpu", weights_only=False)

    fa_cache = np.load(ROOT / "data" / "cache_audit_deberta_embeddings.npz")
    docs_emb = fa_cache["prospective"]
    hyp_a = fa_cache["hyp_a"]
    hyp_b = fa_cache["hyp_b"]
    d_u = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)
    a_u = hyp_a / (np.linalg.norm(hyp_a, axis=1, keepdims=True) + 1e-12)
    b_u = hyp_b / (np.linalg.norm(hyp_b, axis=1, keepdims=True) + 1e-12)
    fa_base = (d_u @ a_u.T - d_u @ b_u.T).astype(float)  # [120, 8]

    def extract_pol(b_val):
        la = b_val["la"].float()
        lb = b_val["lb"].float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        return (sa - sb).astype(float)

    pol_q0_r1 = extract_pol(g_cache["geopolitical_conflict:realization_1"])
    pol_q0_r2 = extract_pol(g_cache["geopolitical_conflict:realization_2"])
    pol_q0 = extract_pol(cache["q0_original_geopolitical"])
    pol_q1 = extract_pol(cache["q1_paraphrase_control"])
    pol_q2 = extract_pol(cache["q2_humanitarian_rights"])
    pol_q3 = extract_pol(cache["q3_realpolitik_statecraft"])
    pol_ihl = extract_pol(cache["disciplinary_paradigms:ihl_humanitarian_law"])
    pol_cook = extract_pol(cache["absurd_structurally_matched_paradigms:culinary_arts"])
    pol_fa = fa_base

    comparisons = {
        "same_fw_diff_realization": (pol_q0_r1, pol_q0_r2, "same_framework"),
        "paraphrase_control_q0_q1": (pol_q0, pol_q1, "paraphrase"),
        "conceptual_shift_q0_q2": (pol_q0, pol_q2, "conceptual_diff_fw"),
        "conceptual_shift_q0_q3": (pol_q0, pol_q3, "conceptual_diff_fw"),
        "disciplinary_shift_q0_ihl": (pol_q0, pol_ihl, "disciplinary_diff_fw"),
        "operator_shift_nli_vs_fa": (pol_q0, pol_fa, "operator_shift"),
        "absurd_control_q0_cooking": (pol_q0, pol_cook, "absurd_control"),
    }

    metrics = ["euclidean", "cosine", "correlation", "standardized", "whitened", "rank"]
    k_sweep = [3, 5, 8, 10, 15, 20, 30, 40, 50]

    # -------------------------------------------------------------------------
    # POINT 12: NEIGHBORHOOD SWEEP ACROSS METRICS
    # -------------------------------------------------------------------------
    print("\n--- Point 12: Neighborhood Overlap Across 6 Metrics & k in [3, 50] ---", flush=True)
    neighborhood_results = {}
    for comp_name, (xA, xB, comp_type) in comparisons.items():
        comp_res = {}
        for m in metrics:
            dist_A = compute_distance_matrix(xA, m)
            dist_B = compute_distance_matrix(xB, m)
            curve = compute_neighborhood_overlap_curve(dist_A, dist_B, k_sweep)
            comp_res[m] = curve
        neighborhood_results[comp_name] = {"type": comp_type, "metrics": comp_res}
        print(f"  {comp_name:<30} | J10 (Euclidean): {comp_res['euclidean'][10]:.4f} | J10 (Cosine): {comp_res['cosine'][10]:.4f} | J10 (Whitened): {comp_res['whitened'][10]:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # POINT 13: ANOMALY RANKING AGREEMENT HIERARCHY
    # -------------------------------------------------------------------------
    print("\n--- Point 13: Anomaly Ranking Agreement Hierarchy ---", flush=True)
    # Predicted hierarchy:
    # r(same fw, diff bank) > r(diff fw, same op) > r(diff op)
    anomaly_methods = ["reconstruction_error", "mahalanobis_distance", "local_outlier_factor"]

    # Compute anomaly scores for each matrix
    anom_scores = {
        "q0_r1": compute_anomaly_scores(pol_q0_r1),
        "q0_r2": compute_anomaly_scores(pol_q0_r2),
        "q0": compute_anomaly_scores(pol_q0),
        "q1": compute_anomaly_scores(pol_q1),
        "q2": compute_anomaly_scores(pol_q2),
        "q3": compute_anomaly_scores(pol_q3),
        "ihl": compute_anomaly_scores(pol_ihl),
        "frameaxis": compute_anomaly_scores(pol_fa),
    }

    anomaly_corrs = {}
    for method in anomaly_methods:
        # 1. Level 1: Same FW, diff bank
        r_same_real = spearmanr(anom_scores["q0_r1"][method], anom_scores["q0_r2"][method])[0]
        r_same_para = spearmanr(anom_scores["q0"][method], anom_scores["q1"][method])[0]
        level_1 = float(np.mean([r_same_real, r_same_para]))

        # 2. Level 2: Diff FW, same operator (NLI)
        r_q0_q2 = spearmanr(anom_scores["q0"][method], anom_scores["q2"][method])[0]
        r_q0_q3 = spearmanr(anom_scores["q0"][method], anom_scores["q3"][method])[0]
        r_q0_ihl = spearmanr(anom_scores["q0"][method], anom_scores["ihl"][method])[0]
        level_2 = float(np.mean([r_q0_q2, r_q0_q3, r_q0_ihl]))

        # 3. Level 3: Diff operator (NLI vs FrameAxis)
        r_diff_op = spearmanr(anom_scores["q0"][method], anom_scores["frameaxis"][method])[0]
        level_3 = float(r_diff_op)

        hierarchy_holds = (level_1 > level_2) and (level_2 > level_3)
        anomaly_corrs[method] = {
            "level_1_same_fw_diff_bank": level_1,
            "level_2_diff_fw_same_op": level_2,
            "level_3_diff_op": level_3,
            "hierarchy_holds": hierarchy_holds,
        }
        print(f"  {method:<25} | L1 (Same FW): {level_1:6.4f} > L2 (Diff FW): {level_2:6.4f} > L3 (Diff Op): {level_3:6.4f} | Holds: {hierarchy_holds}", flush=True)

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "point_12_neighborhood_curves": neighborhood_results,
        "point_13_anomaly_hierarchy": anomaly_corrs,
    }

    out_file = ROOT / "paper" / "kuhnian_metric_anomaly_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[SAVED] Saved metric and anomaly results to {out_file}", flush=True)


if __name__ == "__main__":
    run_all()
