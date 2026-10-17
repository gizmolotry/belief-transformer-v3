#!/usr/bin/env python3
"""Empirical evaluation of Kuhnian paradigms, incommensurability, neighborhood overlap, and anomaly relativity.

This script operationalizes:
1. Paradigms as measurement systems: P = (H, O, M, L)
2. Incommensurability as bounded transformability: Procrustes disparity, affine R^2, subspace angles
3. Kuhnian "world change": k-NN neighborhood Jaccard overlap J_k(x; P_A, P_B)
4. Anomaly relativity: A_P(x) residual variance and cross-paradigm anomaly correlation r(A_P_A, A_P_B)
5. Perspectivally robust invariants: dimensionality, participation ratio, extreme anchor stability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return x / norms


def compute_knn_overlap(X_A: np.ndarray, X_B: np.ndarray, k_values=(5, 10, 15)) -> Dict[str, Any]:
    """Compute document-level k-NN Jaccard neighborhood overlap."""
    N = len(X_A)
    dist_A = cdist(X_A, X_A, metric="euclidean")
    dist_B = cdist(X_B, X_B, metric="euclidean")
    np.fill_diagonal(dist_A, np.inf)
    np.fill_diagonal(dist_B, np.inf)

    results = {}
    for k in k_values:
        jaccards = []
        for i in range(N):
            nn_A = set(np.argsort(dist_A[i])[:k])
            nn_B = set(np.argsort(dist_B[i])[:k])
            intersection = len(nn_A & nn_B)
            union = len(nn_A | nn_B)
            jaccard = intersection / union if union > 0 else 0.0
            jaccards.append(jaccard)
        jaccards = np.array(jaccards)
        results[f"k_{k}"] = {
            "mean": float(np.mean(jaccards)),
            "std": float(np.std(jaccards)),
            "median": float(np.median(jaccards)),
            "q25": float(np.percentile(jaccards, 25)),
            "q75": float(np.percentile(jaccards, 75)),
            "min": float(np.min(jaccards)),
            "max": float(np.max(jaccards)),
            "zero_overlap_count": int(np.sum(jaccards == 0.0)),
            "zero_overlap_pct": float(np.mean(jaccards == 0.0) * 100.0),
        }
    return results


def compute_anomaly_scores(X: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Compute anomaly score A_P(x) = residual energy outside dominant PC subspace."""
    z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components).fit(z)
    recon = pca.inverse_transform(pca.transform(z))
    residuals = np.sum((z - recon) ** 2, axis=1)
    total_energy = np.sum(z**2, axis=1)
    total_energy[total_energy == 0] = 1e-12
    return residuals / total_energy


def compute_affine_predictability(X_A: np.ndarray, X_B: np.ndarray) -> float:
    """Mean R^2 predicting X_B from X_A using linear regression."""
    zA = StandardScaler().fit_transform(X_A)
    zB = StandardScaler().fit_transform(X_B)
    design = np.column_stack([np.ones(len(zA)), zA])
    beta = np.linalg.lstsq(design, zB, rcond=None)[0]
    pred = design @ beta
    res = zB - pred
    r2_per_col = 1.0 - np.sum(res**2, axis=0) / np.sum((zB - zB.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(r2_per_col))


def analyze_paradigm_pair(
    name_A: str, X_A: np.ndarray, name_B: str, X_B: np.ndarray
) -> Dict[str, Any]:
    zA = StandardScaler().fit_transform(X_A)
    zB = StandardScaler().fit_transform(X_B)

    pca_A = PCA().fit(zA)
    pca_B = PCA().fit(zB)

    scores_A = pca_A.transform(zA)
    scores_B = pca_B.transform(zB)

    # Orient PC1
    if pearsonr(scores_A[:, 0], scores_B[:, 0])[0] < 0:
        scores_B[:, 0] = -scores_B[:, 0]
        pca_B.components_[0] = -pca_B.components_[0]

    r_pc1, p_pc1 = pearsonr(scores_A[:, 0], scores_B[:, 0])
    rho_pc1, p_rho_pc1 = spearmanr(scores_A[:, 0], scores_B[:, 0])

    # 2D Procrustes disparity
    _, _, disp_2d = procrustes(scores_A[:, :2], scores_B[:, :2])

    # Subspace angle for PC1 and 2D
    angle_pc1 = np.degrees(
        subspace_angles(pca_A.components_[:1].T, pca_B.components_[:1].T)[0]
    )
    angle_2d = np.degrees(
        subspace_angles(pca_A.components_[:2].T, pca_B.components_[:2].T)[0]
    )

    # Linear affine predictability R^2
    r2_affine = compute_affine_predictability(zA, zB)

    # In-sample CCA
    cca = CCA(n_components=1, scale=False, max_iter=5000)
    ca_A, ca_B = cca.fit_transform(zA, zB)
    r_cca = float(np.abs(pearsonr(ca_A[:, 0], ca_B[:, 0])[0]))

    # Neighborhood Jaccard overlap in full 8D and 2D PCA
    knn_full = compute_knn_overlap(zA, zB)
    knn_2d = compute_knn_overlap(scores_A[:, :2], scores_B[:, :2])

    # Anomaly scores
    anom_A = compute_anomaly_scores(zA, n_components=1)
    anom_B = compute_anomaly_scores(zB, n_components=1)
    r_anom, p_anom = pearsonr(anom_A, anom_B)

    return {
        "paradigm_A": name_A,
        "paradigm_B": name_B,
        "pc1_pearson_r": float(r_pc1),
        "pc1_pearson_p": float(p_pc1),
        "pc1_spearman_rho": float(rho_pc1),
        "pc1_spearman_p": float(p_rho_pc1),
        "procrustes_disparity_2d": float(disp_2d),
        "subspace_angle_pc1_deg": float(angle_pc1),
        "subspace_angle_2d_deg": float(angle_2d),
        "affine_predictability_mean_r2": float(r2_affine),
        "canonical_correlation": float(r_cca),
        "knn_overlap_8d": knn_full,
        "knn_overlap_2d": knn_2d,
        "anomaly_correlation_r": float(r_anom),
        "anomaly_correlation_p": float(p_anom),
    }


def main():
    print("=" * 80)
    print("KUHNIAN PARADIGM BATTERY: INCOMMENSURABILITY, WORLD CHANGE & ANOMALY RELATIVITY")
    print("=" * 80)

    # ---------------------------------------------------------
    # BATTERY 1: Multi-Lens Conceptual Paradigms (Fresh N=120)
    # ---------------------------------------------------------
    print("\n[BATTERY 1] Loading Fresh N=120 Multi-Lens Conceptual Paradigms...")
    fresh_caches = {
        "P0_Military_Security": ROOT / "data/cache_fresh_replication/fresh_120_q0_ideology_embeddings.pt",
        "P1_Paraphrase_Control": ROOT / "data/cache_fresh_replication/fresh_120_q1_paraphrase_embeddings.pt",
        "P2_Humanitarian_Rights": ROOT / "data/cache_fresh_replication/fresh_120_q2_humanitarian_embeddings.pt",
        "P3_Realpolitik_Statecraft": ROOT / "data/cache_fresh_replication/fresh_120_q3_realpolitik_embeddings.pt",
    }

    conceptual_paradigms = {}
    for pname, ppath in fresh_caches.items():
        data = torch.load(ppath, map_location="cpu", weights_only=False)
        logits = data["logits_raw"].float()
        probs = torch.softmax(logits, dim=-1).numpy()
        # Single-side margin of A (stored in logits_raw)
        pol = probs[..., 2] - probs[..., 0]
        conceptual_paradigms[pname] = pol

    # Summary of individual paradigms
    paradigm_summaries = {}
    anomaly_vectors = {}
    for pname, matrix in conceptual_paradigms.items():
        z = StandardScaler().fit_transform(matrix)
        pca = PCA().fit(z)
        evr = pca.explained_variance_ratio_
        pr = float(np.sum(pca.singular_values_**2)**2 / np.sum(pca.singular_values_**4))
        anom = compute_anomaly_scores(z, n_components=1)
        anomaly_vectors[pname] = anom
        paradigm_summaries[pname] = {
            "pc1_evr": float(evr[0]),
            "pc2_evr": float(evr[1]),
            "participation_ratio": pr,
            "mean_anomaly_score": float(np.mean(anom)),
            "std_anomaly_score": float(np.std(anom)),
        }
        print(f"  {pname}: PC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%, PR={pr:.2f}, Mean Anomaly={np.mean(anom):.3f}")

    # Pairwise analysis across conceptual paradigms
    conceptual_pairs = {}
    pnames = list(conceptual_paradigms.keys())
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            pA, pB = pnames[i], pnames[j]
            pair_key = f"{pA}__vs__{pB}"
            res = analyze_paradigm_pair(pA, conceptual_paradigms[pA], pB, conceptual_paradigms[pB])
            conceptual_pairs[pair_key] = res
            print(f"\n--- {pA} vs {pB} ---")
            print(f"  PC1 Pearson r: {res['pc1_pearson_r']:.4f}, Spearman rho: {res['pc1_spearman_rho']:.4f}")
            print(f"  Procrustes Disparity (2D): {res['procrustes_disparity_2d']:.4f}")
            print(f"  PC1 Subspace Angle: {res['subspace_angle_pc1_deg']:.2f} deg")
            print(f"  Affine Mean R^2: {res['affine_predictability_mean_r2']:.4f}")
            print(f"  CCA Canonical r: {res['canonical_correlation']:.4f}")
            print(f"  Neighborhood Jaccard k=10 (8D): mean={res['knn_overlap_8d']['k_10']['mean']:.4f}, zeros={res['knn_overlap_8d']['k_10']['zero_overlap_pct']:.1f}%")
            print(f"  Anomaly Correlation r: {res['anomaly_correlation_r']:.4f} (p={res['anomaly_correlation_p']:.4f})")

    # ---------------------------------------------------------
    # BATTERY 2: Cross-Operator Paradigms (Prospective Confirmatory N=120)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BATTERY 2] Loading Prospective Confirmatory N=120 Cross-Operator Paradigms...")
    # Load prospective NLI
    nli_path = ROOT / "data/cache_prospective_120_q0_q1_logits.pt"
    nli_cache = torch.load(nli_path, map_location="cpu", weights_only=False)
    la_0 = nli_cache["q0_la"].float()
    lb_0 = nli_cache["q0_lb"].float()
    pa_0 = torch.softmax(la_0, dim=-1).numpy()
    pb_0 = torch.softmax(lb_0, dim=-1).numpy()
    sa_0 = pa_0[..., 2] - pa_0[..., 0]
    sb_0 = pb_0[..., 2] - pb_0[..., 0]
    paired_q0 = sa_0 - sb_0
    single_q0 = sa_0

    # Load DeBERTa embeddings for FrameAxis
    emb_cache = np.load(ROOT / "data/cache_audit_deberta_embeddings.npz")
    docs_emb = emb_cache["prospective"]  # [120, 1536]
    hyp_a = emb_cache["hyp_a"]  # [8, 1536]
    hyp_b = emb_cache["hyp_b"]  # [8, 1536]

    # Baseline FrameAxis
    d_u = unit_rows(docs_emb)
    a_u = unit_rows(hyp_a)
    b_u = unit_rows(hyp_b)
    fa_base = d_u @ a_u.T - d_u @ b_u.T  # [120, 8]

    # FrameAxis without Top-1 PC
    mean_doc = docs_emb.mean(axis=0, keepdims=True)
    docs_c = docs_emb - mean_doc
    _, _, vt = np.linalg.svd(docs_c, full_matrices=False)
    docs_no_top1 = docs_c - docs_c @ vt[:1].T @ vt[:1]
    hyp_a_no_top1 = (hyp_a - mean_doc) - (hyp_a - mean_doc) @ vt[:1].T @ vt[:1]
    hyp_b_no_top1 = (hyp_b - mean_doc) - (hyp_b - mean_doc) @ vt[:1].T @ vt[:1]
    fa_no_top1 = unit_rows(docs_no_top1) @ unit_rows(hyp_a_no_top1).T - unit_rows(docs_no_top1) @ unit_rows(hyp_b_no_top1).T

    operator_paradigms = {
        "P_NLI_Paired": paired_q0,
        "P_NLI_Single": single_q0,
        "P_FrameAxis_Baseline": fa_base,
        "P_FrameAxis_Ablated": fa_no_top1,
    }

    operator_summaries = {}
    for pname, matrix in operator_paradigms.items():
        z = StandardScaler().fit_transform(matrix)
        pca = PCA().fit(z)
        evr = pca.explained_variance_ratio_
        pr = float(np.sum(pca.singular_values_**2)**2 / np.sum(pca.singular_values_**4))
        anom = compute_anomaly_scores(z, n_components=1)
        operator_summaries[pname] = {
            "pc1_evr": float(evr[0]),
            "pc2_evr": float(evr[1]),
            "participation_ratio": pr,
            "mean_anomaly_score": float(np.mean(anom)),
            "std_anomaly_score": float(np.std(anom)),
        }
        print(f"  {pname}: PC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%, PR={pr:.2f}, Mean Anomaly={np.mean(anom):.3f}")

    operator_pairs = {}
    op_names = list(operator_paradigms.keys())
    for i in range(len(op_names)):
        for j in range(i + 1, len(op_names)):
            pA, pB = op_names[i], op_names[j]
            pair_key = f"{pA}__vs__{pB}"
            res = analyze_paradigm_pair(pA, operator_paradigms[pA], pB, operator_paradigms[pB])
            operator_pairs[pair_key] = res
            print(f"\n--- {pA} vs {pB} ---")
            print(f"  PC1 Pearson r: {res['pc1_pearson_r']:.4f}, Spearman rho: {res['pc1_spearman_rho']:.4f}")
            print(f"  Procrustes Disparity (2D): {res['procrustes_disparity_2d']:.4f}")
            print(f"  PC1 Subspace Angle: {res['subspace_angle_pc1_deg']:.2f} deg")
            print(f"  Affine Mean R^2: {res['affine_predictability_mean_r2']:.4f}")
            print(f"  CCA Canonical r: {res['canonical_correlation']:.4f}")
            print(f"  Neighborhood Jaccard k=10 (8D): mean={res['knn_overlap_8d']['k_10']['mean']:.4f}, zeros={res['knn_overlap_8d']['k_10']['zero_overlap_pct']:.1f}%")
            print(f"  Anomaly Correlation r: {res['anomaly_correlation_r']:.4f} (p={res['anomaly_correlation_p']:.4f})")

    # ---------------------------------------------------------
    # Perspectival Invariants Across All Paradigms
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERSPECTIVAL INVARIANTS ACROSS PARADIGMS")
    print("=" * 80)

    # Invariant 1: Compression onto low-dimensional manifold
    mean_nli_pc1 = float(np.mean([v["pc1_evr"] for v in paradigm_summaries.values()]))
    mean_nli_pr = float(np.mean([v["participation_ratio"] for v in paradigm_summaries.values()]))
    print(f"Invariant 1 (Rank-1 Low-Dimensionality): Mean PC1 EVR = {mean_nli_pc1*100:.2f}%, Mean PR = {mean_nli_pr:.2f}")

    # Invariant 2: Extremal anchor ordering preservation
    z_p0 = StandardScaler().fit_transform(conceptual_paradigms["P0_Military_Security"])
    sc_p0 = PCA().fit_transform(z_p0)[:, 0]
    top5_idx = np.argsort(sc_p0)[-5:]
    bot5_idx = np.argsort(sc_p0)[:5]

    extremal_retention = {}
    for pname in ["P1_Paraphrase_Control", "P2_Humanitarian_Rights", "P3_Realpolitik_Statecraft"]:
        z_other = StandardScaler().fit_transform(conceptual_paradigms[pname])
        sc_other = PCA().fit_transform(z_other)[:, 0]
        if pearsonr(sc_p0, sc_other)[0] < 0:
            sc_other = -sc_other
        top5_retained = float(np.mean(sc_other[top5_idx] > np.median(sc_other)))
        bot5_retained = float(np.mean(sc_other[bot5_idx] < np.median(sc_other)))
        extremal_retention[pname] = {
            "top5_retained_pct": top5_retained * 100.0,
            "bot5_retained_pct": bot5_retained * 100.0,
            "mean_top5_score": float(np.mean(sc_other[top5_idx])),
            "mean_bot5_score": float(np.mean(sc_other[bot5_idx])),
        }
        print(f"  Extreme Retention in {pname}: Top-5={top5_retained*100:.1f}%, Bot-5={bot5_retained*100:.1f}%")

    output_payload = {
        "metadata": {
            "description": "Empirical evaluation of Kuhnian paradigms, incommensurability, neighborhood overlap, and anomaly relativity.",
            "formula": "G_observed = f(C, H, O, M)",
            "paradigm_formalization": "P = (H, O, M, L)",
        },
        "conceptual_paradigms": {
            "summaries": paradigm_summaries,
            "pairwise_analysis": conceptual_pairs,
        },
        "operator_paradigms": {
            "summaries": operator_summaries,
            "pairwise_analysis": operator_pairs,
        },
        "perspectival_invariants": {
            "mean_nli_pc1_evr": mean_nli_pc1,
            "mean_nli_participation_ratio": mean_nli_pr,
            "extremal_retention": extremal_retention,
        },
    }

    out_path = ROOT / "paper" / "kuhnian_paradigms_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Wrote complete Kuhnian paradigm results to {out_path}")


if __name__ == "__main__":
    main()
