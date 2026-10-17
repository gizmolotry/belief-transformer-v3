#!/usr/bin/env python3
"""Translation Curves, Attenuation Correction, and Complexity Analysis.

Fulfills Points 5 and 6 of the Kuhnian Clarification Battery:
1. Point 5: Attenuation-corrected translation metric:
   C(A, B) = Q_AB / sqrt(Q_AA * Q_BB)
   where Q_AA and Q_BB are the within-framework test-retest / split-half reliabilities
   measured across independent bank realizations and split-half probe sets.
   Evaluated across all 14 translation models (T0 to T13).
2. Point 6: Translation complexity curves:
   Q_train(c) vs Q_test(c) across model complexity c in {T0...T13}.
   Generalization gap Delta Q(c) = Q_train(c) - Q_test(c).
   Threshold complexity c*(tau) for tau in {0.10, 0.25, 0.50, 0.70} on train vs test.
   Identification of overfitting divergence regime.

Outputs:
- paper/kuhnian_translation_complexity_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Translation Model Ladder (T0 through T13)
# -----------------------------------------------------------------------------

class T0_SignFlip:
    """T0: Optimal sign flip per column."""
    def fit(self, X: np.ndarray, Y: np.ndarray):
        signs = []
        for d in range(Y.shape[1]):
            r = pearsonr(X[:, d], Y[:, d])[0]
            s = 1.0 if (not np.isnan(r) and r >= 0) else -1.0
            signs.append(s)
        self.signs_ = np.array(signs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X * self.signs_


class T1_OrthogonalProcrustes:
    """T1: Orthogonal Procrustes isometry."""
    def fit(self, X: np.ndarray, Y: np.ndarray):
        u, _, vt = np.linalg.svd(X.T @ Y)
        self.R_ = u @ vt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R_


class T2_SimilarityTransform:
    """T2: Isotropic scale + rotation + translation."""
    def fit(self, X: np.ndarray, Y: np.ndarray):
        mu_x = np.mean(X, axis=0)
        mu_y = np.mean(Y, axis=0)
        Xc = X - mu_x
        Yc = Y - mu_y
        u, _, vt = np.linalg.svd(Xc.T @ Yc)
        R = u @ vt
        scale = np.trace(Xc.T @ Yc @ R.T) / (np.sum(Xc**2) + 1e-12)
        self.mu_x_ = mu_x
        self.mu_y_ = mu_y
        self.scale_ = scale
        self.R_ = R
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scale_ * ((X - self.mu_x_) @ self.R_) + self.mu_y_


class T3_DiagonalScaleRotation:
    """T3: Diagonal anisotropic scaling + rotation + translation."""
    def fit(self, X: np.ndarray, Y: np.ndarray):
        mu_x = np.mean(X, axis=0)
        mu_y = np.mean(Y, axis=0)
        Xc = X - mu_x
        Yc = Y - mu_y
        std_x = np.std(Xc, axis=0) + 1e-5
        std_y = np.std(Yc, axis=0) + 1e-5
        diag_s = std_y / std_x
        Xs = Xc * diag_s
        u, _, vt = np.linalg.svd(Xs.T @ Yc)
        self.mu_x_ = mu_x
        self.mu_y_ = mu_y
        self.diag_s_ = diag_s
        self.R_ = u @ vt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_x_) * self.diag_s_) @ self.R_ + self.mu_y_


class T11_OptimalTransportBarycentric:
    """T11: Optimal transport barycentric mapping via regularized Sinkhorn."""
    def __init__(self, reg: float = 0.05):
        self.reg = reg

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train_ = X.copy()
        self.Y_train_ = Y.copy()
        C = cdist(X, Y, metric="sqeuclidean")
        C /= np.median(C) + 1e-8
        n, m = len(X), len(Y)
        a = np.ones(n) / n
        b = np.ones(m) / m
        K = np.exp(-C / self.reg)
        u = np.ones(n)
        for _ in range(50):
            v = b / (K.T @ u + 1e-12)
            u = a / (K @ v + 1e-12)
        self.P_ = np.diag(u) @ K @ np.diag(v)
        med_dist = np.median(cdist(X, X))
        self.gamma_ = 1.0 / (2.0 * (med_dist ** 2) + 1e-8)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        gamma = self.gamma_
        K_test = np.exp(-gamma * cdist(X, self.X_train_, metric="sqeuclidean"))
        K_test /= np.sum(K_test, axis=1, keepdims=True) + 1e-12
        row_norm_P = self.P_ / (np.sum(self.P_, axis=1, keepdims=True) + 1e-12)
        mapped_train = row_norm_P @ self.Y_train_
        return K_test @ mapped_train


class T12_PiecewiseLinearCharts:
    """T12: Local piecewise linear charts (k-means Voronoi cells + local affine maps)."""
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, Y: np.ndarray):
        from sklearn.cluster import KMeans
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(X)
        self.models_ = []
        for c in range(self.n_clusters):
            mask = self.kmeans_.labels_ == c
            if np.sum(mask) >= X.shape[1] + 1:
                m = Ridge(alpha=1.0).fit(X[mask], Y[mask])
            else:
                m = Ridge(alpha=1.0).fit(X, Y)
            self.models_.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        clusters = self.kmeans_.predict(X)
        preds = np.zeros((len(X), self.models_[0].predict(X[:1]).shape[1]))
        for i, c in enumerate(clusters):
            preds[i] = self.models_[c].predict(X[i : i + 1])[0]
        return preds


LADDER_MODELS = [
    ("T0_SignFlip", 0, "Sign Flip per Dim", 8),
    ("T1_OrthogonalProcrustes", 1, "Orthogonal Rotation (Isometry)", 28),
    ("T2_SimilarityTransform", 2, "Similarity (Scale + Rot + Shift)", 37),
    ("T3_DiagonalScaleRotation", 3, "Anisotropic Scale + Rot + Shift", 44),
    ("T4_AffineOLS", 4, "Affine Linear OLS", 72),
    ("T5_RidgeCV", 5, "L2 Regularized Affine RidgeCV", 72),
    ("T6_PLS2", 6, "Partial Least Squares (PLS-4)", 36),
    ("T7_CCA", 7, "Canonical Correlation Analysis (CCA-4)", 36),
    ("T8_Poly2Ridge", 8, "Polynomial Degree-2 Ridge", 360),
    ("T9_KernelRidgeRBF", 9, "Kernel Ridge Regression (RBF)", 120),
    ("T10_ShallowMLP", 10, "Shallow MLP (32x16 ReLU)", 832),
    ("T11_OptimalTransport", 11, "Optimal Transport Barycentric", 120),
    ("T12_PiecewiseLinear", 12, "Piecewise Linear Charts (k=3)", 216),
    ("T13_KNNRegressor", 13, "k-Nearest Neighbors (k=5)", 120),
]


def make_model(name: str, dim: int):
    if name == "T0_SignFlip":
        return T0_SignFlip()
    elif name == "T1_OrthogonalProcrustes":
        return T1_OrthogonalProcrustes()
    elif name == "T2_SimilarityTransform":
        return T2_SimilarityTransform()
    elif name == "T3_DiagonalScaleRotation":
        return T3_DiagonalScaleRotation()
    elif name == "T4_AffineOLS":
        return LinearRegression()
    elif name == "T5_RidgeCV":
        return RidgeCV(alphas=np.logspace(-4, 4, 9))
    elif name == "T6_PLS2":
        return PLSRegression(n_components=min(4, dim))
    elif name == "T7_CCA":
        return CCA(n_components=min(4, dim), scale=False, max_iter=2000)
    elif name == "T8_Poly2Ridge":
        return Pipeline([("poly", PolynomialFeatures(degree=2)), ("ridge", Ridge(alpha=10.0))])
    elif name == "T9_KernelRidgeRBF":
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
    elif name == "T10_ShallowMLP":
        return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=42)
    elif name == "T11_OptimalTransport":
        return T11_OptimalTransportBarycentric()
    elif name == "T12_PiecewiseLinear":
        return T12_PiecewiseLinearCharts(n_clusters=3)
    elif name == "T13_KNNRegressor":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate_model_train_and_test(
    model_name: str,
    X_A: np.ndarray,
    X_B: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a translation model recording BOTH train and test performance per fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dim = min(X_A.shape[1], X_B.shape[1])

    train_r2_list = []
    test_r2_list = []
    train_r_list = []
    test_r_list = []
    train_rmse_list = []
    test_rmse_list = []
    test_j10_list = []

    for train_idx, test_idx in kf.split(X_A):
        scaler_A = StandardScaler().fit(X_A[train_idx])
        scaler_B = StandardScaler().fit(X_B[train_idx])

        zA_tr = scaler_A.transform(X_A[train_idx])
        zB_tr = scaler_B.transform(X_B[train_idx])
        zA_te = scaler_A.transform(X_A[test_idx])
        zB_te = scaler_B.transform(X_B[test_idx])

        model = make_model(model_name, dim)
        model.fit(zA_tr, zB_tr)

        if model_name == "T7_CCA":
            cA_tr, _ = model.transform(zA_tr, zB_tr)
            cA_te = model.transform(zA_te)
            W = np.linalg.pinv(cA_tr) @ zB_tr
            pred_tr = cA_tr @ W
            pred_te = cA_te @ W
        else:
            pred_tr = model.predict(zA_tr)
            pred_te = model.predict(zA_te)

        # Train metrics
        tr_r2 = float(r2_score(zB_tr, pred_tr, multioutput="uniform_average"))
        r_tr, _ = pearsonr(pred_tr.ravel(), zB_tr.ravel())
        rmse_tr = float(np.sqrt(np.mean((zB_tr - pred_tr) ** 2)))
        train_r2_list.append(tr_r2)
        train_r_list.append(r_tr if not np.isnan(r_tr) else 0.0)
        train_rmse_list.append(rmse_tr)

        # Test metrics
        te_r2 = float(r2_score(zB_te, pred_te, multioutput="uniform_average"))
        r_te, _ = pearsonr(pred_te.ravel(), zB_te.ravel())
        rmse_te = float(np.sqrt(np.mean((zB_te - pred_te) ** 2)))
        test_r2_list.append(te_r2)
        test_r_list.append(r_te if not np.isnan(r_te) else 0.0)
        test_rmse_list.append(rmse_te)

        # Held-out Jaccard @ k=10
        dist_true = cdist(zB_te, zB_te)
        dist_pred = cdist(pred_te, pred_te)
        np.fill_diagonal(dist_true, np.inf)
        np.fill_diagonal(dist_pred, np.inf)
        k_eff = min(10, len(zB_te) - 1)
        j_fold = []
        for i in range(len(zB_te)):
            nn_t = set(np.argsort(dist_true[i])[:k_eff])
            nn_p = set(np.argsort(dist_pred[i])[:k_eff])
            inter = len(nn_t & nn_p)
            union = len(nn_t | nn_p)
            j_fold.append(inter / union if union > 0 else 0.0)
        test_j10_list.append(np.mean(j_fold))

    return {
        "train_r2_mean": float(np.mean(train_r2_list)),
        "train_r2_std": float(np.std(train_r2_list)),
        "test_r2_mean": float(np.mean(test_r2_list)),
        "test_r2_std": float(np.std(test_r2_list)),
        "train_r_mean": float(np.mean(train_r_list)),
        "test_r_mean": float(np.mean(test_r_list)),
        "train_rmse_mean": float(np.mean(train_rmse_list)),
        "test_rmse_mean": float(np.mean(test_rmse_list)),
        "test_j10_mean": float(np.mean(test_j10_list)),
        "generalization_gap_r2": float(np.mean(train_r2_list) - np.mean(test_r2_list)),
        "generalization_gap_r": float(np.mean(train_r_list) - np.mean(test_r_list)),
        "fold_test_r2": [float(x) for x in test_r2_list],
    }


def compute_attenuation_correction(
    q_ab: float, q_aa: float, q_bb: float
) -> Tuple[float, str]:
    """Computes C(A, B) = q_ab / sqrt(q_aa * q_bb)."""
    denom_sq = q_aa * q_bb
    if denom_sq <= 0:
        return 0.0, "zero_or_negative_reliability"
    denom = np.sqrt(denom_sq)
    if denom < 1e-8:
        return 0.0, "near_zero_denominator"
    val = q_ab / denom
    return float(val), "valid"


def run_translation_and_complexity_battery():
    print("=" * 80)
    print("RUNNING KUHNIAN TRANSLATION CURVES & ATTENUATION CORRECTION (POINTS 5, 6)")
    print("=" * 80)

    # 1. Load Caches
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
        return (sa - sb).astype(float)  # [120, 8]

    pol_q0 = extract_pol(cache["q0_original_geopolitical"])
    pol_q1 = extract_pol(cache["q1_paraphrase_control"])
    pol_q2 = extract_pol(cache["q2_humanitarian_rights"])
    pol_q3 = extract_pol(cache["q3_realpolitik_statecraft"])
    pol_ihl = extract_pol(cache["disciplinary_paradigms:ihl_humanitarian_law"])
    pol_sec = extract_pol(cache["disciplinary_paradigms:strategic_security"])
    pol_fa = fa_base

    # Cooking control for absurd translation
    cook_key = "absurd_structurally_matched_paradigms:culinary_arts"
    pol_cook = extract_pol(cache[cook_key])

    pol_q0_r1 = extract_pol(g_cache["geopolitical_conflict:realization_1"])
    pol_q0_r2 = extract_pol(g_cache["geopolitical_conflict:realization_2"])
    pol_q2_r1 = extract_pol(g_cache["humanitarian_civilian_welfare:realization_1"])
    pol_q2_r2 = extract_pol(g_cache["humanitarian_civilian_welfare:realization_2"])
    pol_q3_r1 = extract_pol(g_cache["strategic_security_deterrence:realization_1"])
    pol_q3_r2 = extract_pol(g_cache["strategic_security_deterrence:realization_2"])

    # -------------------------------------------------------------------------
    # STEP 1: COMPUTE WITHIN-FRAMEWORK RELIABILITY Q_AA, Q_BB
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Evaluating within-framework self-translation reliability (Q_AA, Q_BB)...")
    reliability_pairs = {
        "geopolitical_test_retest": (pol_q0_r1, pol_q0_r2),
        "humanitarian_test_retest": (pol_q2_r1, pol_q2_r2),
        "realpolitik_test_retest": (pol_q3_r1, pol_q3_r2),
        "geopolitical_splithalf": (pol_q0[:, ::2], pol_q0[:, 1::2]),
        "frameaxis_splithalf": (pol_fa[:, ::2], pol_fa[:, 1::2]),
    }

    reliability_results = {}
    for rel_name, (xA, xB) in reliability_pairs.items():
        print(f"  Evaluating self-reliability: {rel_name}...")
        rel_models = {}
        for m_name, c_idx, desc, params in LADDER_MODELS:
            res = evaluate_model_train_and_test(m_name, xA, xB, n_splits=5)
            rel_models[m_name] = res
        reliability_results[rel_name] = rel_models

    # -------------------------------------------------------------------------
    # STEP 2: EVALUATE TRANSITION LADDER (TRAIN + TEST) FOR ALL TRANSITIONS
    # -------------------------------------------------------------------------
    transitions = {
        "paraphrase_q0_to_q1": {
            "src": pol_q0, "tgt": pol_q1,
            "rel_src_key": "geopolitical_test_retest",
            "rel_tgt_key": "geopolitical_test_retest",
            "type": "paraphrase_control",
        },
        "conceptual_q0_to_q2_humanitarian": {
            "src": pol_q0, "tgt": pol_q2,
            "rel_src_key": "geopolitical_test_retest",
            "rel_tgt_key": "humanitarian_test_retest",
            "type": "conceptual_shift",
        },
        "conceptual_q0_to_q3_realpolitik": {
            "src": pol_q0, "tgt": pol_q3,
            "rel_src_key": "geopolitical_test_retest",
            "rel_tgt_key": "realpolitik_test_retest",
            "type": "conceptual_shift",
        },
        "disciplinary_q0_to_ihl": {
            "src": pol_q0, "tgt": pol_ihl,
            "rel_src_key": "geopolitical_test_retest",
            "rel_tgt_key": "humanitarian_test_retest",
            "type": "disciplinary_shift",
        },
        "operator_nli_to_frameaxis": {
            "src": pol_q0, "tgt": pol_fa,
            "rel_src_key": "geopolitical_splithalf",
            "rel_tgt_key": "frameaxis_splithalf",
            "type": "operator_shift",
        },
        "operator_frameaxis_to_nli": {
            "src": pol_fa, "tgt": pol_q0,
            "rel_src_key": "frameaxis_splithalf",
            "rel_tgt_key": "geopolitical_splithalf",
            "type": "operator_shift",
        },
        "absurd_control_q0_to_cooking": {
            "src": pol_q0, "tgt": pol_cook,
            "rel_src_key": "geopolitical_test_retest",
            "rel_tgt_key": "geopolitical_test_retest",
            "type": "absurd_control",
        },
    }

    transition_results = {}
    for trans_name, info in transitions.items():
        print(f"\n[STEP 2] Evaluating transition ladder: {trans_name} ({info['type']})...")
        xA = info["src"]
        xB = info["tgt"]
        rel_A_key = info["rel_src_key"]
        rel_B_key = info["rel_tgt_key"]

        models_res = {}
        for m_name, c_idx, desc, n_params in LADDER_MODELS:
            res = evaluate_model_train_and_test(m_name, xA, xB, n_splits=5)

            q_aa_r2 = reliability_results[rel_A_key][m_name]["test_r2_mean"]
            q_bb_r2 = reliability_results[rel_B_key][m_name]["test_r2_mean"]
            q_ab_r2 = res["test_r2_mean"]

            q_aa_r = reliability_results[rel_A_key][m_name]["test_r_mean"]
            q_bb_r = reliability_results[rel_B_key][m_name]["test_r_mean"]
            q_ab_r = res["test_r_mean"]

            c_r2, status_r2 = compute_attenuation_correction(q_ab_r2, q_aa_r2, q_bb_r2)
            c_r, status_r = compute_attenuation_correction(q_ab_r, q_aa_r, q_bb_r)

            models_res[m_name] = {
                "complexity_index": c_idx,
                "description": desc,
                "parameter_count": n_params,
                "train_r2": res["train_r2_mean"],
                "test_r2": res["test_r2_mean"],
                "test_r2_std": res["test_r2_std"],
                "train_r": res["train_r_mean"],
                "test_r": res["test_r_mean"],
                "train_rmse": res["train_rmse_mean"],
                "test_rmse": res["test_rmse_mean"],
                "test_j10": res["test_j10_mean"],
                "generalization_gap_r2": res["generalization_gap_r2"],
                "generalization_gap_r": res["generalization_gap_r"],
                "reliability_q_aa_r2": q_aa_r2,
                "reliability_q_bb_r2": q_bb_r2,
                "attenuation_corrected_c_r2": c_r2,
                "attenuation_status_r2": status_r2,
                "reliability_q_aa_r": q_aa_r,
                "reliability_q_bb_r": q_bb_r,
                "attenuation_corrected_c_r": c_r,
                "attenuation_status_r": status_r,
            }

        tau_thresholds = [0.10, 0.25, 0.50, 0.70]
        c_star_train_r2 = {}
        c_star_test_r2 = {}
        c_star_train_r = {}
        c_star_test_r = {}

        for tau in tau_thresholds:
            c_star_train_r2[str(tau)] = next(
                (m["complexity_index"] for m in models_res.values() if m["train_r2"] >= tau), None
            )
            c_star_test_r2[str(tau)] = next(
                (m["complexity_index"] for m in models_res.values() if m["test_r2"] >= tau), None
            )
            c_star_train_r[str(tau)] = next(
                (m["complexity_index"] for m in models_res.values() if m["train_r"] >= tau), None
            )
            c_star_test_r[str(tau)] = next(
                (m["complexity_index"] for m in models_res.values() if m["test_r"] >= tau), None
            )

        overfitting_divergence = next(
            (m["complexity_index"] for m in models_res.values() if m["train_r2"] >= 0.5 and m["test_r2"] <= 0.1),
            None,
        )

        transition_results[trans_name] = {
            "type": info["type"],
            "models": models_res,
            "complexity_analysis": {
                "c_star_train_r2": c_star_train_r2,
                "c_star_test_r2": c_star_test_r2,
                "c_star_train_r": c_star_train_r,
                "c_star_test_r": c_star_test_r,
                "overfitting_divergence_index": overfitting_divergence,
            },
        }

    output_path = ROOT / "paper" / "kuhnian_translation_complexity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reliability_benchmarks": reliability_results,
                "transitions": transition_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved translation and complexity results to {output_path}")


if __name__ == "__main__":
    run_translation_and_complexity_battery()
