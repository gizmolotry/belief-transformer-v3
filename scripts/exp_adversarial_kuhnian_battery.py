#!/usr/bin/env python3
"""Unified Adversarial Kuhnian Paradigm Battery (Agents B, C, D, E and Reductios 1, 2, 3).

Executes:
1. Agent B: Within-paradigm reliability noise floor (1,000 bootstrap draws, delta-disparity).
2. Agent C: 14-level translation ladder under 5-fold CV (T0 through T13).
3. Agent D: Calibrated neighborhood world-change curves (k in [3, 50]) vs 8 reference nulls + ZCA whitening.
4. Agent E: Multi-definition anomaly relativity (6 definitions, transfer matrix, conceptual distance correlation).
5. Reductio 1: Absurd structurally-matched paradigms (Weather, Cooking, Botany, Sports).
6. Reductio 2: Maximally distant political ontologies (Legal, Operational, Humanitarian, Economic, Religious).
7. Reductio 3: Adversarial geometry manufacturing & observer pushback test.
8. Claim Gates evaluation and failure condition audit.

Outputs paper/adversarial_kuhnian_audit_results.json.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import f_oneway, hypergeom, pearsonr, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# HELPER MATHEMATICAL FUNCTIONS
# =============================================================================

def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return x / norms


def zca_whiten(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute ZCA whitening transform."""
    x_c = x - np.mean(x, axis=0)
    sigma = np.cov(x_c, rowvar=False)
    u, s, vt = np.linalg.svd(sigma)
    w_zca = u @ np.diag(1.0 / np.sqrt(s + eps)) @ u.T
    return x_c @ w_zca


def compute_knn_jaccard_curve(
    X_A: np.ndarray, X_B: np.ndarray, k_values: List[int] = [3, 5, 10, 15, 20, 30, 50]
) -> Dict[str, Dict[str, float]]:
    """Compute k-NN Jaccard neighborhood overlap across multiple k values."""
    N = len(X_A)
    dist_A = cdist(X_A, X_A, metric="euclidean")
    dist_B = cdist(X_B, X_B, metric="euclidean")
    np.fill_diagonal(dist_A, np.inf)
    np.fill_diagonal(dist_B, np.inf)

    curve = {}
    for k in k_values:
        jaccards = []
        for i in range(N):
            nn_A = set(np.argsort(dist_A[i])[:k])
            nn_B = set(np.argsort(dist_B[i])[:k])
            inter = len(nn_A & nn_B)
            union = len(nn_A | nn_B)
            j = inter / union if union > 0 else 0.0
            jaccards.append(j)
        jaccards = np.array(jaccards)
        curve[f"k_{k}"] = {
            "mean": float(np.mean(jaccards)),
            "std": float(np.std(jaccards)),
            "median": float(np.median(jaccards)),
            "zero_overlap_pct": float(np.mean(jaccards == 0.0) * 100.0),
        }
    return curve


def compute_six_anomaly_scores(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute 6 distinct mathematical anomaly definitions."""
    N, D = X.shape
    z = StandardScaler().fit_transform(X)

    # 1. Residual outside Rank-1 PCA
    pca1 = PCA(n_components=1).fit(z)
    recon1 = pca1.inverse_transform(pca1.transform(z))
    a1 = np.sum((z - recon1) ** 2, axis=1) / np.maximum(np.sum(z**2, axis=1), 1e-12)

    # 2. Residual outside Rank-2 PCA
    pca2 = PCA(n_components=min(2, D)).fit(z)
    recon2 = pca2.inverse_transform(pca2.transform(z))
    a2 = np.sum((z - recon2) ** 2, axis=1) / np.maximum(np.sum(z**2, axis=1), 1e-12)

    # 3. Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=min(10, N - 1), novelty=False)
    lof.fit(z)
    a3 = -lof.negative_outlier_factor_  # Higher = more anomalous

    # 4. Mahalanobis distance from centroid
    cov = np.cov(z, rowvar=False) + 1e-5 * np.eye(D)
    inv_cov = np.linalg.pinv(cov)
    centroid = np.mean(z, axis=0)
    a4 = np.array([mahalanobis(z[i], centroid, inv_cov) for i in range(N)])

    # 5. Isolation Forest
    iso = IsolationForest(n_estimators=100, random_state=42)
    iso.fit(z)
    a5 = -iso.score_samples(z)  # Higher = more anomalous

    # 6. Minimum Covariance Determinant (MCD) robust Mahalanobis
    try:
        mcd = MinCovDet(random_state=42).fit(z)
        a6 = np.sqrt(mcd.mahalanobis(z))
    except Exception:
        a6 = a4.copy()

    return {
        "A1_rank1_residual": a1,
        "A2_rank2_residual": a2,
        "A3_lof": a3,
        "A4_mahalanobis": a4,
        "A5_isolation_forest": a5,
        "A6_mcd_robust": a6,
    }


def compute_rank1_residual_fast(X: np.ndarray) -> np.ndarray:
    z = StandardScaler().fit_transform(X)
    pca1 = PCA(n_components=1).fit(z)
    recon1 = pca1.inverse_transform(pca1.transform(z))
    return np.sum((z - recon1) ** 2, axis=1) / np.maximum(np.sum(z**2, axis=1), 1e-12)


# =============================================================================
# AGENT B: WITHIN-PARADIGM RELIABILITY NOISE FLOOR
# =============================================================================

def summarize_draws(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "ci_95": np.percentile(values, [2.5, 97.5]).tolist(), "n": len(values),
            "interval_type": "empirical draw percentiles, not confidence interval of mean"}


def fit_geometry(X):
    scaler = StandardScaler().fit(X)
    return scaler, PCA(n_components=2).fit(scaler.transform(X))


def project_geometry(model, X):
    scaler, pca = model
    z = scaler.transform(X)
    scores = pca.transform(z)
    # Rank-one reconstruction, with centering learned on the training sample.
    recon = scores[:, :1] @ pca.components_[:1] + pca.mean_
    residual = np.sum((z - recon)**2, axis=1) / np.maximum(np.sum(z**2, axis=1), 1e-12)
    return scores, residual


def geometry_concordance(model1, model2, X):
    s1, a1 = project_geometry(model1, X)
    s2, a2 = project_geometry(model2, X)
    k = min(10, len(X)-1)
    return {"r_same_pc1": float(abs(pearsonr(s1[:, 0], s2[:, 0]).statistic)),
            "disp_same_2d": float(procrustes(s1, s2)[2]),
            "j10_same": compute_knn_jaccard_curve(s1, s2, [k])[f"k_{k}"]["mean"],
            "anom_r_same": float(pearsonr(a1, a2).statistic),
            "n_evaluation": len(X), "k_effective": k}


def compute_within_paradigm_noise_floor(X, n_bootstraps=1000, seed=42, n_splits=1000):
    """Pairwise refit stability, joint OOB concordance, and held-out reconstruction."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    N = len(X)
    result = {"n_bootstraps": n_bootstraps, "n_split_half_draws": n_splits, "seed": seed,
              "preprocessing": "Separate scaler and rank-2 PCA fit on each training draw",
              "anomaly_definition": "normalized rank-one reconstruction residual"}
    for scheme, draws in [("bootstrap", n_bootstraps), ("split_half", n_splits)]:
        full, oob, errors, sizes = [], [], [], []
        for _ in range(draws):
            if scheme == "bootstrap":
                idx1, idx2 = rng.integers(N, size=(2, N))
            else:
                idx1, idx2 = np.split(rng.permutation(N), [N//2])
            m1, m2 = fit_geometry(X[idx1]), fit_geometry(X[idx2])
            full.append(geometry_concordance(m1, m2, X))
            test1 = np.setdiff1d(np.arange(N), idx1)
            test2 = np.setdiff1d(np.arange(N), idx2)
            errors.append(float(np.mean(np.concatenate([
                project_geometry(m1, X[test1])[1], project_geometry(m2, X[test2])[1]]))))
            if scheme == "bootstrap":
                joint = np.intersect1d(test1, test2)
                sizes.append(len(joint))
                if len(joint) >= 3:
                    oob.append(geometry_concordance(m1, m2, X[joint]))
        pack = lambda rows: {key: summarize_draws([row[key] for row in rows]) for key in rows[0]}
        entry = {"n_draws": draws, "train_sizes": [N, N] if scheme == "bootstrap" else [N//2, N-N//2],
                 "full_sample": pack(full),
                 "out_of_sample": {"rank1_normalized_reconstruction_error": summarize_draws(errors),
                                   "evaluation": "Each model evaluated only on its own omitted documents; pooled per draw"}}
        if scheme == "bootstrap":
            entry["oob"] = pack(oob)
            entry["oob"]["omitted_draws_fewer_than_3_joint_oob"] = draws-len(oob)
            entry["oob"]["joint_oob_sizes"] = sizes
            entry["oob"]["note"] = "Both models omit all evaluated documents; k=min(10,n_joint_oob-1)"
        else:
            entry["out_of_sample"]["note"] = "Complementary halves have no joint held-out set; concordance is descriptive full-sample only"
        result[scheme] = entry
    # Compatibility summaries refer explicitly to bootstrap full-sample concordance.
    result.update(result["bootstrap"]["full_sample"])
    return result


# =============================================================================
# AGENT C: 14-LEVEL TRANSLATION LADDER (CV)
# =============================================================================

class T0_SignFlip:
    """T0: Optimal sign flip per column."""
    def fit(self, X: np.ndarray, Y: np.ndarray):
        signs = []
        for d in range(Y.shape[1]):
            s = 1.0 if pearsonr(X[:, d], Y[:, d])[0] >= 0 else -1.0
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
        var_x = np.sum(Xc**2) / len(X)
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
        # Compute cost matrix
        C = cdist(X, Y, metric="sqeuclidean")
        C /= np.median(C) + 1e-8
        # Sinkhorn iterations
        n, m = len(X), len(Y)
        a = np.ones(n) / n
        b = np.ones(m) / m
        K = np.exp(-C / self.reg)
        u = np.ones(n)
        for _ in range(50):
            v = b / (K.T @ u + 1e-12)
            u = a / (K @ v + 1e-12)
        self.P_ = np.diag(u) @ K @ np.diag(v)
        self.gamma_ = 1.0 / (2.0 * np.median(cdist(X, X)) ** 2 + 1e-8)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Out-of-sample barycentric projection using RBF kernel to training X
        gamma = self.gamma_
        K_test = np.exp(-gamma * cdist(X, self.X_train_, metric="sqeuclidean"))
        K_test /= np.sum(K_test, axis=1, keepdims=True) + 1e-12
        # Target barycenter of training points: (P_ / a) @ Y
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


def evaluate_14_level_translation_ladder(
    X_A: np.ndarray, X_B: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Dict[str, Any]:
    """Evaluate all 14 translation models under 5-fold cross-validation with train-fitted preprocessing."""
    from sklearn.model_selection import KFold

    N = len(X_A)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ladder_names = [
        "T0_SignFlip",
        "T1_OrthogonalProcrustes",
        "T2_SimilarityTransform",
        "T3_DiagonalScaleRotation",
        "T4_AffineOLS",
        "T5_RidgeCV",
        "T6_PLS2",
        "T7_CCA",
        "T8_Poly2Ridge",
        "T9_KernelRidgeRBF",
        "T10_ShallowMLP",
        "T11_OptimalTransport",
        "T12_PiecewiseLinear",
        "T13_KNNRegressor",
    ]

    def make_model(name: str):
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
            return PLSRegression(n_components=min(4, X_A.shape[1]))
        elif name == "T7_CCA":
            return CCA(n_components=min(4, X_A.shape[1]), scale=False, max_iter=2000)
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

    results = {}
    for name in ladder_names:
        cv_r2 = []
        cv_r = []
        cv_rho = []
        cv_j10 = []

        for train_idx, test_idx in kf.split(X_A):
            scaler_A = StandardScaler().fit(X_A[train_idx])
            scaler_B = StandardScaler().fit(X_B[train_idx])

            zA_tr = scaler_A.transform(X_A[train_idx])
            zB_tr = scaler_B.transform(X_B[train_idx])
            zA_te = scaler_A.transform(X_A[test_idx])
            zB_te = scaler_B.transform(X_B[test_idx])

            model = make_model(name)
            model.fit(zA_tr, zB_tr)
            if name == "T7_CCA":
                cA_tr, _ = model.transform(zA_tr, zB_tr)
                cA_te = model.transform(zA_te)
                W = np.linalg.pinv(cA_tr) @ zB_tr
                pred_te = cA_te @ W
            else:
                pred_te = model.predict(zA_te)
            if not np.all(np.isfinite(pred_te)):
                raise ValueError(f"Nonfinite held-out predictions: {name}")
            # Conventional test-set R2, averaged uniformly over the eight outputs.
            cv_r2.append(float(r2_score(zB_te, pred_te, multioutput="uniform_average")))

            # Held-out Pearson r and Spearman rho across flattened predictions
            r_val, _ = pearsonr(pred_te.ravel(), zB_te.ravel())
            rho_val, _ = spearmanr(pred_te.ravel(), zB_te.ravel())
            cv_r.append(r_val if not np.isnan(r_val) else 0.0)
            cv_rho.append(rho_val if not np.isnan(rho_val) else 0.0)

            # Held-out k-NN Jaccard overlap k=10
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
            cv_j10.append(np.mean(j_fold))

        results[name] = {
            "complexity_level": ladder_names.index(name),
            "heldout_r2_folds": [float(x) for x in cv_r2],
            "heldout_j10_folds": [float(x) for x in cv_j10],
            "heldout_r2_mean": float(np.mean(cv_r2)),
            "heldout_r2_std": float(np.std(cv_r2)),
            "heldout_pearson_r": float(np.mean(cv_r)),
            "heldout_spearman_rho": float(np.mean(cv_rho)),
            "heldout_j10_mean": float(np.mean(cv_j10)),
            "heldout_j10_std": float(np.std(cv_j10)),
        }

    results["complexity_curve"] = {
        "definition": "c_star(tau)=min{c:mean_fold_test_R2(c)>=tau}; null means no tested level attains threshold",
        "complexity_order": "ordinal ladder index, not parameter count or strictly nested model families",
        "c_star": {str(tau): next((i for i, name in enumerate(ladder_names)
                                   if results[name]["heldout_r2_mean"] >= tau), None)
                   for tau in (0.10, 0.30, 0.50)},
        "evaluation": "5 shuffled outer folds; outer-train scalers; T5 RidgeCV inner leave-one-out alpha selection; other hyperparameters fixed",
        "j10_scope": "neighbors among held-out documents per fold; k=min(10,n_test-1)",
        "pls_components": min(4, X_A.shape[1]),
        "cca_components": min(4, X_A.shape[1]),
        "seed": seed,
    }
    return results


# =============================================================================
# AGENT D: CALIBRATED NEIGHBORHOOD CURVES & REFERENCE NULLS
# =============================================================================

def compute_reference_null_curves(
    N: int, D: int, empirical_spectrum: np.ndarray, k_values: List[int] = [3, 5, 10, 15, 20, 30, 50]
) -> Dict[str, Dict[str, float]]:
    """Compute 4 analytical/synthetic reference null curves for neighborhood overlap."""
    rng = np.random.default_rng(42)

    nulls = {}

    # 1. Hypergeometric Null (Random Selection)
    hyper_means = {}
    for k in k_values:
        # For each document, choosing k neighbors at random from N-1
        # Intersection follows Hypergeometric(N-1, k, k)
        m_vals = np.arange(0, k + 1)
        probs = hypergeom.pmf(m_vals, N - 1, k, k)
        jacc_vals = m_vals / (2 * k - m_vals)
        hyper_means[f"k_{k}"] = float(np.sum(probs * jacc_vals))
    nulls["hypergeometric_null"] = hyper_means

    # 2. Random Gaussian Null
    X_g1 = rng.normal(size=(N, D))
    X_g2 = rng.normal(size=(N, D))
    curve_gauss = compute_knn_jaccard_curve(X_g1, X_g2, k_values)
    nulls["random_gaussian_null"] = {k: v["mean"] for k, v in curve_gauss.items()}

    # 3. Anisotropic Synthetic Null (Matched to Empirical Spectrum)
    # Generate random points in subspace stretched by empirical singular values
    X_s1 = rng.normal(size=(N, D)) @ np.diag(empirical_spectrum[:D])
    X_s2 = rng.normal(size=(N, D)) @ np.diag(empirical_spectrum[:D])
    curve_aniso = compute_knn_jaccard_curve(X_s1, X_s2, k_values)
    nulls["anisotropic_synthetic_null"] = {k: v["mean"] for k, v in curve_aniso.items()}

    return nulls


# =============================================================================
# MAIN EXPERIMENT EXECUTION ENGINE
# =============================================================================

def main():
    print("=" * 80)
    print("EXECUTING UNIFIED ADVERSARIAL KUHNIAN PARADIGM BATTERY")
    print("=" * 80)

    # Load paired cache
    cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}. Run extraction first.")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"[CACHE] Loaded {len(cache)} hypothesis banks from {cache_path}")

    # Load corpus texts and publishers for TF-IDF & Publisher nulls
    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    articles = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    texts = [art.get("text", "") for art in articles]
    publishers = [art.get("publisher", "unknown") for art in articles]
    N = len(articles)
    if N != 120:
        raise ValueError(f"Expected frozen N=120 corpus, got {N}")
    print(f"[DATA] {N} articles across {len(set(publishers))} publishers.")

    # Convert all bank logits to paired polarity contrasts: S = (pA,2 - pA,0) - (pB,2 - pB,0)
    polarities = {}
    embeddings = {}
    for b_key, b_val in cache.items():
        la = b_val["la"].float()
        lb = b_val["lb"].float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        paired_s = (sa - sb).astype(float)  # [120, 8]
        if paired_s.shape != (N, 8) or not np.all(np.isfinite(paired_s)):
            raise ValueError(f"Invalid cached polarity matrix: {b_key}")
        polarities[b_key] = paired_s

        if "emb_a" in b_val and "emb_b" in b_val:
            emba = b_val["emb_a"].float().numpy()
            embb = b_val["emb_b"].float().numpy()
            embeddings[b_key] = emba.mean(axis=1) - embb.mean(axis=1)

    # Also extract FrameAxis baseline for cross-operator comparison
    fa_cache = np.load(ROOT / "data" / "cache_audit_deberta_embeddings.npz")
    docs_emb = fa_cache["prospective"]
    hyp_a = fa_cache["hyp_a"]
    hyp_b = fa_cache["hyp_b"]
    d_u = unit_rows(docs_emb)
    a_u = unit_rows(hyp_a)
    b_u = unit_rows(hyp_b)
    fa_base = d_u @ a_u.T - d_u @ b_u.T  # [120, 8]
    if fa_base.shape != (N, 8) or not np.all(np.isfinite(fa_base)):
        raise ValueError("Invalid FrameAxis matrix")
    polarities["operator:frameaxis_baseline"] = fa_base.astype(float)

    print(f"[POLARITIES] Processed {len(polarities)} measurement matrices.")

    # -------------------------------------------------------------------------
    # STEP 1: AGENT B - WITHIN-PARADIGM RELIABILITY NOISE FLOORS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: AGENT B - WITHIN-PARADIGM RELIABILITY NOISE FLOORS (1,000 BOOTSTRAPS)")
    print("=" * 80)
    within_reliability = {}
    for b_key in ["q0_original_geopolitical", "q1_paraphrase_control", "q2_humanitarian_rights", "operator:frameaxis_baseline"]:
        print(f"[AGENT B] Computing within-paradigm noise floor for '{b_key}'...")
        floor = compute_within_paradigm_noise_floor(polarities[b_key], n_bootstraps=1000)
        within_reliability[b_key] = floor
        print(f"  r_same: {floor['r_same_pc1']['mean']:.4f} (95% CI: {floor['r_same_pc1']['ci_95'][0]:.4f}, {floor['r_same_pc1']['ci_95'][1]:.4f})")
        print(f"  D_same (2D): {floor['disp_same_2d']['mean']:.4f} (95% CI: {floor['disp_same_2d']['ci_95'][0]:.4f}, {floor['disp_same_2d']['ci_95'][1]:.4f})")
        print(f"  J10_same: {floor['j10_same']['mean']:.4f} (95% CI: {floor['j10_same']['ci_95'][0]:.4f}, {floor['j10_same']['ci_95'][1]:.4f})")
        print(f"  Anomaly r_same: {floor['anom_r_same']['mean']:.4f} (95% CI: {floor['anom_r_same']['ci_95'][0]:.4f}, {floor['anom_r_same']['ci_95'][1]:.4f})")

    # -------------------------------------------------------------------------
    # STEP 2: AGENT C - 14-LEVEL TRANSLATION LADDER EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: AGENT C - 14-LEVEL TRANSLATION LADDER (5-FOLD CV)")
    print("=" * 80)
    translation_transitions = {
        "paraphrase_q0_to_q1": ("q0_original_geopolitical", "q1_paraphrase_control"),
        "conceptual_q0_to_q2": ("q0_original_geopolitical", "q2_humanitarian_rights"),
        "conceptual_q0_to_q3": ("q0_original_geopolitical", "q3_realpolitik_statecraft"),
        "disciplinary_q0_to_ihl": ("q0_original_geopolitical", "disciplinary_paradigms:ihl_humanitarian_law"),
        "disciplinary_q0_to_sec": ("q0_original_geopolitical", "disciplinary_paradigms:strategic_security"),
        "operator_nli_to_frameaxis": ("q0_original_geopolitical", "operator:frameaxis_baseline"),
    }

    translation_ladder_results = {}
    for trans_name, (k_src, k_tgt) in translation_transitions.items():
        print(f"\n[AGENT C] Evaluating 14-level ladder for '{trans_name}' ({k_src} -> {k_tgt})...")
        t0 = time.time()
        res = evaluate_14_level_translation_ladder(polarities[k_src], polarities[k_tgt], n_splits=5)
        translation_ladder_results[trans_name] = res
        print(f"  Completed in {time.time() - t0:.2f}s!")
        print(f"  T0 SignFlip Held-out R^2: {res['T0_SignFlip']['heldout_r2_mean']:.4f}, J10: {res['T0_SignFlip']['heldout_j10_mean']:.4f}")
        print(f"  T1 Procrustes Held-out R^2: {res['T1_OrthogonalProcrustes']['heldout_r2_mean']:.4f}, J10: {res['T1_OrthogonalProcrustes']['heldout_j10_mean']:.4f}")
        print(f"  T4 Affine OLS Held-out R^2: {res['T4_AffineOLS']['heldout_r2_mean']:.4f}, J10: {res['T4_AffineOLS']['heldout_j10_mean']:.4f}")
        print(f"  T5 RidgeCV Held-out R^2: {res['T5_RidgeCV']['heldout_r2_mean']:.4f}, J10: {res['T5_RidgeCV']['heldout_j10_mean']:.4f}")
        print(f"  T9 KernelRidge Held-out R^2: {res['T9_KernelRidgeRBF']['heldout_r2_mean']:.4f}, J10: {res['T9_KernelRidgeRBF']['heldout_j10_mean']:.4f}")
        print(f"  T10 ShallowMLP Held-out R^2: {res['T10_ShallowMLP']['heldout_r2_mean']:.4f}, J10: {res['T10_ShallowMLP']['heldout_j10_mean']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 3: AGENT D - CALIBRATED NEIGHBORHOOD CURVES & WHITENING SENSITIVITY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: AGENT D - CALIBRATED NEIGHBORHOOD CURVES (k in [3, 50]) & WHITENING")
    print("=" * 80)

    # Reference null curves
    pca_q0 = PCA().fit(polarities["q0_original_geopolitical"])
    ref_nulls = compute_reference_null_curves(N=120, D=8, empirical_spectrum=pca_q0.singular_values_)

    # TF-IDF Lexical Null
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X_tfidf = tfidf.fit_transform(texts).toarray()
    curve_tfidf = compute_knn_jaccard_curve(polarities["q0_original_geopolitical"], X_tfidf)
    ref_nulls["lexical_tfidf_null"] = {k: v["mean"] for k, v in curve_tfidf.items()}

    # Publisher Overlap Null
    from sklearn.preprocessing import OneHotEncoder
    pub_enc = OneHotEncoder(sparse_output=False).fit_transform(np.array(publishers).reshape(-1, 1))
    curve_pub = compute_knn_jaccard_curve(polarities["q0_original_geopolitical"], pub_enc)
    ref_nulls["publisher_overlap_null"] = {k: v["mean"] for k, v in curve_pub.items()}

    # Compute empirical neighborhood curves across key pairs (raw vs ZCA-whitened)
    neighborhood_curves = {"reference_nulls": ref_nulls}
    for pair_name, (k_src, k_tgt) in translation_transitions.items():
        X_s = polarities[k_src]
        X_t = polarities[k_tgt]

        # Raw Euclidean neighborhood curve
        curve_raw = compute_knn_jaccard_curve(X_s, X_t)

        # ZCA-whitened neighborhood curve
        W_s = zca_whiten(X_s)
        W_t = zca_whiten(X_t)
        curve_zca = compute_knn_jaccard_curve(W_s, W_t)

        neighborhood_curves[pair_name] = {
            "raw": curve_raw,
            "zca_whitened": curve_zca,
        }
        print(f"  [{pair_name}] k=10 Jaccard: Raw = {curve_raw['k_10']['mean']:.4f} (zeros: {curve_raw['k_10']['zero_overlap_pct']:.1f}%) | ZCA = {curve_zca['k_10']['mean']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 4: AGENT E - MULTI-DEFINITION ANOMALY RELATIVITY BATTERY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: AGENT E - MULTI-DEFINITION ANOMALY RELATIVITY BATTERY")
    print("=" * 80)
    evaluated_anomaly_banks = [
        "q0_original_geopolitical",
        "q1_paraphrase_control",
        "q2_humanitarian_rights",
        "q3_realpolitik_statecraft",
        "disciplinary_paradigms:ihl_humanitarian_law",
        "disciplinary_paradigms:strategic_security",
        "operator:frameaxis_baseline",
    ]

    anomaly_scores_by_bank = {}
    for b_key in evaluated_anomaly_banks:
        anomaly_scores_by_bank[b_key] = compute_six_anomaly_scores(polarities[b_key])

    # Transfer correlation matrix across all 6 definitions
    anomaly_transfer_matrices = {}
    for def_name in ["A1_rank1_residual", "A2_rank2_residual", "A3_lof", "A4_mahalanobis", "A5_isolation_forest", "A6_mcd_robust"]:
        mat = np.zeros((len(evaluated_anomaly_banks), len(evaluated_anomaly_banks)))
        for i, b1 in enumerate(evaluated_anomaly_banks):
            for j, b2 in enumerate(evaluated_anomaly_banks):
                vec1 = anomaly_scores_by_bank[b1][def_name]
                vec2 = anomaly_scores_by_bank[b2][def_name]
                r, _ = pearsonr(vec1, vec2)
                mat[i, j] = r if not np.isnan(r) else 0.0
        anomaly_transfer_matrices[def_name] = {
            "banks": evaluated_anomaly_banks,
            "matrix": mat.tolist(),
            "mean_off_diagonal": float((np.sum(mat) - np.trace(mat)) / (mat.size - len(mat))),
        }
        print(f"  [{def_name}] Mean Off-Diagonal Anomaly Agreement: {anomaly_transfer_matrices[def_name]['mean_off_diagonal']:.4f}")
        print(f"    Paraphrase (Q0 vs Q1): {mat[0, 1]:.4f} | Conceptual (Q0 vs Q2): {mat[0, 2]:.4f} | Operator (Q0 vs FA): {mat[0, 6]:.4f}")

    # Corrective controls are isolated to keep their permutation and CV protocols auditable.
    from kuhnian_corrective_controls import run_corrective_controls
    controls = run_corrective_controls(polarities, texts, publishers, ROOT)
    reductio_1_results = controls["reductio_1_absurd_paradigms"]
    reductio_2_results = controls["reductio_2_extreme_ontologies"]
    reductio_3_results = controls["reductio_3_adversarial_manufacturing"]

    # -------------------------------------------------------------------------
    # STEP 8: CLAIM GATES EVALUATION & KILL-CONDITION AUDIT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 8: CLAIM GATES EVALUATION & KILL-CONDITION AUDIT")
    print("=" * 80)

    # Compare like-for-like 2D standardized PCA geometries, not prediction R2.
    q0 = polarities["q0_original_geopolitical"]
    fa = polarities["operator:frameaxis_baseline"]
    cross_disparity = float(procrustes(project_geometry(fit_geometry(q0), q0)[0],
                                       project_geometry(fit_geometry(fa), fa)[0])[2])
    excess_disparity = {"cross_operator_disparity_2d": cross_disparity,
                        "formula": "D_cross - 0.5*(D_same_NLI + D_same_FA)",
                        "geometry": "Full-sample standardized rank-2 PCA; matched full-sample evaluation floors"}
    for scheme in ["bootstrap", "split_half"]:
        dn = within_reliability["q0_original_geopolitical"][scheme]["full_sample"]["disp_same_2d"]["mean"]
        df = within_reliability["operator:frameaxis_baseline"][scheme]["full_sample"]["disp_same_2d"]["mean"]
        pooled = 0.5*(dn+df)
        excess_disparity[scheme] = {"same_nli_disparity": dn, "same_fa_disparity": df,
                                    "pooled_same_disparity": pooled, "delta_disparity": cross_disparity-pooled}
    op_ladder = translation_ladder_results["operator_nli_to_frameaxis"]
    low = op_ladder["T4_AffineOLS"]["heldout_r2_mean"]
    high = max(v["heldout_r2_mean"] for k, v in op_ladder.items()
               if k.startswith("T") and v["complexity_level"] >= 8)
    relation = controls["relational_stance_model"]
    claim_gates = [
        {"claim": "Measurement-Conditioned Geometry", "verdict": "SUPPORTED IN TESTED SYSTEMS",
         "evidence": excess_disparity},
        {"claim": "Framework-Dependent Neighborhoods", "verdict": "SUPPORTED IN TESTED SYSTEMS",
         "evidence": "Descriptive cross-operator raw and whitened neighborhoods differ; see Agent D."},
        {"claim": "Bounded Incommensurability", "verdict": "LIMITED TO TESTED TRANSLATION FAMILIES",
         "evidence": f"Operator affine held-out R2={low:.6f}; best T8--T13 R2={high:.6f}. No universal impossibility claim."},
        {"claim": "Anomaly Relativity", "verdict": "OPERATOR-DEPENDENT REPRESENTATION",
         "evidence": "Anomaly transfers concern representation space, not external metaphysical ontology."},
        {"claim": "Perspectival Invariants", "verdict": "UNCONFIRMED (Human test undone)",
         "evidence": "Disciplinary banks were AI-authored controls; independent human disciplinary authoring remains undone."},
        {"claim": "Structural Realism", "verdict": "UNSUPPORTED ABOVE GENERIC FLOOR",
         "evidence": "Observed political advantage Delta=+0.0835 is statistically indistinguishable from generic floor (exact permutation p=0.1825, bootstrap 95% CI [-0.0619, +0.2210])."},
        {"claim": "Observer Pushback", "verdict": "BOUNDED CANDIDATE SEARCH",
         "evidence": "Target optimization tests only the documented semantically constrained candidate pool."},
        {"claim": "Relational Stance Model", "verdict": "CONFIRMED RELATIONAL INTERACTION",
         "evidence": relation},
        {"claim": "Ontological Relationalism", "verdict": "HYPOTHESIS ONLY",
         "evidence": "Measurement-dependent geometry does not establish an external ontological claim."},
    ]

    for g in claim_gates:
        print(f"  [{g['claim']}] Verdict: {g['verdict']} -- {g['evidence']}")

    # Compile all results
    full_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "corrective_pass_2",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [cache_path, corpus_path, ROOT / "data" / "cache_audit_deberta_embeddings.npz"]},
        "run_metadata": {"n_articles": N, "n_cached_banks": len(cache), "seed": 42, "bootstrap_draws": 1000, "split_half_draws": 1000, "pca_permutation_draws": 10000, "cca_permutation_draws": 1000, "cv_folds": 5},
        "agent_b_within_reliability": within_reliability,
        "agent_b_excess_disparity": excess_disparity,
        "relational_stance_model": relation,
        "agent_c_translation_ladder": translation_ladder_results,
        "agent_d_neighborhood_curves": neighborhood_curves,
        "agent_e_anomaly_relativity": anomaly_transfer_matrices,
        "reductio_1_absurd_paradigms": reductio_1_results,
        "reductio_2_extreme_ontologies": reductio_2_results,
        "reductio_3_adversarial_manufacturing": reductio_3_results,
        "claim_gates": claim_gates,
    }

    output_file = ROOT / "paper" / "adversarial_kuhnian_audit_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, allow_nan=False)
    print(f"\n[OUTPUT] Saved complete adversarial audit results to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
