"""
run_riemannian_eval.py
Canonical Riemannian vs Hadamard vs Flat Euclidean Comparison Suite (N = 60)

Evaluates on the canonical 60-article dataset (data/real_pilot_embeddings.pt) against real
news publisher ground-truth metadata labels.

Compares:
1. Flat Euclidean Distance Baseline (D_euclidean)
2. Deployed Production Hadamard Engine (D_hadamard = D_fused / sqrt(rho_i * rho_j))
3. Theoretical Discretized Riemannian Metric (D_riemannian)

Runs true 1,000-shuffle graph-level distance matrix permutations D[perm, perm] with full
spectral clustering re-fitting on every permutation.
"""

import io
import sys
from pathlib import Path
import json
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Ensure utf-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hadamard_fusion import ConformalMetric
from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig

def evaluate_clustering(D_matrix, y_true, n_seeds=10):
    """Spectral clustering on affinity matrix derived from distance matrix D."""
    N = D_matrix.shape[0]
    n_clusters = len(np.unique(y_true))
    
    sigma = np.median(D_matrix)
    if sigma == 0: sigma = 1e-5
    affinity = np.exp(-(D_matrix ** 2) / (2 * (sigma ** 2)))

    nmis, aris = [], []
    for seed in range(42, 42 + n_seeds):
        spectral = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=seed, n_init=10
        )
        y_pred = spectral.fit_predict(affinity)
        nmis.append(normalized_mutual_info_score(y_true, y_pred))
        aris.append(adjusted_rand_score(y_true, y_pred))

    return float(np.mean(nmis)), float(np.std(nmis)), float(np.mean(aris)), float(np.std(aris)), affinity

def run_graph_permutation_test(D_matrix, y_true, observed_nmi, observed_ari, n_shuffles=1000, rng_seed=42):
    """
    True Graph-Level Permutation Test:
    Jointly permutes distance matrix D[perm, perm] and re-runs SpectralClustering.fit_predict()
    from scratch on every permutation to destroy topological graph correspondence.
    """
    rng = np.random.default_rng(rng_seed)
    N = D_matrix.shape[0]
    n_clusters = len(np.unique(y_true))

    shuffled_nmis = []
    shuffled_aris = []

    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        D_perm = D_matrix[np.ix_(perm, perm)]
        
        sigma = np.median(D_perm)
        if sigma == 0: sigma = 1e-5
        affinity_perm = np.exp(-(D_perm ** 2) / (2 * (sigma ** 2)))

        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42, n_init=10)
        y_pred = spectral.fit_predict(affinity_perm)
        shuffled_nmis.append(normalized_mutual_info_score(y_true, y_pred))
        shuffled_aris.append(adjusted_rand_score(y_true, y_pred))

    p_nmi = float((np.array(shuffled_nmis) >= observed_nmi).sum() + 1) / (n_shuffles + 1)
    p_ari = float((np.array(shuffled_aris) >= observed_ari).sum() + 1) / (n_shuffles + 1)
    return p_nmi, p_ari

def main():
    print("=" * 75)
    print(" CANONICAL CLUSTERING COMPARISON SUITE (N = 60 Real Articles)")
    print("=" * 75)

    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    obs42_path = PROJECT_ROOT / "outputs" / "non_circular_suite" / "exp1_unstructured" / "observer_42.pt"

    if not emb_path.exists():
        raise FileNotFoundError(f"Real embeddings missing: {emb_path}")

    full_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False) # [60, 8, 1536]
    N = full_embeddings.shape[0]

    data_42 = torch.load(obs42_path, map_location="cpu", weights_only=False)
    articles_meta = data_42["article_metadata"]

    # Extract publisher metadata labels for the 60 articles
    pub_labels = [a["publication"] for a in articles_meta][:N]
    unique_pubs = sorted(set(pub_labels))
    pub_to_idx = {p: i for i, p in enumerate(unique_pubs)}
    y_true = np.array([pub_to_idx[p] for p in pub_labels])

    print(f"▶ Input Dataset: {N} real articles across {len(unique_pubs)} real publisher outlets")
    print(f"▶ Outlets: {unique_pubs}")

    mean_cls = full_embeddings.mean(dim=1).numpy() # [60, 1536]

    # --- 1. FLAT EUCLIDEAN DISTANCE BASELINE ---
    D_euclidean = squareform(pdist(mean_cls, metric='euclidean'))
    nmi_euc, std_nmi_euc, ari_euc, std_ari_euc, _ = evaluate_clustering(D_euclidean, y_true)

    # --- 2. DEPLOYED PRODUCTION HADAMARD CONFORMAL ENGINE ---
    D_raw = torch.cdist(torch.tensor(mean_cls), torch.tensor(mean_cls), p=2.0).numpy()
    rho_list = [1.0 / (1.0 + float(np.var(full_embeddings[i].numpy(), axis=0).mean())) for i in range(N)]
    rho_tensor = torch.tensor(rho_list, dtype=torch.float32)
    D_hadamard = ConformalMetric().apply_conformal_scaling(torch.tensor(D_raw), rho_tensor).numpy()
    nmi_had, std_nmi_had, ari_had, std_ari_had, _ = evaluate_clustering(D_hadamard, y_true)

    # --- 3. THEORETICAL DISCRETIZED RIEMANNIC METRIC ---
    spec_result = SpectralPolarity(SpectralPolarityConfig()).compute_batch([full_embeddings[i] for i in range(N)])
    phi = spec_result.singular_values.detach().cpu().numpy() # [60, 8] SVD shear spectrum

    knn_raw_d = kneighbors_graph(mean_cls, n_neighbors=5, mode='distance', include_self=False).toarray()
    knn_graph_d = np.maximum(knn_raw_d, knn_raw_d.T)
    knn_dists_mean = np.array([np.mean(knn_graph_d[i][knn_graph_d[i] > 0]) if np.sum(knn_graph_d[i] > 0) > 0 else 1.0 for i in range(N)])
    rho_riem = 1.0 / (knn_dists_mean + 1e-10)
    rho_riem = (rho_riem - rho_riem.min()) / (rho_riem.max() - rho_riem.min() + 1e-10) * 0.9 + 0.1

    K, lmbd = 5, 2.0
    adj_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if knn_graph_d[i, j] > 0:
                d_base = knn_graph_d[i, j]
                conformal_scale = 1.0 / np.sqrt(rho_riem[i] * rho_riem[j])
                shear_dist = np.linalg.norm(phi[i] - phi[j])
                adj_matrix[i, j] = np.sqrt(conformal_scale * (d_base ** 2) + lmbd * (shear_dist ** 2))

    D_riemannian = dijkstra(adj_matrix, directed=False, unweighted=False)
    non_inf = D_riemannian[~np.isinf(D_riemannian)]
    max_dist = non_inf.max() if len(non_inf) > 0 else 1.0
    D_riemannian[np.isinf(D_riemannian)] = max_dist * 2.0

    nmi_riem, std_nmi_riem, ari_riem, std_ari_riem, _ = evaluate_clustering(D_riemannian, y_true)

    # --- 4. TRUE GRAPH-LEVEL PERMUTATION TEST ---
    p_nmi_had, p_ari_had = run_graph_permutation_test(D_hadamard, y_true, nmi_had, ari_had, n_shuffles=1000)
    p_nmi_riem, p_ari_riem = run_graph_permutation_test(D_riemannian, y_true, nmi_riem, ari_riem, n_shuffles=1000)

    print("\n▶ CANONICAL SIDE-BY-SIDE METRIC PERFORMANCE (N = 60 Real Articles):")
    print(f"  ├── 1. Flat Euclidean Baseline (D_euclidean):   NMI = {nmi_euc:.4f} (±{std_nmi_euc:.4f}) | ARI = {ari_euc:.4f} (±{std_ari_euc:.4f})")
    print(f"  ├── 2. Deployed Hadamard Engine (D_hadamard):   NMI = {nmi_had:.4f} (±{std_nmi_had:.4f}) | ARI = {ari_had:.4f} (±{std_ari_had:.4f})")
    print(f"  └── 3. Theoretical Riemannian (D_riemannian):   NMI = {nmi_riem:.4f} (±{std_nmi_riem:.4f}) | ARI = {ari_riem:.4f} (±{std_ari_riem:.4f})")

    print("\n▶ ABSOLUTE & RELATIVE IMPROVEMENTS OVER EUCLIDEAN BASELINE:")
    print(f"  ├── Deployed Hadamard Gain:    ΔNMI = +{(nmi_had - nmi_euc):.4f} | ΔARI = +{(ari_had - ari_euc):.4f}")
    print(f"  └── Theoretical Riemannian Gain: ΔNMI = +{(nmi_riem - nmi_euc):.4f} | ΔARI = +{(ari_riem - ari_euc):.4f}")

    print("\n▶ TRUE GRAPH-LEVEL PERMUTATION TESTS (1,000 Shuffles D[perm, perm]):")
    print(f"  ├── Hadamard Engine Graph Permutation p:   p_NMI = {p_nmi_had:.6f} | p_ARI = {p_ari_had:.6f}")
    print(f"  └── Riemannian Metric Graph Permutation p: p_NMI = {p_nmi_riem:.6f} | p_ARI = {p_ari_riem:.6f}")

    metrics = {
        "dataset": "real_pilot_embeddings.pt",
        "n_articles": N,
        "n_publishers": len(unique_pubs),
        "euclidean": {"nmi": nmi_euc, "ari": ari_euc},
        "hadamard_deployed": {"nmi": nmi_had, "ari": ari_had, "p_nmi": p_nmi_had, "p_ari": p_ari_had},
        "riemannian_theoretical": {"nmi": nmi_riem, "ari": ari_riem, "p_nmi": p_nmi_riem, "p_ari": p_ari_riem},
    }

    metrics_path = PROJECT_ROOT / "data" / "riemannian_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
