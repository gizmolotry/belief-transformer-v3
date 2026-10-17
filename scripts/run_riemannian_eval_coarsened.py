import sys
import io
from pathlib import Path
import json
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hadamard_fusion import ConformalMetric
from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig

# STRICT ALLSIDES RATINGS MAPPING
# L = Left/Lean Left (-1 or -2)
# C = Center (0)
# R = Right/Lean Right (1 or 2)
ALLSIDES_MAP = {
    'israelnationalnews.com': 'R',
    'thecentersquare.com': 'R',
    'hotair.com': 'R',
    'baltimoresun.com': 'C',
    'npr.org': 'L',
    'haaretz.com': 'L',
    'bostonglobe.com': 'L',
    'abcnews.go.com': 'L',
    'news.sky.com': 'L',
    'foxnews.com': 'R',
    'cnn.com': 'L',
    'bbc.com': 'C',
    'bbc.co.uk': 'C',
    'theguardian.com': 'L',
    'middleeasteye.net': 'L',
    'nypost.com': 'R',
    'dailymail.co.uk': 'R',
    'ynetnews.com': 'L',
    'hindustantimes.com': 'R',
    'aljazeera.com': 'L',
}

def evaluate_clustering(D_matrix, y_true, n_seeds=10):
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
    return float(np.mean(nmis)), float(np.std(nmis)), float(np.mean(aris)), float(np.std(aris))

def main():
    print("=" * 75)
    print(" STRICT COARSENED LABEL CLUSTERING COMPARISON SUITE ")
    print("=" * 75)

    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    full_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False) # [60, 8, 1536]
    
    articles_meta = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 60: break
            articles_meta.append(json.loads(line))

    filtered_embeddings = []
    filtered_labels = []
    
    for i in range(min(60, len(articles_meta))):
        pub = articles_meta[i].get("publication", articles_meta[i].get("source", "Unknown"))
        if pub in ALLSIDES_MAP:
            filtered_embeddings.append(full_embeddings[i])
            filtered_labels.append(ALLSIDES_MAP[pub])
            
    if not filtered_embeddings:
        print("No articles matched the strict publisher mapping.")
        return

    N = len(filtered_embeddings)
    filtered_embeddings = torch.stack(filtered_embeddings)
    mean_cls = filtered_embeddings.mean(dim=1).numpy()
    
    unique_labels = sorted(list(set(filtered_labels)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_true = np.array([label_to_idx[l] for l in filtered_labels])

    print(f"▶ Input Dataset: {N} real articles strictly mapped to AllSides ratings")
    print(f"▶ Categories found: {unique_labels}")
    print(f"▶ Class distribution: {[list(y_true).count(label_to_idx[l]) for l in unique_labels]}")

    # 1. FLAT EUCLIDEAN DISTANCE BASELINE
    D_euclidean = squareform(pdist(mean_cls, metric='euclidean'))
    nmi_euc, std_nmi_euc, ari_euc, std_ari_euc = evaluate_clustering(D_euclidean, y_true)

    # 2. DEPLOYED PRODUCTION HADAMARD CONFORMAL ENGINE
    D_raw = torch.cdist(torch.tensor(mean_cls), torch.tensor(mean_cls), p=2.0).numpy()
    rho_list = [1.0 / (1.0 + float(np.var(filtered_embeddings[i].numpy(), axis=0).mean())) for i in range(N)]
    rho_tensor = torch.tensor(rho_list, dtype=torch.float32)
    D_hadamard = ConformalMetric().apply_conformal_scaling(torch.tensor(D_raw), rho_tensor).numpy()
    nmi_had, std_nmi_had, ari_had, std_ari_had = evaluate_clustering(D_hadamard, y_true)

    # 3. THEORETICAL DISCRETIZED RIEMANNIC METRIC
    spec_result = SpectralPolarity(SpectralPolarityConfig()).compute_batch([filtered_embeddings[i] for i in range(N)])
    phi = spec_result.singular_values.detach().cpu().numpy()

    # Need n_neighbors < N for kneighbors_graph
    n_neigh = min(5, N - 1)
    knn_raw_d = kneighbors_graph(mean_cls, n_neighbors=n_neigh, mode='distance', include_self=False).toarray()
    knn_graph_d = np.maximum(knn_raw_d, knn_raw_d.T)
    knn_dists_mean = np.array([np.mean(knn_graph_d[i][knn_graph_d[i] > 0]) if np.sum(knn_graph_d[i] > 0) > 0 else 1.0 for i in range(N)])
    rho_riem = 1.0 / (knn_dists_mean + 1e-10)
    rho_riem = (rho_riem - rho_riem.min()) / (rho_riem.max() - rho_riem.min() + 1e-10) * 0.9 + 0.1

    K, lmbd = n_neigh, 2.0
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

    nmi_riem, std_nmi_riem, ari_riem, std_ari_riem = evaluate_clustering(D_riemannian, y_true)

    print("\n▶ CANONICAL SIDE-BY-SIDE METRIC PERFORMANCE (Coarsened Labels):")
    print(f"  ├── 1. Flat Euclidean Baseline (D_euclidean):   NMI = {nmi_euc:.4f} (±{std_nmi_euc:.4f}) | ARI = {ari_euc:.4f} (±{std_ari_euc:.4f})")
    print(f"  ├── 2. Deployed Hadamard Engine (D_hadamard):   NMI = {nmi_had:.4f} (±{std_nmi_had:.4f}) | ARI = {ari_had:.4f} (±{std_ari_had:.4f})")
    print(f"  └── 3. Theoretical Riemannian (D_riemannian):   NMI = {nmi_riem:.4f} (±{std_nmi_riem:.4f}) | ARI = {ari_riem:.4f} (±{std_ari_riem:.4f})")

if __name__ == '__main__':
    main()
