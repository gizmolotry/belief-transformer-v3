import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import json
import sys
import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    print("=" * 75)
    print(" CONTROLLED N=60 PUBLISHER COARSENING (NMI/ARI EVAL)")
    print("=" * 75)

    data_path = PROJECT_ROOT / "data" / "controlled_ideo_embeddings_v3_small.pt"
    if not data_path.exists():
        print("Embeddings not found. Run run_controlled_auto_observer_validation.py first.")
        return

    data = torch.load(data_path, map_location="cpu")
    emb = data['ideo_emb']  # [60, 8, 768]
    raw_labels = data['labels']  # e.g. "-2", "-1", "0", "1", "2"
    
    # Coarsen labels into Left, Center, Right
    def coarsen(score):
        try:
            val = float(score)
            if val < 0: return 0  # Left
            elif val == 0: return 1  # Center
            else: return 2  # Right
        except:
            return 1

    true_labels = np.array([coarsen(l) for l in raw_labels])
    N, B, H = emb.shape
    
    # 1. Euclidean: Flatten to [60, 6144]
    X_eucl = emb.reshape(N, B * H).numpy()

    # 2. Hadamard: Variance across bots [60, 768]
    X_hadamard = emb.var(dim=1).numpy()

    # 3. Riemannian: Log-Euclidean mapping of Covariance
    # Covariance per article: [60, 768, 768] is too large, we do it in the bot space (Dual form)
    # Actually, we can use the RKS projection.
    from core.nli_extraction import DirichletConfig, DirichletFusion
    cfg = DirichletConfig(n_bots=B, hidden_dim=H, rks_dim=512, n_observers=1)
    fusion = DirichletFusion(cfg)
    # The Riemannian proxy used in the paper is the variance across O-observers in the RKS space
    # For a deterministic proxy without O-observers, we just use the RKS features themselves
    rks_features = fusion._rks_project(emb.reshape(-1, H)).reshape(N, B, -1)
    X_riem = rks_features.var(dim=1).numpy()

    # Clustering
    def evaluate_clustering(X, name):
        sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
        preds = sc.fit_predict(X)
        nmi = normalized_mutual_info_score(true_labels, preds)
        ari = adjusted_rand_score(true_labels, preds)
        print(f"  {name} Geometry:")
        print(f"    NMI: {nmi:.4f}")
        print(f"    ARI: {ari:.4f}")
        return nmi, ari

    nmi_e, ari_e = evaluate_clustering(X_eucl, "Euclidean")
    nmi_h, ari_h = evaluate_clustering(X_hadamard, "Hadamard")
    nmi_r, ari_r = evaluate_clustering(X_riem, "Riemannian (RKS Var proxy)")

    metrics = {
        "nmi_euclidean": nmi_e, "ari_euclidean": ari_e,
        "nmi_hadamard": nmi_h, "ari_hadamard": ari_h,
        "nmi_riemannian": nmi_r, "ari_riemannian": ari_r,
        "N": N,
        "model_used": "cross-encoder/nli-deberta-v3-small"
    }

    with open(PROJECT_ROOT / "data" / "controlled_clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
