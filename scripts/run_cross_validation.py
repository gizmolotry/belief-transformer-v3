"""
run_cross_validation.py
Non-Circular Held-Out Observer Cross-Validation Suite

Evaluates out-of-fold observer generalization EXCLUSIVELY on the REAL news corpus
(data/real_pilot_embeddings.pt) against REAL news publisher metadata labels (a["publication"]).

Partitions the 8 V-Observers into:
- Subset A (4 observers: Pairs 1, 2, 5, 6)
- Subset B (4 observers: Pairs 3, 4, 7, 8)

Computes out-of-fold clustering performance (NMI & ARI) when evaluating Subset B's
geodesic/Hadamard distance matrix against real external publisher ground-truth labels.
"""

import io
import sys
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Ensure utf-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hadamard_fusion import ConformalMetric

def main():
    print("=" * 70)
    print(" NON-CIRCULAR HELD-OUT OBSERVER CROSS-VALIDATION SUITE")
    print("=" * 70)

    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Real embeddings missing at {emb_path}")

    # Load real embeddings [60, 8, 1536] and observer_42 metadata for publisher ground-truth
    full_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False) # [60, 8, 1536]
    N = full_embeddings.shape[0]

    obs42_file = PROJECT_ROOT / "outputs" / "non_circular_suite" / "exp1_unstructured" / "observer_42.pt"
    if not obs42_file.exists():
        raise FileNotFoundError(f"Observer file missing: {obs42_file}")

    data_42 = torch.load(obs42_file, map_location="cpu", weights_only=False)
    articles_meta = data_42["article_metadata"]

    # Ground-truth REAL publisher labels
    pub_labels = [a["publication"] for a in articles_meta]
    unique_pubs = sorted(set(pub_labels))
    pub_to_idx = {p: i for i, p in enumerate(unique_pubs)}
    y_true = np.array([pub_to_idx[p] for p in pub_labels])[:N]

    print(f"▶ Dataset: {N} real news articles across {len(unique_pubs)} real publisher outlets")
    print(f"▶ Outlets: {unique_pubs}")

    # 1. Full 8-Observer Baseline
    mean_cls_full = full_embeddings.mean(dim=1)
    D_raw_full = torch.cdist(mean_cls_full, mean_cls_full, p=2.0).numpy()
    rho_full = torch.tensor([1.0 / (1.0 + float(np.var(full_embeddings[i].numpy(), axis=0).mean())) for i in range(N)])
    D_conf_full = ConformalMetric().apply_conformal_scaling(torch.tensor(D_raw_full), rho_full).numpy()

    # 2. Subset A (Observers 0, 1, 4, 5)
    subset_A_indices = [0, 1, 4, 5]
    embeds_A = full_embeddings[:, subset_A_indices, :]
    mean_cls_A = embeds_A.mean(dim=1)
    D_raw_A = torch.cdist(mean_cls_A, mean_cls_A, p=2.0).numpy()
    rho_A = torch.tensor([1.0 / (1.0 + float(np.var(embeds_A[i].numpy(), axis=0).mean())) for i in range(N)])
    D_conf_A = ConformalMetric().apply_conformal_scaling(torch.tensor(D_raw_A), rho_A).numpy()

    # 3. Subset B (Held-Out Observers 2, 3, 6, 7)
    subset_B_indices = [2, 3, 6, 7]
    embeds_B = full_embeddings[:, subset_B_indices, :]
    mean_cls_B = embeds_B.mean(dim=1)
    D_raw_B = torch.cdist(mean_cls_B, mean_cls_B, p=2.0).numpy()
    rho_B = torch.tensor([1.0 / (1.0 + float(np.var(embeds_B[i].numpy(), axis=0).mean())) for i in range(N)])
    D_conf_B = ConformalMetric().apply_conformal_scaling(torch.tensor(D_raw_B), rho_B).numpy()

    # Spectral Clustering on Affinities: A = exp(-D^2 / 2*sigma^2)
    def evaluate_clustering(D_matrix):
        sigma = np.median(D_matrix)
        if sigma == 0: sigma = 1e-5
        affinity = np.exp(-(D_matrix**2) / (2 * (sigma**2)))

        spectral = SpectralClustering(
            n_clusters=len(unique_pubs), affinity="precomputed", random_state=42, n_init=10
        )
        y_pred = spectral.fit_predict(affinity)
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))
        return nmi, ari

    nmi_full, ari_full = evaluate_clustering(D_conf_full)
    nmi_A, ari_A = evaluate_clustering(D_conf_A)
    nmi_B, ari_B = evaluate_clustering(D_conf_B)

    print("\n▶ REAL PUBLISHER GROUND-TRUTH CROSS-VALIDATION RESULTS:")
    print(f"  ├── Full 8-Observer Suite:   NMI = {nmi_full:.4f} | ARI = {ari_full:.4f}")
    print(f"  ├── Subset A (4 Observers):  NMI = {nmi_A:.4f} | ARI = {ari_A:.4f}")
    print(f"  └── Subset B (4 Observers):  NMI = {nmi_B:.4f} | ARI = {ari_B:.4f}")

    metrics = {
        "dataset": "real_pilot_embeddings.pt",
        "ground_truth_target": "publisher_metadata_publication",
        "n_articles": N,
        "n_publishers": len(unique_pubs),
        "full_suite": {"nmi": nmi_full, "ari": ari_full},
        "subset_A": {"nmi": nmi_A, "ari": ari_A},
        "subset_B_held_out": {"nmi": nmi_B, "ari": ari_B},
    }

    result_file = PROJECT_ROOT / "data" / "cross_val_metrics.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nNon-circular cross-validation metrics saved to: {result_file}")

if __name__ == "__main__":
    main()
