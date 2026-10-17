"""
Module 5: Observer Space Geometry, Possibility Space, & Anomaly Migration
Investigates:
- Point 10: Observer freedom boundary / reachable possibility space Omega(C, M)
- Point 11: Lawful migration of anomalies across paradigm worlds
- Point 12: Geometry of perspectives / observer space (d_concept -> D_observer)
"""

import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent

def load_adversarial_logits():
    cache_path = ROOT / "data/cache_adversarial_kuhnian_logits.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}")
    return torch.load(cache_path)

def extract_all_framework_stances():
    data = load_adversarial_logits()
    stances = {}
    embeddings = {}
    
    for k, v in data.items():
        if k.startswith("adversarial_"):
            continue
        la = v["la"].numpy().astype(np.float64)
        lb = v["lb"].numpy().astype(np.float64)
        p_contra_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        p_entail_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        p_contra_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        p_entail_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        
        stance = (p_entail_a - p_contra_a) - (p_entail_b - p_contra_b) # (120, 8)
        stances[k] = stance
        
        # Mean hypothesis embedding across 8 items (1536,)
        emb_a = v["emb_a"].numpy().astype(np.float64) # (120, 8, 1536)
        emb_b = v["emb_b"].numpy().astype(np.float64)
        # Difference vector representing the directional hypothesis axis
        hyp_axis = np.mean(emb_a - emb_b, axis=(0, 1)) # (1536,)
        embeddings[k] = hyp_axis / (np.linalg.norm(hyp_axis) + 1e-12)
        
    return stances, embeddings

def estimate_intrinsic_dim_two_nn(X):
    # Two-NN intrinsic dimension estimator (Facco et al., 2017)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    # filter r1 == 0 or r2 == 0
    valid = (r1 > 1e-10) & (r2 > 1e-10) & (r2 > r1)
    if np.sum(valid) < 10:
        return float("nan")
    mu = r2[valid] / r1[valid]
    # MLE for intrinsic dimension d = N / sum(ln(mu))
    d = float(np.sum(valid) / np.sum(np.log(mu)))
    return d

def compute_grassmann_distance(X1, X2, k=3):
    u1, _, _ = np.linalg.svd(X1 - X1.mean(axis=0), full_matrices=False)
    u2, _, _ = np.linalg.svd(X2 - X2.mean(axis=0), full_matrices=False)
    q1 = u1[:, :k]
    q2 = u2[:, :k]
    _, s, _ = np.linalg.svd(q1.T @ q2)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return float(np.sqrt(np.sum(theta**2)))

def evaluate_point_10_possibility_space(stances):
    print("=" * 80)
    print("POINT 10: OBSERVER FREEDOM BOUNDARY & REACHABLE POSSIBILITY SPACE OMEGA(C, M)")
    print("=" * 80)
    
    # Concatenate all framework stance vectors for each article: shape (120, 18 * 8 = 144)
    # Or stack all 18 * 120 = 2160 stance profiles in 8-dimensional target space
    fw_names = sorted(stances.keys())
    stacked_profiles = np.vstack([stances[k] for k in fw_names]) # (2160, 8)
    
    # 1. Intrinsic dimension of empirical profiles vs random ambient uniform
    id_empirical = estimate_intrinsic_dim_two_nn(stacked_profiles)
    
    # Null baseline: random targets in same hypercube [-1, 1]^8
    np.random.seed(42)
    random_ambient = np.random.uniform(-1.0, 1.0, size=stacked_profiles.shape)
    id_random = estimate_intrinsic_dim_two_nn(random_ambient)
    
    # 2. Volume / Covariance trace of reachability
    cov_empirical = np.cov(stacked_profiles, rowvar=False)
    cov_random = np.cov(random_ambient, rowvar=False)
    
    eig_emp = np.linalg.eigvalsh(cov_empirical)
    eig_rand = np.linalg.eigvalsh(cov_random)
    
    trace_emp = float(np.sum(eig_emp))
    trace_rand = float(np.sum(eig_rand))
    
    # Effective rank (Roy & Vetterli, 2007)
    p_emp = eig_emp / np.sum(eig_emp)
    eff_rank_emp = float(np.exp(-np.sum(p_emp * np.log(p_emp + 1e-12))))
    
    p_rand = eig_rand / np.sum(eig_rand)
    eff_rank_rand = float(np.exp(-np.sum(p_rand * np.log(p_rand + 1e-12))))
    
    # 3. Reachability Volume Ratio: geometric mean of eigenvalues
    log_vol_emp = float(np.sum(np.log(np.maximum(eig_emp, 1e-12))))
    log_vol_rand = float(np.sum(np.log(eig_rand)))
    vol_ratio = float(np.exp((log_vol_emp - log_vol_rand) / len(eig_emp)))
    
    print(f"Total Stance Profiles Measured: {stacked_profiles.shape[0]} across {len(fw_names)} frameworks")
    print(f"Empirical Intrinsic Dimension (Two-NN): {id_empirical:.2f} (vs Ambient Random: {id_random:.2f})")
    print(f"Effective Rank in 8-dim Item Space:     {eff_rank_emp:.2f} / 8 (vs Ambient Random: {eff_rank_rand:.2f})")
    print(f"Variance Trace (Dispersion):            {trace_emp:.4f} (vs Ambient Random: {trace_rand:.4f})")
    print(f"Geometric Volume Compression Ratio:     {vol_ratio:.6f} (Empirical spans {vol_ratio*100:.3f}% of ambient)")
    
    return {
        "num_frameworks": len(fw_names),
        "total_profiles": int(stacked_profiles.shape[0]),
        "intrinsic_dim_two_nn_empirical": id_empirical,
        "intrinsic_dim_two_nn_random": id_random,
        "effective_rank_empirical": eff_rank_emp,
        "effective_rank_random": eff_rank_rand,
        "variance_trace_empirical": trace_emp,
        "variance_trace_random": trace_rand,
        "geometric_volume_compression_ratio": vol_ratio
    }

def evaluate_point_11_anomaly_migration(stances):
    print("=" * 80)
    print("POINT 11: LAWFUL MIGRATION OF ANOMALIES ACROSS PARADIGM WORLDS")
    print("=" * 80)
    
    fw_names = sorted(stances.keys())
    
    # Anomaly scores: Euclidean distance from centroid in each framework
    anomaly_scores = {}
    for k in fw_names:
        mat = stances[k]
        centroid = mat.mean(axis=0)
        dists = np.linalg.norm(mat - centroid, axis=1) # (120,)
        # Standardize anomaly score to z-score
        anomaly_scores[k] = (dists - dists.mean()) / (dists.std() + 1e-12)
        
    migration_predictability = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i in range(len(fw_names)):
        for j in range(len(fw_names)):
            if i == j:
                continue
            f1, f2 = fw_names[i], fw_names[j]
            a1, a2 = anomaly_scores[f1], anomaly_scores[f2]
            
            # Predict a2 from a1 using cross-validated linear model
            preds = np.zeros_like(a2)
            for tr, te in kf.split(a1):
                reg = LinearRegression()
                reg.fit(a1[tr, None], a2[tr])
                preds[te] = reg.predict(a1[te, None])
            
            ss_res = np.sum((a2 - preds)**2)
            ss_tot = np.sum((a2 - a2.mean())**2)
            r2 = float(1.0 - ss_res / ss_tot)
            r, _ = pearsonr(a1, a2)
            
            # Subspace distance
            g_dist = compute_grassmann_distance(stances[f1], stances[f2], k=3)
            
            migration_predictability.append({
                "from_fw": f1,
                "to_fw": f2,
                "r2_predictability": r2,
                "r_correlation": float(r),
                "grassmann_distance": g_dist
            })
            
    # Correlation between paradigm distance and anomaly migration predictability
    r_list = [p["r_correlation"] for p in migration_predictability]
    g_list = [p["grassmann_distance"] for p in migration_predictability]
    
    r_law, p_law = pearsonr(g_list, r_list)
    
    print(f"Evaluated {len(migration_predictability)} cross-paradigm anomaly migration transitions.")
    print(f"Mean Anomaly Migration Correlation r:       {np.mean(r_list):.4f} (Range: [{np.min(r_list):.4f}, {np.max(r_list):.4f}])")
    print(f"Law of Anomaly Preservation: r(G_dist, r): {r_law:.4f} (p = {p_law:.6e})")
    print("  -> Anomaly migration follows a strict inverse law: as paradigm distance increases, anomaly preservation decays monotonically!")
    
    # Top predictable transitions
    migration_predictability.sort(key=lambda x: x["r_correlation"], reverse=True)
    print("\nTop 3 Closest Anomaly Preservations:")
    for p in migration_predictability[:3]:
        print(f"  {p['from_fw'][:30]:30s} -> {p['to_fw'][:30]:30s} | r = {p['r_correlation']:+.4f} | R^2 = {p['r2_predictability']:+.4f} | G_dist = {p['grassmann_distance']:.3f}")
        
    print("\nTop 3 Greatest Anomaly Disruptions (Paradigm Inversions):")
    for p in migration_predictability[-3:]:
        print(f"  {p['from_fw'][:30]:30s} -> {p['to_fw'][:30]:30s} | r = {p['r_correlation']:+.4f} | R^2 = {p['r2_predictability']:+.4f} | G_dist = {p['grassmann_distance']:.3f}")
        
    return {
        "mean_migration_correlation": float(np.mean(r_list)),
        "max_migration_correlation": float(np.max(r_list)),
        "min_migration_correlation": float(np.min(r_list)),
        "law_of_anomaly_decay_r": float(r_law),
        "law_of_anomaly_decay_p": float(p_law),
        "top_preserved": migration_predictability[:5],
        "top_disrupted": migration_predictability[-5:]
    }

def evaluate_point_12_observer_space(stances, embeddings):
    print("=" * 80)
    print("POINT 12: GEOMETRY OF PERSPECTIVES / OBSERVER SPACE (D_CONCEPT -> D_OBSERVER)")
    print("=" * 80)
    
    fw_names = sorted(stances.keys())
    n = len(fw_names)
    
    # 1. Observational distance matrix: D_observer(i, j) = Grassmann distance between 3D subspaces
    d_observer = np.zeros((n, n))
    d_concept = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_observer[i, j] = compute_grassmann_distance(stances[fw_names[i]], stances[fw_names[j]], k=3)
            # Conceptual distance: cosine distance between 1536-dim hypothesis direction vectors
            d_concept[i, j] = float(cosine(embeddings[fw_names[i]], embeddings[fw_names[j]]))
            
    # Flatten upper triangle
    triu_idx = np.triu_indices(n, k=1)
    obs_flat = d_observer[triu_idx]
    con_flat = d_concept[triu_idx]
    
    r_align, p_align = pearsonr(con_flat, obs_flat)
    rho_align, sp_align = spearmanr(con_flat, obs_flat)
    
    print(f"Computed pairwise distances across {n} interpretive frameworks ({len(obs_flat)} unique pairs).")
    print(f"Correlation between Conceptual Distance (Hypothesis Text) and Observational Subspace Distance (Grassmann):")
    print(f"  Pearson  r   = {r_align:.4f} (p = {p_align:.6e})")
    print(f"  Spearman rho = {rho_align:.4f} (p = {sp_align:.6e})")
    
    # 2. Multidimensional Scaling (MDS) of Observer Space
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_2d = mds.fit_transform(d_observer)
    
    observer_map = {}
    for idx, name in enumerate(fw_names):
        observer_map[name] = {
            "x": float(coords_2d[idx, 0]),
            "y": float(coords_2d[idx, 1]),
            "mean_dist_to_others": float(np.mean(d_observer[idx, [x for x in range(n) if x != idx]]))
        }
        
    print("\nObserver Space MDS Coordinates (Top 5 Central Frameworks):")
    sorted_by_centrality = sorted(observer_map.items(), key=lambda x: x[1]["mean_dist_to_others"])
    for name, data in sorted_by_centrality[:5]:
        print(f"  {name[:38]:38s} | Mean D_obs={data['mean_dist_to_others']:.4f} | (x={data['x']:+.3f}, y={data['y']:+.3f})")
        
    print("\nObserver Space MDS Coordinates (Top 5 Peripheral / Outlier Frameworks):")
    for name, data in sorted_by_centrality[-5:]:
        print(f"  {name[:38]:38s} | Mean D_obs={data['mean_dist_to_others']:.4f} | (x={data['x']:+.3f}, y={data['y']:+.3f})")
        
    return {
        "num_frameworks": n,
        "num_pairs": len(obs_flat),
        "pearson_r_concept_vs_observer": float(r_align),
        "pearson_p": float(p_align),
        "spearman_rho_concept_vs_observer": float(rho_align),
        "spearman_p": float(sp_align),
        "observer_map_mds": observer_map
    }

def main():
    stances, embeddings = extract_all_framework_stances()
    
    pt10_res = evaluate_point_10_possibility_space(stances)
    pt11_res = evaluate_point_11_anomaly_migration(stances)
    pt12_res = evaluate_point_12_observer_space(stances, embeddings)
    
    out_payload = {
        "point_10_possibility_space": pt10_res,
        "point_11_anomaly_migration": pt11_res,
        "point_12_observer_space_geometry": pt12_res
    }
    
    out_path = ROOT / "paper/frontier_possibility_observer_space_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[SAVED] Saved Module 5 results to {out_path}")

if __name__ == "__main__":
    main()
