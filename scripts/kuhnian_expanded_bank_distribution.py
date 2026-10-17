#!/usr/bin/env python3
"""Expanded Reference Bank Distribution and 3-Level CCA Statistical Inference.

Fulfills Points 7 and 8 of the Kuhnian Clarification Battery:
1. Point 7: Expanded reference bank distribution across 20-25 banks in 5 classes:
   - Real political frameworks (11 banks: Q0, Q1, Q2, Q3, IHL, Sec, Legal, Ops, Welfare, Econ, Rel)
   - Topic-matched non-political frameworks (3 banks: Peace, Realist IR, Media framing)
   - Syntactically matched absurd frameworks (4 banks: Meteorology, Culinary, Botany, Sports)
   - Adversarial / distractors (2 banks: Inverted PC1, Orthogonal distractors)
   - Shuffled / nonsense controls
   Computes distribution of PC1 variance, Participation Ratio, out-of-sample CCA to Q0,
   neighborhood overlap (J10), publisher ANOVA (F, eta2), and anomaly agreement (Spearman rho).
2. Point 8: Multi-level statistical inference on the CCA gap (Delta r = r_political - r_absurd):
   - Article bootstrap (B=1,000)
   - Bank bootstrap (B=1,000)
   - Combined Article + Bank cluster bootstrap (B=1,000)
   - Exact label permutation test (10,000 draws)

Outputs:
- paper/kuhnian_expanded_bank_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_corpus_and_cache():
    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    articles = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    publishers = [art.get("publisher", "unknown") for art in articles]
    N = len(publishers)

    cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    polarities = {}
    for k, b_val in cache.items():
        la = b_val["la"].float()
        lb = b_val["lb"].float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        polarities[k] = (sa - sb).astype(float)  # [120, 8]

    # Generate additional nonsense / shuffled banks to reach 25 banks
    rng = np.random.default_rng(42)
    # Shuffled row bank: shuffle articles per pair
    pol_shuff = np.zeros_like(polarities["q0_original_geopolitical"])
    for d in range(8):
        pol_shuff[:, d] = rng.permutation(polarities["q0_original_geopolitical"][:, d])
    polarities["nonsense:shuffled_articles"] = pol_shuff

    # White noise Gaussian bank
    polarities["nonsense:gaussian_noise"] = rng.normal(0, 1, size=(N, 8))

    return publishers, polarities


def compute_bank_metrics(X: np.ndarray, Q0: np.ndarray, publishers: List[str], kf: KFold) -> Dict[str, Any]:
    N, D = X.shape

    # 1. PCA: PC1 variance ratio and Participation Ratio
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    cov = (Z.T @ Z) / (N - 1)
    evals, _ = np.linalg.eigh(cov)
    evals = np.maximum(evals[::-1], 0)
    total_var = np.sum(evals) + 1e-12
    pc1_var_ratio = float(evals[0] / total_var)
    pr = float((np.sum(evals) ** 2) / (np.sum(evals ** 2) + 1e-12))

    # 2. Out-of-sample CCA to Q0 (5-fold CV)
    Z_q0 = StandardScaler().fit_transform(Q0)
    canon_r_folds = []
    for tr, te in kf.split(Z):
        cca = CCA(n_components=1, scale=False, max_iter=1000)
        cca.fit(Z[tr], Z_q0[tr])
        u_te, v_te = cca.transform(Z[te], Z_q0[te])
        r, _ = pearsonr(u_te[:, 0], v_te[:, 0])
        canon_r_folds.append(float(r if not np.isnan(r) else 0.0))
    canon_r_mean = float(np.mean(canon_r_folds))

    # 3. Neighborhood Overlap J10 with Q0
    dist_x = cdist(Z, Z)
    dist_q0 = cdist(Z_q0, Z_q0)
    np.fill_diagonal(dist_x, np.inf)
    np.fill_diagonal(dist_q0, np.inf)
    j_list = []
    for i in range(N):
        nn_x = set(np.argsort(dist_x[i])[:10])
        nn_q = set(np.argsort(dist_q0[i])[:10])
        inter = len(nn_x & nn_q)
        union = len(nn_x | nn_q)
        j_list.append(inter / union if union > 0 else 0.0)
    j10_mean = float(np.mean(j_list))

    # 4. Publisher separation on PC1 (ANOVA F-stat and eta^2)
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(Z)[:, 0]
    unique_pubs = list(set(publishers))
    groups = [pc1_scores[np.array(publishers) == p] for p in unique_pubs if np.sum(np.array(publishers) == p) >= 3]
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        # eta^2 = SS_between / SS_total
        grand_mean = np.mean(pc1_scores)
        ss_tot = np.sum((pc1_scores - grand_mean) ** 2)
        ss_bet = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta2 = float(ss_bet / (ss_tot + 1e-12))
        f_stat = float(f_stat)
    else:
        f_stat, eta2, p_val = 0.0, 0.0, 1.0

    # 5. Anomaly ranking agreement with Q0
    # Reconstruction anomaly score from 1D PCA
    Z_rec = pc1_scores[:, None] @ pca.components_
    anom_x = np.mean((Z - Z_rec) ** 2, axis=1)

    pca_q0 = PCA(n_components=1)
    q0_pc1 = pca_q0.fit_transform(Z_q0)[:, 0]
    q0_rec = q0_pc1[:, None] @ pca_q0.components_
    anom_q0 = np.mean((Z_q0 - q0_rec) ** 2, axis=1)

    spearman_anom, _ = spearmanr(anom_x, anom_q0)
    spearman_anom = float(spearman_anom if not np.isnan(spearman_anom) else 0.0)

    return {
        "pc1_variance_ratio": pc1_var_ratio,
        "participation_ratio": pr,
        "cca_heldout_r": canon_r_mean,
        "cca_fold_r": canon_r_folds,
        "j10_overlap_q0": j10_mean,
        "publisher_anova_f": f_stat,
        "publisher_anova_eta2": eta2,
        "publisher_anova_p": float(p_val),
        "anomaly_spearman_rho": spearman_anom,
    }


def run_expanded_bank_distribution_and_inference():
    print("=" * 80)
    print("EXPANDED REFERENCE BANK DISTRIBUTION & CCA 3-LEVEL INFERENCE (POINTS 7, 8)")
    print("=" * 80, flush=True)

    publishers, polarities = load_corpus_and_cache()
    N = len(publishers)
    kf = KFold(5, shuffle=True, random_state=42)
    Q0 = polarities["q0_original_geopolitical"]

    bank_classes = {
        "political": [
            "q0_original_geopolitical",
            "q1_paraphrase_control",
            "q2_humanitarian_rights",
            "q3_realpolitik_statecraft",
            "disciplinary_paradigms:ihl_humanitarian_law",
            "disciplinary_paradigms:strategic_security",
            "extreme_political_ontologies:legal_culpability",
            "extreme_political_ontologies:operational_efficacy",
            "extreme_political_ontologies:humanitarian_welfare",
            "extreme_political_ontologies:economic_cost",
            "extreme_political_ontologies:religious_obligation",
        ],
        "disciplinary_non_political": [
            "disciplinary_paradigms:peace_conflict_studies",
            "disciplinary_paradigms:realist_international_relations",
            "disciplinary_paradigms:media_framing_analysis",
        ],
        "absurd_structurally_matched": [
            "absurd_structurally_matched_paradigms:meteorology",
            "absurd_structurally_matched_paradigms:culinary_arts",
            "absurd_structurally_matched_paradigms:botanical_taxonomy",
            "absurd_structurally_matched_paradigms:sports_tactics",
        ],
        "adversarial_distractors": [
            "adversarial_manufacturing_candidates:adv_inverted_pc1",
            "adversarial_manufacturing_candidates:adv_orthogonal_distractors",
        ],
        "nonsense_controls": [
            "nonsense:shuffled_articles",
            "nonsense:gaussian_noise",
        ],
    }

    # Evaluate all banks
    bank_results = {}
    print(f"\n[EVALUATING {sum(len(v) for v in bank_classes.values())} BANKS]...", flush=True)
    for cat, banks in bank_classes.items():
        print(f"\n--- Category: {cat} ({len(banks)} banks) ---", flush=True)
        for b_name in banks:
            res = compute_bank_metrics(polarities[b_name], Q0, publishers, kf)
            bank_results[b_name] = {"category": cat, **res}
            print(f"  {b_name[:45]:<45} | PC1: {res['pc1_variance_ratio']:.4f} | PR: {res['participation_ratio']:.2f} | CCA: {res['cca_heldout_r']:.4f} | J10: {res['j10_overlap_q0']:.4f} | F: {res['publisher_anova_f']:.2f}", flush=True)

    # Category summaries
    cat_summaries = {}
    for cat in bank_classes.keys():
        b_keys = [k for k, v in bank_results.items() if v["category"] == cat]
        cat_summaries[cat] = {
            "n_banks": len(b_keys),
            "pc1_var_mean": float(np.mean([bank_results[k]["pc1_variance_ratio"] for k in b_keys])),
            "pc1_var_std": float(np.std([bank_results[k]["pc1_variance_ratio"] for k in b_keys])),
            "pr_mean": float(np.mean([bank_results[k]["participation_ratio"] for k in b_keys])),
            "cca_mean": float(np.mean([bank_results[k]["cca_heldout_r"] for k in b_keys])),
            "cca_std": float(np.std([bank_results[k]["cca_heldout_r"] for k in b_keys])),
            "j10_mean": float(np.mean([bank_results[k]["j10_overlap_q0"] for k in b_keys])),
            "publisher_f_mean": float(np.mean([bank_results[k]["publisher_anova_f"] for k in b_keys])),
            "publisher_eta2_mean": float(np.mean([bank_results[k]["publisher_anova_eta2"] for k in b_keys])),
        }

    # -------------------------------------------------------------------------
    # POINT 8: 3-LEVEL INFERENCE ON CCA GAP (POLITICAL VS ABSURD)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POINT 8: MULTI-LEVEL STATISTICAL INFERENCE ON CCA GAP (POLITICAL VS ABSURD)")
    print("=" * 80, flush=True)

    pol_banks = bank_classes["political"]
    # Exclude Q0 and Q1 from political comparison to avoid self-correlation / paraphrase inflation
    # Use distinct substantive political frameworks: Q2, Q3, IHL, Sec, Legal, Ops, Welfare, Econ, Rel (9 banks)
    pol_eval_banks = [b for b in pol_banks if b not in ["q0_original_geopolitical", "q1_paraphrase_control"]]
    abs_banks = bank_classes["absurd_structurally_matched"]

    obs_pol_cca = float(np.mean([bank_results[k]["cca_heldout_r"] for k in pol_eval_banks]))
    obs_abs_cca = float(np.mean([bank_results[k]["cca_heldout_r"] for k in abs_banks]))
    obs_delta_cca = obs_pol_cca - obs_abs_cca
    print(f"  Observed Substantive Political CCA: {obs_pol_cca:.4f} (N={len(pol_eval_banks)} banks)")
    print(f"  Observed Absurd Matched CCA:        {obs_abs_cca:.4f} (N={len(abs_banks)} banks)")
    print(f"  Observed CCA Delta:                {obs_delta_cca:+.4f}", flush=True)

    # Fast CCA function for bootstraps
    def eval_cca_fast(X_pol_list, X_abs_list, Q0_mat, n_folds=5):
        kf_boot = KFold(n_folds, shuffle=True, random_state=42)
        Z_q0 = StandardScaler().fit_transform(Q0_mat)

        pol_r_list = []
        for X in X_pol_list:
            Z = StandardScaler().fit_transform(X)
            fold_r = []
            for tr, te in kf_boot.split(Z):
                cca = CCA(n_components=1, scale=False, max_iter=500)
                cca.fit(Z[tr], Z_q0[tr])
                u, v = cca.transform(Z[te], Z_q0[te])
                r, _ = pearsonr(u[:, 0], v[:, 0])
                fold_r.append(r if not np.isnan(r) else 0.0)
            pol_r_list.append(np.mean(fold_r))

        abs_r_list = []
        for X in X_abs_list:
            Z = StandardScaler().fit_transform(X)
            fold_r = []
            for tr, te in kf_boot.split(Z):
                cca = CCA(n_components=1, scale=False, max_iter=500)
                cca.fit(Z[tr], Z_q0[tr])
                u, v = cca.transform(Z[te], Z_q0[te])
                r, _ = pearsonr(u[:, 0], v[:, 0])
                fold_r.append(r if not np.isnan(r) else 0.0)
            abs_r_list.append(np.mean(fold_r))

        return float(np.mean(pol_r_list)), float(np.mean(abs_r_list))

    rng = np.random.default_rng(42)
    B = 1000

    # Level 1: Article Bootstrap (B=1,000)
    print("  [LEVEL 1] Running Article Bootstrap (B=1,000)...", flush=True)
    t0 = time.time()
    art_boot_deltas = []
    # Pre-sample bootstrap indices
    for b_idx in range(B):
        idx = rng.choice(N, size=N, replace=True)
        # To make article bootstrap fast across 1000 runs:
        # Use mean Pearson r of first canonical variates fitted on full sample or 2-fold
        # Since full 5-fold CCA on 1000 * 13 banks = 65,000 fits, we compute full-sample SVD CCA with shrinkage
        # SVD-based regularized CCA:
        def rcca(X, Y, reg=0.01):
            Zx = StandardScaler().fit_transform(X)
            Zy = StandardScaler().fit_transform(Y)
            Cxx = (Zx.T @ Zx) / (len(X) - 1) + reg * np.eye(Zx.shape[1])
            Cyy = (Zy.T @ Zy) / (len(Y) - 1) + reg * np.eye(Zy.shape[1])
            Cxy = (Zx.T @ Zy) / (len(X) - 1)
            Rx = np.linalg.cholesky(Cxx)
            Ry = np.linalg.cholesky(Cyy)
            M = np.linalg.inv(Rx) @ Cxy @ np.linalg.inv(Ry).T
            s = np.linalg.svd(M, compute_uv=False)
            return float(s[0])

        q0_b = Q0[idx]
        pol_r_b = [rcca(polarities[k][idx], q0_b) for k in pol_eval_banks]
        abs_r_b = [rcca(polarities[k][idx], q0_b) for k in abs_banks]
        art_boot_deltas.append(np.mean(pol_r_b) - np.mean(abs_r_b))
    art_boot_deltas = np.array(art_boot_deltas)
    p_art = float((1 + np.sum(art_boot_deltas <= 0)) / (B + 1))
    print(f"    Done in {time.time() - t0:.2f}s! Mean Delta: {np.mean(art_boot_deltas):.4f}, 95% CI: [{np.percentile(art_boot_deltas, 2.5):.4f}, {np.percentile(art_boot_deltas, 97.5):.4f}], p = {p_art:.4f}", flush=True)

    # Level 2: Bank Bootstrap (B=1,000)
    print("  [LEVEL 2] Running Bank Bootstrap (B=1,000)...", flush=True)
    t0 = time.time()
    bank_boot_deltas = []
    pol_eval_r_fixed = np.array([bank_results[k]["cca_heldout_r"] for k in pol_eval_banks])
    abs_r_fixed = np.array([bank_results[k]["cca_heldout_r"] for k in abs_banks])
    for b_idx in range(B):
        sample_pol = rng.choice(pol_eval_r_fixed, size=len(pol_eval_r_fixed), replace=True)
        sample_abs = rng.choice(abs_r_fixed, size=len(abs_r_fixed), replace=True)
        bank_boot_deltas.append(np.mean(sample_pol) - np.mean(sample_abs))
    bank_boot_deltas = np.array(bank_boot_deltas)
    p_bank = float((1 + np.sum(bank_boot_deltas <= 0)) / (B + 1))
    print(f"    Done in {time.time() - t0:.2f}s! Mean Delta: {np.mean(bank_boot_deltas):.4f}, 95% CI: [{np.percentile(bank_boot_deltas, 2.5):.4f}, {np.percentile(bank_boot_deltas, 97.5):.4f}], p = {p_bank:.4f}", flush=True)

    # Level 3: Combined Article + Bank Cluster Bootstrap (B=1,000)
    print("  [LEVEL 3] Running Combined Article + Bank Cluster Bootstrap (B=1,000)...", flush=True)
    t0 = time.time()
    cluster_boot_deltas = []
    for b_idx in range(B):
        idx = rng.choice(N, size=N, replace=True)
        q0_b = Q0[idx]
        boot_pol_keys = rng.choice(pol_eval_banks, size=len(pol_eval_banks), replace=True)
        boot_abs_keys = rng.choice(abs_banks, size=len(abs_banks), replace=True)
        pol_r_b = [rcca(polarities[k][idx], q0_b) for k in boot_pol_keys]
        abs_r_b = [rcca(polarities[k][idx], q0_b) for k in boot_abs_keys]
        cluster_boot_deltas.append(np.mean(pol_r_b) - np.mean(abs_r_b))
    cluster_boot_deltas = np.array(cluster_boot_deltas)
    p_cluster = float((1 + np.sum(cluster_boot_deltas <= 0)) / (B + 1))
    print(f"    Done in {time.time() - t0:.2f}s! Mean Delta: {np.mean(cluster_boot_deltas):.4f}, 95% CI: [{np.percentile(cluster_boot_deltas, 2.5):.4f}, {np.percentile(cluster_boot_deltas, 97.5):.4f}], p = {p_cluster:.4f}", flush=True)

    # Level 4: Exact Bank Label Permutation Test (10,000 draws)
    print("  [LEVEL 4] Running Bank Label Permutation Test (10,000 draws)...", flush=True)
    t0 = time.time()
    all_eval_r = np.concatenate([pol_eval_r_fixed, abs_r_fixed])
    n_p = len(pol_eval_r_fixed)
    n_tot = len(all_eval_r)
    perm_deltas = []
    for _ in range(10000):
        perm_r = rng.permutation(all_eval_r)
        delta_perm = np.mean(perm_r[:n_p]) - np.mean(perm_r[n_p:])
        perm_deltas.append(delta_perm)
    perm_deltas = np.array(perm_deltas)
    p_perm = float((1 + np.sum(perm_deltas >= obs_delta_cca - 1e-12)) / (10000 + 1))
    print(f"    Done in {time.time() - t0:.2f}s! Null Mean: {np.mean(perm_deltas):.4f}, Null 95% CI: [{np.percentile(perm_deltas, 2.5):.4f}, {np.percentile(perm_deltas, 97.5):.4f}], p = {p_perm:.4f}", flush=True)

    inference_summary = {
        "observed_political_cca_mean": obs_pol_cca,
        "observed_absurd_cca_mean": obs_abs_cca,
        "observed_delta_cca": obs_delta_cca,
        "article_bootstrap": {
            "mean_delta": float(np.mean(art_boot_deltas)),
            "std_delta": float(np.std(art_boot_deltas)),
            "ci_95": np.percentile(art_boot_deltas, [2.5, 97.5]).tolist(),
            "p_value_one_sided": p_art,
            "interpretation": "Articles resampled with replacement; tests whether article sampling variance explains the gap.",
        },
        "bank_bootstrap": {
            "mean_delta": float(np.mean(bank_boot_deltas)),
            "std_delta": float(np.std(bank_boot_deltas)),
            "ci_95": np.percentile(bank_boot_deltas, [2.5, 97.5]).tolist(),
            "p_value_one_sided": p_bank,
            "interpretation": "Banks resampled with replacement; tests whether bank wording selection explains the gap.",
        },
        "cluster_bootstrap": {
            "mean_delta": float(np.mean(cluster_boot_deltas)),
            "std_delta": float(np.std(cluster_boot_deltas)),
            "ci_95": np.percentile(cluster_boot_deltas, [2.5, 97.5]).tolist(),
            "p_value_one_sided": p_cluster,
            "interpretation": "Joint resampling of articles and banks; accounts for both sources of variance.",
        },
        "permutation_test": {
            "n_permutations": 10000,
            "null_mean": float(np.mean(perm_deltas)),
            "null_std": float(np.std(perm_deltas)),
            "null_95_ci": np.percentile(perm_deltas, [2.5, 97.5]).tolist(),
            "p_value_exact_perm": p_perm,
            "interpretation": "Exact label permutation across political and absurd banks.",
        },
    }

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "individual_banks": bank_results,
        "category_summaries": cat_summaries,
        "inference_on_cca_gap": inference_summary,
    }

    out_file = ROOT / "paper" / "kuhnian_expanded_bank_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[SAVED] Saved expanded bank results to {out_file}", flush=True)


if __name__ == "__main__":
    run_expanded_bank_distribution_and_inference()
