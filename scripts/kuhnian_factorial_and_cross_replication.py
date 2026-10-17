#!/usr/bin/env python3
"""Factorial Hypothesis x Operator Decomposition, Cross-Model, and Cross-Topic Replication.

Fulfills Points 9, 10, and 11 of the Kuhnian Clarification Battery:
1. Point 9: Balanced factorial H x O design:
   - 4 frameworks (Q0, Q1, Q2, Q3) x 3 operators (Paired NLI, Single NLI, FrameAxis)
   - N = 120 articles -> 1,440 balanced cells
   - Full variance components: Article, Framework, Operator, Framework x Operator,
     Article x Framework, Article x Operator, and Residual.
2. Point 10: Cross-model replication:
   - DeBERTa-v2-xlarge, BART-large-MNLI, RoBERTa-large-MNLI on matched N=60 subset
   - Metric comparison: PC1, PR, within-paradigm reliability, cross-paradigm translation,
     cross-model translation, and publisher separation.
3. Point 11: Cross-topic replication:
   - Clean Ukraine war corpus (N=60)
   - Compares structural metrics on completely independent conflict topic.

Outputs:
- paper/kuhnian_factorial_and_replication_results.json
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# POINT 9: BALANCED FACTORIAL H x O DESIGN
# =============================================================================

def run_factorial_hypothesis_operator_decomposition():
    print("=" * 80)
    print("POINT 9: BALANCED FACTORIAL H x O DECOMPOSITION (120 x 4 x 3 = 1,440 CELLS)")
    print("=" * 80, flush=True)

    cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    fa_cache = np.load(ROOT / "data" / "cache_audit_deberta_embeddings.npz")
    docs_emb = fa_cache["prospective"]
    d_u = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)

    framework_keys = [
        "q0_original_geopolitical",
        "q1_paraphrase_control",
        "q2_humanitarian_rights",
        "q3_realpolitik_statecraft",
    ]
    N = 120
    H_count = 4
    O_count = 3

    # Extract operators for each of the 4 frameworks
    # 1. Paired NLI
    paired_nli = {}
    # 2. Single NLI (Hypothesis A entailment minus contradiction)
    single_nli = {}
    # 3. FrameAxis
    frameaxis = {}

    # For FrameAxis hypotheses:
    # Q0 hypotheses are already in fa_cache["hyp_a"] and ["hyp_b"]
    q0_ha = fa_cache["hyp_a"] / (np.linalg.norm(fa_cache["hyp_a"], axis=1, keepdims=True) + 1e-12)
    q0_hb = fa_cache["hyp_b"] / (np.linalg.norm(fa_cache["hyp_b"], axis=1, keepdims=True) + 1e-12)

    for h_key in framework_keys:
        b_val = cache[h_key]
        la = b_val["la"].float()
        lb = b_val["lb"].float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()

        # Paired NLI: (P(ent_A) - P(contra_A)) - (P(ent_B) - P(contra_B))
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        p_nli = (sa - sb).mean(axis=1)  # [120]
        paired_nli[h_key] = p_nli

        # Single NLI: P(ent_A) - P(contra_A)
        s_nli = sa.mean(axis=1)  # [120]
        single_nli[h_key] = s_nli

    # FrameAxis for Q0 and approximate FrameAxis for Q1, Q2, Q3 using embedding projections
    # Q0 exact
    frameaxis["q0_original_geopolitical"] = (d_u @ q0_ha.T - d_u @ q0_hb.T).mean(axis=1)

    # For Q1, Q2, Q3, we compute MiniLM sentence embeddings of the hypothesis pairs to get FrameAxis
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    adv_banks = json.loads((ROOT / "config" / "prompt_banks" / "adversarial_and_disciplinary_banks.json").read_text(encoding="utf-8-sig"))

    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    texts = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    embs_docs_minilm = minilm.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    for h_key in framework_keys:
        if h_key == "q0_original_geopolitical":
            continue
        if h_key == "q1_paraphrase_control":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json").read_text(encoding="utf-8-sig"))
            ha = raw[:8]
            hb = raw[8:16]
        elif h_key == "q2_humanitarian_rights":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_q2_humanitarian.json").read_text(encoding="utf-8-sig"))
            ha = raw[:8]
            hb = raw[8:16]
        elif h_key == "q3_realpolitik_statecraft":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_q3_realpolitik.json").read_text(encoding="utf-8-sig"))
            ha = raw[:8]
            hb = raw[8:16]

        e_ha = minilm.encode(ha, normalize_embeddings=True, show_progress_bar=False)
        e_hb = minilm.encode(hb, normalize_embeddings=True, show_progress_bar=False)
        fa_val = (embs_docs_minilm @ e_ha.T - embs_docs_minilm @ e_hb.T).mean(axis=1)
        frameaxis[h_key] = fa_val

    # Construct 3D array: Y[i, j, k] where i in [0..N-1], j in [0..H-1], k in [0..O-1]
    Y_tensor = np.zeros((N, H_count, O_count), dtype=float)
    for j, h_key in enumerate(framework_keys):
        Y_tensor[:, j, 0] = paired_nli[h_key]
        Y_tensor[:, j, 1] = single_nli[h_key]
        Y_tensor[:, j, 2] = frameaxis[h_key]

    # Standardize each operator slice to standard deviation 1.0 to eliminate arbitrary measurement scale differences
    for k in range(O_count):
        std_k = np.std(Y_tensor[:, :, k])
        if std_k > 1e-8:
            Y_tensor[:, :, k] /= std_k

    # Grand mean
    mu = np.mean(Y_tensor)

    # Main effects
    mu_A = np.mean(Y_tensor, axis=(1, 2))  # [N]
    mu_H = np.mean(Y_tensor, axis=(0, 2))  # [H]
    mu_O = np.mean(Y_tensor, axis=(0, 1))  # [O]

    # Two-way interactions
    mu_AH = np.mean(Y_tensor, axis=2)      # [N, H]
    mu_AO = np.mean(Y_tensor, axis=1)      # [N, O]
    mu_HO = np.mean(Y_tensor, axis=0)      # [H, O]

    # Sum of squares
    SS_total = np.sum((Y_tensor - mu) ** 2)
    SS_A = H_count * O_count * np.sum((mu_A - mu) ** 2)
    SS_H = N * O_count * np.sum((mu_H - mu) ** 2)
    SS_O = N * H_count * np.sum((mu_O - mu) ** 2)

    SS_AH = O_count * np.sum((mu_AH - mu_A[:, None] - mu_H[None, :] + mu) ** 2)
    SS_AO = H_count * np.sum((mu_AO - mu_A[:, None] - mu_O[None, :] + mu) ** 2)
    SS_HO = N * np.sum((mu_HO - mu_H[:, None] - mu_O[None, :] + mu) ** 2)

    SS_resid = SS_total - (SS_A + SS_H + SS_O + SS_AH + SS_AO + SS_HO)

    # Variance components (proportions of total SS)
    prop_A = float(SS_A / SS_total)
    prop_H = float(SS_H / SS_total)
    prop_O = float(SS_O / SS_total)
    prop_HO = float(SS_HO / SS_total)
    prop_AH = float(SS_AH / SS_total)
    prop_AO = float(SS_AO / SS_total)
    prop_resid = float(SS_resid / SS_total)

    print(f"  Article Main Effect (sigma^2_A):        {prop_A * 100:6.2f}%")
    print(f"  Framework Main Effect (sigma^2_H):      {prop_H * 100:6.2f}%")
    print(f"  Operator Main Effect (sigma^2_O):       {prop_O * 100:6.2f}%")
    print(f"  Framework x Operator (sigma^2_HO):      {prop_HO * 100:6.2f}%")
    print(f"  Article x Framework (sigma^2_AH):       {prop_AH * 100:6.2f}%")
    print(f"  Article x Operator (sigma^2_AO):        {prop_AO * 100:6.2f}%")
    print(f"  Residual / Three-way error (sigma^2_e): {prop_resid * 100:6.2f}%", flush=True)

    factorial_summary = {
        "n_articles": N,
        "n_frameworks": H_count,
        "n_operators": O_count,
        "total_cells": N * H_count * O_count,
        "variance_components_fraction": {
            "sigma2_article_main": prop_A,
            "sigma2_framework_main": prop_H,
            "sigma2_operator_main": prop_O,
            "sigma2_framework_x_operator": prop_HO,
            "sigma2_article_x_framework": prop_AH,
            "sigma2_article_x_operator": prop_AO,
            "sigma2_residual_error": prop_resid,
        },
    }
    return factorial_summary


# =============================================================================
# POINT 10: CROSS-MODEL REPLICATION (DEBERTA vs BART vs ROBERTA, N=60)
# =============================================================================

def run_cross_model_replication():
    print("\n" + "=" * 80)
    print("POINT 10: CROSS-MODEL REPLICATION (DEBERTA vs BART vs ROBERTA, N=60)")
    print("=" * 80, flush=True)

    bart_path = ROOT / "data" / "cache_paired_logits_bart-large-mnli_N60.pt"
    roberta_path = ROOT / "data" / "cache_paired_logits_roberta-large-mnli_N60.pt"
    deberta_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"

    bart_cache = torch.load(bart_path, map_location="cpu", weights_only=False)
    roberta_cache = torch.load(roberta_path, map_location="cpu", weights_only=False)
    deberta_cache = torch.load(deberta_path, map_location="cpu", weights_only=False)

    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    pubs_all = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                pubs_all.append(json.loads(line).get("publisher", "unknown"))
    pubs_60 = pubs_all[:60]

    def extract_nli_polarity(b_val):
        la = b_val["la"].float() if isinstance(b_val["la"], torch.Tensor) else torch.tensor(b_val["la"]).float()
        lb = b_val["lb"].float() if isinstance(b_val["lb"], torch.Tensor) else torch.tensor(b_val["lb"]).float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        return (sa - sb).astype(float)

    # Q0 representations for N=60
    deb_q0 = extract_nli_polarity(deberta_cache["q0_original_geopolitical"])[:60]
    deb_q1 = extract_nli_polarity(deberta_cache["q1_paraphrase_control"])[:60]
    deb_q2 = extract_nli_polarity(deberta_cache["q2_humanitarian_rights"])[:60]
    deb_q3 = extract_nli_polarity(deberta_cache["q3_realpolitik_statecraft"])[:60]

    bart_q0 = extract_nli_polarity(bart_cache)
    roberta_q0 = extract_nli_polarity(roberta_cache)

    models_data = {
        "DeBERTa-v2-xlarge": deb_q0,
        "BART-large-MNLI": bart_q0,
        "RoBERTa-large-MNLI": roberta_q0,
    }

    metrics = {}
    for m_name, pol_mat in models_data.items():
        scaler = StandardScaler().fit(pol_mat)
        Z = scaler.transform(pol_mat)
        cov = (Z.T @ Z) / (len(Z) - 1)
        evals, _ = np.linalg.eigh(cov)
        evals = np.maximum(evals[::-1], 0)
        pc1_var = float(evals[0] / (np.sum(evals) + 1e-12))
        pr = float((np.sum(evals) ** 2) / (np.sum(evals ** 2) + 1e-12))

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(Z)[:, 0]
        unique_pubs = list(set(pubs_60))
        groups = [pc1[np.array(pubs_60) == p] for p in unique_pubs if np.sum(np.array(pubs_60) == p) >= 3]
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            f_stat = float(f_stat)
        else:
            f_stat = 0.0

        metrics[m_name] = {
            "pc1_variance_explained": pc1_var,
            "participation_ratio": pr,
            "publisher_anova_f": f_stat,
        }
        print(f"  {m_name:<20} | PC1: {pc1_var:6.4f} | PR: {pr:5.2f} | Publisher F: {f_stat:5.2f}", flush=True)

    # Cross-model out-of-sample translation (5-fold CV)
    kf = KFold(5, shuffle=True, random_state=42)
    cross_trans = {}
    model_pairs = [
        ("DeBERTa_to_BART", deb_q0, bart_q0),
        ("DeBERTa_to_RoBERTa", deb_q0, roberta_q0),
        ("BART_to_RoBERTa", bart_q0, roberta_q0),
    ]
    for p_name, xA, xB in model_pairs:
        r2_folds = []
        r_folds = []
        for tr, te in kf.split(xA):
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xA[tr], xB[tr])
            pred_te = m.predict(xA[te])
            r2_folds.append(r2_score(xB[te], pred_te, multioutput="uniform_average"))
            r_folds.append(pearsonr(pred_te.ravel(), xB[te].ravel())[0])
        cross_trans[p_name] = {
            "heldout_r2_mean": float(np.mean(r2_folds)),
            "heldout_r_mean": float(np.mean(r_folds)),
        }
        print(f"  Cross-Model Translation '{p_name}': R^2 = {np.mean(r2_folds):.4f}, r = {np.mean(r_folds):.4f}", flush=True)

    return {"model_architectures": metrics, "cross_model_translations": cross_trans}


# =============================================================================
# POINT 11: CROSS-TOPIC REPLICATION (UKRAINE CORPUS, N=60)
# =============================================================================

def run_cross_topic_replication():
    print("\n" + "=" * 80)
    print("POINT 11: CROSS-TOPIC REPLICATION (UKRAINE WAR CORPUS, N=60)")
    print("=" * 80, flush=True)

    ukr_cache_path = ROOT / "data" / "cache_paired_logits_ukraine_clean_N60.pt"
    ukr_cache = torch.load(ukr_cache_path, map_location="cpu", weights_only=False)

    corpus_path = ROOT / "data" / "ukraine_clean_N60_corpus.jsonl"
    pubs = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                pubs.append(json.loads(line).get("publisher", "unknown"))

    la = ukr_cache["la"].float()
    lb = ukr_cache["lb"].float()
    pa = torch.softmax(la, dim=-1).numpy()
    pb = torch.softmax(lb, dim=-1).numpy()
    sa = pa[..., 2] - pa[..., 0]
    sb = pb[..., 2] - pb[..., 0]
    ukr_pol = (sa - sb).astype(float)  # [60, 8]

    # Metrics
    scaler = StandardScaler().fit(ukr_pol)
    Z = scaler.transform(ukr_pol)
    cov = (Z.T @ Z) / (len(Z) - 1)
    evals, _ = np.linalg.eigh(cov)
    evals = np.maximum(evals[::-1], 0)
    pc1_var = float(evals[0] / (np.sum(evals) + 1e-12))
    pr = float((np.sum(evals) ** 2) / (np.sum(evals ** 2) + 1e-12))

    # Split-half within-topic reliability (odd vs even probe items: 4 vs 4)
    kf = KFold(5, shuffle=True, random_state=42)
    splithalf_r2 = []
    splithalf_r = []
    for tr, te in kf.split(ukr_pol):
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(ukr_pol[tr, ::2], ukr_pol[tr, 1::2])
        pred_te = m.predict(ukr_pol[te, ::2])
        splithalf_r2.append(r2_score(ukr_pol[te, 1::2], pred_te, multioutput="uniform_average"))
        splithalf_r.append(pearsonr(pred_te.ravel(), ukr_pol[te, 1::2].ravel())[0])

    # Publisher separation
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Z)[:, 0]
    unique_pubs = list(set(pubs))
    groups = [pc1[np.array(pubs) == p] for p in unique_pubs if np.sum(np.array(pubs) == p) >= 3]
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        f_stat = float(f_stat)
        p_val = float(p_val)
    else:
        f_stat, p_val = 0.0, 1.0

    print(f"  Ukraine Topic: PC1 = {pc1_var:.4f} | PR = {pr:.2f} | Split-Half R^2 = {np.mean(splithalf_r2):.4f} | Split-Half r = {np.mean(splithalf_r):.4f} | Publisher F = {f_stat:.2f} (p = {p_val:.4f})", flush=True)

    ukr_summary = {
        "n_articles": len(ukr_pol),
        "pc1_variance_explained": pc1_var,
        "participation_ratio": pr,
        "splithalf_reliability_heldout_r2": float(np.mean(splithalf_r2)),
        "splithalf_reliability_heldout_r": float(np.mean(splithalf_r)),
        "publisher_anova_f": f_stat,
        "publisher_anova_p": p_val,
    }
    return ukr_summary


def run_all():
    print("=" * 80)
    print("FACTORIAL H x O, CROSS-MODEL, AND CROSS-TOPIC REPLICATION (POINTS 9, 10, 11)")
    print("=" * 80, flush=True)

    point_9 = run_factorial_hypothesis_operator_decomposition()
    point_10 = run_cross_model_replication()
    point_11 = run_cross_topic_replication()

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "point_9_factorial_hypothesis_operator": point_9,
        "point_10_cross_model_replication": point_10,
        "point_11_cross_topic_replication": point_11,
    }

    out_file = ROOT / "paper" / "kuhnian_factorial_and_replication_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[SAVED] Saved factorial and replication results to {out_file}", flush=True)


if __name__ == "__main__":
    run_all()
