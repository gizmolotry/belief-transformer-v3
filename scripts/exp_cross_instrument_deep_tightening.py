"""
scratch/test_scrutiny_ablations.py

Comprehensive execution script for:
1. Anisotropy-removal FrameAxis ablations (centering, top PC removal, whitening)
2. VADER sentiment regression and residual publisher separation ANOVA
3. Permutation CCA and Cross-Validated (out-of-sample) CCA
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_shingles(text: str, k: int = 5) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < k:
        return set([' '.join(words)])
    return set(' '.join(words[i:i+k]) for i in range(len(words) - k + 1))

def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

class SentenceTransformerSuite:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        
    def extract_frameaxis_matrices(
        self,
        texts: List[str],
        hyp_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        e_x = self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        ha_list = [p[0] for p in hyp_pairs]
        hb_list = [p[1] for p in hyp_pairs]
        e_ha = self.model.encode(ha_list, normalize_embeddings=True, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        e_hb = self.model.encode(hb_list, normalize_embeddings=True, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        
        cos_a = torch.matmul(e_x, e_ha.T)
        cos_b = torch.matmul(e_x, e_hb.T)
        fa_diff = (cos_a - cos_b).cpu().numpy()
        
        v_k = e_ha - e_hb
        v_norm = F.normalize(v_k, p=2, dim=1)
        fa_proj = torch.matmul(e_x, v_norm.T).cpu().numpy()
        return fa_diff, fa_proj, e_x

def extract_article_text(record: Dict[str, Any]) -> str:
    candidates = ["content", "text", "body", "article", "raw_content"]
    for k in candidates:
        v = record.get(k)
        if v and isinstance(v, str) and len(v.strip()) > 50:
            return v.strip()
    raise ValueError(f"Could not extract valid text from record: {list(record.keys())}")

def compute_pca(matrix: np.ndarray, standardize: bool = True):
    mean = matrix.mean(axis=0, keepdims=True)
    if standardize:
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        transformed = (matrix - mean) / std
    else:
        transformed = matrix - mean
    cov = np.cov(transformed, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 1e-12)
    eigvecs = eigvecs[:, idx]
    scores = transformed @ eigvecs
    evr = eigvals / eigvals.sum()
    pr = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2)))
    return scores, evr, eigvecs, pr

def compute_eta2(scores: np.ndarray, groups: list) -> float:
    grand_mean = np.mean(scores)
    ss_tot = np.sum((scores - grand_mean) ** 2)
    ss_bet = 0.0
    for pub in set(groups):
        idxs = [i for i, g in enumerate(groups) if g == pub]
        ss_bet += len(idxs) * (np.mean(scores[idxs]) - grand_mean) ** 2
    return float(ss_bet / ss_tot) if ss_tot > 0 else 0.0

# -----------------------------------------------------------------------------
# 1. Feature Extractor
# -----------------------------------------------------------------------------
class FeatureExtractorSuite:
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli"):
        print(f"Loading tokenizer & models for: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
        self.encoder = self.nli_model.deberta if hasattr(self.nli_model, "deberta") else self.nli_model.base_model
        
    @torch.no_grad()
    def encode_sentences(self, texts: List[str], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = 16
        normed_list = []
        raw_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = self.encoder(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            raw_list.append(mean_pooled)
            normed = F.normalize(mean_pooled, p=2, dim=1)
            normed_list.append(normed)
        return torch.cat(normed_list, dim=0), torch.cat(raw_list, dim=0)

def main():
    extractor = FeatureExtractorSuite("microsoft/deberta-v2-xlarge-mnli")
    
    # Load canonical Q0 pairs from probe_hypotheses.json
    q0_raw = json.load(open("probe_hypotheses.json", encoding="utf-8"))
    q0_pairs = [(q0_raw[2*i], q0_raw[2*i+1]) for i in range(8)]
    
    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print("\n--- Loading Corpora ---")
    # Prospective N=120
    prosp_path = "data/prospective_confirmatory_120_corpus.jsonl"
    prosp_records = [json.loads(line) for line in open(prosp_path, encoding="utf-8")]
    prosp_texts = [extract_article_text(r) for r in prosp_records]
    prosp_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in prosp_records]
    
    nli_cache = torch.load("data/cache_prospective_120_q0_q1_logits.pt")
    la0, lb0 = nli_cache["q0_la"].float(), nli_cache["q0_lb"].float()
    nli_prosp_m = ((torch.softmax(la0, dim=-1)[..., 2] - torch.softmax(la0, dim=-1)[..., 0]) - 
                   (torch.softmax(lb0, dim=-1)[..., 2] - torch.softmax(lb0, dim=-1)[..., 0])).numpy()
    s_nli_prosp, evr_nli_prosp, _, _ = compute_pca(nli_prosp_m, standardize=True)
    pc1_nli_prosp = s_nli_prosp[:, 0]
    
    # Controlled Balanced N=60
    ctrl_path = "data/controlled_N60_corpus.jsonl"
    ctrl_records = [json.loads(line) for line in open(ctrl_path, encoding="utf-8")]
    ctrl_texts = [extract_article_text(r) for r in ctrl_records]
    ctrl_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in ctrl_records]
    
    ctrl_cache = torch.load("data/cache_paired_logits_deberta-v2-xlarge-mnli_N60.pt", weights_only=False)
    la_c = torch.tensor(ctrl_cache["la"]).float()
    lb_c = torch.tensor(ctrl_cache["lb"]).float()
    nli_ctrl_m = ((torch.softmax(la_c, dim=-1)[..., 2] - torch.softmax(la_c, dim=-1)[..., 0]) - 
                  (torch.softmax(lb_c, dim=-1)[..., 2] - torch.softmax(lb_c, dim=-1)[..., 0])).numpy()
    s_nli_ctrl, evr_nli_ctrl, _, _ = compute_pca(nli_ctrl_m, standardize=True)
    pc1_nli_ctrl = s_nli_ctrl[:, 0]

    # Clean Ukraine N=60
    ukr_path = "data/ukraine_clean_N60_corpus.jsonl"
    ukr_records = [json.loads(line) for line in open(ukr_path, encoding="utf-8")]
    ukr_texts = [extract_article_text(r) for r in ukr_records]
    ukr_cache = torch.load("data/cache_paired_logits_ukraine_clean_N60.pt", weights_only=False)
    la_u = torch.tensor(ukr_cache["la"]).float()
    lb_u = torch.tensor(ukr_cache["lb"]).float()
    nli_ukr_m = ((torch.softmax(la_u, dim=-1)[..., 2] - torch.softmax(la_u, dim=-1)[..., 0]) - 
                 (torch.softmax(lb_u, dim=-1)[..., 2] - torch.softmax(lb_u, dim=-1)[..., 0])).numpy()
    s_nli_ukr, _, _, _ = compute_pca(nli_ukr_m, standardize=True)
    pc1_nli_ukr = s_nli_ukr[:, 0]

    # Clean N=83 subset
    import glob
    prior_files = [f for f in glob.glob("data/*.jsonl") if "prospective" not in f and "real_corpus" not in f]
    prior_shingles = []
    for pf in prior_files:
        for line in open(pf, encoding="utf-8"):
            try:
                d = json.loads(line)
                txt = d.get("text") or d.get("content") or ""
                if txt and len(txt) > 50:
                    prior_shingles.append(get_shingles(txt))
            except:
                pass
    prosp_shingles = [get_shingles(t) for t in prosp_texts]
    clean_83_idx = []
    for i, ps in enumerate(prosp_shingles):
        max_j = max((jaccard(ps, pjs) for pjs in prior_shingles), default=0.0)
        if max_j < 0.50:
            clean_83_idx.append(i)
    clean_83_idx = np.array(clean_83_idx)

    # Encode texts
    print("Encoding texts with DeBERTa...")
    e_prosp, raw_prosp = extractor.encode_sentences(prosp_texts)
    e_ctrl, raw_ctrl = extractor.encode_sentences(ctrl_texts)
    
    # Encode hypotheses
    ha_list = [p[0] for p in q0_pairs]
    hb_list = [p[1] for p in q0_pairs]
    e_ha, raw_ha = extractor.encode_sentences(ha_list)
    e_hb, raw_hb = extractor.encode_sentences(hb_list)

    # =========================================================================
    # PART 1: ANISOTROPY-REMOVAL FRAMEAXIS ABLATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: ANISOTROPY-REMOVAL ABLATIONS ON DEBERTA FRAMEAXIS")
    print("=" * 80)

    def run_frameaxis_pipeline(
        e_docs: torch.Tensor,
        e_ha_p: torch.Tensor,
        e_hb_p: torch.Tensor,
        label: str,
        nli_pc1: np.ndarray,
        raw_docs: torch.Tensor,
        pubs: list = None
    ) -> Dict[str, Any]:
        cos_a = torch.matmul(e_docs, e_ha_p.T)
        cos_b = torch.matmul(e_docs, e_hb_p.T)
        fa_diff = (cos_a - cos_b).cpu().numpy()
        
        # PCA
        scores, evr, eigvecs, pr = compute_pca(fa_diff, standardize=True)
        pc1 = scores[:, 0]
        pc2 = scores[:, 1]
        
        # Orient signs so positive correlates with NLI if positive
        r_nli, p_nli = pearsonr(pc1, nli_pc1)
        r_orig = r_nli
        if r_nli < 0:
            pc1 = -pc1
            r_nli = -r_nli
        rho_nli, sp_nli = spearmanr(pc1, nli_pc1)

        # Norm correlation
        norms = torch.norm(raw_docs, p=2, dim=1).cpu().numpy()
        r_norm, p_norm = pearsonr(pc1, norms)
        
        # Mean embedding cosine correlation
        mean_emb = F.normalize(raw_docs.mean(dim=0, keepdim=True), p=2, dim=1)
        cos_mean = torch.matmul(F.normalize(raw_docs, p=2, dim=1), mean_emb.T).squeeze(-1).cpu().numpy()
        r_mean, p_mean = pearsonr(pc1, cos_mean)

        res = {
            "label": label,
            "pc1_evr": float(evr[0]),
            "pc2_evr": float(evr[1]),
            "participation_ratio": float(pr),
            "r_with_nli_pc1": float(r_orig),
            "rho_with_nli_pc1": float(rho_nli),
            "r_with_raw_norm": float(r_norm),
            "p_with_raw_norm": float(p_norm),
            "r_with_mean_cos": float(r_mean),
            "p_with_mean_cos": float(p_mean),
            "pc1_scores": pc1,
            "fa_diff": fa_diff
        }
        
        if pubs is not None:
            # ANOVA
            pub_counts = {}
            for p in pubs:
                pub_counts[p] = pub_counts.get(p, 0) + 1
            if len(set(pub_counts.values())) == 1 and list(pub_counts.values())[0] == 5:
                # Controlled balanced
                groups = [pc1[[i for i, p in enumerate(pubs) if p == pub]] for pub in sorted(list(set(pubs)))]
                f_stat, p_val = f_oneway(*groups)
                eta2 = compute_eta2(pc1, pubs)
                res["anova_f"] = float(f_stat)
                res["anova_p"] = float(p_val)
                res["eta2"] = float(eta2)
            else:
                # Frequent pubs >= 2
                frequent_pubs = [p for p, c in pub_counts.items() if c >= 2]
                freq_indices = [i for i, p in enumerate(pubs) if p in frequent_pubs]
                freq_pubs_list = [pubs[i] for i in freq_indices]
                groups = [pc1[[i for i, p in enumerate(freq_pubs_list) if p == pub]] for pub in frequent_pubs]
                f_stat, p_val = f_oneway(*groups)
                eta2 = compute_eta2(pc1[freq_indices], freq_pubs_list)
                res["anova_f"] = float(f_stat)
                res["anova_p"] = float(p_val)
                res["eta2"] = float(eta2)
        return res

    # 1. Baseline: Raw L2-normalized
    base_prosp = run_frameaxis_pipeline(e_prosp, e_ha, e_hb, "Baseline (Raw Normalized)", pc1_nli_prosp, raw_prosp, prosp_pubs)
    base_pc1_prosp = base_prosp["pc1_scores"]

    # Transformation functions on raw embeddings:
    # A. Centering (Subtract corpus mean)
    mean_vec = raw_prosp.mean(dim=0, keepdim=True)
    raw_centered = raw_prosp - mean_vec
    e_centered = F.normalize(raw_centered, p=2, dim=1)
    e_ha_c = F.normalize(raw_ha - mean_vec, p=2, dim=1)
    e_hb_c = F.normalize(raw_hb - mean_vec, p=2, dim=1)
    res_center = run_frameaxis_pipeline(e_centered, e_ha_c, e_hb_c, "Centering (Mean Subtraction)", pc1_nli_prosp, raw_prosp, prosp_pubs)

    # B. Top-1 PC Removal (Mu & Viswanath 2018 / Arora et al. 2017)
    # SVD on centered embeddings
    U, S, V = torch.pca_lowrank(raw_centered, q=10, center=False)
    # V[:, 0] is top PC direction
    top1 = V[:, 0:1]  # [d, 1]
    raw_no_top1 = raw_centered - torch.matmul(raw_centered, top1) @ top1.T
    e_no_top1 = F.normalize(raw_no_top1, p=2, dim=1)
    ha_no_top1 = F.normalize((raw_ha - mean_vec) - torch.matmul(raw_ha - mean_vec, top1) @ top1.T, p=2, dim=1)
    hb_no_top1 = F.normalize((raw_hb - mean_vec) - torch.matmul(raw_hb - mean_vec, top1) @ top1.T, p=2, dim=1)
    res_top1 = run_frameaxis_pipeline(e_no_top1, ha_no_top1, hb_no_top1, "Top-1 PC Removal", pc1_nli_prosp, raw_prosp, prosp_pubs)

    # C. Top-2 PC Removal
    top2 = V[:, 0:2]
    raw_no_top2 = raw_centered - torch.matmul(raw_centered, top2) @ top2.T
    e_no_top2 = F.normalize(raw_no_top2, p=2, dim=1)
    ha_no_top2 = F.normalize((raw_ha - mean_vec) - torch.matmul(raw_ha - mean_vec, top2) @ top2.T, p=2, dim=1)
    hb_no_top2 = F.normalize((raw_hb - mean_vec) - torch.matmul(raw_hb - mean_vec, top2) @ top2.T, p=2, dim=1)
    res_top2 = run_frameaxis_pipeline(e_no_top2, ha_no_top2, hb_no_top2, "Top-2 PC Removal", pc1_nli_prosp, raw_prosp, prosp_pubs)

    # D. Top-3 PC Removal
    top3 = V[:, 0:3]
    raw_no_top3 = raw_centered - torch.matmul(raw_centered, top3) @ top3.T
    e_no_top3 = F.normalize(raw_no_top3, p=2, dim=1)
    ha_no_top3 = F.normalize((raw_ha - mean_vec) - torch.matmul(raw_ha - mean_vec, top3) @ top3.T, p=2, dim=1)
    hb_no_top3 = F.normalize((raw_hb - mean_vec) - torch.matmul(raw_hb - mean_vec, top3) @ top3.T, p=2, dim=1)
    res_top3 = run_frameaxis_pipeline(e_no_top3, ha_no_top3, hb_no_top3, "Top-3 PC Removal", pc1_nli_prosp, raw_prosp, prosp_pubs)

    # E. ZCA Whitening on raw_centered
    cov = (raw_centered.T @ raw_centered) / (raw_centered.size(0) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    eps = 1e-5
    inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
    raw_zca = raw_centered @ inv_sqrt
    e_zca = F.normalize(raw_zca, p=2, dim=1)
    ha_zca = F.normalize((raw_ha - mean_vec) @ inv_sqrt, p=2, dim=1)
    hb_zca = F.normalize((raw_hb - mean_vec) @ inv_sqrt, p=2, dim=1)
    res_zca = run_frameaxis_pipeline(e_zca, ha_zca, hb_zca, "ZCA Whitening", pc1_nli_prosp, raw_prosp, prosp_pubs)

    ablations = [base_prosp, res_center, res_top1, res_top2, res_top3, res_zca]
    print("\nPROSPECTIVE N=120 ABLATION TABLE:")
    print(f"{'Condition':<28} | {'PC1 EVR':<9} | {'PC2 EVR':<9} | {'PR':<6} | {'r(Orig FA)':<10} | {'r(NLI)':<8} | {'r(Norm)':<9} | {'r(MeanCos)':<10}")
    print("-" * 105)
    for ab in ablations:
        r_orig_fa, _ = pearsonr(ab["pc1_scores"], base_pc1_prosp)
        print(f"{ab['label']:<28} | {ab['pc1_evr']*100:>7.2f}% | {ab['pc2_evr']*100:>7.2f}% | {ab['participation_ratio']:>6.3f} | {r_orig_fa:>10.4f} | {ab['r_with_nli_pc1']:>8.4f} | {ab['r_with_raw_norm']:>9.4f} | {ab['r_with_mean_cos']:>10.4f}")

    # Now also run ablations on CONTROLLED BALANCED N=60 to check publisher separation!
    print("\nCONTROLLED BALANCED N=60 ABLATION (PUBLISHER SEPARATION):")
    base_ctrl = run_frameaxis_pipeline(e_ctrl, e_ha, e_hb, "Baseline (Raw Normalized)", pc1_nli_ctrl, raw_ctrl, ctrl_pubs)
    mean_ctrl = raw_ctrl.mean(dim=0, keepdim=True)
    raw_c_ctrl = raw_ctrl - mean_ctrl
    U_c, S_c, V_c = torch.pca_lowrank(raw_c_ctrl, q=10, center=False)
    
    # Centering
    res_c_center = run_frameaxis_pipeline(F.normalize(raw_c_ctrl, p=2, dim=1), 
                                          F.normalize(raw_ha - mean_ctrl, p=2, dim=1), 
                                          F.normalize(raw_hb - mean_ctrl, p=2, dim=1), 
                                          "Centering", pc1_nli_ctrl, raw_ctrl, ctrl_pubs)
    # Top-1 removal
    t1_c = V_c[:, 0:1]
    res_c_top1 = run_frameaxis_pipeline(F.normalize(raw_c_ctrl - (raw_c_ctrl @ t1_c) @ t1_c.T, p=2, dim=1),
                                        F.normalize((raw_ha - mean_ctrl) - ((raw_ha - mean_ctrl) @ t1_c) @ t1_c.T, p=2, dim=1),
                                        F.normalize((raw_hb - mean_ctrl) - ((raw_hb - mean_ctrl) @ t1_c) @ t1_c.T, p=2, dim=1),
                                        "Top-1 PC Removal", pc1_nli_ctrl, raw_ctrl, ctrl_pubs)
    # Top-2 removal
    t2_c = V_c[:, 0:2]
    res_c_top2 = run_frameaxis_pipeline(F.normalize(raw_c_ctrl - (raw_c_ctrl @ t2_c) @ t2_c.T, p=2, dim=1),
                                        F.normalize((raw_ha - mean_ctrl) - ((raw_ha - mean_ctrl) @ t2_c) @ t2_c.T, p=2, dim=1),
                                        F.normalize((raw_hb - mean_ctrl) - ((raw_hb - mean_ctrl) @ t2_c) @ t2_c.T, p=2, dim=1),
                                        "Top-2 PC Removal", pc1_nli_ctrl, raw_ctrl, ctrl_pubs)
    
    ctrl_ablations = [base_ctrl, res_c_center, res_c_top1, res_c_top2]
    print(f"{'Condition':<25} | {'PC1 EVR':<9} | {'PR':<6} | {'r(NLI)':<8} | {'ANOVA F':<8} | {'p-val':<8} | {'eta^2':<8}")
    print("-" * 85)
    for ab in ctrl_ablations:
        print(f"{ab['label']:<25} | {ab['pc1_evr']*100:>7.2f}% | {ab['participation_ratio']:>6.3f} | {ab['r_with_nli_pc1']:>8.4f} | {ab['anova_f']:>8.3f} | {ab['anova_p']:>8.4f} | {ab['eta2']:>8.4f}")

    # =========================================================================
    # PART 2: VADER SENTIMENT REGRESSION & RESIDUAL PUBLISHER SEPARATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: VADER SENTIMENT REGRESSION & RESIDUAL PUBLISHER SEPARATION")
    print("=" * 80)
    analyzer = SentimentIntensityAnalyzer()
    
    def test_sentiment_regression(pc1_scores: np.ndarray, texts: List[str], pubs: list, dataset_label: str):
        vader_scores = [analyzer.polarity_scores(t) for t in texts]
        compound = np.array([v["compound"] for v in vader_scores])
        pos = np.array([v["pos"] for v in vader_scores])
        neg = np.array([v["neg"] for v in vader_scores])
        
        # 1. Raw PC1 ANOVA
        pub_counts = {}
        for p in pubs:
            pub_counts[p] = pub_counts.get(p, 0) + 1
        
        is_balanced = (len(set(pub_counts.values())) == 1 and list(pub_counts.values())[0] == 5)
        if is_balanced:
            pub_set = sorted(list(set(pubs)))
            idxs = list(range(len(pubs)))
            pub_list = pubs
        else:
            frequent_pubs = [p for p, c in pub_counts.items() if c >= 2]
            idxs = [i for i, p in enumerate(pubs) if p in frequent_pubs]
            pub_set = frequent_pubs
            pub_list = [pubs[i] for i in idxs]
            
        def eval_anova(y, name):
            sub_y = y[idxs]
            groups = [sub_y[[i for i, p in enumerate(pub_list) if p == pub]] for pub in pub_set]
            f_stat, p_val = f_oneway(*groups)
            kw_stat, kw_p = kruskal(*groups)
            eta2 = compute_eta2(sub_y, pub_list)
            return {"name": name, "f_stat": float(f_stat), "p_val": float(p_val), "kw_h": float(kw_stat), "kw_p": float(kw_p), "eta2": float(eta2)}

        raw_res = eval_anova(pc1_scores, "Raw PC1")

        # 2. Regress out Compound Sentiment
        # y = b0 + b1 * compound + eps
        X_comp = np.column_stack([np.ones(len(pc1_scores)), compound])
        beta_comp, _, _, _ = np.linalg.lstsq(X_comp, pc1_scores, rcond=None)
        res_comp = pc1_scores - (X_comp @ beta_comp)
        r_rem_comp, p_rem_comp = pearsonr(res_comp, compound)
        comp_res = eval_anova(res_comp, "Residual after Compound Sentiment")
        comp_res["r2_sentiment"] = float(1.0 - np.var(res_comp) / np.var(pc1_scores))

        # 3. Regress out Negative Sentiment
        X_neg = np.column_stack([np.ones(len(pc1_scores)), neg])
        beta_neg, _, _, _ = np.linalg.lstsq(X_neg, pc1_scores, rcond=None)
        res_neg = pc1_scores - (X_neg @ beta_neg)
        neg_res = eval_anova(res_neg, "Residual after Negative Sentiment")
        neg_res["r2_sentiment"] = float(1.0 - np.var(res_neg) / np.var(pc1_scores))

        # 4. Multivariate Regression: pos + neg + compound
        X_all = np.column_stack([np.ones(len(pc1_scores)), pos, neg, compound])
        beta_all, _, _, _ = np.linalg.lstsq(X_all, pc1_scores, rcond=None)
        res_all = pc1_scores - (X_all @ beta_all)
        all_res = eval_anova(res_all, "Residual after All Sentiment (Pos+Neg+Comp)")
        all_res["r2_sentiment"] = float(1.0 - np.var(res_all) / np.var(pc1_scores))

        print(f"\nSentiment Control on {dataset_label}:")
        print(f"{'Condition':<45} | {'ANOVA F':<8} | {'p-val':<8} | {'eta^2':<8} | {'KW H':<8} | {'KW p':<8}")
        print("-" * 95)
        for r in [raw_res, comp_res, neg_res, all_res]:
            print(f"{r['name']:<45} | {r['f_stat']:>8.3f} | {r['p_val']:>8.4f} | {r['eta2']:>8.4f} | {r['kw_h']:>8.3f} | {r['kw_p']:>8.4f}")
        return [raw_res, comp_res, neg_res, all_res]

    ctrl_sent_nli = test_sentiment_regression(pc1_nli_ctrl, ctrl_texts, ctrl_pubs, "Controlled Balanced N=60 (NLI PC1)")
    ctrl_sent_fadeb = test_sentiment_regression(base_ctrl["pc1_scores"], ctrl_texts, ctrl_pubs, "Controlled Balanced N=60 (DeBERTa FrameAxis PC1)")
    prosp_sent_nli = test_sentiment_regression(pc1_nli_prosp, prosp_texts, prosp_pubs, "Prospective N=120 (NLI PC1)")

    # =========================================================================
    # PART 3: PERMUTATION & CROSS-VALIDATED (OUT-OF-SAMPLE) CCA
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: PERMUTATION & CROSS-VALIDATED (OUT-OF-SAMPLE) CCA")
    print("=" * 80)

    def run_validated_cca(
        X: np.ndarray,
        Y: np.ndarray,
        label: str,
        n_perms: int = 1000,
        n_repeats: int = 50,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        N, K = X.shape
        # Standardize columns
        X_std = (X - X.mean(axis=0)) / np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
        Y_std = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) < 1e-12, 1.0, Y.std(axis=0))
        
        cca = CCA(n_components=1)
        cca.fit(X_std, Y_std)
        x_c, y_c = cca.transform(X_std, Y_std)
        r_train, p_train = pearsonr(x_c[:, 0], y_c[:, 0])

        # Permutation test (shuffle rows of Y)
        perm_corrs = []
        for i in range(n_perms):
            perm_idx = np.random.permutation(N)
            cca_p = CCA(n_components=1)
            cca_p.fit(X_std, Y_std[perm_idx])
            xp_c, yp_c = cca_p.transform(X_std, Y_std[perm_idx])
            r_p, _ = pearsonr(xp_c[:, 0], yp_c[:, 0])
            perm_corrs.append(r_p)
        perm_corrs = np.array(perm_corrs)
        perm_p = float((np.sum(perm_corrs >= r_train) + 1) / (n_perms + 1))
        null_q95 = float(np.percentile(perm_corrs, 95))
        null_mean = float(np.mean(perm_corrs))

        # Repeated K-fold Cross-Validation (Out-of-sample)
        cv_test_corrs = []
        for rep in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
            test_x_proj = np.zeros(N)
            test_y_proj = np.zeros(N)
            for train_idx, test_idx in kf.split(X_std):
                cca_cv = CCA(n_components=1)
                cca_cv.fit(X_std[train_idx], Y_std[train_idx])
                # Check sign convention on train
                xt_c, yt_c = cca_cv.transform(X_std[train_idx], Y_std[train_idx])
                sign = np.sign(np.corrcoef(xt_c[:, 0], yt_c[:, 0])[0, 1])
                
                # Transform test
                x_te, y_te = cca_cv.transform(X_std[test_idx], Y_std[test_idx])
                test_x_proj[test_idx] = x_te[:, 0]
                test_y_proj[test_idx] = y_te[:, 0] * sign
            r_cv, _ = pearsonr(test_x_proj, test_y_proj)
            cv_test_corrs.append(r_cv)
            
        cv_test_corrs = np.array(cv_test_corrs)
        mean_cv_test_r = float(np.mean(cv_test_corrs))
        std_cv_test_r = float(np.std(cv_test_corrs))
        ci95_low = float(np.percentile(cv_test_corrs, 2.5))
        ci95_high = float(np.percentile(cv_test_corrs, 97.5))

        print(f"{label:<32} | In-Sample r: {r_train:.4f} | Perm Null (mean={null_mean:.4f}, q95={null_q95:.4f}, p={perm_p:.4f}) | Out-of-Sample CV r: {mean_cv_test_r:.4f} +/- {std_cv_test_r:.4f} [95% CI: {ci95_low:.4f}, {ci95_high:.4f}]")
        return {
            "label": label,
            "in_sample_r_canon": float(r_train),
            "perm_null_mean": null_mean,
            "perm_null_q95": null_q95,
            "perm_p_value": perm_p,
            "out_of_sample_cv_mean_r": mean_cv_test_r,
            "out_of_sample_cv_std_r": std_cv_test_r,
            "out_of_sample_cv_95ci": [ci95_low, ci95_high]
        }

    cca_prosp_deb = run_validated_cca(nli_prosp_m, base_prosp["fa_diff"], "Prospective N=120 (DeBERTa)")
    cca_ctrl_deb = run_validated_cca(nli_ctrl_m, base_ctrl["fa_diff"], "Controlled Balanced N=60 (DeBERTa)")
    cca_clean83_deb = run_validated_cca(nli_prosp_m[clean_83_idx], base_prosp["fa_diff"][clean_83_idx], "Clean N=83 (DeBERTa)")
    
    # Also for MiniLM on Prospective N=120
    minilm_extractor = SentenceTransformerSuite("sentence-transformers/all-MiniLM-L6-v2")
    fa_diff_min, _, _ = minilm_extractor.extract_frameaxis_matrices(prosp_texts, q0_pairs)
    cca_prosp_min = run_validated_cca(nli_prosp_m, fa_diff_min, "Prospective N=120 (MiniLM)")

    # Save all results to json
    results = {
        "anisotropy_ablations_prospective_n120": [
            {k: v for k, v in ab.items() if k not in ["pc1_scores", "fa_diff"]} for ab in ablations
        ],
        "anisotropy_ablations_controlled_n60": [
            {k: v for k, v in ab.items() if k not in ["pc1_scores", "fa_diff"]} for ab in ctrl_ablations
        ],
        "sentiment_controlled_publisher_separation": {
            "controlled_balanced_n60_nli": ctrl_sent_nli,
            "controlled_balanced_n60_fa_deberta": ctrl_sent_fadeb,
            "prospective_n120_nli": prosp_sent_nli
        },
        "validated_cca": {
            "prospective_n120_deberta": cca_prosp_deb,
            "controlled_n60_deberta": cca_ctrl_deb,
            "clean_n83_deberta": cca_clean83_deb,
            "prospective_n120_minilm": cca_prosp_min
        }
    }
    out_file = "paper/cross_instrument_deep_tightening_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll tightening ablation results saved to {out_file}!")

if __name__ == "__main__":
    main()
