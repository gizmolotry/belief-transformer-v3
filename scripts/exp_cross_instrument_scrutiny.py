"""
scripts/exp_cross_instrument_scrutiny.py

Deep Scrutiny of Cross-Instrument Alignment (NLI vs FrameAxis)
--------------------------------------------------------------
Addresses the reviewer's core questions:
1. Sign convention & axis-alignment:
   - Axis-by-axis correlations r(NLI_k, FA_k) for k=1..8.
   - PCA loading profiles v_1 (NLI) vs w_1 (FA).
   - Unconstrained Pearson r and Spearman rho (without forcing positive |r|).
   - Canonical Correlation Analysis (CCA) between 8D NLI and 8D FA.
   - Inter-axis correlation matrices for NLI vs FA (anisotropy / common-mode check).

2. Bootstrap stability & outlier fragility:
   - 2,000-draw bootstrap 95% CIs for Pearson r and Spearman rho.
   - Leave-one-out (LOO) influence diagnostics: min, max, std of r_{-i}, top influential articles.

3. Substantive signal vs degenerate noise of FrameAxis PC1:
   - Correlate NLI PC1 and FA PC1 (DeBERTa and MiniLM) with:
     * Article length (word count, character count)
     * Lexical sentiment (VADER compound, pos, neg)
     * Surface metrics (type-token ratio, stopword ratio)
     * Projection onto corpus mean embedding (anisotropy marker)
     * Publisher identity: ANOVA F, eta^2, Kruskal-Wallis, and ideological group contrast
       (Arab/Pro-Palestinian vs Israeli/Pro-Israel vs Western wire).

Evaluated on:
- Prospective N=120 (full Q0 and Q1, plus clean N=83 subset)
- Clean Ukraine N=60
- Controlled Balanced N=60
"""

import os
import json
import glob
import re
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal, ttest_ind
from sklearn.cross_decomposition import CCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_shingles(text: str, k: int = 5) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < k:
        return set([' '.join(words)])
    return set(' '.join(words[i:i+k]) for i in range(len(words) - k + 1))

def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_article_text(record: Dict[str, Any]) -> str:
    candidates = ["content", "text", "body", "article", "raw_content"]
    for k in candidates:
        v = record.get(k)
        if v and isinstance(v, str) and len(v.strip()) > 50:
            return v.strip()
    raise ValueError(f"Could not extract valid text from record: {list(record.keys())}")

def compute_pca(matrix: np.ndarray, standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns scores, eigvals, eigvecs."""
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
    return scores, eigvals, eigvecs

# -----------------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------------
class FeatureExtractorSuite:
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli"):
        print(f"Loading tokenizer & models for: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
        self.encoder = self.nli_model.deberta if hasattr(self.nli_model, "deberta") else self.nli_model.base_model
        
    @torch.no_grad()
    def encode_sentences(self, texts: List[str], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns normalized embeddings and raw mean-pooled embeddings."""
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

    @torch.no_grad()
    def extract_frameaxis_matrices(
        self,
        texts: List[str],
        hyp_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        e_x, raw_x = self.encode_sentences(texts)
        ha_list = [p[0] for p in hyp_pairs]
        hb_list = [p[1] for p in hyp_pairs]
        e_ha, _ = self.encode_sentences(ha_list)
        e_hb, _ = self.encode_sentences(hb_list)
        
        cos_a = torch.matmul(e_x, e_ha.T)
        cos_b = torch.matmul(e_x, e_hb.T)
        fa_diff = (cos_a - cos_b).cpu().numpy()
        
        v_k = e_ha - e_hb
        v_norm = F.normalize(v_k, p=2, dim=1)
        fa_proj = torch.matmul(e_x, v_norm.T).cpu().numpy()
        return fa_diff, fa_proj, e_x, raw_x

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

# -----------------------------------------------------------------------------
# Detailed Statistical Diagnostics
# -----------------------------------------------------------------------------
def run_detailed_cross_instrument_battery(
    nli_matrix: np.ndarray,
    fa_matrix: np.ndarray,
    name: str = "Analysis"
) -> Dict[str, Any]:
    N, K = nli_matrix.shape
    
    # 1. Axis-by-axis correlations
    axis_corrs = []
    for k in range(K):
        r, p = pearsonr(nli_matrix[:, k], fa_matrix[:, k])
        rho, sp = spearmanr(nli_matrix[:, k], fa_matrix[:, k])
        axis_corrs.append({
            "axis_index": k,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(sp)
        })
        
    # 2. Inter-axis correlation matrices
    nli_corr_mat = np.corrcoef(nli_matrix, rowvar=False)
    fa_corr_mat = np.corrcoef(fa_matrix, rowvar=False)
    
    # Upper triangular off-diagonals
    triu_idx = np.triu_indices(K, k=1)
    nli_off_diags = nli_corr_mat[triu_idx]
    fa_off_diags = fa_corr_mat[triu_idx]
    
    # 3. PCA on standardized matrices
    nli_scores, nli_evals, nli_evecs = compute_pca(nli_matrix, standardize=True)
    fa_scores, fa_evals, fa_evecs = compute_pca(fa_matrix, standardize=True)
    
    s_nli = nli_scores[:, 0]
    s_fa = fa_scores[:, 0]
    
    # PCA on raw matrices
    nli_scores_raw, _, nli_evecs_raw = compute_pca(nli_matrix, standardize=False)
    fa_scores_raw, _, fa_evecs_raw = compute_pca(fa_matrix, standardize=False)
    
    # Check raw vs standardized correlation
    r_raw, p_raw = pearsonr(nli_scores_raw[:, 0], fa_scores_raw[:, 0])
    rho_raw, sp_raw = spearmanr(nli_scores_raw[:, 0], fa_scores_raw[:, 0])
    
    r_std, p_std = pearsonr(s_nli, s_fa)
    rho_std, sp_std = spearmanr(s_nli, s_fa)
    
    # Check correlation with PC2, PC3
    r_nli1_fa2, p_nli1_fa2 = pearsonr(s_nli, fa_scores[:, 1])
    r_nli1_fa3, p_nli1_fa3 = pearsonr(s_nli, fa_scores[:, 2]) if K > 2 else (0.0, 1.0)
    
    # 4. Canonical Correlation Analysis (CCA)
    cca = CCA(n_components=min(K, 4))
    cca.fit(nli_matrix, fa_matrix)
    nli_c, fa_c = cca.transform(nli_matrix, fa_matrix)
    cca_corrs = [float(np.corrcoef(nli_c[:, i], fa_c[:, i])[0, 1]) for i in range(min(K, 4))]
    
    # 5. Bootstrap Diagnostics (2000 draws)
    n_boot = 2000
    rng = np.random.default_rng(SEED)
    boot_r = np.zeros(n_boot)
    boot_rho = np.zeros(n_boot)
    
    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        # recompute PCA on bootstrap sample to avoid data leakage
        sc_n, _, _ = compute_pca(nli_matrix[idx], standardize=True)
        sc_f, _, _ = compute_pca(fa_matrix[idx], standardize=True)
        # align sign to positive correlation with full-sample score
        if np.corrcoef(sc_n[:, 0], s_nli[idx])[0, 1] < 0:
            sc_n[:, 0] = -sc_n[:, 0]
        if np.corrcoef(sc_f[:, 0], s_fa[idx])[0, 1] < 0:
            sc_f[:, 0] = -sc_f[:, 0]
        boot_r[b] = np.corrcoef(sc_n[:, 0], sc_f[:, 0])[0, 1]
        boot_rho[b], _ = spearmanr(sc_n[:, 0], sc_f[:, 0])
        
    ci_r_low, ci_r_high = float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))
    ci_rho_low, ci_rho_high = float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5))
    
    # 6. Leave-One-Out (LOO) Influence
    loo_r = np.zeros(N)
    loo_rho = np.zeros(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        r_i, _ = pearsonr(s_nli[mask], s_fa[mask])
        rho_i, _ = spearmanr(s_nli[mask], s_fa[mask])
        loo_r[i] = r_i
        loo_rho[i] = rho_i
        
    delta_r = loo_r - r_std
    top_influential_idx = np.argsort(np.abs(delta_r))[::-1][:5].tolist()
    
    return {
        "name": name,
        "n_samples": N,
        "n_axes": K,
        "axis_by_axis_correlations": axis_corrs,
        "mean_axis_by_axis_r": float(np.mean([x["pearson_r"] for x in axis_corrs])),
        "mean_axis_by_axis_rho": float(np.mean([x["spearman_rho"] for x in axis_corrs])),
        "nli_inter_axis_off_diagonals": {
            "mean": float(np.mean(nli_off_diags)),
            "std": float(np.std(nli_off_diags)),
            "min": float(np.min(nli_off_diags)),
            "max": float(np.max(nli_off_diags)),
            "frac_positive": float(np.mean(nli_off_diags > 0))
        },
        "fa_inter_axis_off_diagonals": {
            "mean": float(np.mean(fa_off_diags)),
            "std": float(np.std(fa_off_diags)),
            "min": float(np.min(fa_off_diags)),
            "max": float(np.max(fa_off_diags)),
            "frac_positive": float(np.mean(fa_off_diags > 0))
        },
        "pca_loadings": {
            "nli_v1": nli_evecs[:, 0].tolist(),
            "fa_w1": fa_evecs[:, 0].tolist(),
            "nli_v1_sign_pattern": [int(x > 0) for x in nli_evecs[:, 0]],
            "fa_w1_sign_pattern": [int(x > 0) for x in fa_evecs[:, 0]]
        },
        "cross_instrument_correlations": {
            "raw_pearson_r": float(r_raw),
            "raw_pearson_p": float(p_raw),
            "raw_spearman_rho": float(rho_raw),
            "raw_spearman_p": float(sp_raw),
            "std_pearson_r": float(r_std),
            "std_pearson_p": float(p_std),
            "std_spearman_rho": float(rho_std),
            "std_spearman_p": float(sp_std),
            "nli_pc1_vs_fa_pc2_r": float(r_nli1_fa2),
            "nli_pc1_vs_fa_pc2_p": float(p_nli1_fa2),
            "nli_pc1_vs_fa_pc3_r": float(r_nli1_fa3),
            "nli_pc1_vs_fa_pc3_p": float(p_nli1_fa3),
            "cca_canonical_correlations": cca_corrs
        },
        "bootstrap_2000": {
            "pearson_r_mean": float(np.mean(boot_r)),
            "pearson_r_std": float(np.std(boot_r)),
            "pearson_r_ci95": [ci_r_low, ci_r_high],
            "spearman_rho_mean": float(np.mean(boot_rho)),
            "spearman_rho_std": float(np.std(boot_rho)),
            "spearman_rho_ci95": [ci_rho_low, ci_rho_high]
        },
        "leave_one_out_influence": {
            "loo_r_min": float(np.min(loo_r)),
            "loo_r_max": float(np.max(loo_r)),
            "loo_r_range": float(np.max(loo_r) - np.min(loo_r)),
            "top_influential_indices": top_influential_idx,
            "max_abs_delta_r": float(np.max(np.abs(delta_r)))
        },
        "s_nli": s_nli,
        "s_fa": s_fa
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STARTING DEEP SCRUTINY OF CROSS-INSTRUMENT ALIGNMENT")
    print("=" * 80)
    
    extractor = FeatureExtractorSuite("microsoft/deberta-v2-xlarge-mnli")
    minilm_extractor = SentenceTransformerSuite("sentence-transformers/all-MiniLM-L6-v2")
    vader = SentimentIntensityAnalyzer()
    
    # 1. Load Hypotheses Banks
    q0_raw = json.load(open("probe_hypotheses.json", encoding="utf-8"))
    q0_pairs = [(q0_raw[2*i], q0_raw[2*i+1]) for i in range(8)]
    q1_raw = json.load(open("config/prompt_banks/probe_hypotheses_paraphrase_a.json", encoding="utf-8"))
    q1_pairs = [(q1_raw[2*i], q1_raw[2*i+1]) for i in range(8)]
    
    import yaml
    ukr_yaml = yaml.safe_load(open("config/ukraine_cross_topic_queries.yaml", encoding="utf-8"))
    ukr_pairs = [(v["A"], v["B"]) for v in ukr_yaml["queries"].values()]
    
    # 2. Prospective N=120 Corpus
    print("\n--- Loading & Processing Prospective N=120 ---")
    prosp_path = "data/prospective_confirmatory_120_corpus.jsonl"
    prosp_records = [json.loads(line) for line in open(prosp_path, encoding="utf-8")]
    prosp_texts = [extract_article_text(r) for r in prosp_records]
    prosp_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in prosp_records]
    
    # Shingle filter for clean N=83
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
    
    # Load cached NLI logits for Prospective N=120
    nli_cache = torch.load("data/cache_prospective_120_q0_q1_logits.pt")
    la0, lb0 = nli_cache["q0_la"].float(), nli_cache["q0_lb"].float()
    nli_q0_m = ((torch.softmax(la0, dim=-1)[..., 2] - torch.softmax(la0, dim=-1)[..., 0]) - 
                (torch.softmax(lb0, dim=-1)[..., 2] - torch.softmax(lb0, dim=-1)[..., 0])).numpy()
    
    la1, lb1 = nli_cache["q1_la"].float(), nli_cache["q1_lb"].float()
    nli_q1_m = ((torch.softmax(la1, dim=-1)[..., 2] - torch.softmax(la1, dim=-1)[..., 0]) - 
                (torch.softmax(lb1, dim=-1)[..., 2] - torch.softmax(lb1, dim=-1)[..., 0])).numpy()
    
    # Extract FrameAxis matrices and embeddings
    print("Extracting DeBERTa FrameAxis for Prospective N=120 (Q0 & Q1)...")
    fa_diff_q0, fa_proj_q0, e_x_deb, raw_x_deb = extractor.extract_frameaxis_matrices(prosp_texts, q0_pairs)
    fa_diff_q1, fa_proj_q1, _, _ = extractor.extract_frameaxis_matrices(prosp_texts, q1_pairs)
    
    print("Extracting MiniLM FrameAxis for Prospective N=120 (Q0)...")
    fa_diff_m0, fa_proj_m0, e_x_min = minilm_extractor.extract_frameaxis_matrices(prosp_texts, q0_pairs)
    
    # Run battery on Prospective N=120 conditions
    print("\n--- Running Battery on Prospective N=120 ---")
    bat_q0_deb = run_detailed_cross_instrument_battery(nli_q0_m, fa_diff_q0, "Prospective N=120 Q0 (DeBERTa)")
    bat_q1_deb = run_detailed_cross_instrument_battery(nli_q1_m, fa_diff_q1, "Prospective N=120 Q1 (DeBERTa)")
    bat_clean83_deb = run_detailed_cross_instrument_battery(nli_q0_m[clean_83_idx], fa_diff_q0[clean_83_idx], "Clean N=83 Q0 (DeBERTa)")
    bat_q0_min = run_detailed_cross_instrument_battery(nli_q0_m, fa_diff_m0, "Prospective N=120 Q0 (MiniLM)")
    
    # 3. Clean Ukraine N=60 Corpus
    print("\n--- Loading & Processing Clean Ukraine N=60 ---")
    ukr_path = "data/ukraine_clean_N60_corpus.jsonl"
    ukr_records = [json.loads(line) for line in open(ukr_path, encoding="utf-8")]
    ukr_texts = [extract_article_text(r) for r in ukr_records]
    ukr_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in ukr_records]
    
    ukr_cache = torch.load("data/cache_paired_logits_ukraine_clean_N60.pt", weights_only=False)
    la_u = torch.tensor(ukr_cache["la"]).float()
    lb_u = torch.tensor(ukr_cache["lb"]).float()
    nli_ukr_m = ((torch.softmax(la_u, dim=-1)[..., 2] - torch.softmax(la_u, dim=-1)[..., 0]) - 
                 (torch.softmax(lb_u, dim=-1)[..., 2] - torch.softmax(lb_u, dim=-1)[..., 0])).numpy()
    
    fa_diff_ukr, fa_proj_ukr, e_x_ukr, raw_x_ukr = extractor.extract_frameaxis_matrices(ukr_texts, ukr_pairs)
    bat_ukr = run_detailed_cross_instrument_battery(nli_ukr_m, fa_diff_ukr, "Clean Ukraine N=60 (DeBERTa)")
    
    # 4. Controlled Balanced N=60 Corpus
    print("\n--- Loading & Processing Controlled Balanced N=60 ---")
    ctrl_path = "data/controlled_N60_corpus.jsonl"
    ctrl_records = [json.loads(line) for line in open(ctrl_path, encoding="utf-8")]
    ctrl_texts = [extract_article_text(r) for r in ctrl_records]
    ctrl_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in ctrl_records]
    
    ctrl_cache = torch.load("data/cache_paired_logits_deberta-v2-xlarge-mnli_N60.pt", weights_only=False)
    la_c = torch.tensor(ctrl_cache["la"]).float()
    lb_c = torch.tensor(ctrl_cache["lb"]).float()
    nli_ctrl_m = ((torch.softmax(la_c, dim=-1)[..., 2] - torch.softmax(la_c, dim=-1)[..., 0]) - 
                  (torch.softmax(lb_c, dim=-1)[..., 2] - torch.softmax(lb_c, dim=-1)[..., 0])).numpy()
    
    fa_diff_ctrl, fa_proj_ctrl, e_x_ctrl, raw_x_ctrl = extractor.extract_frameaxis_matrices(ctrl_texts, q0_pairs)
    bat_ctrl = run_detailed_cross_instrument_battery(nli_ctrl_m, fa_diff_ctrl, "Controlled Balanced N=60 (DeBERTa)")
    
    # -------------------------------------------------------------------------
    # 5. External Signal vs Noise Diagnostic Battery
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUBSTANTIVE SIGNAL VS DEGENERATE NOISE BATTERY (PROSPECTIVE N=120)")
    print("=" * 80)
    
    # Extract metadata features for Prospective N=120
    word_counts = np.array([len(re.findall(r'\b\w+\b', t)) for t in prosp_texts], dtype=float)
    char_counts = np.array([len(t) for t in prosp_texts], dtype=float)
    
    # VADER sentiment
    vader_compounds = []
    vader_pos = []
    vader_neg = []
    for t in prosp_texts:
        res = vader.polarity_scores(t[:2000])
        vader_compounds.append(res["compound"])
        vader_pos.append(res["pos"])
        vader_neg.append(res["neg"])
    vader_compounds = np.array(vader_compounds, dtype=float)
    vader_pos = np.array(vader_pos, dtype=float)
    vader_neg = np.array(vader_neg, dtype=float)
    
    # Surface lexical features: type-token ratio
    ttrs = []
    for t in prosp_texts:
        words = re.findall(r'\b\w+\b', t.lower())
        ttrs.append(len(set(words)) / max(len(words), 1))
    ttrs = np.array(ttrs, dtype=float)
    
    # Embedding geometry: projection on mean vector & norm
    mean_vec_deb = e_x_deb.mean(dim=0, keepdim=True)
    mean_vec_deb = F.normalize(mean_vec_deb, p=2, dim=1)
    cos_to_mean_deb = torch.matmul(e_x_deb, mean_vec_deb.T).squeeze().cpu().numpy()
    raw_norms_deb = torch.norm(raw_x_deb, p=2, dim=1).cpu().numpy()
    
    # Extract PC1 scores for Prospective N=120
    s_nli = bat_q0_deb["s_nli"]
    s_fa_deb = bat_q0_deb["s_fa"]
    s_fa_min = bat_q0_min["s_fa"]
    
    # Correlate external features with NLI PC1, FrameAxis DeBERTa PC1, FrameAxis MiniLM PC1
    def get_corrs(feat: np.ndarray, name: str) -> Dict[str, Any]:
        r_nli, p_nli = pearsonr(feat, s_nli)
        r_fadeb, p_fadeb = pearsonr(feat, s_fa_deb)
        r_famin, p_famin = pearsonr(feat, s_fa_min)
        return {
            "feature": name,
            "corr_nli_pc1": {"r": float(r_nli), "p": float(p_nli)},
            "corr_fa_deberta_pc1": {"r": float(r_fadeb), "p": float(p_fadeb)},
            "corr_fa_minilm_pc1": {"r": float(r_famin), "p": float(p_famin)}
        }
        
    feature_corrs = [
        get_corrs(word_counts, "Word Count"),
        get_corrs(char_counts, "Character Count"),
        get_corrs(vader_compounds, "VADER Compound Sentiment"),
        get_corrs(vader_pos, "VADER Positive Sentiment"),
        get_corrs(vader_neg, "VADER Negative Sentiment"),
        get_corrs(ttrs, "Type-Token Ratio (Lexical Diversity)"),
        get_corrs(cos_to_mean_deb, "Cosine Similarity to Mean Embedding (Anisotropy Axis)"),
        get_corrs(raw_norms_deb, "Raw Hidden Embedding L2 Norm")
    ]
    
    # 6. Publisher Separation Analysis
    pub_counts = {}
    for p in prosp_pubs:
        pub_counts[p] = pub_counts.get(p, 0) + 1
    frequent_pubs = [p for p, c in pub_counts.items() if c >= 2]
    freq_indices = [i for i, p in enumerate(prosp_pubs) if p in frequent_pubs]
    freq_pubs_list = [prosp_pubs[i] for i in freq_indices]
    
    # Group scores by publisher
    groups_nli = [s_nli[[i for i, p in enumerate(freq_pubs_list) if p == pub]] for pub in frequent_pubs]
    groups_fadeb = [s_fa_deb[[i for i, p in enumerate(freq_pubs_list) if p == pub]] for pub in frequent_pubs]
    groups_famin = [s_fa_min[[i for i, p in enumerate(freq_pubs_list) if p == pub]] for pub in frequent_pubs]
    
    f_nli, p_f_nli = f_oneway(*groups_nli)
    f_fadeb, p_f_fadeb = f_oneway(*groups_fadeb)
    f_famin, p_f_famin = f_oneway(*groups_famin)
    
    kw_nli, p_kw_nli = kruskal(*groups_nli)
    kw_fadeb, p_kw_fadeb = kruskal(*groups_fadeb)
    kw_famin, p_kw_famin = kruskal(*groups_famin)
    
    def compute_eta2(scores: np.ndarray, groups: List[str]) -> float:
        grand_mean = np.mean(scores)
        ss_tot = np.sum((scores - grand_mean) ** 2)
        ss_bet = 0.0
        for pub in set(groups):
            idxs = [i for i, g in enumerate(groups) if g == pub]
            ss_bet += len(idxs) * (np.mean(scores[idxs]) - grand_mean) ** 2
        return float(ss_bet / ss_tot) if ss_tot > 0 else 0.0
        
    eta2_nli = compute_eta2(s_nli[freq_indices], freq_pubs_list)
    eta2_fadeb = compute_eta2(s_fa_deb[freq_indices], freq_pubs_list)
    eta2_famin = compute_eta2(s_fa_min[freq_indices], freq_pubs_list)
    
    # Controlled N=60 (12 publishers x 5 articles balanced)
    s_ctrl_nli = bat_ctrl["s_nli"]
    s_ctrl_fadeb = bat_ctrl["s_fa"]
    ctrl_pub_set = sorted(list(set(ctrl_pubs)))
    groups_ctrl_nli = [s_ctrl_nli[[i for i, p in enumerate(ctrl_pubs) if p == pub]] for pub in ctrl_pub_set]
    groups_ctrl_fadeb = [s_ctrl_fadeb[[i for i, p in enumerate(ctrl_pubs) if p == pub]] for pub in ctrl_pub_set]
    
    f_ctrl_nli, p_f_ctrl_nli = f_oneway(*groups_ctrl_nli)
    f_ctrl_fadeb, p_f_ctrl_fadeb = f_oneway(*groups_ctrl_fadeb)
    eta2_ctrl_nli = compute_eta2(s_ctrl_nli, ctrl_pubs)
    eta2_ctrl_fadeb = compute_eta2(s_ctrl_fadeb, ctrl_pubs)
    
    # Ideological Contrast Grouping on Prospective N=120
    pro_pal_matches = ["middle east monitor", "wafa", "al jazeera", "palestine chronicle", "jurist", "middle east eye", "electronic intifada"]
    pro_isr_matches = ["israel national news", "israel today", "cleveland jewish news", "jerusalem post", "times of israel", "ynet", "i24"]
    
    group_pal_idx = [i for i, p in enumerate(prosp_pubs) if any(m in p.lower() for m in pro_pal_matches)]
    group_isr_idx = [i for i, p in enumerate(prosp_pubs) if any(m in p.lower() for m in pro_isr_matches)]
    
    print(f"Identified {len(group_pal_idx)} Pro-Palestinian outlets and {len(group_isr_idx)} Pro-Israel outlets in Prospective N=120.")
    
    t_nli, p_t_nli = ttest_ind(s_nli[group_pal_idx], s_nli[group_isr_idx], equal_var=False)
    t_fadeb, p_t_fadeb = ttest_ind(s_fa_deb[group_pal_idx], s_fa_deb[group_isr_idx], equal_var=False)
    t_famin, p_t_famin = ttest_ind(s_fa_min[group_pal_idx], s_fa_min[group_isr_idx], equal_var=False)
    
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
        return float((np.mean(x) - np.mean(y)) / pool_sd) if pool_sd > 0 else 0.0
        
    d_nli = cohen_d(s_nli[group_pal_idx], s_nli[group_isr_idx])
    d_fadeb = cohen_d(s_fa_deb[group_pal_idx], s_fa_deb[group_isr_idx])
    d_famin = cohen_d(s_fa_min[group_pal_idx], s_fa_min[group_isr_idx])
    
    publisher_separation = {
        "prospective_n120_anova_frequent": {
            "n_frequent_pubs": len(frequent_pubs),
            "n_articles": len(freq_indices),
            "nli": {"f_stat": float(f_nli), "p_val": float(p_f_nli), "eta_squared": eta2_nli, "kw_h": float(kw_nli), "kw_p": float(p_kw_nli)},
            "fa_deberta": {"f_stat": float(f_fadeb), "p_val": float(p_f_fadeb), "eta_squared": eta2_fadeb, "kw_h": float(kw_fadeb), "kw_p": float(p_kw_fadeb)},
            "fa_minilm": {"f_stat": float(f_famin), "p_val": float(p_f_famin), "eta_squared": eta2_famin, "kw_h": float(kw_famin), "kw_p": float(p_kw_famin)}
        },
        "controlled_n60_anova_balanced": {
            "nli": {"f_stat": float(f_ctrl_nli), "p_val": float(p_f_ctrl_nli), "eta_squared": eta2_ctrl_nli},
            "fa_deberta": {"f_stat": float(f_ctrl_fadeb), "p_val": float(p_f_ctrl_fadeb), "eta_squared": eta2_ctrl_fadeb}
        },
        "ideological_contrast_group_test": {
            "n_pal": len(group_pal_idx),
            "n_isr": len(group_isr_idx),
            "nli": {
                "mean_pal": float(np.mean(s_nli[group_pal_idx])),
                "mean_isr": float(np.mean(s_nli[group_isr_idx])),
                "t_stat": float(t_nli),
                "p_val": float(p_t_nli),
                "cohen_d": d_nli
            },
            "fa_deberta": {
                "mean_pal": float(np.mean(s_fa_deb[group_pal_idx])),
                "mean_isr": float(np.mean(s_fa_deb[group_isr_idx])),
                "t_stat": float(t_fadeb),
                "p_val": float(p_t_fadeb),
                "cohen_d": d_fadeb
            },
            "fa_minilm": {
                "mean_pal": float(np.mean(s_fa_min[group_pal_idx])),
                "mean_isr": float(np.mean(s_fa_min[group_isr_idx])),
                "t_stat": float(t_famin),
                "p_val": float(p_t_famin),
                "cohen_d": d_famin
            }
        }
    }
    
    for bat in [bat_q0_deb, bat_q1_deb, bat_clean83_deb, bat_q0_min, bat_ukr, bat_ctrl]:
        bat.pop("s_nli", None)
        bat.pop("s_fa", None)
        
    final_report = {
        "metadata": {
            "description": "Deep scrutiny of cross-instrument alignment (NLI vs FrameAxis), sign convention, bootstrap stability, LOO influence, and external signal vs degenerate noise.",
            "encoder_deberta": "microsoft/deberta-v2-xlarge-mnli",
            "encoder_minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "date": "2026-09-14"
        },
        "cross_instrument_batteries": {
            "prospective_n120_q0_deberta": bat_q0_deb,
            "prospective_n120_q1_deberta": bat_q1_deb,
            "clean_n83_q0_deberta": bat_clean83_deb,
            "prospective_n120_q0_minilm": bat_q0_min,
            "clean_ukraine_n60_deberta": bat_ukr,
            "controlled_n60_deberta": bat_ctrl
        },
        "feature_correlations": feature_corrs,
        "publisher_separation": publisher_separation
    }
    
    out_path = "paper/cross_instrument_scrutiny_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSuccessfully wrote scrutiny results to {out_path}!")

if __name__ == "__main__":
    main()
