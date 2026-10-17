"""
scripts/exp_frameaxis_and_compositional_baseline.py

Matched FrameAxis Baseline & Minimal Compositional Controls Benchmark
---------------------------------------------------------------------
Addresses the central novelty objection:
1. FrameAxis-style semantic-axis projection:
   v_k = e(h_A) - e(h_B)
   F_k(x) = cos(e(x), e(h_A)) - cos(e(x), e(h_B))
   P_k(x) = cos(e(x), v_k)

2. Evaluated on IDENTICAL datasets:
   - Prospective Confirmatory N=120 (full and clean N=83 subset)
   - Balanced Controlled N=60 (12 publishers x 5 articles)
   - Clean Ukraine War N=60 (54 publishers)
   Across both Q0 (canonical) and Q1 (paraphrase) banks.

3. Complete Statistical Battery:
   - PCA Eigenspectrum & Participation Ratio (PR)
   - 10,000-draw column & publisher-block permutation null gates
   - Q0 <-> Q1 paraphrase stability (Pearson r, Spearman rho, Procrustes disparity, subspace angle)
   - 35 disjoint 4-vs-4 partition agreement
   - Cross-instrument correlation: r(PC1_NLI, PC1_FA)

4. Minimal Compositional Perturbation Suite:
   - 16 minimal pairs across 4 categories:
     * Agent-Patient Reversal (4 pairs)
     * Polarity Negation (4 pairs)
     * Causal Direction Reversal (4 pairs)
     * Quantifier / Modality Inversion (4 pairs)
   - Evaluated on NLI vs FrameAxis: directional sensitivity, sign flips, cosine collapse.
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
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PERMUTATIONS = 10000
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Helper Functions
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

def compute_pca_metrics(matrix: np.ndarray, standardize: bool = False) -> Dict[str, Any]:
    N, D = matrix.shape
    mean = matrix.mean(axis=0, keepdims=True)
    if standardize:
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        transformed = (matrix - mean) / std
        cov = np.cov(transformed, rowvar=False)
    else:
        transformed = matrix - mean
        cov = np.cov(transformed, rowvar=False)
        
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 1e-12)
    eigvecs = eigvecs[:, idx]
    
    total_var = eigvals.sum()
    evr = (eigvals / total_var).tolist()
    pr = float((eigvals.sum()) ** 2 / (eigvals ** 2).sum())
    
    scores = transformed @ eigvecs
    return {
        "standardized": standardize,
        "evr": evr,
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]) if D > 1 else 0.0,
        "pr": pr,
        "eigvals": eigvals.tolist(),
        "eigvecs": eigvecs,
        "scores": scores,
        "transformed": transformed
    }

def run_permutation_nulls(
    matrix: np.ndarray,
    publishers: List[str] = None,
    n_perms: int = N_PERMUTATIONS,
    standardize: bool = False
) -> Dict[str, Any]:
    N, D = matrix.shape
    obs = compute_pca_metrics(matrix, standardize=standardize)
    obs_evr = np.array(obs["evr"])
    
    null_col_evr1 = np.zeros(n_perms)
    null_col_evr2 = np.zeros(n_perms)
    null_col_pr = np.zeros(n_perms)
    
    rng = np.random.default_rng(SEED)
    for p in range(n_perms):
        shuffled = np.zeros_like(matrix)
        for d in range(D):
            shuffled[:, d] = rng.permutation(matrix[:, d])
        res = compute_pca_metrics(shuffled, standardize=standardize)
        null_col_evr1[p] = res["evr"][0]
        null_col_evr2[p] = res["evr"][1] if D > 1 else 0.0
        null_col_pr[p] = res["pr"]
        
    p_col_1 = float((np.sum(null_col_evr1 >= obs_evr[0]) + 1) / (n_perms + 1))
    p_col_2 = float((np.sum(null_col_evr2 >= obs_evr[1]) + 1) / (n_perms + 1)) if D > 1 else 1.0
    q95_col_1 = float(np.percentile(null_col_evr1, 95))
    q95_col_2 = float(np.percentile(null_col_evr2, 95)) if D > 1 else 0.0
    
    out = {
        "standardized": standardize,
        "obs_pc1_evr": float(obs_evr[0]),
        "obs_pc2_evr": float(obs_evr[1]) if D > 1 else 0.0,
        "obs_pr": float(obs["pr"]),
        "col_null_mean_pr": float(np.mean(null_col_pr)),
        "col_null_pc1_p": p_col_1,
        "col_null_pc1_q95": q95_col_1,
        "col_null_pc2_p": p_col_2,
        "col_null_pc2_q95": q95_col_2,
        "detected_rank_col": int((obs_evr[0] > q95_col_1) + ((obs_evr[1] > q95_col_2) if D > 1 else 0))
    }
    
    if publishers is not None:
        unique_pubs = list(set(publishers))
        pub_indices = {pub: [i for i, p in enumerate(publishers) if p == pub] for pub in unique_pubs}
        null_pub_evr1 = np.zeros(n_perms)
        null_pub_evr2 = np.zeros(n_perms)
        
        for p in range(n_perms):
            shuffled = np.zeros_like(matrix)
            for pub, idxs in pub_indices.items():
                for d in range(D):
                    shuffled[idxs, d] = rng.permutation(matrix[idxs, d])
            res = compute_pca_metrics(shuffled, standardize=standardize)
            null_pub_evr1[p] = res["evr"][0]
            null_pub_evr2[p] = res["evr"][1] if D > 1 else 0.0
            
        p_pub_1 = float((np.sum(null_pub_evr1 >= obs_evr[0]) + 1) / (n_perms + 1))
        p_pub_2 = float((np.sum(null_pub_evr2 >= obs_evr[1]) + 1) / (n_perms + 1)) if D > 1 else 1.0
        q95_pub_1 = float(np.percentile(null_pub_evr1, 95))
        q95_pub_2 = float(np.percentile(null_pub_evr2, 95)) if D > 1 else 0.0
        
        out["pub_null_pc1_p"] = p_pub_1
        out["pub_null_pc1_q95"] = q95_pub_1
        out["pub_null_pc2_p"] = p_pub_2
        out["pub_null_pc2_q95"] = q95_pub_2
        out["detected_rank_pub"] = int((obs_evr[0] > q95_pub_1) + ((obs_evr[1] > q95_pub_2) if D > 1 else 0))
        
    return out

def evaluate_cross_bank_alignment(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    standardize: bool = False
) -> Dict[str, Any]:
    pca_a = compute_pca_metrics(matrix_a, standardize=standardize)
    pca_b = compute_pca_metrics(matrix_b, standardize=standardize)
    
    score_a = pca_a["scores"][:, 0]
    score_b = pca_b["scores"][:, 0]
    
    # Check sign alignment
    r_raw, p_raw = pearsonr(score_a, score_b)
    if r_raw < 0:
        score_b = -score_b
        r_raw = -r_raw
        
    rho, rho_p = spearmanr(score_a, score_b)
    
    # 2-PC Procrustes disparity
    k_dim = min(2, matrix_a.shape[1])
    xy_a = pca_a["scores"][:, :k_dim]
    xy_b = pca_b["scores"][:, :k_dim]
    _, _, disparity = procrustes(xy_a, xy_b)
    
    # Subspace angle between PC1 loading vectors
    v1 = pca_a["eigvecs"][:, 0]
    v2 = pca_b["eigvecs"][:, 0]
    cos_angle = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, 0.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    
    return {
        "standardized": standardize,
        "pearson_r": float(r_raw),
        "pearson_p": float(p_raw),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "procrustes_disparity": float(disparity),
        "subspace_angle_deg": angle_deg
    }

def evaluate_disjoint_splits(matrix: np.ndarray, standardize: bool = False) -> Dict[str, Any]:
    N, D = matrix.shape
    assert D == 8, f"Expected 8 axes, got {D}"
    all_axes = list(range(8))
    
    corrs = []
    # Unique 4-vs-4 complementary partitions: index 0 always in first half
    for split_a in combinations(all_axes, 4):
        if 0 not in split_a:
            continue
        split_b = [x for x in all_axes if x not in split_a]
        
        pca_a = compute_pca_metrics(matrix[:, split_a], standardize=standardize)
        pca_b = compute_pca_metrics(matrix[:, split_b], standardize=standardize)
        
        sc_a = pca_a["scores"][:, 0]
        sc_b = pca_b["scores"][:, 0]
        r, _ = pearsonr(sc_a, sc_b)
        corrs.append(abs(r))
        
    corrs = np.array(corrs)
    return {
        "standardized": standardize,
        "n_splits": len(corrs),
        "mean_r": float(np.mean(corrs)),
        "std_r": float(np.std(corrs)),
        "median_r": float(np.median(corrs)),
        "min_r": float(np.min(corrs)),
        "max_r": float(np.max(corrs)),
        "frac_ge_07": float(np.mean(corrs >= 0.70)),
        "frac_ge_08": float(np.mean(corrs >= 0.80))
    }

# -----------------------------------------------------------------------------
# Models and Extractors
# -----------------------------------------------------------------------------
class FeatureExtractorSuite:
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli"):
        print(f"Loading tokenizer & models for: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
        self.encoder = self.nli_model.deberta if hasattr(self.nli_model, "deberta") else self.nli_model.base_model
        
    @torch.no_grad()
    def encode_sentences(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Encode texts into mean-pooled, L2-normalized vectors."""
        inputs = self.tokenizer(
            texts,
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
        normed = F.normalize(mean_pooled, p=2, dim=1)
        return normed

    @torch.no_grad()
    def extract_nli_matrix(self, texts: List[str], hyp_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Extract audited paired NLI matrix: (s_A - s_B) across all pairs."""
        N = len(texts)
        K = len(hyp_pairs)
        matrix = np.zeros((N, K), dtype=np.float32)
        
        # DeBERTa order: 0: Contradiction, 1: Neutral, 2: Entailment
        for k, (h_a, h_b) in enumerate(hyp_pairs):
            inputs_a = self.tokenizer(texts, [h_a]*N, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            out_a = self.nli_model(**inputs_a)
            probs_a = F.softmax(out_a.logits, dim=-1)
            s_a = (probs_a[:, 2] - probs_a[:, 0]).cpu().numpy()
            
            inputs_b = self.tokenizer(texts, [h_b]*N, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            out_b = self.nli_model(**inputs_b)
            probs_b = F.softmax(out_b.logits, dim=-1)
            s_b = (probs_b[:, 2] - probs_b[:, 0]).cpu().numpy()
            
            matrix[:, k] = s_a - s_b
            
        return matrix

    @torch.no_grad()
    def extract_frameaxis_matrices(
        self,
        texts: List[str],
        hyp_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract FrameAxis matrices:
        1. Cosine Difference: F_k(x) = cos(e(x), e(h_A)) - cos(e(x), e(h_B))
        2. Normalized Projection: P_k(x) = cos(e(x), e(h_A) - e(h_B))
        """
        N = len(texts)
        K = len(hyp_pairs)
        
        # Encode articles in batches
        print(f"Encoding {N} documents with encoder backbone...")
        batch_size = 16
        e_texts = []
        for i in range(0, N, batch_size):
            batch = texts[i:i+batch_size]
            emb = self.encode_sentences(batch)
            e_texts.append(emb)
        e_x = torch.cat(e_texts, dim=0) # [N, D]
        
        # Encode hypothesis pairs
        ha_list = [p[0] for p in hyp_pairs]
        hb_list = [p[1] for p in hyp_pairs]
        e_ha = self.encode_sentences(ha_list) # [K, D]
        e_hb = self.encode_sentences(hb_list) # [K, D]
        
        # 1. Cosine difference: e_x @ e_ha.T - e_x @ e_hb.T
        cos_a = torch.matmul(e_x, e_ha.T) # [N, K]
        cos_b = torch.matmul(e_x, e_hb.T) # [N, K]
        fa_diff = (cos_a - cos_b).cpu().numpy()
        
        # 2. Normalized vector projection: v_k = e_ha - e_hb
        v_k = e_ha - e_hb # [K, D]
        v_norm = F.normalize(v_k, p=2, dim=1) # [K, D]
        fa_proj = torch.matmul(e_x, v_norm.T).cpu().numpy() # [N, K]
        
        return fa_diff, fa_proj

class SentenceTransformerExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        
    def encode_sentences(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        
    def extract_frameaxis_matrices(
        self,
        texts: List[str],
        hyp_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(texts)
        K = len(hyp_pairs)
        
        e_x = self.encode_sentences(texts) # [N, D]
        ha_list = [p[0] for p in hyp_pairs]
        hb_list = [p[1] for p in hyp_pairs]
        e_ha = self.encode_sentences(ha_list) # [K, D]
        e_hb = self.encode_sentences(hb_list) # [K, D]
        
        cos_a = torch.matmul(e_x, e_ha.T)
        cos_b = torch.matmul(e_x, e_hb.T)
        fa_diff = (cos_a - cos_b).cpu().numpy()
        
        v_k = e_ha - e_hb
        v_norm = F.normalize(v_k, p=2, dim=1)
        fa_proj = torch.matmul(e_x, v_norm.T).cpu().numpy()
        
        return fa_diff, fa_proj

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STARTING MATCHED FRAMEAXIS BASELINE & COMPOSITIONAL CONTROLS BENCHMARK")
    print("=" * 80)
    
    extractor = FeatureExtractorSuite("microsoft/deberta-v2-xlarge-mnli")
    minilm_extractor = SentenceTransformerExtractor("sentence-transformers/all-MiniLM-L6-v2")
    
    # -------------------------------------------------------------------------
    # 1. Load Hypotheses Banks
    # -------------------------------------------------------------------------
    # Q0 Canonical: probe_hypotheses.json (list of 16 strings)
    q0_raw = json.load(open("probe_hypotheses.json", encoding="utf-8"))
    q0_pairs = [(q0_raw[2*i], q0_raw[2*i+1]) for i in range(8)]
    
    # Q1 Paraphrase: config/prompt_banks/probe_hypotheses_paraphrase_a.json (list of 16 strings)
    q1_raw = json.load(open("config/prompt_banks/probe_hypotheses_paraphrase_a.json", encoding="utf-8"))
    q1_pairs = [(q1_raw[2*i], q1_raw[2*i+1]) for i in range(8)]
    
    # Ukraine: config/ukraine_cross_topic_queries.yaml
    import yaml
    ukr_yaml = yaml.safe_load(open("config/ukraine_cross_topic_queries.yaml", encoding="utf-8"))
    ukr_pairs = [(v["A"], v["B"]) for v in ukr_yaml["queries"].values()]
        
    print(f"Loaded Q0: {len(q0_pairs)} pairs, Q1: {len(q1_pairs)} pairs, Ukraine: {len(ukr_pairs)} pairs.")

    # -------------------------------------------------------------------------
    # 2. Part 1: Minimal Compositional Perturbation Suite
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: MINIMAL COMPOSITIONAL PERTURBATION SUITE (16 PAIRS)")
    print("=" * 80)
    
    compositional_cases = [
        # Category 1: Agent-Patient Reversal
        {
            "category": "Agent-Patient Reversal",
            "name": "Israel vs Hamas Attack",
            "base": "Israel attacked Hamas militants in Gaza today.",
            "pert": "Hamas militants attacked Israel in Gaza today.",
            "hyp_a": "This statement portrays Israeli military actions as legitimate self-defense against terrorism.",
            "hyp_b": "This statement portrays Israeli military actions as disproportionate aggression against civilians.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Agent-Patient Reversal",
            "name": "Ukraine vs Russia Strike",
            "base": "Ukrainian forces struck Russian military positions along the border.",
            "pert": "Russian forces struck Ukrainian military positions along the border.",
            "hyp_a": "This article portrays Ukrainian military actions as legitimate sovereign defense.",
            "hyp_b": "This article portrays Ukrainian military actions as hostile escalation.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Agent-Patient Reversal",
            "name": "Police vs Protesters Detainment",
            "base": "The police forces aggressively detained peaceful anti-war protesters.",
            "pert": "Peaceful anti-war protesters aggressively detained the police forces.",
            "hyp_a": "This statement frames state authorities as violent oppressors.",
            "hyp_b": "This statement frames protesters as disruptive aggressors.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Agent-Patient Reversal",
            "name": "Occupier vs Residents Eviction",
            "base": "The military occupiers forcibly evicted indigenous residents from their homes.",
            "pert": "Indigenous residents forcibly evicted the military occupiers from their homes.",
            "hyp_a": "This statement frames Israel as a colonial occupying power violating rights.",
            "hyp_b": "This statement frames security operations as lawful enforcement.",
            "expected_sign_delta": 1.0
        },
        
        # Category 2: Polarity Negation
        {
            "category": "Polarity Negation",
            "name": "Military Operation Justified",
            "base": "The military strike was fully justified under international humanitarian law.",
            "pert": "The military strike was not justified under international humanitarian law.",
            "hyp_a": "This article describes military responses as proportionate and legally justified.",
            "hyp_b": "This article emphasizes violations of international law and illegality.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Polarity Negation",
            "name": "Deliberate Civilian Targeting",
            "base": "Armed forces deliberately target civilian infrastructure during operations.",
            "pert": "Armed forces do not deliberately target civilian infrastructure during operations.",
            "hyp_a": "This article suggests forces deliberately target civilians.",
            "hyp_b": "This article suggests civilian casualties are unintended collateral damage.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Polarity Negation",
            "name": "Ceasefire Compliance",
            "base": "The government strictly complied with all agreed ceasefire obligations.",
            "pert": "The government did not comply with any agreed ceasefire obligations.",
            "hyp_a": "This article emphasizes compliance with international agreements and restraint.",
            "hyp_b": "This article emphasizes flagrant violations of agreements and aggression.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Polarity Negation",
            "name": "Civilian Casualties Caused",
            "base": "The intense bombing caused catastrophic civilian casualties and deaths.",
            "pert": "The intense bombing caused no civilian casualties or deaths.",
            "hyp_a": "This article emphasizes catastrophic civilian casualties and collective punishment.",
            "hyp_b": "This article emphasizes surgical precision and absence of civilian harm.",
            "expected_sign_delta": 1.0
        },
        
        # Category 3: Causal Direction Reversal
        {
            "category": "Causal Direction Reversal",
            "name": "Casualties Resulting From Strikes",
            "base": "Extensive civilian casualties directly resulted from the airstrikes.",
            "pert": "The airstrikes directly resulted from extensive civilian casualties.",
            "hyp_a": "The military airstrikes directly caused civilian casualties.",
            "hyp_b": "Pre-existing civilian casualties provoked the military airstrikes.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Causal Direction Reversal",
            "name": "Sanctions Caused Economic Collapse",
            "base": "The devastating economic collapse was caused by punitive foreign sanctions.",
            "pert": "Punitive foreign sanctions were caused by the devastating economic collapse.",
            "hyp_a": "Foreign sanctions caused severe economic devastation and collective suffering.",
            "hyp_b": "Domestic economic mismanagement necessitated foreign sanctions.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Causal Direction Reversal",
            "name": "Blockade Provoked Attack",
            "base": "The crushing maritime blockade provoked the armed cross-border attack.",
            "pert": "The armed cross-border attack provoked the crushing maritime blockade.",
            "hyp_a": "The cross-border attack was a predictable response provoked by the blockade.",
            "hyp_b": "The maritime blockade was an essential defensive measure triggered by the attack.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Causal Direction Reversal",
            "name": "Refugee Crisis From Incursion",
            "base": "A mass humanitarian refugee crisis was created by the ground incursion.",
            "pert": "The ground incursion was created by a mass humanitarian refugee crisis.",
            "hyp_a": "The military incursion generated massive refugee displacement and suffering.",
            "hyp_b": "Refugee flows required a military ground incursion for stabilization.",
            "expected_sign_delta": 1.0
        },
        
        # Category 4: Quantifier / Modality Inversion
        {
            "category": "Quantifier Inversion",
            "name": "Avoidable Civilian Deaths",
            "base": "All civilian deaths and injuries in the conflict were completely avoidable.",
            "pert": "No civilian deaths and injuries in the conflict were avoidable.",
            "hyp_a": "This article frames civilian casualties as avoidable tragic atrocities.",
            "hyp_b": "This article frames civilian casualties as an unavoidable military necessity.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Quantifier Inversion",
            "name": "International Condemnation",
            "base": "All international legal bodies united in condemning the military aggression.",
            "pert": "No international legal bodies condemned the military aggression.",
            "hyp_a": "This article emphasizes universal global condemnation of military violations.",
            "hyp_b": "This article emphasizes broad international support and legal legitimacy.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Quantifier Inversion",
            "name": "Destruction Degree",
            "base": "Heavy artillery completely destroyed and pulverized the entire refugee camp.",
            "pert": "Heavy artillery minimally impacted and lightly damaged the refugee camp.",
            "hyp_a": "This article emphasizes complete destruction and disproportionate devastation.",
            "hyp_b": "This article emphasizes restrained, minimal tactical impact.",
            "expected_sign_delta": 1.0
        },
        {
            "category": "Quantifier Inversion",
            "name": "Hospital Infiltration",
            "base": "Combatants systematically operate from inside every hospital in the sector.",
            "pert": "Combatants never operate from inside any hospital in the sector.",
            "hyp_a": "This article claims healthcare facilities are used as operational bases.",
            "hyp_b": "This article asserts healthcare facilities are free of military presence.",
            "expected_sign_delta": 1.0
        }
    ]
    
    comp_results = []
    print(f"Evaluating {len(compositional_cases)} minimal compositional pairs...")
    
    for case in compositional_cases:
        base_txt = case["base"]
        pert_txt = case["pert"]
        h_a = case["hyp_a"]
        h_b = case["hyp_b"]
        exp_sign = case["expected_sign_delta"]
        
        emb_base = extractor.encode_sentences([base_txt])[0]
        emb_pert = extractor.encode_sentences([pert_txt])[0]
        cos_similarity = float(torch.dot(emb_base, emb_pert).item())
        
        nli_m = extractor.extract_nli_matrix([base_txt, pert_txt], [(h_a, h_b)])
        nli_base = float(nli_m[0, 0])
        nli_pert = float(nli_m[1, 0])
        nli_delta = float(nli_base - nli_pert)
        nli_flipped = bool((nli_base > 0 and nli_pert < 0) or (nli_base < 0 and nli_pert > 0))
        nli_dir = bool((nli_delta * exp_sign) > 0)
        
        fa_diff, fa_proj = extractor.extract_frameaxis_matrices([base_txt, pert_txt], [(h_a, h_b)])
        fa_diff_base = float(fa_diff[0, 0])
        fa_diff_pert = float(fa_diff[1, 0])
        fa_diff_delta = float(fa_diff_base - fa_diff_pert)
        fa_diff_flipped = bool((fa_diff_base > 0 and fa_diff_pert < 0) or (fa_diff_base < 0 and fa_diff_pert > 0))
        fa_diff_dir = bool((fa_diff_delta * exp_sign) > 0)
        
        fa_proj_base = float(fa_proj[0, 0])
        fa_proj_pert = float(fa_proj[1, 0])
        fa_proj_delta = float(fa_proj_base - fa_proj_pert)
        fa_proj_flipped = bool((fa_proj_base > 0 and fa_proj_pert < 0) or (fa_proj_base < 0 and fa_proj_pert > 0))
        fa_proj_dir = bool((fa_proj_delta * exp_sign) > 0)
        
        # MiniLM evaluation
        emb_base_m = minilm_extractor.encode_sentences([base_txt])[0]
        emb_pert_m = minilm_extractor.encode_sentences([pert_txt])[0]
        minilm_cos_sim = float(torch.dot(emb_base_m, emb_pert_m).item())
        
        m_diff, m_proj = minilm_extractor.extract_frameaxis_matrices([base_txt, pert_txt], [(h_a, h_b)])
        m_diff_base = float(m_diff[0, 0])
        m_diff_pert = float(m_diff[1, 0])
        m_diff_delta = float(m_diff_base - m_diff_pert)
        m_diff_flipped = bool((m_diff_base > 0 and m_diff_pert < 0) or (m_diff_base < 0 and m_diff_pert > 0))
        m_diff_dir = bool((m_diff_delta * exp_sign) > 0)
        
        m_proj_base = float(m_proj[0, 0])
        m_proj_pert = float(m_proj[1, 0])
        m_proj_delta = float(m_proj_base - m_proj_pert)
        m_proj_flipped = bool((m_proj_base > 0 and m_proj_pert < 0) or (m_proj_base < 0 and m_proj_pert > 0))
        m_proj_dir = bool((m_proj_delta * exp_sign) > 0)

        entry = {
            "category": case["category"],
            "name": case["name"],
            "base_text": base_txt,
            "pert_text": pert_txt,
            "target_hyp_a": h_a,
            "target_hyp_b": h_b,
            "expected_sign_delta": exp_sign,
            "cosine_similarity_base_pert": cos_similarity,
            "nli_base": nli_base,
            "nli_pert": nli_pert,
            "nli_delta": nli_delta,
            "nli_directional_concordance": nli_dir,
            "nli_sign_flipped": nli_flipped,
            "fa_diff_base": fa_diff_base,
            "fa_diff_pert": fa_diff_pert,
            "fa_diff_delta": fa_diff_delta,
            "fa_diff_directional_concordance": fa_diff_dir,
            "fa_diff_sign_flipped": fa_diff_flipped,
            "fa_proj_base": fa_proj_base,
            "fa_proj_pert": fa_proj_pert,
            "fa_proj_delta": fa_proj_delta,
            "fa_proj_directional_concordance": fa_proj_dir,
            "fa_proj_sign_flipped": fa_proj_flipped,
            "minilm_cosine_similarity": minilm_cos_sim,
            "minilm_fa_diff_base": m_diff_base,
            "minilm_fa_diff_pert": m_diff_pert,
            "minilm_fa_diff_delta": m_diff_delta,
            "minilm_fa_diff_directional_concordance": m_diff_dir,
            "minilm_fa_diff_sign_flipped": m_diff_flipped,
            "minilm_fa_proj_base": m_proj_base,
            "minilm_fa_proj_pert": m_proj_pert,
            "minilm_fa_proj_delta": m_proj_delta,
            "minilm_fa_proj_directional_concordance": m_proj_dir,
            "minilm_fa_proj_sign_flipped": m_proj_flipped
        }
        comp_results.append(entry)
        print(f"[{case['category']}] {case['name']}: DeBERTa CosSim={cos_similarity:.4f} | NLI Delta={nli_delta:+.4f} (Dir={nli_dir}, Flip={nli_flipped}) | MiniLM CosSim={minilm_cos_sim:.4f} | MiniLM FA Delta={m_diff_delta:+.4f} (Dir={m_diff_dir}, Flip={m_diff_flipped})")

    categories = sorted(list(set(c["category"] for c in comp_results)))
    category_summary_deberta = {}
    category_summary_minilm = {}
    for cat in categories:
        items = [c for c in comp_results if c["category"] == cat]
        category_summary_deberta[cat] = {
            "count": len(items),
            "mean_cosine_similarity": float(np.mean([x["cosine_similarity_base_pert"] for x in items])),
            "mean_abs_nli_delta": float(np.mean([abs(x["nli_delta"]) for x in items])),
            "nli_directional_concordance_rate": float(np.mean([x["nli_directional_concordance"] for x in items])),
            "nli_sign_flip_rate": float(np.mean([x["nli_sign_flipped"] for x in items])),
            "mean_abs_fa_diff_delta": float(np.mean([abs(x["fa_diff_delta"]) for x in items])),
            "fa_diff_directional_concordance_rate": float(np.mean([x["fa_diff_directional_concordance"] for x in items])),
            "fa_diff_sign_flip_rate": float(np.mean([x["fa_diff_sign_flipped"] for x in items])),
            "mean_abs_fa_proj_delta": float(np.mean([abs(x["fa_proj_delta"]) for x in items])),
            "fa_proj_directional_concordance_rate": float(np.mean([x["fa_proj_directional_concordance"] for x in items])),
            "fa_proj_sign_flip_rate": float(np.mean([x["fa_proj_sign_flipped"] for x in items]))
        }
        category_summary_minilm[cat] = {
            "count": len(items),
            "mean_cosine_similarity": float(np.mean([x["minilm_cosine_similarity"] for x in items])),
            "mean_abs_fa_diff_delta": float(np.mean([abs(x["minilm_fa_diff_delta"]) for x in items])),
            "fa_diff_directional_concordance_rate": float(np.mean([x["minilm_fa_diff_directional_concordance"] for x in items])),
            "fa_diff_sign_flip_rate": float(np.mean([x["minilm_fa_diff_sign_flipped"] for x in items])),
            "mean_abs_fa_proj_delta": float(np.mean([abs(x["minilm_fa_proj_delta"]) for x in items])),
            "fa_proj_directional_concordance_rate": float(np.mean([x["minilm_fa_proj_directional_concordance"] for x in items])),
            "fa_proj_sign_flip_rate": float(np.mean([x["minilm_fa_proj_sign_flipped"] for x in items]))
        }
    
    comp_artifact = {
        "metadata": {
            "comparator": "FrameAxis-style sentence-level semantic-projection baseline",
            "clarification": "Diverges from Kwak et al. (2021) by embedding whole propositions and whole documents rather than aggregating word-level dictionary counts.",
            "models": ["microsoft/deberta-v2-xlarge-mnli", "sentence-transformers/all-MiniLM-L6-v2"],
            "primary_operator": "F_k(x) = cos(e(x), e(h_A)) - cos(e(x), e(h_B)) [FrameAxis cosine difference]",
            "sensitivity_operator": "P_k(x) = cos(e(x), e(h_A) - e(h_B)) = <e(x), v_k> / (||e(x)||_2 ||v_k||_2) = F_k(x) / ||v_k||_2 where ||e(x)||_2 = 1 [Normalized vector projection]",
            "normalization_note": "All document and hypothesis representations are L2-normalized unit vectors. P_k is a strictly positive scalar scaling (1/||v_k||_2) of F_k per axis; under column standardization, Z(P) and Z(F) are mathematically identical.",
            "benchmark_status": "Diagnostic benchmark (N=4 per category, 16 total minimal pairs) assessing directional responsiveness rather than a powered inferential test.",
            "n_pairs": len(comp_results)
        },
        "category_summary": {
            "deberta_v2_xlarge": category_summary_deberta,
            "minilm_l6_v2": category_summary_minilm
        },
        "cases": comp_results
    }
    with open("paper/compositional_controls_results.json", "w", encoding="utf-8") as f:
        json.dump(comp_artifact, f, indent=2)
    print("Saved compositional controls to paper/compositional_controls_results.json.")

    # -------------------------------------------------------------------------
    # 3. Part 2: Matched FrameAxis Corpus Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: MATCHED FRAMEAXIS BASELINE ON CORPORA")
    print("=" * 80)
    
    prosp_path = "data/prospective_confirmatory_120_corpus.jsonl"
    prosp_records = [json.loads(line) for line in open(prosp_path, encoding="utf-8")]
    prosp_texts = [extract_article_text(r) for r in prosp_records]
    prosp_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in prosp_records]
    print(f"Loaded Prospective Corpus: {len(prosp_texts)} articles across {len(set(prosp_pubs))} publishers.")

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
    print(f"Identified clean non-syndicated subset: N={len(clean_83_idx)} articles.")

    nli_cache = torch.load("data/cache_prospective_120_q0_q1_logits.pt")
    la0, lb0 = nli_cache["q0_la"].float(), nli_cache["q0_lb"].float()
    nli_q0_m = ((torch.softmax(la0, dim=-1)[..., 2] - torch.softmax(la0, dim=-1)[..., 0]) - 
                (torch.softmax(lb0, dim=-1)[..., 2] - torch.softmax(lb0, dim=-1)[..., 0])).numpy()
    
    la1, lb1 = nli_cache["q1_la"].float(), nli_cache["q1_lb"].float()
    nli_q1_m = ((torch.softmax(la1, dim=-1)[..., 2] - torch.softmax(la1, dim=-1)[..., 0]) - 
                (torch.softmax(lb1, dim=-1)[..., 2] - torch.softmax(lb1, dim=-1)[..., 0])).numpy()

    print("Extracting FrameAxis Q0 on Prospective N=120...")
    fa_diff_q0, fa_proj_q0 = extractor.extract_frameaxis_matrices(prosp_texts, q0_pairs)
    print("Extracting FrameAxis Q1 on Prospective N=120...")
    fa_diff_q1, fa_proj_q1 = extractor.extract_frameaxis_matrices(prosp_texts, q1_pairs)

    def eval_pair(ma, mb):
        return {
            "raw": evaluate_cross_bank_alignment(ma, mb, standardize=False),
            "standardized": evaluate_cross_bank_alignment(ma, mb, standardize=True)
        }

    print("\n--- Running Permutation Nulls & Dimensionality on Prospective N=120 ---")
    results_prosp = {}
    
    for label, m in [
        ("NLI_Q0_full", nli_q0_m),
        ("NLI_Q1_full", nli_q1_m),
        ("FrameAxis_Diff_Q0_full", fa_diff_q0),
        ("FrameAxis_Diff_Q1_full", fa_diff_q1),
        ("FrameAxis_Proj_Q0_full", fa_proj_q0),
        ("FrameAxis_Proj_Q1_full", fa_proj_q1),
        ("NLI_Q0_clean83", nli_q0_m[clean_83_idx]),
        ("NLI_Q1_clean83", nli_q1_m[clean_83_idx]),
        ("FrameAxis_Diff_Q0_clean83", fa_diff_q0[clean_83_idx]),
        ("FrameAxis_Diff_Q1_clean83", fa_diff_q1[clean_83_idx]),
        ("FrameAxis_Proj_Q0_clean83", fa_proj_q0[clean_83_idx]),
        ("FrameAxis_Proj_Q1_clean83", fa_proj_q1[clean_83_idx])
    ]:
        print(f"Evaluating {label} (raw + standardized)...")
        results_prosp[label] = {
            "dimensionality_raw": run_permutation_nulls(m, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(m, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(m, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(m, standardize=True)
        }

    print("\n--- Evaluating Q0 <-> Q1 Paraphrase Stability ---")
    paraphrase_stability = {
        "NLI_full": eval_pair(nli_q0_m, nli_q1_m),
        "NLI_clean83": eval_pair(nli_q0_m[clean_83_idx], nli_q1_m[clean_83_idx]),
        "FrameAxis_Diff_full": eval_pair(fa_diff_q0, fa_diff_q1),
        "FrameAxis_Diff_clean83": eval_pair(fa_diff_q0[clean_83_idx], fa_diff_q1[clean_83_idx]),
        "FrameAxis_Proj_full": eval_pair(fa_proj_q0, fa_proj_q1),
        "FrameAxis_Proj_clean83": eval_pair(fa_proj_q0[clean_83_idx], fa_proj_q1[clean_83_idx])
    }
    
    print("\n--- Evaluating Cross-Instrument Alignment (NLI vs FrameAxis) ---")
    cross_instrument = {
        "full_Q0_NLI_vs_FA_Diff": eval_pair(nli_q0_m, fa_diff_q0),
        "full_Q0_NLI_vs_FA_Proj": eval_pair(nli_q0_m, fa_proj_q0),
        "full_Q1_NLI_vs_FA_Diff": eval_pair(nli_q1_m, fa_diff_q1),
        "full_Q1_NLI_vs_FA_Proj": eval_pair(nli_q1_m, fa_proj_q1),
        "clean83_Q0_NLI_vs_FA_Diff": eval_pair(nli_q0_m[clean_83_idx], fa_diff_q0[clean_83_idx]),
        "clean83_Q0_NLI_vs_FA_Proj": eval_pair(nli_q0_m[clean_83_idx], fa_proj_q0[clean_83_idx])
    }

    print("\n--- Evaluating MiniLM FrameAxis on Prospective N=120 ---")
    fa_diff_minilm_q0, fa_proj_minilm_q0 = minilm_extractor.extract_frameaxis_matrices(prosp_texts, q0_pairs)
    fa_diff_minilm_q1, fa_proj_minilm_q1 = minilm_extractor.extract_frameaxis_matrices(prosp_texts, q1_pairs)

    minilm_prosp_results = {
        "MiniLM_Diff_Q0_full": {
            "dimensionality_raw": run_permutation_nulls(fa_diff_minilm_q0, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_diff_minilm_q0, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_diff_minilm_q0, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_diff_minilm_q0, standardize=True)
        },
        "MiniLM_Diff_Q1_full": {
            "dimensionality_raw": run_permutation_nulls(fa_diff_minilm_q1, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_diff_minilm_q1, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_diff_minilm_q1, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_diff_minilm_q1, standardize=True)
        },
        "MiniLM_Proj_Q0_full": {
            "dimensionality_raw": run_permutation_nulls(fa_proj_minilm_q0, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_proj_minilm_q0, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_proj_minilm_q0, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_proj_minilm_q0, standardize=True)
        },
        "MiniLM_Proj_Q1_full": {
            "dimensionality_raw": run_permutation_nulls(fa_proj_minilm_q1, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_proj_minilm_q1, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_proj_minilm_q1, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_proj_minilm_q1, standardize=True)
        },
        "paraphrase_stability_Diff": eval_pair(fa_diff_minilm_q0, fa_diff_minilm_q1),
        "paraphrase_stability_Proj": eval_pair(fa_proj_minilm_q0, fa_proj_minilm_q1),
        "cross_instrument_NLI_vs_MiniLM_Diff": eval_pair(nli_q0_m, fa_diff_minilm_q0),
        "cross_instrument_NLI_vs_MiniLM_Proj": eval_pair(nli_q0_m, fa_proj_minilm_q0)
    }

    # -------------------------------------------------------------------------
    # 4. Part 3: Controlled N=60 Corpus (Publisher Null Gate)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: CONTROLLED BALANCED N=60 CORPUS (PUBLISHER NULL GATE)")
    print("=" * 80)
    ctrl_path = "data/controlled_N60_corpus.jsonl"
    ctrl_records = [json.loads(line) for line in open(ctrl_path, encoding="utf-8")]
    ctrl_texts = [extract_article_text(r) for r in ctrl_records]
    ctrl_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in ctrl_records]
    
    nli_ctrl_cache = torch.load("data/cache_paired_logits_deberta-v2-xlarge-mnli_N60.pt", weights_only=False)
    la_c = torch.tensor(nli_ctrl_cache["la"]).float()
    lb_c = torch.tensor(nli_ctrl_cache["lb"]).float()
    nli_ctrl_m = ((torch.softmax(la_c, dim=-1)[..., 2] - torch.softmax(la_c, dim=-1)[..., 0]) - 
                  (torch.softmax(lb_c, dim=-1)[..., 2] - torch.softmax(lb_c, dim=-1)[..., 0])).numpy()
    
    fa_diff_ctrl, fa_proj_ctrl = extractor.extract_frameaxis_matrices(ctrl_texts, q0_pairs)
    
    results_ctrl = {
        "NLI": {
            "dimensionality_raw": run_permutation_nulls(nli_ctrl_m, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(nli_ctrl_m, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(nli_ctrl_m, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(nli_ctrl_m, standardize=True)
        },
        "FrameAxis_Diff": {
            "dimensionality_raw": run_permutation_nulls(fa_diff_ctrl, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_diff_ctrl, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_diff_ctrl, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_diff_ctrl, standardize=True)
        },
        "FrameAxis_Proj": {
            "dimensionality_raw": run_permutation_nulls(fa_proj_ctrl, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_proj_ctrl, publishers=ctrl_pubs, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_proj_ctrl, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_proj_ctrl, standardize=True)
        },
        "cross_instrument_NLI_vs_FA_Diff": eval_pair(nli_ctrl_m, fa_diff_ctrl),
        "cross_instrument_NLI_vs_FA_Proj": eval_pair(nli_ctrl_m, fa_proj_ctrl)
    }

    # -------------------------------------------------------------------------
    # 5. Part 4: Clean Ukraine War N=60 Corpus
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: CLEAN UKRAINE WAR N=60 CORPUS")
    print("=" * 80)
    ukr_path = "data/ukraine_clean_N60_corpus.jsonl"
    ukr_records = [json.loads(line) for line in open(ukr_path, encoding="utf-8")]
    ukr_texts = [extract_article_text(r) for r in ukr_records]
    ukr_pubs = [r.get("publisher") or r.get("publication") or "unknown" for r in ukr_records]
    
    nli_ukr_cache = torch.load("data/cache_paired_logits_ukraine_clean_N60.pt", weights_only=False)
    la_u = torch.tensor(nli_ukr_cache["la"]).float()
    lb_u = torch.tensor(nli_ukr_cache["lb"]).float()
    nli_ukr_m = ((torch.softmax(la_u, dim=-1)[..., 2] - torch.softmax(la_u, dim=-1)[..., 0]) - 
                 (torch.softmax(lb_u, dim=-1)[..., 2] - torch.softmax(lb_u, dim=-1)[..., 0])).numpy()
    
    fa_diff_ukr, fa_proj_ukr = extractor.extract_frameaxis_matrices(ukr_texts, ukr_pairs)
    
    results_ukr = {
        "NLI": {
            "dimensionality_raw": run_permutation_nulls(nli_ukr_m, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(nli_ukr_m, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(nli_ukr_m, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(nli_ukr_m, standardize=True)
        },
        "FrameAxis_Diff": {
            "dimensionality_raw": run_permutation_nulls(fa_diff_ukr, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_diff_ukr, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_diff_ukr, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_diff_ukr, standardize=True)
        },
        "FrameAxis_Proj": {
            "dimensionality_raw": run_permutation_nulls(fa_proj_ukr, n_perms=N_PERMUTATIONS, standardize=False),
            "dimensionality_standardized": run_permutation_nulls(fa_proj_ukr, n_perms=N_PERMUTATIONS, standardize=True),
            "disjoint_35_splits_raw": evaluate_disjoint_splits(fa_proj_ukr, standardize=False),
            "disjoint_35_splits_standardized": evaluate_disjoint_splits(fa_proj_ukr, standardize=True)
        },
        "cross_instrument_NLI_vs_FA_Diff": eval_pair(nli_ukr_m, fa_diff_ukr),
        "cross_instrument_NLI_vs_FA_Proj": eval_pair(nli_ukr_m, fa_proj_ukr)
    }

    # -------------------------------------------------------------------------
    # 6. Save Complete Artifacts
    # -------------------------------------------------------------------------
    full_artifact = {
        "metadata": {
            "comparator": "FrameAxis-style sentence-level semantic-projection baseline",
            "clarification": "Diverges from Kwak et al. (2021) by embedding whole propositions and whole documents rather than aggregating word-level dictionary counts.",
            "encoder": "microsoft/deberta-v2-xlarge-mnli",
            "device": DEVICE,
            "permutations": N_PERMUTATIONS,
            "primary_operator": "F_k(x) = cos(e(x), e(h_A)) - cos(e(x), e(h_B)) [FrameAxis cosine difference]",
            "sensitivity_operator": "P_k(x) = cos(e(x), e(h_A) - e(h_B)) = <e(x), v_k> / (||e(x)||_2 ||v_k||_2) = F_k(x) / ||v_k||_2 where ||e(x)||_2 = 1 [Normalized vector projection]",
            "normalization_note": "All document and hypothesis representations are L2-normalized unit vectors. P_k is a strictly positive scalar scaling (1/||v_k||_2) of F_k per axis; under column standardization, Z(P) and Z(F) are mathematically identical.",
            "benchmark_status": "Diagnostic benchmark (N=4 per category, 16 total minimal pairs) assessing directional responsiveness rather than a powered inferential test.",
            "description": "Matched FrameAxis-style baseline vs NLI across Prospective N=120, Clean N=83, Controlled N=60, and Clean Ukraine N=60 under both raw covariance and column-standardized correlation matrices."
        },
        "prospective_n120": {
            "conditions": results_prosp,
            "paraphrase_stability": paraphrase_stability,
            "cross_instrument_alignment": cross_instrument
        },
        "controlled_n60": results_ctrl,
        "clean_ukraine_n60": results_ukr,
        "sentence_transformers_minilm": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "compositional_controls": category_summary_minilm,
            "prospective_n120": minilm_prosp_results
        }
    }
    
    out_path = "paper/frameaxis_baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_artifact, f, indent=2)
        
    print(f"\nSaved complete FrameAxis benchmark results to {out_path}!")
    print("=" * 80)
    print("EXPERIMENT EXECUTION COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
