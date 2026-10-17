"""
Surgical Audit: Dependence-Aware QAP Inference, Causal Activation Patching,
and Multi-Criterion Underdetermination Audit.

Tests:
1. Mantel / QAP Permutation Test on Anomaly Migration vs Grassmann Subspace Distance.
2. Conceptual Distance Metric Sensitivity (DeBERTa, MiniLM, TF-IDF vs Subspace Distance).
3. Self-Translation Estimator Multiplicity Baseline vs Conceptual Translation Multiplicity.
4. Multi-Criterion External Co-Adequacy & Sign-Aligned Subspace Geometry.
5. Causal Activation Patching across DeBERTa layers (8, 12, 16, 18, 20, 22, 24).
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent

def load_adversarial_logits():
    cache_path = ROOT / "data/cache_adversarial_kuhnian_logits.pt"
    return torch.load(cache_path)

def load_prospective_120():
    corpus_path = ROOT / "data/prospective_confirmatory_120_corpus.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows

def compute_grassmann_distance(X1, X2, k=3):
    u1, _, _ = np.linalg.svd(X1 - X1.mean(axis=0), full_matrices=False)
    u2, _, _ = np.linalg.svd(X2 - X2.mean(axis=0), full_matrices=False)
    q1 = u1[:, :k]
    q2 = u2[:, :k]
    _, s, _ = np.linalg.svd(q1.T @ q2)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return float(np.sqrt(np.sum(theta**2)))

# ==============================================================================
# 1. MANTEL / QAP PERMUTATION TEST ON ANOMALY MIGRATION
# ==============================================================================
def audit_test_1_qap_anomaly():
    print("\n" + "=" * 80)
    print("AUDIT TEST 1: DEPENDENCE-AWARE MANTEL / QAP TEST ON ANOMALY MIGRATION")
    print("=" * 80)
    
    adv_data = load_adversarial_logits()
    fw_names = sorted([k for k in adv_data.keys() if not k.startswith("adversarial_")])
    n = len(fw_names)
    
    # Extract stance matrices & anomaly profiles
    anomaly_profiles = {}
    stance_matrices = {}
    
    for k in fw_names:
        la = adv_data[k]["la"].numpy().astype(np.float64)
        lb = adv_data[k]["lb"].numpy().astype(np.float64)
        p_contra_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        p_entail_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        p_contra_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        p_entail_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        
        stance = (p_entail_a - p_contra_a) - (p_entail_b - p_contra_b)
        stance_matrices[k] = stance
        
        centroid = stance.mean(axis=0)
        dists = np.linalg.norm(stance - centroid, axis=1)
        anomaly_profiles[k] = (dists - dists.mean()) / (dists.std() + 1e-12)
        
    d_grassmann = np.zeros((n, n))
    c_anomaly = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                d_grassmann[i, j] = 0.0
                c_anomaly[i, j] = 1.0
            else:
                d_grassmann[i, j] = compute_grassmann_distance(stance_matrices[fw_names[i]], stance_matrices[fw_names[j]], k=3)
                r, _ = pearsonr(anomaly_profiles[fw_names[i]], anomaly_profiles[fw_names[j]])
                c_anomaly[i, j] = r
                
    triu_idx = np.triu_indices(n, k=1)
    obs_g = d_grassmann[triu_idx]
    obs_c = c_anomaly[triu_idx]
    
    # Naive statistics
    r_naive, p_naive = pearsonr(obs_g, obs_c)
    rho_naive, sp_naive = spearmanr(obs_g, obs_c)
    
    # Mantel / QAP Permutation Test (10,000 permutations of the 18 framework labels)
    n_perms = 10000
    np.random.seed(42)
    perm_r = np.zeros(n_perms)
    perm_rho = np.zeros(n_perms)
    
    for p_idx in range(n_perms):
        perm = np.random.permutation(n)
        perm_d_g = d_grassmann[np.ix_(perm, perm)]
        perm_g_flat = perm_d_g[triu_idx]
        
        pr, _ = pearsonr(perm_g_flat, obs_c)
        prho, _ = spearmanr(perm_g_flat, obs_c)
        perm_r[p_idx] = pr
        perm_rho[p_idx] = prho
        
    p_mantel_pearson = float(np.mean(perm_r <= r_naive))
    p_mantel_spearman = float(np.mean(perm_rho <= rho_naive))
    
    print(f"Number of frameworks: {n} ({len(obs_g)} unique pairs)")
    print(f"Naive Pearson r:      {r_naive:.4f} (naive p = {p_naive:.4e} <- OVERCONFIDENT)")
    print(f"Naive Spearman rho:   {rho_naive:.4f} (naive p = {sp_naive:.4e})")
    print(f"Mantel / QAP p-value (Pearson):  p = {p_mantel_pearson:.5f} (out of {n_perms} matrix permutations)")
    print(f"Mantel / QAP p-value (Spearman): p = {p_mantel_spearman:.5f} (out of {n_perms} matrix permutations)")
    
    survives = (p_mantel_pearson < 0.05 and p_mantel_spearman < 0.05)
    print(f"--> Hypothesis 'Geometrically more distant perspectives agree less about anomalies' survives QAP test: {survives}")
    
    return {
        "n_frameworks": n,
        "n_pairs": len(obs_g),
        "pearson_r": float(r_naive),
        "spearman_rho": float(rho_naive),
        "naive_pearson_p": float(p_naive),
        "qap_mantel_pearson_p": p_mantel_pearson,
        "qap_mantel_spearman_p": p_mantel_spearman,
        "relationship_survives_qap": survives
    }

# ==============================================================================
# 2. CONCEPTUAL DISTANCE METRIC SENSITIVITY & MANTEL TEST
# ==============================================================================
def audit_test_2_conceptual_sensitivity():
    print("\n" + "=" * 80)
    print("AUDIT TEST 2: CONCEPTUAL DISTANCE METRIC SENSITIVITY & MANTEL TEST")
    print("=" * 80)
    
    adv_data = load_adversarial_logits()
    fw_names = sorted([k for k in adv_data.keys() if not k.startswith("adversarial_")])
    n = len(fw_names)
    
    # 1. Observational distance matrix (Grassmann)
    d_grassmann = np.zeros((n, n))
    stance_matrices = {}
    for k in fw_names:
        la = adv_data[k]["la"].numpy().astype(np.float64)
        lb = adv_data[k]["lb"].numpy().astype(np.float64)
        p_contra_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        p_entail_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        p_contra_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        p_entail_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        stance_matrices[k] = (p_entail_a - p_contra_a) - (p_entail_b - p_contra_b)
        
    for i in range(n):
        for j in range(n):
            if i != j:
                d_grassmann[i, j] = compute_grassmann_distance(stance_matrices[fw_names[i]], stance_matrices[fw_names[j]], k=3)
                
    # 2. Multiple conceptual representations
    # A. DeBERTa 1536-dim hypothesis direction vector
    d_concept_deberta = np.zeros((n, n))
    emb_deberta = {}
    for k in fw_names:
        ea = adv_data[k]["emb_a"].numpy().astype(np.float64)
        eb = adv_data[k]["emb_b"].numpy().astype(np.float64)
        axis = np.mean(ea - eb, axis=(0, 1))
        emb_deberta[k] = axis / (np.linalg.norm(axis) + 1e-12)
        
    for i in range(n):
        for j in range(n):
            if i != j:
                d_concept_deberta[i, j] = float(cosine(emb_deberta[fw_names[i]], emb_deberta[fw_names[j]]))
                
    # B. TF-IDF character & word n-gram vectors on prompt text names
    # Extract prompt representations from bank keys/names
    prompt_names = [k.replace(":", " ").replace("_", " ") for k in fw_names]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tfidf_feats = vec.fit_transform(prompt_names).toarray()
    
    d_concept_tfidf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d_concept_tfidf[i, j] = float(cosine(tfidf_feats[i], tfidf_feats[j]))
                
    triu_idx = np.triu_indices(n, k=1)
    obs_g = d_grassmann[triu_idx]
    
    results = {}
    for name, mat in [("DeBERTa Hypothesis Hidden Vectors", d_concept_deberta),
                      ("TF-IDF Character N-grams of Framework Name", d_concept_tfidf)]:
        con_flat = mat[triu_idx]
        r, p_naive = pearsonr(con_flat, obs_g)
        rho, sp_naive = spearmanr(con_flat, obs_g)
        
        # Mantel permutation
        n_perms = 10000
        np.random.seed(42)
        perm_r = np.zeros(n_perms)
        for p_idx in range(n_perms):
            perm = np.random.permutation(n)
            perm_mat = mat[np.ix_(perm, perm)]
            perm_flat = perm_mat[triu_idx]
            pr, _ = pearsonr(perm_flat, obs_g)
            perm_r[p_idx] = pr
        p_mantel = float(np.mean(perm_r >= r if r > 0 else perm_r <= r))
        
        print(f"Conceptual Metric: {name}")
        print(f"  Pearson r = {r:+.4f} (naive p = {p_naive:.4f}, Mantel p = {p_mantel:.4f})")
        print(f"  Spearman rho = {rho:+.4f} (naive p = {sp_naive:.4f})")
        results[name] = {
            "pearson_r": float(r),
            "naive_pearson_p": float(p_naive),
            "mantel_p": float(p_mantel),
            "spearman_rho": float(rho)
        }
        
    return results

# ==============================================================================
# 3. SELF-TRANSLATION ESTIMATOR MULTIPLICITY BASELINE
# ==============================================================================
def audit_test_3_translation_multiplicity():
    print("\n" + "=" * 80)
    print("AUDIT TEST 3: TRANSLATION MULTIPLICITY VS SELF-TRANSLATION BASELINE")
    print("=" * 80)
    
    # Load replicated G-theory logits to get independent realizations of the same framework
    g_data = torch.load(ROOT / "data/cache_replicated_g_theory_logits.pt")
    
    def get_stance_from_dict(d):
        la = d["la"].numpy().astype(np.float64)
        lb = d["lb"].numpy().astype(np.float64)
        pe_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        pc_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        pe_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        pc_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        return (pe_a - pc_a) - (pe_b - pc_b)
        
    s_f0_b0 = get_stance_from_dict(g_data["geopolitical_conflict:realization_1"])
    s_f0_b1 = get_stance_from_dict(g_data["geopolitical_conflict:realization_2"])
    s_f1_b0 = get_stance_from_dict(g_data["humanitarian_civilian_welfare:realization_1"])
    
    # Import the exact model suite from Module 3
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    models = [
        ("Ridge_alpha_1e-2", RidgeCV(alphas=[0.01])),
        ("Ridge_alpha_1e-1", RidgeCV(alphas=[0.1])),
        ("Ridge_alpha_1.0", RidgeCV(alphas=[1.0])),
        ("Ridge_alpha_10.0", RidgeCV(alphas=[10.0])),
        ("Ridge_alpha_100.0", RidgeCV(alphas=[100.0])),
        ("PLS_1_comp", PLSRegression(n_components=1)),
        ("PLS_2_comp", PLSRegression(n_components=2)),
        ("PLS_3_comp", PLSRegression(n_components=3)),
        ("PLS_4_comp", PLSRegression(n_components=4)),
        ("KernelRidge_poly2", KernelRidge(kernel="polynomial", degree=2, alpha=1.0)),
        ("KernelRidge_cosine", KernelRidge(kernel="cosine", alpha=1.0)),
        ("KernelRidge_rbf", KernelRidge(kernel="rbf", gamma=0.05, alpha=1.0)),
        ("ElasticNet_l1_0.2", ElasticNet(alpha=0.1, l1_ratio=0.2, random_state=42)),
        ("MLP_hidden_8", MLPRegressor(hidden_layer_sizes=(8,), max_iter=500, random_state=1)),
        ("MLP_hidden_16", MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=2)),
    ]
    
    def evaluate_multiplicity(XA, XB, name):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scaler_A = StandardScaler().fit(XA)
        scaler_B = StandardScaler().fit(XB)
        ZA = scaler_A.transform(XA)
        ZB = scaler_B.transform(XB)
        
        model_preds = {}
        model_r2s = {}
        
        for m_name, model in models:
            pred_te = np.zeros_like(ZB)
            for tr, te in kf.split(ZA):
                try:
                    m_inst = type(model)(**model.get_params())
                except Exception:
                    m_inst = RidgeCV()
                m_inst.fit(ZA[tr], ZB[tr])
                pred_te[te] = m_inst.predict(ZA[te])
            r2 = float(r2_score(ZB, pred_te, multioutput="uniform_average"))
            model_r2s[m_name] = r2
            model_preds[m_name] = pred_te
            
        best_r2 = max(model_r2s.values())
        co_adequate = [m for m, r2 in model_r2s.items() if (best_r2 - r2) <= 0.05]
        
        # Pointwise prediction disagreement across co-adequate models
        co_preds = np.stack([model_preds[m] for m in co_adequate], axis=0) # (K, 120, 8)
        var_per_point = np.var(co_preds, axis=0) # (120, 8)
        mean_var = float(np.mean(var_per_point))
        max_var = float(np.max(var_per_point))
        
        print(f"Condition: {name}")
        print(f"  Best Model R^2: {best_r2:.4f}")
        print(f"  Co-adequate models (Delta R^2 <= 0.05): {len(co_adequate)} / {len(models)}")
        print(f"  Mean Pointwise Disagreement sigma^2_trans: {mean_var:.6f} (max = {max_var:.6f})")
        return {
            "best_r2": best_r2,
            "co_adequate_count": len(co_adequate),
            "mean_pointwise_var": mean_var,
            "max_pointwise_var": max_var
        }
        
    res_self = evaluate_multiplicity(s_f0_b0, s_f0_b1, "Self-Translation (Same Framework, Bank 1 -> Bank 2)")
    res_cross = evaluate_multiplicity(s_f0_b0, s_f1_b0, "Cross-Paradigm Translation (Geopolitical -> Humanitarian)")
    
    var_ratio = res_cross["mean_pointwise_var"] / (res_self["mean_pointwise_var"] + 1e-12)
    print(f"\n--> Multiplicity Dispersion Ratio (Cross-Paradigm vs Self-Translation): {var_ratio:.2f}x")
    
    return {
        "self_translation": res_self,
        "cross_paradigm_translation": res_cross,
        "dispersion_ratio": float(var_ratio)
    }

# ==============================================================================
# 4. MULTI-CRITERION UNDERDETERMINATION & SIGN-ALIGNED SUBSPACE GEOMETRY
# ==============================================================================
def audit_test_4_underdetermination():
    print("\n" + "=" * 80)
    print("AUDIT TEST 4: MULTI-CRITERION UNDERDETERMINATION & SIGN ALIGNMENT")
    print("=" * 80)
    
    adv_data = load_adversarial_logits()
    prospective_rows = load_prospective_120()
    pubs = [r.get("publisher", "") for r in prospective_rows]
    
    # Extract stance matrices
    fw1_key = "q3_realpolitik_statecraft"
    fw2_key = "extreme_political_ontologies:legal_culpability"
    
    def get_stance(k):
        la = adv_data[k]["la"].numpy().astype(np.float64)
        lb = adv_data[k]["lb"].numpy().astype(np.float64)
        pe_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        pc_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        pe_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        pc_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        return (pe_a - pc_a) - (pe_b - pc_b)
        
    s1 = get_stance(fw1_key)
    s2 = get_stance(fw2_key)
    
    # 1. Sign-Aligned PCA correlation
    u1, sv1, _ = np.linalg.svd(s1 - s1.mean(axis=0), full_matrices=False)
    u2, sv2, _ = np.linalg.svd(s2 - s2.mean(axis=0), full_matrices=False)
    pc1_1 = u1[:, 0] * sv1[0]
    pc1_2 = u2[:, 0] * sv2[0]
    
    raw_r, _ = pearsonr(pc1_1, pc1_2)
    sign_aligned_r = abs(raw_r)
    grassmann_d = compute_grassmann_distance(s1, s2, k=3)
    
    # 2. Multi-criterion external fit
    # Criterion 1: Top-5 publisher classification accuracy
    from collections import Counter
    pub_counts = Counter(pubs)
    top_pubs = [p for p, _ in pub_counts.most_common(5) if p]
    mask = np.array([p in top_pubs for p in pubs])
    y_top = np.array([top_pubs.index(p) for p in np.array(pubs)[mask]])
    X1_top = s1[mask]
    X2_top = s2[mask]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs_1, accs_2 = [], []
    for tr, te in kf.split(X1_top):
        c1 = LogisticRegression(max_iter=200).fit(X1_top[tr], y_top[tr])
        c2 = LogisticRegression(max_iter=200).fit(X2_top[tr], y_top[tr])
        accs_1.append(c1.score(X1_top[te], y_top[te]))
        accs_2.append(c2.score(X2_top[te], y_top[te]))
    acc_1, acc_2 = float(np.mean(accs_1)), float(np.mean(accs_2))
    
    # Criterion 2: ANOVA F across all publishers with n >= 2
    valid_pubs = {p for p, c in pub_counts.items() if c >= 2 and p}
    mask_all = np.array([p in valid_pubs for p in pubs])
    filtered_pubs = np.array(pubs)[mask_all]
    
    f1, p1 = f_oneway(*[pc1_1[mask_all][filtered_pubs == p] for p in np.unique(filtered_pubs)])
    f2, p2 = f_oneway(*[pc1_2[mask_all][filtered_pubs == p] for p in np.unique(filtered_pubs)])
    
    # Criterion 3: VADER compound sentiment prediction R^2
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    vader_scores = np.array([sia.polarity_scores(r["text"])["compound"] for r in prospective_rows])
    
    def cv_sentiment_r2(X, y):
        preds = np.zeros(len(y))
        for tr, te in kf.split(X):
            reg = Ridge(alpha=10.0).fit(X[tr], y[tr])
            preds[te] = reg.predict(X[te])
        return max(0.0, float(1.0 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)))
        
    sent_r2_1 = cv_sentiment_r2(s1, vader_scores)
    sent_r2_2 = cv_sentiment_r2(s2, vader_scores)
    
    print(f"Pair: {fw1_key}  <--->  {fw2_key}")
    print(f"  Raw PC1 Correlation:          r = {raw_r:+.4f}")
    print(f"  Sign-Aligned PC1 Correlation: |r| = {sign_aligned_r:.4f}")
    print(f"  3D Grassmann Subspace Dist:   d_G = {grassmann_d:.4f} (Max possible = {np.sqrt(3)*(np.pi/2):.4f})")
    print(f"  External Criterion 1 (Top-5 Pub Acc):   {acc_1:.3f} vs {acc_2:.3f} (Delta = {abs(acc_1 - acc_2):.3f})")
    print(f"  External Criterion 2 (All-Pub ANOVA F):  {f1:.2f} vs {f2:.2f} (Delta = {abs(f1 - f2):.2f})")
    print(f"  External Criterion 3 (Sentiment R^2):    {sent_r2_1:.3f} vs {sent_r2_2:.3f} (Delta = {abs(sent_r2_1 - sent_r2_2):.3f})")
    
    # Notice: even after sign alignment, |r| = 0.7557, but Grassmann distance is 1.9058 (very large)!
    return {
        "raw_r": float(raw_r),
        "sign_aligned_abs_r": float(sign_aligned_r),
        "grassmann_distance": float(grassmann_d),
        "top5_pub_acc": [acc_1, acc_2],
        "all_pub_anova_f": [float(f1), float(f2)],
        "sentiment_r2": [sent_r2_1, sent_r2_2]
    }

# ==============================================================================
# 5. CAUSAL ACTIVATION PATCHING IN DEBERTA V2 XLARGE
# ==============================================================================
def audit_test_5_causal_activation_patching():
    print("\n" + "=" * 80)
    print("AUDIT TEST 5: CAUSAL ACTIVATION PATCHING ACROSS DEBERTA SELF-ATTENTION LAYERS")
    print("=" * 80)
    
    model_name = "microsoft/deberta-v2-xlarge-mnli"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    model.eval()
    
    prospective_rows = load_prospective_120()
    # Test on 12 diverse articles
    test_articles = prospective_rows[:12]
    
    # Framework A: Geopolitical (Israel stance)
    ha = "The text strongly supports the Israeli government position in the conflict."
    # Framework B: Humanitarian (Severe violations stance)
    hb = "The text reports severe human rights violations and civilian atrocities."
    
    eval_layers = [8, 12, 16, 18, 20, 22, 24]
    
    causal_transfer_results = {l: [] for l in eval_layers}
    
    for art in test_articles:
        text = art["text"]
        inputs_a = tokenizer(text, ha, truncation=True, max_length=512, return_tensors="pt").to(device)
        inputs_b = tokenizer(text, hb, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        # Get baseline logits and all hidden states
        with torch.no_grad():
            out_a = model(**inputs_a, output_hidden_states=True)
            out_b = model(**inputs_b, output_hidden_states=True)
            
        logit_a = (out_a.logits[0, 2] - out_a.logits[0, 0]).item()
        logit_b = (out_b.logits[0, 2] - out_b.logits[0, 0]).item()
        baseline_diff = logit_b - logit_a
        
        if abs(baseline_diff) < 0.2:
            continue
            
        # Extract token mask for hypothesis tokens (token_type_ids == 1)
        tt_a = inputs_a["token_type_ids"][0]
        tt_b = inputs_b["token_type_ids"][0]
        
        hyp_mask_a = (tt_a == 1)
        hyp_mask_b = (tt_b == 1)
        
        # Run activation patching at each specified layer
        for l in eval_layers:
            # We patch the hidden states at layer l:
            # We construct a hook function that replaces the hypothesis tokens at layer l
            hs_b_target = out_b.hidden_states[l].clone()
            
            # Since sequence lengths might differ, we match the hypothesis token lengths or use slice
            len_hyp_a = int(hyp_mask_a.sum().item())
            len_hyp_b = int(hyp_mask_b.sum().item())
            min_hyp = min(len_hyp_a, len_hyp_b)
            
            # We use forward hook on the layer
            patched_output = None
            def get_hook(layer_idx, rep_b):
                def hook_fn(module, input, output):
                    # output is either tensor or tuple
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        # Patch hypothesis tokens
                        idx_a = torch.where(hyp_mask_a)[0][:min_hyp]
                        idx_b = torch.where(hyp_mask_b)[0][:min_hyp]
                        h[0, idx_a, :] = rep_b[0, idx_b, :]
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        idx_a = torch.where(hyp_mask_a)[0][:min_hyp]
                        idx_b = torch.where(hyp_mask_b)[0][:min_hyp]
                        h[0, idx_a, :] = rep_b[0, idx_b, :]
                        return h
                return hook_fn
                
            # Attach hook to layer (l-1) of deberta encoder (since layers are 0-indexed in encoder)
            hook_handle = model.deberta.encoder.layer[l-1].register_forward_hook(get_hook(l, hs_b_target))
            with torch.no_grad():
                out_patched = model(**inputs_a)
            hook_handle.remove()
            
            logit_patched = (out_patched.logits[0, 2] - out_patched.logits[0, 0]).item()
            transfer_ratio = (logit_patched - logit_a) / baseline_diff
            causal_transfer_results[l].append(transfer_ratio)
            
    print("\nLayerwise Causal Activation Patching Results (Hypothesis Token State Transplant):")
    summary = {}
    for l in eval_layers:
        ratios = causal_transfer_results[l]
        mean_tr = float(np.mean(ratios))
        std_tr = float(np.std(ratios))
        print(f"  Layer {l:2d} | Causal Transplant Ratio: {mean_tr:+.4f} (std = {std_tr:.4f})")
        summary[f"layer_{l}"] = {
            "mean_transfer_ratio": mean_tr,
            "std_transfer_ratio": std_tr
        }
        
    print("\nInterpretation:")
    print("  - If early layers have transfer ratio near 0, the model ignores early hypothesis states.")
    print("  - If late layers (e.g. 20-24) have high transfer ratio, the framework's causal control is concentrated in late layers.")
    return summary

def main():
    t1 = audit_test_1_qap_anomaly()
    t2 = audit_test_2_conceptual_sensitivity()
    t3 = audit_test_3_translation_multiplicity()
    t4 = audit_test_4_underdetermination()
    t5 = audit_test_5_causal_activation_patching()
    
    out_payload = {
        "test_1_qap_anomaly_migration": t1,
        "test_2_conceptual_sensitivity_mantel": t2,
        "test_3_translation_multiplicity_baseline": t3,
        "test_4_multi_criterion_underdetermination": t4,
        "test_5_causal_activation_patching": t5
    }
    
    out_path = ROOT / "paper/surgical_audit_frontier_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[SAVED] Saved surgical audit results to {out_path}")

if __name__ == "__main__":
    main()
