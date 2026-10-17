"""
Module 4: Empirical Underdetermination & Mechanistic Constitution vs Readout
Investigates:
- Point 4: Empirical Underdetermination (equivalent external fit, divergent geometry)
- Point 6: Mechanistic Constitution vs Readout (layerwise transformer hidden state tracking in DeBERTa)
"""

import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import f_oneway, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent

def load_adversarial_logits():
    cache_path = ROOT / "data/cache_adversarial_kuhnian_logits.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}")
    return torch.load(cache_path)

def load_prospective_120():
    corpus_path = ROOT / "data/prospective_confirmatory_120_corpus.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    pubs = [r.get("publisher", "") for r in rows]
    return rows, pubs

def compute_subspace_distance(X1, X2, k=3):
    u1, _, _ = np.linalg.svd(X1 - X1.mean(axis=0), full_matrices=False)
    u2, _, _ = np.linalg.svd(X2 - X2.mean(axis=0), full_matrices=False)
    q1 = u1[:, :k]
    q2 = u2[:, :k]
    _, s, _ = np.linalg.svd(q1.T @ q2)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return float(np.sqrt(np.sum(theta**2)))

def jaccard_similarity(set1, set2):
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)

def evaluate_point_4_underdetermination():
    print("=" * 80)
    print("POINT 4: EMPIRICAL UNDERDETERMINATION (EQUIVALENT FIT, DIVERGENT GEOMETRY)")
    print("=" * 80)
    
    adv_data = load_adversarial_logits()
    prospective_rows, prospective_pubs = load_prospective_120()
    
    from collections import Counter
    pub_counts = Counter(prospective_pubs)
    valid_pubs = {p for p, c in pub_counts.items() if c >= 2 and p}
    pub_mask = np.array([p in valid_pubs for p in prospective_pubs])
    filtered_pubs = np.array([p for p in prospective_pubs if p in valid_pubs])
    
    framework_metrics = {}
    stance_matrices = {}
    
    for fw_key, fw_val in adv_data.items():
        if fw_key.startswith("adversarial_"):
            continue
        la = fw_val["la"].numpy().astype(np.float64)
        lb = fw_val["lb"].numpy().astype(np.float64)
        p_contra_a = np.exp(la[:, :, 0]) / np.sum(np.exp(la), axis=-1)
        p_entail_a = np.exp(la[:, :, 2]) / np.sum(np.exp(la), axis=-1)
        p_contra_b = np.exp(lb[:, :, 0]) / np.sum(np.exp(lb), axis=-1)
        p_entail_b = np.exp(lb[:, :, 2]) / np.sum(np.exp(lb), axis=-1)
        
        stance_raw = ((p_entail_a - p_contra_a) - (p_entail_b - p_contra_b)).astype(np.float64)
        stance_matrices[fw_key] = stance_raw
        
        stance_centered = stance_raw - stance_raw.mean(axis=0)
        u, s, vt = np.linalg.svd(stance_centered, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        
        pc1_sub = pc1[pub_mask]
        groups = [pc1_sub[filtered_pubs == pub] for pub in np.unique(filtered_pubs)]
        f_stat, f_p = f_oneway(*groups)
        
        top_pubs = [p for p, _ in pub_counts.most_common(5) if p]
        top_mask = np.array([p in top_pubs for p in prospective_pubs])
        y_top = np.array([top_pubs.index(p) for p in np.array(prospective_pubs)[top_mask]])
        X_top = stance_raw[top_mask]
        
        cv_accs = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(X_top):
            clf = LogisticRegression(C=1.0, max_iter=200)
            clf.fit(X_top[tr], y_top[tr])
            cv_accs.append(clf.score(X_top[te], y_top[te]))
        ext_acc = float(np.mean(cv_accs))
        
        dists = np.linalg.norm(stance_centered, axis=1)
        anom_top10 = np.argsort(dists)[-12:]
        
        framework_metrics[fw_key] = {
            "anova_F": float(f_stat),
            "anova_p": float(f_p),
            "pub_acc": ext_acc,
            "pc1": pc1,
            "anom_top10": set(anom_top10.tolist())
        }
        print(f"Framework: {fw_key[:38]:38s} | ANOVA F={f_stat:5.2f} (p={f_p:.4f}) | Top-5 Pub Acc={ext_acc:.3f}")
    
    fw_keys = list(framework_metrics.keys())
    underdetermined_pairs = []
    
    for i in range(len(fw_keys)):
        for j in range(i + 1, len(fw_keys)):
            k1, k2 = fw_keys[i], fw_keys[j]
            m1, m2 = framework_metrics[k1], framework_metrics[k2]
            
            delta_acc = abs(m1["pub_acc"] - m2["pub_acc"])
            delta_F = abs(m1["anova_F"] - m2["anova_F"])
            
            r_pc1, _ = pearsonr(m1["pc1"], m2["pc1"])
            grassmann = compute_subspace_distance(stance_matrices[k1], stance_matrices[k2], k=3)
            jaccard = jaccard_similarity(m1["anom_top10"], m2["anom_top10"])
            
            underdetermined_pairs.append({
                "fw1": k1,
                "fw2": k2,
                "pub_acc_1": m1["pub_acc"],
                "pub_acc_2": m2["pub_acc"],
                "delta_acc": float(delta_acc),
                "anova_F_1": m1["anova_F"],
                "anova_F_2": m2["anova_F"],
                "delta_F": float(delta_F),
                "r_pc1": float(r_pc1),
                "grassmann_dist": float(grassmann),
                "jaccard_anomalies": float(jaccard)
            })
    
    co_adequate = [p for p in underdetermined_pairs if p["delta_acc"] <= 0.05 and p["delta_F"] <= 0.6]
    co_adequate.sort(key=lambda x: x["r_pc1"])
    
    print(f"\nFound {len(co_adequate)} empirically co-adequate framework pairs (|Delta Acc| <= 0.05, |Delta F| <= 0.60).")
    print("\nTop Underdetermined Framework Pairs (Equivalent External Fit, Divergent Internal Geometry):")
    for p in co_adequate[:5]:
        print(f"Pair: {p['fw1']}  <--->  {p['fw2']}")
        print(f"  External Fit: Acc={p['pub_acc_1']:.3f} vs {p['pub_acc_2']:.3f} (Delta={p['delta_acc']:.3f}) | F={p['anova_F_1']:.2f} vs {p['anova_F_2']:.2f}")
        print(f"  Internal Geometry: r(PC1) = {p['r_pc1']:+.4f} | Grassmann Dist = {p['grassmann_dist']:.4f} | Anomaly Jaccard = {p['jaccard_anomalies']:.3f}\n")
        
    return {
        "framework_metrics": {k: {mk: mv for mk, mv in v.items() if mk != "pc1" and mk != "anom_top10"} for k, v in framework_metrics.items()},
        "top_co_adequate_divergent_pairs": co_adequate[:10]
    }


def evaluate_point_6_mechanistic_constitution():
    print("=" * 80)
    print("POINT 6: MECHANISTIC CONSTITUTION VS READOUT (LAYERWISE DEBERTA TRACKING)")
    print("=" * 80)
    
    model_name = "microsoft/deberta-v2-xlarge-mnli"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    model.eval()
    
    prospective_rows, _ = load_prospective_120()
    selected_articles = prospective_rows[:24]
    
    hyp_pairs = [
        {
            "name": "Geopolitical",
            "ha": "The text strongly supports the Israeli government position in the conflict.",
            "hb": "The text strongly supports the Palestinian perspective in the conflict."
        },
        {
            "name": "Humanitarian",
            "ha": "The text reports full compliance with international humanitarian standards.",
            "hb": "The text reports severe human rights violations and civilian atrocities."
        }
    ]
    
    eval_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    num_layers = len(eval_layers)
    results_by_pair = {}
    
    for pair in hyp_pairs:
        pair_name = pair["name"]
        ha = pair["ha"]
        hb = pair["hb"]
        print(f"\n--- Tracking Layerwise State Evolution for Hypothesis Pair: {pair_name} ---")
        
        art_penetration = np.zeros((len(selected_articles), num_layers))
        cls_divergence = np.zeros((len(selected_articles), num_layers))
        
        final_stance_logits = []
        layer_cls_reps = [[] for _ in range(num_layers)]
        layer_art_reps = [[] for _ in range(num_layers)]
        
        for art_idx, art_row in enumerate(selected_articles):
            text = art_row["text"]
            inputs_a = tokenizer(text, ha, truncation=True, max_length=512, return_tensors="pt").to(device)
            inputs_b = tokenizer(text, hb, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out_a = model(**inputs_a, output_hidden_states=True)
                out_b = model(**inputs_b, output_hidden_states=True)
            
            logits_a = out_a.logits.squeeze(0).cpu().numpy()
            logits_b = out_b.logits.squeeze(0).cpu().numpy()
            final_stance = (logits_a[2] - logits_a[0]) - (logits_b[2] - logits_b[0])
            final_stance_logits.append(final_stance)
            
            tt_a = inputs_a["token_type_ids"].squeeze(0).cpu().numpy()
            tt_b = inputs_b["token_type_ids"].squeeze(0).cpu().numpy()
            
            art_mask_a = (tt_a == 0)
            art_mask_a[0] = False
            art_mask_b = (tt_b == 0)
            art_mask_b[0] = False
            
            for l_idx, layer_num in enumerate(eval_layers):
                hs_a = out_a.hidden_states[layer_num].squeeze(0)
                hs_b = out_b.hidden_states[layer_num].squeeze(0)
                
                cls_a = hs_a[0].cpu().numpy()
                cls_b = hs_b[0].cpu().numpy()
                
                cls_div = 1.0 - np.dot(cls_a, cls_b) / (np.linalg.norm(cls_a) * np.linalg.norm(cls_b) + 1e-12)
                cls_divergence[art_idx, l_idx] = cls_div
                
                art_a = hs_a[art_mask_a].mean(dim=0).cpu().numpy()
                art_b = hs_b[art_mask_b].mean(dim=0).cpu().numpy()
                
                art_pen = 1.0 - np.dot(art_a, art_b) / (np.linalg.norm(art_a) * np.linalg.norm(art_b) + 1e-12)
                art_penetration[art_idx, l_idx] = art_pen
                
                layer_cls_reps[l_idx].append(cls_a - cls_b)
                layer_art_reps[l_idx].append(art_a - art_b)
        
        final_y = np.array(final_stance_logits)
        probe_r2_cls = []
        probe_r2_art = []
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        
        for l_idx, layer_num in enumerate(eval_layers):
            X_cls = np.array(layer_cls_reps[l_idx])
            X_art = np.array(layer_art_reps[l_idx])
            
            def cv_fit(X, y):
                preds = np.zeros_like(y)
                for tr, te in kf.split(X):
                    reg = Ridge(alpha=10.0)
                    reg.fit(X[tr], y[tr])
                    preds[te] = reg.predict(X[te])
                ss_res = np.sum((y - preds)**2)
                ss_tot = np.sum((y - y.mean())**2)
                return max(0.0, float(1.0 - ss_res / ss_tot))
            
            r2_c = cv_fit(X_cls, final_y)
            r2_a = cv_fit(X_art, final_y)
            probe_r2_cls.append(r2_c)
            probe_r2_art.append(r2_a)
            
            mean_pen = float(np.mean(art_penetration[:, l_idx]))
            mean_cdiv = float(np.mean(cls_divergence[:, l_idx]))
            print(f"Layer {layer_num:2d} | Art Token Penetration: {mean_pen:.5f} | CLS Divergence: {mean_cdiv:.5f} | Probe R^2 (Art): {r2_a:.4f} | Probe R^2 (CLS): {r2_c:.4f}")
        
        results_by_pair[pair_name] = {
            "eval_layers": eval_layers,
            "mean_art_token_penetration": [float(x) for x in np.mean(art_penetration, axis=0)],
            "mean_cls_divergence": [float(x) for x in np.mean(cls_divergence, axis=0)],
            "probe_r2_article_tokens": probe_r2_art,
            "probe_r2_cls_token": probe_r2_cls,
            "readout_vs_constitutive_summary": {
                "layer_0_penetration": float(np.mean(art_penetration[:, 0])),
                "layer_12_penetration": float(np.mean(art_penetration[:, 6])),
                "layer_24_penetration": float(np.mean(art_penetration[:, -1])),
                "midpoint_penetration_pct": float(np.mean(art_penetration[:, 6]) / (np.mean(art_penetration[:, -1]) + 1e-12) * 100),
                "midpoint_cls_probe_r2": float(probe_r2_cls[6]),
                "final_cls_probe_r2": float(probe_r2_cls[-1])
            }
        }
        
    return results_by_pair

def main():
    pt4_res = evaluate_point_4_underdetermination()
    pt6_res = evaluate_point_6_mechanistic_constitution()
    
    out_payload = {
        "point_4_empirical_underdetermination": pt4_res,
        "point_6_mechanistic_constitution": pt6_res
    }
    
    out_path = ROOT / "paper/frontier_underdetermination_mechanisms_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[SAVED] Saved Module 4 results to {out_path}")

if __name__ == "__main__":
    main()
