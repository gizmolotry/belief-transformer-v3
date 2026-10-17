import torch
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import argparse

def project_out(vectors, direction):
    direction = direction / torch.norm(direction)
    if vectors.dim() == 3:
        proj = (vectors * direction.view(1, 1, -1)).sum(dim=2, keepdim=True)
        return vectors - proj * direction.view(1, 1, -1)
    else:
        proj = (vectors * direction).sum(dim=1, keepdim=True)
        return vectors - proj * direction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='data/controlled_N_dense_corpus.jsonl')
    parser.add_argument('--embeds', default='data/controlled_ideo_embeddings_dense.pt')
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    
    # 1. Load MBFC continuous scores
    mbfc_scores = {}
    mbfc_path = root / "data" / "mbfc_continuous_scores.json"
    if mbfc_path.exists():
        with open(mbfc_path, "r") as f:
            mbfc_scores = json.load(f)
            
    # 2. Load articles to get publishers
    with open(root / args.corpus, "r", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f]
        
    publishers = [a.get("publication") or a.get("source") for a in articles]
    
    # 3. Load the embeddings
    embeds_path = root / args.embeds
    if not embeds_path.exists():
        print(f"Error: {embeds_path} not found.")
        print("Please extract embeddings first.")
        return
        
    payload = torch.load(embeds_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "ideo_emb" in payload:
            ideo_emb = payload["ideo_emb"]
            if not isinstance(ideo_emb, torch.Tensor):
                ideo_emb = torch.tensor(ideo_emb)
        elif "nli_pairs" in payload:
            pairs = payload["nli_pairs"]
            if isinstance(pairs[0], dict) and "cls_per_bot" in pairs[0]:
                ideo_emb = torch.stack([p["cls_per_bot"] for p in pairs], dim=0)
            elif isinstance(pairs[0], dict) and "embedding" in pairs[0]:
                ideo_emb = torch.stack([p["embedding"] for p in pairs], dim=0)
            else:
                raise ValueError("Unrecognized structure inside nli_pairs")
        else:
            raise ValueError(f"Unrecognized payload dictionary keys: {list(payload.keys())}")
    elif isinstance(payload, torch.Tensor):
        ideo_emb = payload
    else:
        ideo_emb = torch.tensor(payload)
    
    if ideo_emb.dim() == 2:
        # Infer dimensions based on 8 hypotheses
        N = ideo_emb.shape[0] // 8
        ideo_emb = ideo_emb.view(N, 8, -1)
        
    print(f"Loaded Ideology Embeddings: {ideo_emb.shape}")
    
    # 4. Compute semantic PC1 (across all vectors) to remove anisotropy
    flat_emb = ideo_emb.reshape(-1, ideo_emb.shape[-1])
    center = flat_emb.mean(dim=0)
    flat_emb_c = flat_emb - center
    U, S, V = torch.svd(flat_emb_c)
    semantic_pc1 = V[:, 0]
    
    ideo_resid = project_out(ideo_emb, semantic_pc1)
    
    # 5. Compute the primary ideological axis
    article_ideo = ideo_resid.mean(dim=1)
    
    a_center = article_ideo.mean(dim=0)
    a_c = article_ideo - a_center
    U_i, S_i, V_i = torch.svd(a_c)
    ideology_axis = V_i[:, 0]
    
    # 6. Project articles onto the ideology axis
    geometric_scores = (article_ideo * ideology_axis).sum(dim=1).numpy()
    
    # Ensure polarity
    cnn_idx = [i for i, p in enumerate(publishers) if p == "cnn.com"]
    fox_idx = [i for i, p in enumerate(publishers) if p == "foxnews.com"]
    if cnn_idx and fox_idx:
        if geometric_scores[cnn_idx].mean() > geometric_scores[fox_idx].mean():
            geometric_scores = -geometric_scores
            
    # 7. Aggregate by publisher
    pub_geo_scores = {}
    pub_mbfc_scores = {}
    for i, pub in enumerate(publishers):
        if pub in mbfc_scores:
            if pub not in pub_geo_scores:
                pub_geo_scores[pub] = []
            pub_geo_scores[pub].append(geometric_scores[i])
            pub_mbfc_scores[pub] = mbfc_scores[pub]
            
    # Compute publisher means
    print("\n--- Publisher Level Continuous Mapping ---")
    x_geo, y_mbfc = [], []
    for pub in pub_geo_scores.keys():
        mean_geo = np.mean(pub_geo_scores[pub])
        val_mbfc = pub_mbfc_scores[pub]
        x_geo.append(mean_geo)
        y_mbfc.append(val_mbfc)
        print(f"{pub.ljust(25)} | Geo: {mean_geo:>6.3f} | MBFC: {val_mbfc:>6.2f}")
        
    x_geo = np.array(x_geo)
    y_mbfc = np.array(y_mbfc)
    
    if len(x_geo) >= 8:
        p_r, p_p = pearsonr(x_geo, y_mbfc)
        s_r, s_p = spearmanr(x_geo, y_mbfc)
        
        print("\n--- Continuous Geometric Correlation (N={}) ---".format(len(x_geo)))
        print(f"Pearson r:  {p_r:.3f} (p={p_p:.4f})")
        print(f"Spearman rho: {s_r:.3f} (p={s_p:.4f})")
    else:
        print(f"\nInsufficient N ({len(x_geo)}) for correlation.")
        
if __name__ == '__main__':
    main()
