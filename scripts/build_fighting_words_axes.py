import numpy as np
import json
import spacy
from collections import Counter, defaultdict
import os
import argparse
from scipy.stats import dirichlet

def get_dirichlet_log_odds_ratio(counts_t, counts_bg, n_t, n_bg, prior):
    """
    Computes log odds ratio stabilized by a Dirichlet prior.
    """
    delta = {}
    alpha_0 = sum(prior.values())
    for w in set(counts_t.keys()).union(counts_bg.keys()):
        alpha_w = prior.get(w, 0.01) # fallback smooth
        
        c_t = counts_t.get(w, 0)
        c_bg = counts_bg.get(w, 0)
        
        num = (c_t + alpha_w) / (n_t + alpha_0 - (c_t + alpha_w))
        den = (c_bg + alpha_w) / (n_bg + alpha_0 - (c_bg + alpha_w))
        
        delta[w] = np.log(num) - np.log(den)
    return delta

def build_fighting_words_axes(corpus_path, output_path, n_phrases=50):
    print("Loading SpaCy for phrase extraction...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. You must run: python -m spacy download en_core_web_sm")
        return
        
    print(f"Reading corpus from {corpus_path}...")
    articles_by_cluster = defaultdict(list)
    background_counts = Counter()
    n_bg_total = 0
    
    with open(corpus_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000: # Limit for memory/speed during skeleton execution
                break
            doc = json.loads(line)
            text = doc.get("content", "")
            cluster = doc.get("topic_cluster", 0) # Fallback to 0 if topic missing
            
            # Extract phrases (noun chunks)
            parsed = nlp(text)
            phrases = [chunk.text.lower() for chunk in parsed.noun_chunks if len(chunk.text.split()) > 1]
            
            articles_by_cluster[cluster].append({
                "id": doc.get("article_id", str(i)),
                "text": text,
                "phrases": phrases
            })
            
            background_counts.update(phrases)
            n_bg_total += len(phrases)
            
    print(f"Extracted phrases. Computing Dirichlet prior from {n_bg_total} total background phrases.")
    
    # Establish Prior based on background distribution (proportional to total alpha_0 = 100)
    alpha_0_param = 100.0
    prior = {w: (c / n_bg_total) * alpha_0_param for w, c in background_counts.items()}
    
    axes_embeddings = []
    
    # Normally we'd use a real SentenceTransformer here, mocked for skeleton
    print("Mocking SentenceTransformer projection...")
    embedding_dim = 1536
    
    for cluster, articles in articles_by_cluster.items():
        print(f"\nProcessing Cluster {cluster} (n={len(articles)} articles)")
        counts_t = Counter()
        n_t_total = 0
        
        for art in articles:
            counts_t.update(art["phrases"])
            n_t_total += len(art["phrases"])
            
        counts_rest = background_counts - counts_t
        n_rest = n_bg_total - n_t_total
        
        delta = get_dirichlet_log_odds_ratio(counts_t, counts_rest, n_t_total, n_rest, prior)
        
        # Sort phrases by log-odds to find most discriminative for this cluster
        top_phrases = sorted(delta.items(), key=lambda x: x[1], reverse=True)[:n_phrases]
        print(f"Top 5 phrases for Cluster {cluster}: {[p[0] for p in top_phrases[:5]]}")
        
        # In a real run, we find exemplar sentences containing these phrases
        # and embed them with SentenceTransformer. For skeleton, generate a random projection vector.
        cluster_axis = np.random.normal(0, 1, embedding_dim)
        cluster_axis = cluster_axis / np.linalg.norm(cluster_axis)
        axes_embeddings.append(cluster_axis)
        
    axes_embeddings = np.array(axes_embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, axes_embeddings)
    print(f"\nSaved {axes_embeddings.shape[0]} fighting word axes to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/real_corpus.jsonl")
    parser.add_argument("--output", default="data/fighting_words_axes.npy")
    args = parser.parse_args()
    build_fighting_words_axes(args.corpus, args.output)
