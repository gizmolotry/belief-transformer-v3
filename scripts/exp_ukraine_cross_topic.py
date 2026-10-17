from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig
from scripts.run_gpu_validation import (
    NEGATIVE_CONTROL_PAIRS,
    RANDOM_CONTROL_PAIRS,
)
from scripts.verify_axis_coherence_exact import (
    remove_global_pc1,
    axis_coherence_matrix,
    subset_coherence,
    exhaustive_upper_tail,
)


def load_ukraine_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "ukraine_cross_topic_queries.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pairs = []
    for k, v in data["queries"].items():
        pairs.append((v["A"], v["B"]))
    return pairs


def build_ukraine_corpus(
    output_path: Path,
    target_count: int = 60,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing Ukraine corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Harvesting Ukraine conflict articles from real_corpus.jsonl...")
    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"
    
    seen_hashes = set()
    articles = []
    pub_counts: dict[str, int] = {}
    max_per_pub = 12

    ukr_primary = ["ukraine", "ukrainian"]
    ukr_secondary = ["russia", "russian", "putin", "zelensky", "kyiv", "donbas", "crimea", "nato"]

    with open(real_corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            content = data.get("content") or data.get("text") or data.get("body") or ""
            title = data.get("title") or ""
            full_text = f"{title}\n\n{content}".strip()
            
            if len(content) < 400:
                continue

            full_lower = full_text.lower()
            has_primary = any(kw in full_lower for kw in ukr_primary)
            sec_count = sum(1 for kw in ukr_secondary if kw in full_lower)

            if not (has_primary and sec_count >= 2):
                continue

            fp = hashlib.sha256(content.strip()[:500].encode("utf-8")).hexdigest()[:16]
            if fp in seen_hashes:
                continue
            seen_hashes.add(fp)

            pub = data.get("publisher") or data.get("source") or "unknown"
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            pub_counts[pub] = pub_counts.get(pub, 0) + 1
            articles.append({
                "id": f"ukraine_{len(articles):03d}",
                "title": title,
                "publisher": pub,
                "content": content,
                "text": content,
            })

            if len(articles) >= target_count:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
    print(f"[CORPUS] Saved {len(articles)} Ukraine articles across {len(pub_counts)} publishers to {output_path}")
    return articles


def main():
    parser = argparse.ArgumentParser(description="Cross-topic replication on Ukraine war corpus.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus-output", type=Path, default=PROJECT_ROOT / "data" / "ukraine_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "ukraine_cross_topic_results.json")
    args = parser.parse_args()

    ukraine_pairs = load_ukraine_pairs()
    print(f"[SETUP] Loaded {len(ukraine_pairs)} Ukraine contrastive pairs.")

    articles = build_ukraine_corpus(args.corpus_output, target_count=60)
    cache_path = PROJECT_ROOT / "data" / "ukraine_N60_embeddings.pt"

    if cache_path.exists():
        print(f"[CACHE] Loading cached embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ukr_emb = cached["ukr_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
        ukr_logits = cached.get("ukr_logits")
    else:
        print(f"[EXTRACT] Initializing extractor with {args.model_name} on cuda...")
        config = ExtractionConfig(
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_cls_tokens=True,
        )
        extractor = UnifiedNLIExtractor(config)

        print("[EXTRACT] 1. Ukraine War pairs (8 axes)...")
        extractor.hypothesis_pairs = ukraine_pairs
        extractor.n_bots = len(ukraine_pairs)
        batch_ukr = extractor.extract_batch(articles, show_progress=True)
        ukr_emb = batch_ukr["cls_per_bot"].cpu()
        ukr_logits = batch_ukr["logits_raw"].cpu()

        print("[EXTRACT] 2. Structural Control pairs (8 axes)...")
        extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
        extractor.n_bots = len(NEGATIVE_CONTROL_PAIRS)
        ctrl_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

        print("[EXTRACT] 3. Random Control pairs (8 axes)...")
        extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
        extractor.n_bots = len(RANDOM_CONTROL_PAIRS)
        rand_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ukr_emb": ukr_emb,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
                "ukr_logits": ukr_logits,
            },
            cache_path,
        )
        print(f"[SAVE] Cached Ukraine embeddings to {cache_path}")

    # Coherence evaluation across 24 pooled axes
    residuals, pc1_evr = remove_global_pc1([ukr_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    c_ukr = subset_coherence(pooled_matrix, range(8))
    c_ctrl = subset_coherence(pooled_matrix, range(8, 16))
    c_rand = subset_coherence(pooled_matrix, range(16, 24))

    print("\n" + "=" * 70)
    print("CROSS-TOPIC (UKRAINE) REPLICATION RESULTS")
    print("=" * 70)
    print(f"Ukraine Coherence        : {c_ukr:+.6f}")
    print(f"Structure Control        : {c_ctrl:+.6f}")
    print(f"Random Control           : {c_rand:+.6f}")
    print(f"PC1 Explained Variance   : {pc1_evr:.4f}")

    print("[NULL] Running exhaustive exact upper-tail enumeration (735,471 combinations)...")
    exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=c_ukr)
    print(f"Exact Upper Tail Count   : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"Exact p-value            : {exact_null['exact_upper_tail_probability']:.6e}")
    print("=" * 70)

    # 1. Embedding Score PCA
    normed_ukr = torch.nn.functional.normalize(residuals[0], p=2, dim=-1)
    scores = normed_ukr.mean(dim=-1).numpy()
    pca_emb = PCA().fit(scores)
    evr_emb = pca_emb.explained_variance_ratio_

    # 2. Logits Polarity PCA: P(E) - P(C)
    pca_logits_data = None
    if ukr_logits is not None:
        probs = torch.softmax(ukr_logits.float(), dim=-1).numpy()
        polarity_matrix = probs[..., 2] - probs[..., 0]  # [N, 8]
        pca_pol = PCA().fit(polarity_matrix)
        evr_pol = pca_pol.explained_variance_ratio_
        pca_logits_data = {
            "pc1_evr": float(evr_pol[0]),
            "pc2_evr": float(evr_pol[1]),
            "cum_2pc": float(np.sum(evr_pol[:2])),
            "all_evr": [float(x) for x in evr_pol],
        }
        print(f"Logits Polarity PCA      : PC1={evr_pol[0]*100:.1f}%, PC2={evr_pol[1]*100:.1f}%, Cum2={np.sum(evr_pol[:2])*100:.1f}%")

    results = {
        "dataset": "ukraine_N60_corpus",
        "n_articles": len(articles),
        "coherence": {
            "ukraine": c_ukr,
            "structure": c_ctrl,
            "random": c_rand,
        },
        "exact_null": exact_null,
        "pca_embedding": {
            "pc1_evr": float(evr_emb[0]),
            "pc2_evr": float(evr_emb[1]),
            "cum_2pc": float(np.sum(evr_emb[:2])),
            "all_evr": [float(x) for x in evr_emb],
        },
        "pca_logits_polarity": pca_logits_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Saved Ukraine cross-topic results to {args.output}")


if __name__ == "__main__":
    main()
