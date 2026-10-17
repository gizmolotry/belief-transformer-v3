from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig
from scripts.run_gpu_validation import (
    IDEOLOGICAL_PAIRS,
    NEGATIVE_CONTROL_PAIRS,
    RANDOM_CONTROL_PAIRS,
)
from scripts.verify_axis_coherence_exact import (
    remove_global_pc1,
    axis_coherence_matrix,
    subset_coherence,
    exhaustive_upper_tail,
)


def get_seen_fingerprints() -> set[str]:
    seen = set()
    for fname in ["controlled_N_dense_corpus.jsonl", "controlled_N60_corpus.jsonl"]:
        fpath = PROJECT_ROOT / "data" / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("content") or data.get("text") or data.get("body") or ""
                    url = data.get("url") or data.get("link") or ""
                    if text:
                        seen.add(hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16])
                    if url:
                        seen.add(hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16])
    return seen


def build_fresh_unseen_corpus(
    output_path: Path,
    target_count: int = 120,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing fresh unseen corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    seen_fps = get_seen_fingerprints()
    print(f"[CORPUS] Identified {len(seen_fps)} previously seen article fingerprints to exclude.")

    fresh_articles = []
    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    pub_counts: dict[str, int] = {}
    max_per_pub = 15

    with open(real_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("content") or data.get("text") or data.get("body") or ""
            url = data.get("url") or data.get("link") or ""
            if not text or len(text.strip()) < 150:
                continue

            fp_text = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
            fp_url = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16] if url else ""

            if fp_text in seen_fps or fp_url in seen_fps:
                continue

            pub = data.get("publisher") or data.get("publication") or data.get("source") or "unknown"
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            data["text"] = text
            data["content"] = text
            fresh_articles.append(data)
            pub_counts[pub] = pub_counts.get(pub, 0) + 1

            if len(fresh_articles) >= target_count:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in fresh_articles:
            f.write(json.dumps(a) + "\n")
    print(f"[CORPUS] Assembled and saved {len(fresh_articles)} strictly unseen articles across {len(pub_counts)} publishers to {output_path}")
    return fresh_articles


def main():
    parser = argparse.ArgumentParser(description="Preregistered out-of-sample replication on fresh unseen data.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus-output", type=Path, default=PROJECT_ROOT / "data" / "fresh_unseen_120_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "fresh_unseen_replication_results.json")
    args = parser.parse_args()

    articles = build_fresh_unseen_corpus(args.corpus_output, target_count=120)

    cache_path = PROJECT_ROOT / "data" / "fresh_unseen_120_embeddings.pt"

    if cache_path.exists():
        print(f"[CACHE] Loading cached embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ideo_emb = cached["ideo_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
        ideo_logits = cached.get("ideo_logits")
    else:
        print(f"[EXTRACT] Initializing extractor with {args.model_name} on cuda...")
        config = ExtractionConfig(
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_cls_tokens=True,
        )
        extractor = UnifiedNLIExtractor(config)

        print("[EXTRACT] 1. Ideological pairs (8 axes)...")
        extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
        extractor.n_bots = len(IDEOLOGICAL_PAIRS)
        batch_ideo = extractor.extract_batch(articles, show_progress=True)
        ideo_emb = batch_ideo["cls_per_bot"].cpu()
        ideo_logits = batch_ideo["logits_raw"].cpu()

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
                "ideo_emb": ideo_emb,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
                "ideo_logits": ideo_logits,
            },
            cache_path,
        )
        print(f"[SAVE] Saved embeddings to {cache_path}")

    # Coherence evaluation
    residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    ideology = subset_coherence(pooled_matrix, range(8))
    structure = subset_coherence(pooled_matrix, range(8, 16))
    random = subset_coherence(pooled_matrix, range(16, 24))

    print("\n" + "=" * 70)
    print("FRESH UNSEEN DATA REPLICATION RESULTS (N=120 articles)")
    print("=" * 70)
    print(f"Ideology Coherence       : {ideology:+.6f}")
    print(f"Structure Coherence      : {structure:+.6f}")
    print(f"Random Coherence         : {random:+.6f}")
    print(f"PC1 Explained Variance   : {pc1_evr:.4f}")

    print("[NULL] Running exhaustive exact upper-tail enumeration (735,471 combinations)...")
    exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=ideology)
    print(f"Exact Upper Tail Count   : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"Exact p-value            : {exact_null['exact_upper_tail_probability']:.6e}")
    print("=" * 70)

    # 1. Embedding Score PCA
    normed_ideo = torch.nn.functional.normalize(residuals[0], p=2, dim=-1)
    scores = normed_ideo.mean(dim=-1).numpy()
    pca_emb = PCA().fit(scores)
    evr_emb = pca_emb.explained_variance_ratio_

    # 2. Logits Polarity PCA: P(E) - P(C)
    pca_logits_data = None
    if ideo_logits is not None:
        probs = torch.softmax(ideo_logits.float(), dim=-1).numpy()
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
        "dataset": "fresh_unseen_120_articles",
        "n_articles": len(articles),
        "coherence": {
            "ideology": ideology,
            "structure": structure,
            "random": random,
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
    print(f"[DONE] Saved fresh unseen replication results to {args.output}")


if __name__ == "__main__":
    main()
