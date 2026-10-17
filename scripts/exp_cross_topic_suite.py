from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig
from scripts.run_gpu_validation import NEGATIVE_CONTROL_PAIRS, RANDOM_CONTROL_PAIRS
from scripts.verify_axis_coherence_exact import (
    remove_global_pc1,
    axis_coherence_matrix,
    subset_coherence,
    exhaustive_upper_tail,
)
from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


def load_yaml_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def build_filtered_corpus(
    output_path: Path,
    keywords: list[str],
    target_count: int = 60,
    max_per_pub: int = 6,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Harvesting articles for keywords {keywords[:3]} from real_corpus.jsonl...")
    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    seen_hashes = set()
    articles = []
    pub_counts: dict[str, int] = {}

    with open(real_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title} {content}".lower()

            if not any(kw in text for kw in keywords):
                continue

            h = hashlib.sha256(f"{title}|{content[:200]}".encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            pub = item.get("publisher", "Unknown")
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            pub_counts[pub] = pub_counts.get(pub, 0) + 1
            # Ensure text field exists
            if "text" not in item:
                item["text"] = f"{title}\n\n{content}"
            articles.append(item)

            if len(articles) >= target_count:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")

    print(f"[CORPUS] Saved {len(articles)} articles across {len(pub_counts)} publishers to {output_path}")
    return articles


def evaluate_topic(
    topic_name: str,
    pairs_ideo: list[tuple[str, str]],
    corpus: list[dict[str, Any]],
    cache_path: Path,
    extractor: UnifiedNLIExtractor,
    permutations: int = 10_000,
) -> dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"EVALUATING TOPIC: {topic_name}")
    print("=" * 75)

    publishers = [a.get("publisher", "unknown") for a in corpus]

    if cache_path.exists():
        print(f"[CACHE] Loading cached representations from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ideo_emb = cached["ideo_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
        ideo_logits = cached["ideo_logits"]
    else:
        pairs_ctrl = NEGATIVE_CONTROL_PAIRS
        pairs_rand = RANDOM_CONTROL_PAIRS

        print(f"[EXTRACT] 1/3 Extracting {topic_name} Ideological pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_ideo
        extractor.n_bots = len(pairs_ideo)
        batch_ideo = extractor.extract_batch(corpus, show_progress=True)

        print("[EXTRACT] 2/3 Extracting Structural Control pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_ctrl
        extractor.n_bots = len(pairs_ctrl)
        batch_ctrl = extractor.extract_batch(corpus, show_progress=True)

        print("[EXTRACT] 3/3 Extracting Random Control pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_rand
        extractor.n_bots = len(pairs_rand)
        batch_rand = extractor.extract_batch(corpus, show_progress=True)

        ideo_emb = batch_ideo["cls_per_bot"].cpu()
        ideo_logits = batch_ideo["logits_raw"].cpu()
        ctrl_emb = batch_ctrl["cls_per_bot"].cpu()
        rand_emb = batch_rand["cls_per_bot"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ideo_emb": ideo_emb,
                "ideo_logits": ideo_logits,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
            },
            cache_path,
        )
        print(f"[SAVE] Cached extraction to {cache_path}")

    # 1. High-Dimensional CLS Embedding Axis Coherence
    print("\n--- STEP 1: HIGH-DIMENSIONAL CLS AXIS COHERENCE EVALUATION ---")
    residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    obs_ideo = subset_coherence(pooled_matrix, range(8))
    obs_ctrl = subset_coherence(pooled_matrix, range(8, 16))
    obs_rand = subset_coherence(pooled_matrix, range(16, 24))

    print(f"  Cone Anisotropy (Global PC1 EVR) : {pc1_evr*100:.1f}%")
    print(f"  {topic_name} Ideology Coherence : {obs_ideo:+.6f}")
    print(f"  Structural Control Coherence    : {obs_ctrl:+.6f}")
    print(f"  Random Control Coherence        : {obs_rand:+.6f}")

    print("\n[NULL] Running exhaustive exact upper-tail enumeration (735,471 combinations)...")
    exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_ideo)
    print(f"  Exact Upper Tail Count          : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"  Exact p-value                   : {exact_null['exact_upper_tail_probability']:.6e}")

    # 2. Polarity Score Dimensionality Gating (10,000 draws)
    print(f"\n--- STEP 2: POLARITY SCORE PCA & DIMENSIONALITY NULL GATES ({permutations} DRAWS) ---")
    probs = torch.softmax(ideo_logits.float(), dim=-1).numpy()
    pol_matrix = probs[..., 2] - probs[..., 0]  # [N, 8]

    dim_gate = run_dimensionality_null_test(
        pol_matrix, topic_name, publishers=publishers, draws=permutations
    )

    c1 = dim_gate["components"][0]
    c2 = dim_gate["components"][1]
    print(f"  Polarity PC1 EVR                : {c1['observed_variance_ratio']*100:.2f}% (p = {c1['column_null']['p_value']:.4f}, null mean: {c1['column_null']['mean_variance_ratio']*100:.2f}%)")
    print(f"  Polarity PC2 EVR                : {c2['observed_variance_ratio']*100:.2f}% (p = {c2['column_null']['p_value']:.4f}, null mean: {c2['column_null']['mean_variance_ratio']*100:.2f}%)")
    print(f"  Detected Rank                   : {dim_gate['detected_rank_column_null']}")
    print(f"  Participation Ratio             : {dim_gate['observed_participation_ratio']:.2f} (null mean: {dim_gate['column_null_mean_participation_ratio']:.2f})")

    pca = PCA().fit(pol_matrix)
    scores = pca.transform(pol_matrix)[:, 0]
    min_idx = int(np.argmin(scores))
    max_idx = int(np.argmax(scores))

    print("\n--- Extreme Articles along PC1 ---")
    print(f"  Min Score ({scores[min_idx]:+.3f}): [{corpus[min_idx].get('publisher', 'unknown')}] {corpus[min_idx].get('title', corpus[min_idx].get('text', '')[:80])}")
    print(f"  Max Score ({scores[max_idx]:+.3f}): [{corpus[max_idx].get('publisher', 'unknown')}] {corpus[max_idx].get('title', corpus[max_idx].get('text', '')[:80])}")

    return {
        "topic": topic_name,
        "n_articles": len(corpus),
        "n_publishers": len(set(publishers)),
        "anisotropy_pc1_evr": float(pc1_evr),
        "coherence": {
            "ideology": float(obs_ideo),
            "structural": float(obs_ctrl),
            "random": float(obs_rand),
        },
        "exact_null": exact_null,
        "dimensionality_gate": dim_gate,
        "extremes": {
            "min_article": {
                "score": float(scores[min_idx]),
                "publisher": corpus[min_idx].get("publisher", "unknown"),
                "snippet": corpus[min_idx].get("title", corpus[min_idx].get("text", "")[:150]),
            },
            "max_article": {
                "score": float(scores[max_idx]),
                "publisher": corpus[max_idx].get("publisher", "unknown"),
                "snippet": corpus[max_idx].get("title", corpus[max_idx].get("text", "")[:150]),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Topic Replication Suite: Climate Politics & Abortion Rights")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    print(f"[SETUP] Initializing UnifiedNLIExtractor with {args.model_name} on {args.device}...")
    config = ExtractionConfig(
        model_name=args.model_name,
        device=args.device,
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    # 1. Climate Politics & Energy Transition
    climate_pairs = load_yaml_pairs(PROJECT_ROOT / "config" / "climate_framing_queries.yaml")
    climate_corpus = build_filtered_corpus(
        output_path=PROJECT_ROOT / "data" / "climate_N60_corpus.jsonl",
        keywords=[
            "climate change", "global warming", "fossil fuel", "carbon emission",
            "renewable energy", "greenhouse gas", "clean energy", "climate crisis",
        ],
        target_count=60,
        max_per_pub=6,
    )
    climate_res = evaluate_topic(
        topic_name="Climate Politics & Energy Transition",
        pairs_ideo=climate_pairs,
        corpus=climate_corpus,
        cache_path=PROJECT_ROOT / "data" / "cache_cross_topic" / "climate_N60_deberta.pt",
        extractor=extractor,
        permutations=args.permutations,
    )
    out_climate = PROJECT_ROOT / "paper" / "cross_topic_climate_replication.json"
    with open(out_climate, "w", encoding="utf-8") as f:
        json.dump(climate_res, f, indent=2)
    print(f"[SUCCESS] Saved climate replication results to {out_climate}")

    # 2. Abortion Rights & Reproductive Healthcare
    abortion_pairs = load_yaml_pairs(PROJECT_ROOT / "config" / "abortion_framing_queries.yaml")
    abortion_corpus = build_filtered_corpus(
        output_path=PROJECT_ROOT / "data" / "abortion_N60_corpus.jsonl",
        keywords=[
            "abortion", "reproductive right", "pro-choice", "pro-life",
            "roe v. wade", "roe v wade", "planned parenthood",
        ],
        target_count=60,
        max_per_pub=6,
    )
    abortion_res = evaluate_topic(
        topic_name="Abortion Rights & Reproductive Healthcare",
        pairs_ideo=abortion_pairs,
        corpus=abortion_corpus,
        cache_path=PROJECT_ROOT / "data" / "cache_cross_topic" / "abortion_N60_deberta.pt",
        extractor=extractor,
        permutations=args.permutations,
    )
    out_abortion = PROJECT_ROOT / "paper" / "cross_topic_abortion_replication.json"
    with open(out_abortion, "w", encoding="utf-8") as f:
        json.dump(abortion_res, f, indent=2)
    print(f"[SUCCESS] Saved abortion replication results to {out_abortion}")

    print("\n" + "=" * 75)
    print("ALL CROSS-TOPIC EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
