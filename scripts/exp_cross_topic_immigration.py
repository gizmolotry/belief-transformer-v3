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


def load_immigration_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "immigration_framing_queries.yaml"
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def build_immigration_corpus(
    output_path: Path,
    target_count: int = 60,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing immigration corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Harvesting immigration articles from real_corpus.jsonl...")
    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    keywords = ["immigration", "border patrol", "migrant", "asylum", "deportation", "undocumented"]
    seen_hashes = set()
    articles = []
    pub_counts: dict[str, int] = {}
    max_per_pub = 6

    with open(real_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("content") or data.get("text") or data.get("body") or ""
            if not text or len(text.strip()) < 200:
                continue

            text_lower = text.lower()
            if not any(kw in text_lower for kw in keywords):
                continue

            # Exclude Middle East conflict articles
            if any(term in text_lower for term in ["gaza", "hamas", "netanyahu", "tel aviv", "idf"]):
                continue

            fp = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
            if fp in seen_hashes:
                continue
            seen_hashes.add(fp)

            pub = data.get("publisher") or data.get("publication") or data.get("source") or "unknown"
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            data["text"] = text
            data["content"] = text
            articles.append(data)
            pub_counts[pub] = pub_counts.get(pub, 0) + 1

            if len(articles) >= target_count:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
    print(f"[CORPUS] Assembled {len(articles)} immigration articles across {len(pub_counts)} publishers to {output_path}")
    return articles


def main():
    parser = argparse.ArgumentParser(description="Cross-topic replication on US immigration discourse.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus-output", type=Path, default=PROJECT_ROOT / "data" / "immigration_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "cross_topic_immigration_replication.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    articles = build_immigration_corpus(args.corpus_output, target_count=60)
    publishers = [a.get("publisher", "unknown") for a in articles]

    cache_path = PROJECT_ROOT / "data" / "cache_cross_topic" / "immigration_N60_deberta.pt"

    if cache_path.exists():
        print(f"[CACHE] Loading cached representations from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ideo_emb = cached["ideo_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
        ideo_logits = cached["ideo_logits"]
    else:
        print(f"[EXTRACT] Initializing extractor with {args.model_name} on {args.device}...")
        config = ExtractionConfig(
            model_name=args.model_name,
            device=args.device,
            use_cls_tokens=True,
        )
        extractor = UnifiedNLIExtractor(config)

        pairs_ideo = load_immigration_pairs()
        pairs_ctrl = NEGATIVE_CONTROL_PAIRS
        pairs_rand = RANDOM_CONTROL_PAIRS

        print("[EXTRACT] 1/3 Extracting Immigration Ideological pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_ideo
        extractor.n_bots = len(pairs_ideo)
        batch_ideo = extractor.extract_batch(articles, show_progress=True)

        print("[EXTRACT] 2/3 Extracting Structural Control pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_ctrl
        extractor.n_bots = len(pairs_ctrl)
        batch_ctrl = extractor.extract_batch(articles, show_progress=True)

        print("[EXTRACT] 3/3 Extracting Random Control pairs (8 axes)...")
        extractor.hypothesis_pairs = pairs_rand
        extractor.n_bots = len(pairs_rand)
        batch_rand = extractor.extract_batch(articles, show_progress=True)

        ideo_emb = batch_ideo["cls_per_bot"].cpu()
        ideo_logits = batch_ideo["logits_raw"].cpu()
        ctrl_emb = batch_ctrl["cls_per_bot"].cpu()
        rand_emb = batch_rand["cls_per_bot"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": args.model_name,
                "ideo_emb": ideo_emb,
                "ideo_logits": ideo_logits,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
            },
            cache_path,
        )
        print(f"[SAVE] Cached extraction to {cache_path}")

    print("\n" + "=" * 75)
    print("STEP 1: HIGH-DIMENSIONAL CLS AXIS COHERENCE EVALUATION")
    print("=" * 75)

    residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    obs_ideo = subset_coherence(pooled_matrix, range(8))
    obs_ctrl = subset_coherence(pooled_matrix, range(8, 16))
    obs_rand = subset_coherence(pooled_matrix, range(16, 24))

    print(f"  Cone Anisotropy (Global PC1 EVR) : {pc1_evr*100:.1f}%")
    print(f"  Immigration Ideology Coherence  : {obs_ideo:+.6f}")
    print(f"  Structural Control Coherence    : {obs_ctrl:+.6f}")
    print(f"  Random Control Coherence        : {obs_rand:+.6f}")

    print("\n[NULL] Running exhaustive exact upper-tail enumeration (735,471 combinations)...")
    exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_ideo)
    print(f"  Exact Upper Tail Count          : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"  Exact p-value                   : {exact_null['exact_upper_tail_probability']:.6e}")

    print("\n" + "=" * 75)
    print("STEP 2: POLARITY SCORE PCA & DIMENSIONALITY NULL GATES (10,000 PERMUTATIONS)")
    print("=" * 75)

    probs = torch.softmax(ideo_logits.float(), dim=-1).numpy()
    pol_matrix = probs[..., 2] - probs[..., 0]  # [N, 8]

    dim_gate = run_dimensionality_null_test(
        pol_matrix, "US Immigration (N=60)", publishers=publishers, draws=args.permutations
    )

    c1 = dim_gate["components"][0]
    c2 = dim_gate["components"][1]
    print(f"  Polarity PC1 EVR                : {c1['observed_variance_ratio']*100:.2f}% (p = {c1['column_null']['p_value']:.4f}, null mean: {c1['column_null']['mean_variance_ratio']*100:.2f}%)")
    print(f"  Polarity PC2 EVR                : {c2['observed_variance_ratio']*100:.2f}% (p = {c2['column_null']['p_value']:.4f}, null mean: {c2['column_null']['mean_variance_ratio']*100:.2f}%)")
    print(f"  Detected Rank                   : {dim_gate['detected_rank_column_null']}")
    print(f"  Participation Ratio             : {dim_gate['observed_participation_ratio']:.2f} (null mean: {dim_gate['column_null_mean_participation_ratio']:.2f})")

    # Sample extreme articles
    pca = PCA().fit(pol_matrix)
    scores = pca.transform(pol_matrix)[:, 0]
    min_idx = int(np.argmin(scores))
    max_idx = int(np.argmax(scores))

    print("\n--- Extreme Articles along PC1 ---")
    print(f"  Min Score ({scores[min_idx]:+.3f}): [{articles[min_idx].get('publisher', 'unknown')}] {articles[min_idx].get('title', articles[min_idx]['text'][:80])}")
    print(f"  Max Score ({scores[max_idx]:+.3f}): [{articles[max_idx].get('publisher', 'unknown')}] {articles[max_idx].get('title', articles[max_idx]['text'][:80])}")
    print("=" * 75)

    output_payload = {
        "topic": "US Immigration & Border Policy",
        "dataset": "immigration_N60_corpus",
        "n_articles": len(articles),
        "model": args.model_name,
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
                "publisher": articles[min_idx].get("publisher", "unknown"),
                "snippet": articles[min_idx]["text"][:200],
            },
            "max_article": {
                "score": float(scores[max_idx]),
                "publisher": articles[max_idx].get("publisher", "unknown"),
                "snippet": articles[max_idx]["text"][:200],
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved cross-topic immigration replication report to {args.output}")


if __name__ == "__main__":
    main()
