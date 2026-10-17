from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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


def load_weather_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "domain_null_weather_queries.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pairs = []
    for k, v in data["queries"].items():
        pairs.append((v["A"], v["B"]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Domain-null weather hypothesis bank ablation.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "domain_null_weather_results.json")
    args = parser.parse_args()

    cache_path = PROJECT_ROOT / "data" / "domain_null_weather_embeddings.pt"

    weather_pairs = load_weather_pairs()
    print(f"[SETUP] Loaded {len(weather_pairs)} domain-null weather pairs.")

    if cache_path.exists():
        print(f"[CACHE] Loading cached embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ideo_emb = cached["ideo_emb"]
        weather_emb = cached["weather_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
    else:
        print(f"[EXTRACT] Initializing extractor with {args.model_name} on cuda...")
        config = ExtractionConfig(
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_cls_tokens=True,
        )
        extractor = UnifiedNLIExtractor(config)

        articles = []
        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

        baseline_cache = PROJECT_ROOT / "data" / "controlled_ideo_embeddings_xlarge.pt"
        if baseline_cache.exists():
            print(f"[CACHE] Reusing baseline embeddings from {baseline_cache}...")
            base_data = torch.load(baseline_cache, map_location="cpu", weights_only=False)
            ideo_emb = base_data["ideo_emb"]
            ctrl_emb = base_data["ctrl_emb"]
            rand_emb = base_data["rand_emb"]
        else:
            print("[EXTRACT] 1. Ideological pairs (8 axes)...")
            extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
            extractor.n_bots = len(IDEOLOGICAL_PAIRS)
            ideo_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

            print("[EXTRACT] 3. Structural Control pairs (8 axes)...")
            extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
            extractor.n_bots = len(NEGATIVE_CONTROL_PAIRS)
            ctrl_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

            print("[EXTRACT] 4. Random Control pairs (8 axes)...")
            extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
            extractor.n_bots = len(RANDOM_CONTROL_PAIRS)
            rand_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

        # 2. Weather Domain-Null
        print("[EXTRACT] 2. Domain-Null Weather pairs (8 axes)...")
        extractor.hypothesis_pairs = weather_pairs
        extractor.n_bots = len(weather_pairs)
        weather_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ideo_emb": ideo_emb,
                "weather_emb": weather_emb,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
            },
            cache_path,
        )
        print(f"[SAVE] Cached extracted embeddings to {cache_path}")

    # 1. Canonical Tripartite Test for Domain-Null Weather: [weather, structural_ctrl, random_ctrl]
    residuals_w, pc1_evr_w = remove_global_pc1([weather_emb, ctrl_emb, rand_emb])
    pooled_w = axis_coherence_matrix(torch.cat(residuals_w, dim=1))
    c_weather = subset_coherence(pooled_w, range(0, 8))
    c_struct_w = subset_coherence(pooled_w, range(8, 16))
    c_rand_w = subset_coherence(pooled_w, range(16, 24))
    print("[NULL] Running exhaustive exact upper-tail enumeration for Weather (735,471 combinations)...")
    exact_w = exhaustive_upper_tail(pooled_w, subset_size=8, observed=c_weather)

    # 2. Canonical Tripartite Test for Ideology Baseline: [ideology, structural_ctrl, random_ctrl]
    residuals_i, pc1_evr_i = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_i = axis_coherence_matrix(torch.cat(residuals_i, dim=1))
    c_ideo = subset_coherence(pooled_i, range(0, 8))
    c_struct_i = subset_coherence(pooled_i, range(8, 16))
    c_rand_i = subset_coherence(pooled_i, range(16, 24))
    print("[NULL] Running exhaustive exact upper-tail enumeration for Ideology (735,471 combinations)...")
    exact_i = exhaustive_upper_tail(pooled_i, subset_size=8, observed=c_ideo)

    print("\n" + "=" * 75)
    print("DOMAIN-NULL WEATHER HYPOTHESIS TEST RESULTS (VS TARGET IDEOLOGY)")
    print("=" * 75)
    print(f"{'Condition':<25} | {'Target':<12} | {'Structure':<12} | {'Random':<12} | {'Exact p':<12}")
    print("-" * 75)
    print(f"{'Ideology (Ground Truth)':<25} | {c_ideo:+.6f}    | {c_struct_i:+.6f}    | {c_rand_i:+.6f}    | {exact_i['exact_upper_tail_probability']:.2e}")
    print(f"{'Weather (Domain-Null)':<25} | {c_weather:+.6f}    | {c_struct_w:+.6f}    | {c_rand_w:+.6f}    | {exact_w['exact_upper_tail_probability']:.2e}")
    print("=" * 75)

    results = {
        "domain_null_weather": {
            "coherences": {
                "weather": c_weather,
                "structural_control": c_struct_w,
                "random_control": c_rand_w,
            },
            "pc1_evr": pc1_evr_w,
            "exact_null": exact_w,
        },
        "ideology_baseline": {
            "coherences": {
                "ideology": c_ideo,
                "structural_control": c_struct_i,
                "random_control": c_rand_i,
            },
            "pc1_evr": pc1_evr_i,
            "exact_null": exact_i,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Saved domain-null results to {args.output}")


if __name__ == "__main__":
    main()
