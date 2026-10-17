from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

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


def extract_or_load_embeddings(
    model_name: str,
    corpus_path: Path,
    cache_path: Path,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    if cache_path.exists():
        print(f"[CACHE] Loading cached embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return {
            "ideo_emb": cached["ideo_emb"],
            "ctrl_emb": cached["ctrl_emb"],
            "rand_emb": cached["rand_emb"],
        }

    print(f"[MODEL] Initializing extractor with model: {model_name} on {device}...")
    config = ExtractionConfig(
        model_name=model_name,
        device=device if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    articles = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[DATA] Loaded {len(articles)} articles from {corpus_path}")

    # 1. Ideological
    print("[EXTRACT] Extracting Ideological pairs (8 axes)...")
    extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
    extractor.n_bots = len(IDEOLOGICAL_PAIRS)
    ideo_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

    # 2. Structural control
    print("[EXTRACT] Extracting Structural Control pairs (8 axes)...")
    extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
    extractor.n_bots = len(NEGATIVE_CONTROL_PAIRS)
    ctrl_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

    # 3. Random control
    print("[EXTRACT] Extracting Random Control pairs (8 axes)...")
    extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
    extractor.n_bots = len(RANDOM_CONTROL_PAIRS)
    rand_emb = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "ideo_emb": ideo_emb,
        "ctrl_emb": ctrl_emb,
        "rand_emb": rand_emb,
    }
    torch.save(payload, cache_path)
    print(f"[SAVE] Cached extracted embeddings to {cache_path}")
    return payload


def evaluate_coherence(
    ideo_emb: torch.Tensor,
    ctrl_emb: torch.Tensor,
    rand_emb: torch.Tensor,
) -> dict[str, Any]:
    residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    expected_axes = ideo_emb.shape[1]
    ideology = subset_coherence(pooled_matrix, range(expected_axes))
    structure = subset_coherence(pooled_matrix, range(expected_axes, 2 * expected_axes))
    random = subset_coherence(pooled_matrix, range(2 * expected_axes, 3 * expected_axes))

    print(f"  Ideology Coherence       : {ideology:+.6f}")
    print(f"  Structure Coherence      : {structure:+.6f}")
    print(f"  Random Coherence         : {random:+.6f}")
    print(f"  PC1 Explained Variance   : {pc1_evr:.4f}")

    print("[NULL] Running exhaustive exact upper-tail enumeration (735,471 combinations)...")
    exact_null = exhaustive_upper_tail(
        pooled_matrix, subset_size=expected_axes, observed=ideology
    )
    print(f"  Exact Upper Tail Count   : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"  Exact p-value            : {exact_null['exact_upper_tail_probability']:.6e}")

    return {
        "pc1_explained_variance_ratio": pc1_evr,
        "observed_coherence": {
            "ideology": ideology,
            "structure": structure,
            "random": random,
        },
        "exact_null": exact_null,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-model axis coherence replication.")
    parser.add_argument("--model-name", type=str, default="roberta-large-mnli")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_slug = args.model_name.replace("/", "_")
    cache_path = PROJECT_ROOT / "data" / f"controlled_ideo_embeddings_{model_slug}.pt"
    if args.output is None:
        output_path = PROJECT_ROOT / "paper" / f"cross_model_coherence_{model_slug}.json"
    else:
        output_path = args.output

    data = extract_or_load_embeddings(args.model_name, args.corpus, cache_path)
    results = evaluate_coherence(data["ideo_emb"], data["ctrl_emb"], data["rand_emb"])

    output_payload = {
        "model_name": args.model_name,
        "corpus_path": str(args.corpus),
        "embedding_shape": list(data["ideo_emb"].shape),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[DONE] Saved full results to {output_path}")


if __name__ == "__main__":
    main()
