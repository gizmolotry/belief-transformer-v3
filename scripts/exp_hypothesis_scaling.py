from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig
from scripts.run_gpu_validation import IDEOLOGICAL_PAIRS

# 1-8: Canonical Q0 pairs
Q0_PAIRS = IDEOLOGICAL_PAIRS

# 9-16: Q1 paraphrase pairs (loaded from config/prompt_banks/probe_hypotheses_paraphrase_a.json)
with open(PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json", "r", encoding="utf-8") as f:
    _raw_q1 = json.load(f)
Q1_PAIRS = [(_raw_q1[2 * i], _raw_q1[2 * i + 1]) for i in range(8)]

# 17-24: Q2 paraphrase pairs (loaded from config/prompt_banks/probe_hypotheses_paraphrase_b.json)
with open(PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_b.json", "r", encoding="utf-8") as f:
    _raw_q2 = json.load(f)
Q2_PAIRS = [(_raw_q2[2 * i], _raw_q2[2 * i + 1]) for i in range(8)]

# 25-32: 8 newly authored dialectical pairs
EXPANDED_PAIRS = [
    (
        "This article frames civilian infrastructure destruction as deliberate military policy.",
        "This article frames civilian infrastructure destruction as unavoidable tactical collateral.",
    ),
    (
        "This article emphasizes third-party state complicity in military funding and arms supplies.",
        "This article emphasizes legitimate international security alliances and mutual defense treaties.",
    ),
    (
        "This article characterizes regional escalation as initiated by militant proxies.",
        "This article characterizes regional escalation as provoked by sovereign military expansionism.",
    ),
    (
        "This article portrays humanitarian aid blockades as systematic starvation and collective punishment.",
        "This article portrays humanitarian aid restrictions as necessary security screening against contraband.",
    ),
    (
        "This article frames international court proceedings as universal legal justice and accountability.",
        "This article frames international court proceedings as politicized anti-state bias and jurisdictional overreach.",
    ),
    (
        "This article describes hostage taking as a grave war crime violating fundamental human rights.",
        "This article describes hostage taking as a tactical leverage mechanism in asymmetrical warfare.",
    ),
    (
        "This article depicts mainstream media coverage as biased toward Western geopolitical interests.",
        "This article depicts mainstream media coverage as objective, rigorous, and balanced across both sides.",
    ),
    (
        "This article highlights grassroots civilian solidarity and non-violent political resistance.",
        "This article highlights state military dominance and armed combat supremacy.",
    ),
]

ALL_32_PAIRS = Q0_PAIRS + Q1_PAIRS + Q2_PAIRS + EXPANDED_PAIRS


def extract_32_embeddings(
    model_name: str = "microsoft/deberta-v2-xlarge-mnli",
    corpus_path: Path = PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl",
    cache_path: Path = PROJECT_ROOT / "data" / "hypothesis_scaling_32_embeddings.pt",
    device: str = "cuda",
) -> torch.Tensor:
    if cache_path.exists():
        print(f"[CACHE] Loading cached 32-pair embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["emb_32"]

    print(f"[EXTRACT] Extracting all 32 hypothesis pairs with {model_name} on {device}...")
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

    extractor.hypothesis_pairs = ALL_32_PAIRS
    extractor.n_bots = len(ALL_32_PAIRS)
    emb_32 = extractor.extract_batch(articles, show_progress=True)["cls_per_bot"].cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "emb_32": emb_32}, cache_path)
    print(f"[SAVE] Cached 32-pair embeddings of shape {emb_32.shape} to {cache_path}")
    return emb_32


def analyze_hypothesis_subset(
    tensor_k: torch.Tensor,
) -> dict[str, Any]:
    """
    tensor_k: [N, K, D]
    """
    N, K, D = tensor_k.shape

    # 1. Global PC1 removal
    flat = tensor_k.reshape(-1, D)
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, s, v = torch.svd(centered)
    pc1 = v[:, 0]
    proj = torch.matmul(flat, pc1).unsqueeze(1) * pc1.unsqueeze(0)
    resid = (flat - proj).reshape(N, K, D)
    pc1_evr = float((s[0] ** 2) / (s ** 2).sum())

    # 2. Axis Coherence (mean pairwise cosine after PC1 removal)
    normed = torch.nn.functional.normalize(resid, p=2, dim=-1)
    cos_sims = []
    for i in range(K):
        for j in range(i + 1, K):
            sim = torch.sum(normed[:, i, :] * normed[:, j, :], dim=-1).mean().item()
            cos_sims.append(sim)
    mean_coherence = float(np.mean(cos_sims))

    # 3. Dimensionality of Article Representation in K dimensions
    # For each article, take the K axis scores (mean along D of normed projection)
    article_matrix = normed.mean(dim=-1).numpy()  # [N, K]
    pca = PCA()
    pca.fit(article_matrix)
    evr = pca.explained_variance_ratio_

    pc1_var = float(evr[0])
    pc2_var = float(evr[1]) if len(evr) > 1 else 0.0
    cum_2pc = float(np.sum(evr[:2]))

    # Participation ratio / effective rank
    eigenvalues = pca.explained_variance_
    participation_ratio = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    # Components to 80% and 90% variance
    cum_evr = np.cumsum(evr)
    n_80 = int(np.searchsorted(cum_evr, 0.80) + 1)
    n_90 = int(np.searchsorted(cum_evr, 0.90) + 1)

    return {
        "k_hypotheses": K,
        "pc1_removal_evr": pc1_evr,
        "mean_axis_coherence": mean_coherence,
        "article_pca_pc1_variance": pc1_var,
        "article_pca_pc2_variance": pc2_var,
        "article_pca_cum_2pc_variance": cum_2pc,
        "participation_ratio": participation_ratio,
        "components_to_80_pct": n_80,
        "components_to_90_pct": n_90,
        "full_evr": [float(x) for x in evr[: min(10, K)]],
    }


def main():
    parser = argparse.ArgumentParser(description="Hypothesis-count scaling ablation.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "hypothesis_scaling_results.json")
    args = parser.parse_args()

    emb_32 = extract_32_embeddings(
        model_name=args.model_name,
        corpus_path=args.corpus,
    )

    scaling_results = {}

    # K = 4: evaluate 5 diverse 4-pair subsets and report average + spread
    k4_subsets = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 3, 4, 7],
    ]
    k4_runs = [analyze_hypothesis_subset(emb_32[:, sub, :]) for sub in k4_subsets]
    scaling_results["K=4"] = {
        "mean_coherence": float(np.mean([r["mean_axis_coherence"] for r in k4_runs])),
        "mean_cum_2pc": float(np.mean([r["article_pca_cum_2pc_variance"] for r in k4_runs])),
        "mean_participation_ratio": float(np.mean([r["participation_ratio"] for r in k4_runs])),
        "subsets_evaluated": k4_runs,
    }

    # K = 8: canonical Q0
    k8_res = analyze_hypothesis_subset(emb_32[:, :8, :])
    scaling_results["K=8"] = k8_res

    # K = 16: Q0 + Q1
    k16_res = analyze_hypothesis_subset(emb_32[:, :16, :])
    scaling_results["K=16"] = k16_res

    # K = 32: All 32 pairs
    k32_res = analyze_hypothesis_subset(emb_32[:, :32, :])
    scaling_results["K=32"] = k32_res

    print("\n" + "=" * 70)
    print("HYPOTHESIS COUNT SCALING ABLATION SUMMARY")
    print("=" * 70)
    print(f"K=4  | Coherence: {scaling_results['K=4']['mean_coherence']:+.4f} | 2-PC Var: {scaling_results['K=4']['mean_cum_2pc']:.3f} | PR: {scaling_results['K=4']['mean_participation_ratio']:.2f}")
    print(f"K=8  | Coherence: {scaling_results['K=8']['mean_axis_coherence']:+.4f} | 2-PC Var: {scaling_results['K=8']['article_pca_cum_2pc_variance']:.3f} | PR: {scaling_results['K=8']['participation_ratio']:.2f}")
    print(f"K=16 | Coherence: {scaling_results['K=16']['mean_axis_coherence']:+.4f} | 2-PC Var: {scaling_results['K=16']['article_pca_cum_2pc_variance']:.3f} | PR: {scaling_results['K=16']['participation_ratio']:.2f}")
    print(f"K=32 | Coherence: {scaling_results['K=32']['mean_axis_coherence']:+.4f} | 2-PC Var: {scaling_results['K=32']['article_pca_cum_2pc_variance']:.3f} | PR: {scaling_results['K=32']['participation_ratio']:.2f}")
    print("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scaling_results, f, indent=2)
    print(f"[DONE] Saved full scaling results to {args.output}")


if __name__ == "__main__":
    main()
