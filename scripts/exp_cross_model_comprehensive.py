from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr
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


MODELS = [
    "microsoft/deberta-v2-xlarge-mnli",
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
]


def load_framing_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "framing_queries.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def extract_or_load_model_data(
    model_name: str,
    articles: list[dict[str, Any]],
    cache_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    slug = model_name.replace("/", "_")
    cache_path = cache_dir / f"cross_model_{slug}_N60.pt"
    if cache_path.exists():
        print(f"[{slug}] Loading cached representations from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"\n[{slug}] Initializing extraction with {model_name} on {device}...")
    config = ExtractionConfig(
        model_name=model_name,
        device=device if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    pairs_ideo = load_framing_pairs()
    pairs_ctrl = NEGATIVE_CONTROL_PAIRS
    pairs_rand = RANDOM_CONTROL_PAIRS

    print(f"[{slug}] 1/3 Extracting Ideological Framing pairs (8 axes)...")
    extractor.hypothesis_pairs = pairs_ideo
    extractor.n_bots = len(pairs_ideo)
    batch_ideo = extractor.extract_batch(articles, show_progress=True)

    print(f"[{slug}] 2/3 Extracting Structural Control pairs (8 axes)...")
    extractor.hypothesis_pairs = pairs_ctrl
    extractor.n_bots = len(pairs_ctrl)
    batch_ctrl = extractor.extract_batch(articles, show_progress=True)

    print(f"[{slug}] 3/3 Extracting Random Control pairs (8 axes)...")
    extractor.hypothesis_pairs = pairs_rand
    extractor.n_bots = len(pairs_rand)
    batch_rand = extractor.extract_batch(articles, show_progress=True)

    payload = {
        "model_name": model_name,
        "ideo_emb": batch_ideo["cls_per_bot"].cpu(),
        "ideo_logits": batch_ideo["logits_raw"].cpu(),
        "ctrl_emb": batch_ctrl["cls_per_bot"].cpu(),
        "ctrl_logits": batch_ctrl["logits_raw"].cpu(),
        "rand_emb": batch_rand["cls_per_bot"].cpu(),
        "rand_logits": batch_rand["logits_raw"].cpu(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[{slug}] Saved cached extraction to {cache_path}")
    return payload


def compute_polarity_matrix(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=-1).numpy()
    return probs[..., 2] - probs[..., 0]


def main():
    parser = argparse.ArgumentParser(description="Cross-model comprehensive replication.")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "cross_model_comprehensive_replication.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    articles = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[DATA] Loaded {len(articles)} articles from {args.corpus}")
    publishers = [a.get("publisher", "unknown") for a in articles]

    cache_dir = PROJECT_ROOT / "data" / "cache_cross_model"

    per_model_results = {}
    model_polarities = {}
    model_projections = {}
    model_pcas = {}

    for model_name in MODELS:
        data = extract_or_load_model_data(model_name, articles, cache_dir, device=args.device)
        ideo_emb = data["ideo_emb"]
        ctrl_emb = data["ctrl_emb"]
        rand_emb = data["rand_emb"]
        ideo_logits = data["ideo_logits"]

        # 1. Coherence evaluation
        residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
        pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))
        obs_ideo = subset_coherence(pooled_matrix, range(8))
        obs_ctrl = subset_coherence(pooled_matrix, range(8, 16))
        obs_rand = subset_coherence(pooled_matrix, range(16, 24))
        exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_ideo)

        # 2. Polarity score dimensionality null gate
        pol_matrix = compute_polarity_matrix(ideo_logits)
        model_polarities[model_name] = pol_matrix

        dim_gate = run_dimensionality_null_test(
            pol_matrix, f"{model_name} (N=60)", publishers=publishers, draws=args.permutations
        )

        pca = PCA().fit(pol_matrix)
        proj = pca.transform(pol_matrix)
        model_pcas[model_name] = pca
        model_projections[model_name] = proj

        c1 = dim_gate["components"][0]
        c2 = dim_gate["components"][1]
        print(f"\n--- Model: {model_name} ---")
        print(f"  Cone Anisotropy (Global PC1 EVR) : {pc1_evr*100:.1f}%")
        print(f"  Ideology Coherence              : {obs_ideo:+.6f}")
        print(f"  Exact Upper Tail p-value        : {exact_null['exact_upper_tail_probability']:.6e} ({exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']})")
        print(f"  Polarity PC1 EVR                : {c1['observed_variance_ratio']*100:.2f}% (p = {c1['column_null']['p_value']:.4f})")
        print(f"  Polarity PC2 EVR                : {c2['observed_variance_ratio']*100:.2f}% (p = {c2['column_null']['p_value']:.4f})")
        print(f"  Detected Rank                   : {dim_gate['detected_rank_column_null']}")
        print(f"  Participation Ratio             : {dim_gate['observed_participation_ratio']:.2f} (null mean: {dim_gate['column_null_mean_participation_ratio']:.2f})")

        per_model_results[model_name] = {
            "cone_anisotropy_pc1_evr": float(pc1_evr),
            "coherence_ideology": float(obs_ideo),
            "coherence_structural": float(obs_ctrl),
            "coherence_random": float(obs_rand),
            "exact_null": exact_null,
            "dimensionality_gate": dim_gate,
        }

    # 3. Pairwise cross-model alignment
    print("\n" + "=" * 75)
    print("CROSS-MODEL ARTICLE STANCE ALIGNMENT & RANKING STABILITY")
    print("=" * 75)

    pairwise_comparisons = {}
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i >= j:
                continue
            pair_key = f"{m1}__vs__{m2}"
            proj1 = model_projections[m1].copy()
            proj2 = model_projections[m2].copy()
            pca1 = model_pcas[m1]
            pca2 = model_pcas[m2]

            # Fix sign alignment for PC1
            if np.corrcoef(proj1[:, 0], proj2[:, 0])[0, 1] < 0:
                proj2[:, 0] = -proj2[:, 0]
                pca2.components_[0] = -pca2.components_[0]

            r_val, r_p = pearsonr(proj1[:, 0], proj2[:, 0])
            rho_val, rho_p = spearmanr(proj1[:, 0], proj2[:, 0])

            _, _, proc_disp = procrustes(proj1[:, :2], proj2[:, :2])
            angle_rad = subspace_angles(pca1.components_[:1].T, pca2.components_[:1].T)[0]
            angle_deg = float(np.degrees(angle_rad))

            print(f"Comparison: {m1.split('/')[-1]} vs {m2.split('/')[-1]}")
            print(f"  PC1 Article Score Pearson r    : {r_val:.4f} (p = {r_p:.4e})")
            print(f"  PC1 Article Score Spearman rho : {rho_val:.4f} (p = {rho_p:.4e})")
            print(f"  2-PC Procrustes Disparity      : {proc_disp:.4f}")
            print(f"  PC1 Subspace Angle             : {angle_deg:.2f}°")

            pairwise_comparisons[pair_key] = {
                "model_1": m1,
                "model_2": m2,
                "pearson_r": float(r_val),
                "pearson_p": float(r_p),
                "spearman_rho": float(rho_val),
                "spearman_p": float(rho_p),
                "procrustes_disparity_2pc": float(proc_disp),
                "subspace_angle_pc1_deg": float(angle_deg),
            }

    output_payload = {
        "dataset": "controlled_N60_corpus",
        "n_articles": len(articles),
        "models_evaluated": MODELS,
        "per_model_results": per_model_results,
        "pairwise_comparisons": pairwise_comparisons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved cross-model replication report to {args.output}")


if __name__ == "__main__":
    main()
