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


def load_q0_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "framing_queries.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def load_q1_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(raw[2 * i], raw[2 * i + 1]) for i in range(8)]


def extract_or_load_bank(
    bank_name: str,
    pairs: list[tuple[str, str]],
    articles: list[dict[str, Any]],
    extractor: UnifiedNLIExtractor,
    cache_dir: Path,
) -> dict[str, torch.Tensor]:
    cache_path = cache_dir / f"fresh_120_{bank_name}_embeddings.pt"
    if cache_path.exists():
        print(f"[{bank_name.upper()}] Loading cached embeddings from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"[{bank_name.upper()}] Extracting forward pass for {len(pairs)} pairs on {len(articles)} articles...")
    extractor.hypothesis_pairs = pairs
    extractor.n_bots = len(pairs)
    batch = extractor.extract_batch(articles, show_progress=True)
    payload = {
        "cls_per_bot": batch["cls_per_bot"].cpu(),
        "logits_raw": batch["logits_raw"].cpu(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[{bank_name.upper()}] Saved cache to {cache_path}")
    return payload


def compute_paired_polarity(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=-1).numpy()
    return probs[..., 2] - probs[..., 0]


def main():
    parser = argparse.ArgumentParser(description="Fresh-data preregistered replication of Q0 and Q1.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "fresh_unseen_120_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "fresh_preregistered_q0_q1_replication.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    articles = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[DATA] Loaded {len(articles)} fresh unseen articles from {args.corpus}")
    publishers = [a.get("publisher", "unknown") for a in articles]

    q0_pairs = load_q0_pairs()
    q1_pairs = load_q1_pairs()
    ctrl_pairs = NEGATIVE_CONTROL_PAIRS
    rand_pairs = RANDOM_CONTROL_PAIRS

    config = ExtractionConfig(
        model_name=args.model_name,
        device=args.device,
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)
    cache_dir = PROJECT_ROOT / "data" / "cache_fresh_replication"

    q0_data = extract_or_load_bank("q0_ideology", q0_pairs, articles, extractor, cache_dir)
    q1_data = extract_or_load_bank("q1_paraphrase", q1_pairs, articles, extractor, cache_dir)
    ctrl_data = extract_or_load_bank("ctrl_structure", ctrl_pairs, articles, extractor, cache_dir)
    rand_data = extract_or_load_bank("rand_control", rand_pairs, articles, extractor, cache_dir)

    ctrl_emb = ctrl_data["cls_per_bot"]
    rand_emb = rand_data["cls_per_bot"]

    print("\n" + "=" * 75)
    print("STEP 1: EXACT COMBINATORIAL AXIS COHERENCE EVALUATION")
    print("=" * 75)

    def run_coherence_suite(name: str, ideo_emb: torch.Tensor):
        residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
        pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))
        obs_ideo = subset_coherence(pooled_matrix, range(8))
        obs_ctrl = subset_coherence(pooled_matrix, range(8, 16))
        obs_rand = subset_coherence(pooled_matrix, range(16, 24))
        exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_ideo)
        print(f"[{name}] Coherence: Ideology={obs_ideo:+.6f}, Structural={obs_ctrl:+.6f}, Random={obs_rand:+.6f}")
        print(f"[{name}] Exact Upper Tail: {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']} (p = {exact_null['exact_upper_tail_probability']:.6e})")
        return {
            "coherence_ideology": float(obs_ideo),
            "coherence_structural": float(obs_ctrl),
            "coherence_random": float(obs_rand),
            "global_pc1_evr": float(pc1_evr),
            "exact_null": exact_null,
        }

    q0_coherence = run_coherence_suite("Q0 Canonical", q0_data["cls_per_bot"])
    q1_coherence = run_coherence_suite("Q1 Paraphrase", q1_data["cls_per_bot"])

    print("\n" + "=" * 75)
    print("STEP 2: SYSTEMATIC DIMENSIONALITY NULL GATES (10,000 PERMUTATIONS)")
    print("=" * 75)

    q0_pol = compute_paired_polarity(q0_data["logits_raw"])
    q1_pol = compute_paired_polarity(q1_data["logits_raw"])

    q0_dim_gate = run_dimensionality_null_test(
        q0_pol, "Q0 Canonical (Fresh N=120)", publishers=publishers, draws=args.permutations
    )
    q1_dim_gate = run_dimensionality_null_test(
        q1_pol, "Q1 Paraphrase (Fresh N=120)", publishers=publishers, draws=args.permutations
    )

    c0_1 = q0_dim_gate['components'][0]
    c0_2 = q0_dim_gate['components'][1]
    c1_1 = q1_dim_gate['components'][0]
    c1_2 = q1_dim_gate['components'][1]
    print(f"[Q0 Gate] PC1 EVR={c0_1['observed_variance_ratio']*100:.2f}% (p={c0_1['column_null']['p_value']:.4f}), PC2 EVR={c0_2['observed_variance_ratio']*100:.2f}% (p={c0_2['column_null']['p_value']:.4f})")
    print(f"[Q1 Gate] PC1 EVR={c1_1['observed_variance_ratio']*100:.2f}% (p={c1_1['column_null']['p_value']:.4f}), PC2 EVR={c1_2['observed_variance_ratio']*100:.2f}% (p={c1_2['column_null']['p_value']:.4f})")

    print("\n" + "=" * 75)
    print("STEP 3: CROSS-BANK GEOMETRIC STABILITY (Q0 vs Q1 ON FRESH DATA)")
    print("=" * 75)

    pca_q0 = PCA().fit(q0_pol)
    pca_q1 = PCA().fit(q1_pol)

    proj_q0 = pca_q0.transform(q0_pol)
    proj_q1 = pca_q1.transform(q1_pol)

    if np.corrcoef(proj_q0[:, 0], proj_q1[:, 0])[0, 1] < 0:
        proj_q1[:, 0] = -proj_q1[:, 0]
        pca_q1.components_[0] = -pca_q1.components_[0]

    r_pc1, p_pearson = pearsonr(proj_q0[:, 0], proj_q1[:, 0])
    rho_pc1, p_spearman = spearmanr(proj_q0[:, 0], proj_q1[:, 0])

    mtx1, mtx2, proc_disp = procrustes(proj_q0[:, :2], proj_q1[:, :2])

    angle_rad = subspace_angles(pca_q0.components_[:1].T, pca_q1.components_[:1].T)[0]
    angle_deg = float(np.degrees(angle_rad))

    print(f"PC1 Article Score Pearson r    : {r_pc1:.4f} (p = {p_pearson:.4e})")
    print(f"PC1 Article Score Spearman rho : {rho_pc1:.4f} (p = {p_spearman:.4e})")
    print(f"2-PC Procrustes Disparity      : {proc_disp:.4f}")
    print(f"PC1 Subspace Angle             : {angle_deg:.2f}°")
    print("=" * 75)

    payload = {
        "dataset": "fresh_unseen_120_corpus",
        "n_articles": len(articles),
        "model": args.model_name,
        "q0_coherence": q0_coherence,
        "q1_coherence": q1_coherence,
        "q0_dimensionality_gate": q0_dim_gate,
        "q1_dimensionality_gate": q1_dim_gate,
        "cross_bank_stability": {
            "pearson_r_pc1": float(r_pc1),
            "pearson_p": float(p_pearson),
            "spearman_rho_pc1": float(rho_pc1),
            "spearman_p": float(p_spearman),
            "procrustes_disparity_2pc": float(proc_disp),
            "subspace_angle_pc1_deg": float(angle_deg),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SUCCESS] Saved preregistered replication report to {args.output}")


if __name__ == "__main__":
    main()
