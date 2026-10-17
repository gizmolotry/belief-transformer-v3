from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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
from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


def load_json_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return [(raw[2 * i], raw[2 * i + 1]) for i in range(8)]


def load_yaml_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def main():
    parser = argparse.ArgumentParser(description="Multi-lens independent hypothesis bank replication.")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "fresh_unseen_120_corpus.jsonl")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "independent_authored_banks_q0_q1_q2_q3.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    articles = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[DATA] Loaded {len(articles)} fresh unseen articles from {args.corpus}")
    publishers = [a.get("publisher", "unknown") for a in articles]

    banks = {
        "Q0_Military_Conflict": {
            "description": "Original Geopolitical / Military Conflict Framing",
            "pairs": load_yaml_pairs(PROJECT_ROOT / "config" / "framing_queries.yaml"),
            "cached_file": PROJECT_ROOT / "data" / "cache_fresh_replication" / "fresh_120_q0_ideology_embeddings.pt",
        },
        "Q1_Paraphrase": {
            "description": "Lexical / Syntactic Paraphrase Bank",
            "pairs": load_json_pairs(PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"),
            "cached_file": PROJECT_ROOT / "data" / "cache_fresh_replication" / "fresh_120_q1_paraphrase_embeddings.pt",
        },
        "Q2_Humanitarian_Rights": {
            "description": "Humanitarian / Human Rights & International Law Framework",
            "pairs": load_json_pairs(PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_q2_humanitarian.json"),
            "cached_file": PROJECT_ROOT / "data" / "cache_fresh_replication" / "fresh_120_q2_humanitarian_embeddings.pt",
        },
        "Q3_Realpolitik_Statecraft": {
            "description": "Geopolitical Realpolitik & Sovereign Statecraft Framework",
            "pairs": load_json_pairs(PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_q3_realpolitik.json"),
            "cached_file": PROJECT_ROOT / "data" / "cache_fresh_replication" / "fresh_120_q3_realpolitik_embeddings.pt",
        },
    }

    extractor = None

    extracted_data = {}
    for name, binfo in banks.items():
        cache_p = binfo["cached_file"]
        if cache_p.exists():
            print(f"[{name}] Loading from cache: {cache_p}...")
            payload = torch.load(cache_p, map_location="cpu", weights_only=False)
        else:
            if extractor is None:
                print(f"[INIT] Initializing {args.model_name} extractor on {args.device}...")
                config = ExtractionConfig(
                    model_name=args.model_name,
                    device=args.device,
                    use_cls_tokens=True,
                )
                extractor = UnifiedNLIExtractor(config)

            print(f"[{name}] Extracting forward pass for {len(binfo['pairs'])} pairs on {len(articles)} articles...")
            extractor.hypothesis_pairs = binfo["pairs"]
            extractor.n_bots = len(binfo["pairs"])
            batch = extractor.extract_batch(articles, show_progress=True)
            payload = {
                "cls_per_bot": batch["cls_per_bot"].cpu(),
                "logits_raw": batch["logits_raw"].cpu(),
            }
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_p)
            print(f"[{name}] Saved cache to {cache_p}")

        extracted_data[name] = payload

    print("\n" + "=" * 75)
    print("STEP 1: DIMENSIONALITY & NULL GATES PER INDEPENDENT CONCEPTUAL LENS")
    print("=" * 75)

    bank_polarities = {}
    bank_pcas = {}
    bank_projections = {}
    bank_gates = {}

    for name, payload in extracted_data.items():
        logits = payload["logits_raw"]
        probs = torch.softmax(logits.float(), dim=-1).numpy()
        pol = probs[..., 2] - probs[..., 0]  # [N, 8]
        bank_polarities[name] = pol

        gate = run_dimensionality_null_test(
            pol, f"{name} (Fresh N=120)", publishers=publishers, draws=args.permutations
        )
        bank_gates[name] = gate

        pca = PCA().fit(pol)
        proj = pca.transform(pol)
        bank_pcas[name] = pca
        bank_projections[name] = proj

        c1 = gate["components"][0]
        c2 = gate["components"][1]
        print(f"[{name}]")
        print(f"  PC1 EVR: {c1['observed_variance_ratio']*100:.2f}% (p = {c1['column_null']['p_value']:.4f}, null mean: {c1['column_null']['mean_variance_ratio']*100:.2f}%)")
        print(f"  PC2 EVR: {c2['observed_variance_ratio']*100:.2f}% (p = {c2['column_null']['p_value']:.4f}, null mean: {c2['column_null']['mean_variance_ratio']*100:.2f}%)")
        print(f"  Detected Rank: {gate['detected_rank_column_null']}")
        print(f"  Participation Ratio: {gate['observed_participation_ratio']:.2f} (null: {gate['column_null_mean_participation_ratio']:.2f})")

    print("\n" + "=" * 75)
    print("STEP 2: PAIRWISE CONCEPTUAL LENS ALIGNMENT & STANCE INVARIANCE MATRIX")
    print("=" * 75)

    bank_keys = list(banks.keys())
    alignment_matrix = {}

    for i, k1 in enumerate(bank_keys):
        for j, k2 in enumerate(bank_keys):
            if i >= j:
                continue
            pair_key = f"{k1}__vs__{k2}"
            proj1 = bank_projections[k1].copy()
            proj2 = bank_projections[k2].copy()
            pca1 = bank_pcas[k1]
            pca2 = bank_pcas[k2]

            # Fix sign alignment for PC1
            if np.corrcoef(proj1[:, 0], proj2[:, 0])[0, 1] < 0:
                proj2[:, 0] = -proj2[:, 0]
                pca2.components_[0] = -pca2.components_[0]

            r_val, r_p = pearsonr(proj1[:, 0], proj2[:, 0])
            rho_val, rho_p = spearmanr(proj1[:, 0], proj2[:, 0])

            _, _, proc_disp = procrustes(proj1[:, :2], proj2[:, :2])
            angle_rad = subspace_angles(pca1.components_[:1].T, pca2.components_[:1].T)[0]
            angle_deg = float(np.degrees(angle_rad))

            print(f"Pair: {k1} vs {k2}")
            print(f"  Pearson r (PC1 stance)       : {r_val:.4f} (p = {r_p:.4e})")
            print(f"  Spearman rho (rank order)    : {rho_val:.4f} (p = {rho_p:.4e})")
            print(f"  2-PC Procrustes Disparity    : {proc_disp:.4f}")
            print(f"  PC1 Subspace Angle           : {angle_deg:.2f}°")

            alignment_matrix[pair_key] = {
                "lens_1": k1,
                "lens_2": k2,
                "pearson_r": float(r_val),
                "pearson_p": float(r_p),
                "spearman_rho": float(rho_val),
                "spearman_p": float(rho_p),
                "procrustes_disparity_2pc": float(proc_disp),
                "subspace_angle_pc1_deg": float(angle_deg),
            }

    output_payload = {
        "dataset": "fresh_unseen_120_corpus",
        "n_articles": len(articles),
        "model": args.model_name,
        "banks": {k: v["description"] for k, v in banks.items()},
        "bank_gates": bank_gates,
        "pairwise_alignments": alignment_matrix,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved multi-lens replication report to {args.output}")


if __name__ == "__main__":
    main()
