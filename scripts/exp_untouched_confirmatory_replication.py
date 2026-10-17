from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


def collect_touched_fingerprints() -> set[str]:
    seen = set()
    files_to_check = [
        "data/controlled_N60_corpus.jsonl",
        "data/fresh_unseen_120_corpus.jsonl",
        "data/immigration_N60_corpus.jsonl",
        "data/climate_N60_corpus.jsonl",
        "data/abortion_N60_corpus.jsonl",
        "data/real_articles.jsonl",
    ]
    for rel_path in files_to_check:
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    title = item.get("title", "").strip().lower()
                    content = (item.get("content", "") or item.get("text", "")).strip()
                    h = hashlib.sha256(f"{title}|{content[:200]}".encode("utf-8")).hexdigest()
                    seen.add(h)
                    if "url" in item and item["url"]:
                        seen.add(item["url"])
                    if title:
                        seen.add(title)
                except Exception:
                    pass
    return seen


def build_untouched_corpus(output_path: Path, target_count: int = 120) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing untouched corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Harvesting {target_count} strictly untouched articles from real_corpus.jsonl...")
    seen_fingerprints = collect_touched_fingerprints()
    print(f"[CORPUS] Excluded {len(seen_fingerprints)} previously touched fingerprints.")

    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"
    keywords = ["israel", "palestin", "gaza", "hamas", "netanyahu", "idf", "hostage", "ceasefire"]

    candidates = []
    pub_counts: dict[str, int] = {}
    max_per_pub = 3

    with open(real_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "").strip().lower()
            content = (item.get("content", "") or item.get("text", "")).strip()
            text = f"{title} {content}".lower()

            if not any(kw in text for kw in keywords):
                continue

            h = hashlib.sha256(f"{title}|{content[:200]}".encode("utf-8")).hexdigest()
            if h in seen_fingerprints or item.get("url", "") in seen_fingerprints or title in seen_fingerprints:
                continue

            pub = item.get("publisher", "Unknown")
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            pub_counts[pub] = pub_counts.get(pub, 0) + 1
            if "text" not in item:
                item["text"] = f"{item.get('title', '')}\n\n{content}"
            candidates.append(item)

            if len(candidates) >= target_count:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in candidates:
            f.write(json.dumps(a) + "\n")

    print(f"[CORPUS] Harvested and saved {len(candidates)} genuinely untouched articles across {len(pub_counts)} publishers to {output_path}")
    return candidates


def evaluate_complementary_splits(score_matrix: np.ndarray) -> dict[str, Any]:
    N, K = score_matrix.shape
    assert K == 8
    all_axes = list(range(8))
    combos = list(itertools.combinations(all_axes, 4))
    seen = set()
    unique_pairs = []
    for c in combos:
        c_set = frozenset(c)
        comp_set = frozenset(set(all_axes) - c_set)
        if c_set not in seen and comp_set not in seen:
            seen.add(c_set)
            seen.add(comp_set)
            unique_pairs.append((list(c), list(set(all_axes) - c_set)))

    pearsons = []
    spearmans = []
    split_details = []

    for s1, s2 in unique_pairs:
        pca1 = PCA(n_components=1).fit(score_matrix[:, s1])
        pca2 = PCA(n_components=1).fit(score_matrix[:, s2])
        z1 = pca1.transform(score_matrix[:, s1])[:, 0]
        z2 = pca2.transform(score_matrix[:, s2])[:, 0]
        r, p_r = pearsonr(z1, z2)
        rho, p_rho = spearmanr(z1, z2)
        abs_r = float(abs(r))
        abs_rho = float(abs(rho))
        pearsons.append(abs_r)
        spearmans.append(abs_rho)
        split_details.append({
            "split_1": s1,
            "split_2": s2,
            "pearson_r": float(r),
            "abs_pearson_r": abs_r,
            "spearman_rho": float(rho),
            "abs_spearman_rho": abs_rho,
        })

    pearsons = np.array(pearsons)
    spearmans = np.array(spearmans)
    return {
        "n_partitions": len(unique_pairs),
        "pearson_abs_mean": float(pearsons.mean()),
        "pearson_abs_std": float(pearsons.std()),
        "pearson_abs_median": float(np.median(pearsons)),
        "pearson_abs_min": float(pearsons.min()),
        "pearson_abs_max": float(pearsons.max()),
        "spearman_abs_mean": float(spearmans.mean()),
        "spearman_abs_std": float(spearmans.std()),
        "spearman_abs_median": float(np.median(spearmans)),
        "spearman_abs_min": float(spearmans.min()),
        "spearman_abs_max": float(spearmans.max()),
        "fraction_ge_0_70": float((pearsons >= 0.70).mean()),
        "fraction_ge_0_80": float((pearsons >= 0.80).mean()),
        "splits": split_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Confirmatory replication on genuinely untouched articles.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    corpus_path = PROJECT_ROOT / "data" / "untouched_confirmatory_120_corpus.jsonl"
    corpus = build_untouched_corpus(corpus_path, target_count=120)
    publishers = [a.get("publisher", "unknown") for a in corpus]
    q0_pairs = load_q0_pairs()

    print(f"\n[SETUP] Initializing extractor with microsoft/deberta-v2-xlarge-mnli on {args.device}...")
    config = ExtractionConfig(
        model_name="microsoft/deberta-v2-xlarge-mnli",
        device=args.device,
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    cache_path = PROJECT_ROOT / "data" / "cache_untouched_replication.pt"
    if cache_path.exists():
        print(f"[CACHE] Loading cached representations from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        ideo_emb = cached["ideo_emb"]
        ctrl_emb = cached["ctrl_emb"]
        rand_emb = cached["rand_emb"]
        logits_a = cached["logits_a"]
        logits_b = cached["logits_b"]
    else:
        print("\n[EXTRACT] 1/3 Extracting Ideological pairs (8 axes)...")
        extractor.hypothesis_pairs = q0_pairs
        extractor.n_bots = len(q0_pairs)
        batch_ideo = extractor.extract_batch(corpus, show_progress=True)
        ideo_emb = batch_ideo["cls_per_bot"].cpu()
        logits_a = batch_ideo["logits_raw"].cpu()

        print("[EXTRACT] 2/3 Extracting Structural Control pairs (8 axes)...")
        extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
        extractor.n_bots = len(NEGATIVE_CONTROL_PAIRS)
        batch_ctrl = extractor.extract_batch(corpus, show_progress=True)
        ctrl_emb = batch_ctrl["cls_per_bot"].cpu()

        print("[EXTRACT] 3/3 Extracting Random Control pairs (8 axes)...")
        extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
        extractor.n_bots = len(RANDOM_CONTROL_PAIRS)
        batch_rand = extractor.extract_batch(corpus, show_progress=True)
        rand_emb = batch_rand["cls_per_bot"].cpu()

        # Extract opposing hypothesis logits for audited paired contrast
        print("[EXTRACT] Extracting opposing hypothesis B logits for paired contrast...")
        # Flip pairs (B, A) so logits_raw captures B
        flipped_pairs = [(b, a) for (a, b) in q0_pairs]
        extractor.hypothesis_pairs = flipped_pairs
        extractor.n_bots = len(flipped_pairs)
        batch_flipped = extractor.extract_batch(corpus, show_progress=True)
        logits_b = batch_flipped["logits_raw"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ideo_emb": ideo_emb,
                "ctrl_emb": ctrl_emb,
                "rand_emb": rand_emb,
                "logits_a": logits_a,
                "logits_b": logits_b,
            },
            cache_path,
        )
        print(f"[SAVE] Saved extraction cache to {cache_path}")

    # 1. High-Dimensional Hidden-State Axis Coherence
    print("\n" + "=" * 75)
    print("STEP 1: HIGH-DIMENSIONAL HIDDEN-STATE AXIS COHERENCE EVALUATION")
    print("=" * 75)

    residuals, pc1_evr = remove_global_pc1([ideo_emb, ctrl_emb, rand_emb])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))

    obs_ideo = subset_coherence(pooled_matrix, range(8))
    obs_ctrl = subset_coherence(pooled_matrix, range(8, 16))
    obs_rand = subset_coherence(pooled_matrix, range(16, 24))

    print(f"  Cone Anisotropy (Global PC1 EVR) : {pc1_evr*100:.1f}%")
    print(f"  Ideology Coherence              : {obs_ideo:+.6f}")
    print(f"  Structural Control Coherence    : {obs_ctrl:+.6f}")
    print(f"  Random Control Coherence        : {obs_rand:+.6f}")

    exact_null = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_ideo)
    print(f"  Exact Upper Tail Count          : {exact_null['upper_tail_count_ge_observed']} / {exact_null['total_subsets']}")
    print(f"  Exact p-value                   : {exact_null['exact_upper_tail_probability']:.6e}")

    # 2. Polarity Score Dimensionality Gating (Under both Single-Side and Audited Paired Contrast)
    print("\n" + "=" * 75)
    print("STEP 2: POLARITY SCORE DIMENSIONALITY NULL GATES (10,000 PERMUTATIONS)")
    print("=" * 75)

    probs_a = torch.softmax(logits_a.float(), dim=-1).numpy()
    probs_b = torch.softmax(logits_b.float(), dim=-1).numpy()

    # DeBERTa: index 0: Contradiction, 2: Entailment
    s_A = (probs_a[..., 2] - probs_a[..., 0]).astype(np.float64)
    s_B = (probs_b[..., 2] - probs_b[..., 0]).astype(np.float64)
    s_diff = (s_A - s_B).astype(np.float64)

    print("\n--- A. Audited Paired Displacement (s_A - s_B) ---")
    null_diff = run_dimensionality_null_test(
        s_diff, "Untouched N=120 Paired Contrast", publishers=publishers, draws=args.permutations
    )
    c_diff_1 = null_diff["components"][0]
    c_diff_2 = null_diff["components"][1]
    print(f"  PC1 EVR: {c_diff_1['observed_variance_ratio']*100:.2f}% (p = {c_diff_1['column_null']['p_value']:.4f})")
    print(f"  PC2 EVR: {c_diff_2['observed_variance_ratio']*100:.2f}% (p = {c_diff_2['column_null']['p_value']:.4f})")
    print(f"  Detected Rank: k = {null_diff['detected_rank_column_null']}")
    print(f"  Participation Ratio: {null_diff['observed_participation_ratio']:.2f}")

    print("\n--- B. Single-Side Polarity (s_A) ---")
    null_sa = run_dimensionality_null_test(
        s_A, "Untouched N=120 Single-Side A", publishers=publishers, draws=args.permutations
    )
    c_sa_1 = null_sa["components"][0]
    c_sa_2 = null_sa["components"][1]
    print(f"  PC1 EVR: {c_sa_1['observed_variance_ratio']*100:.2f}% (p = {c_sa_1['column_null']['p_value']:.4f})")
    print(f"  PC2 EVR: {c_sa_2['observed_variance_ratio']*100:.2f}% (p = {c_sa_2['column_null']['p_value']:.4f})")
    print(f"  Detected Rank: k = {null_sa['detected_rank_column_null']}")
    print(f"  Participation Ratio: {null_sa['observed_participation_ratio']:.2f}")

    # 3. Complementary Disjoint-Bank Agreement (35 unique 4-vs-4 partitions)
    print("\n" + "=" * 75)
    print("STEP 3: COMPLEMENTARY DISJOINT-BANK AGREEMENT (35 UNIQUE 4-vs-4 PARTITIONS)")
    print("=" * 75)

    disjoint_diff = evaluate_complementary_splits(s_diff)
    print(f"Audited Paired Displacement (s_A - s_B) across 35 disjoint splits:")
    print(f"  Pearson |r|  : mean = {disjoint_diff['pearson_abs_mean']:.4f} +/- {disjoint_diff['pearson_abs_std']:.4f}, median = {disjoint_diff['pearson_abs_median']:.4f}, min = {disjoint_diff['pearson_abs_min']:.4f}, max = {disjoint_diff['pearson_abs_max']:.4f}")
    print(f"  Spearman |rho|: mean = {disjoint_diff['spearman_abs_mean']:.4f} +/- {disjoint_diff['spearman_abs_std']:.4f}, median = {disjoint_diff['spearman_abs_median']:.4f}, min = {disjoint_diff['spearman_abs_min']:.4f}, max = {disjoint_diff['spearman_abs_max']:.4f}")
    print(f"  Fraction with |r| >= 0.70: {disjoint_diff['fraction_ge_0_70']*100:.1f}%")

    disjoint_sa = evaluate_complementary_splits(s_A)
    print(f"\nSingle-Side Polarity (s_A) across 35 disjoint splits:")
    print(f"  Pearson |r|  : mean = {disjoint_sa['pearson_abs_mean']:.4f} +/- {disjoint_sa['pearson_abs_std']:.4f}, median = {disjoint_sa['pearson_abs_median']:.4f}, min = {disjoint_sa['pearson_abs_min']:.4f}, max = {disjoint_sa['pearson_abs_max']:.4f}")
    print(f"  Spearman |rho|: mean = {disjoint_sa['spearman_abs_mean']:.4f} +/- {disjoint_sa['spearman_abs_std']:.4f}, median = {disjoint_sa['spearman_abs_median']:.4f}, min = {disjoint_sa['spearman_abs_min']:.4f}, max = {disjoint_sa['spearman_abs_max']:.4f}")

    output_payload = {
        "dataset": "untouched_confirmatory_120_corpus",
        "n_articles": len(corpus),
        "n_publishers": len(set(publishers)),
        "coherence": {
            "ideology": float(obs_ideo),
            "structural": float(obs_ctrl),
            "random": float(obs_rand),
            "exact_upper_tail_count": int(exact_null["upper_tail_count_ge_observed"]),
            "total_subsets": int(exact_null["total_subsets"]),
            "exact_p_value": float(exact_null["exact_upper_tail_probability"]),
            "anisotropy_pc1_evr": float(pc1_evr),
        },
        "dimensionality_gates": {
            "audited_paired_contrast": {
                "pc1_evr": float(c_diff_1["observed_variance_ratio"]),
                "pc1_p_value": float(c_diff_1["column_null"]["p_value"]),
                "pc2_evr": float(c_diff_2["observed_variance_ratio"]),
                "pc2_p_value": float(c_diff_2["column_null"]["p_value"]),
                "detected_rank": int(null_diff["detected_rank_column_null"]),
                "participation_ratio": float(null_diff["observed_participation_ratio"]),
            },
            "single_side_polarity": {
                "pc1_evr": float(c_sa_1["observed_variance_ratio"]),
                "pc1_p_value": float(c_sa_1["column_null"]["p_value"]),
                "pc2_evr": float(c_sa_2["observed_variance_ratio"]),
                "pc2_p_value": float(c_sa_2["column_null"]["p_value"]),
                "detected_rank": int(null_sa["detected_rank_column_null"]),
                "participation_ratio": float(null_sa["observed_participation_ratio"]),
            },
        },
        "complementary_disjoint_splits": {
            "audited_paired_contrast": disjoint_diff,
            "single_side_polarity": disjoint_sa,
        },
    }

    out_file = PROJECT_ROOT / "paper" / "untouched_confirmatory_replication.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved untouched confirmatory replication report to {out_file}")


if __name__ == "__main__":
    main()
