from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
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


def load_q1_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(raw[2 * i], raw[2 * i + 1]) for i in range(8)]


def load_exclusion_registry() -> set[str]:
    reg_path = PROJECT_ROOT / "data" / "touched_fingerprints_registry.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"Exclusion registry not found at {reg_path}")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["fingerprints"])


def sample_prospective_corpus(
    output_path: Path,
    target_count: int = 120,
    max_per_pub: int = 8,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing prospective corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Sampling {target_count} prospective articles from real_corpus.jsonl...")
    excluded_fps = load_exclusion_registry()
    print(f"[CORPUS] Loaded {len(excluded_fps)} excluded fingerprints.")

    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"
    keywords = ["palestine", "palestinian", "gaza", "israel", "israeli", "hamas"]

    articles = []
    pub_counts: dict[str, int] = {}
    skipped_excluded = 0

    with open(real_corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
            except Exception:
                continue

            content = data.get("content") or data.get("text") or data.get("body") or ""
            title = data.get("title") or ""
            full_text = f"{title}\n\n{content}".strip()
            text_lower = full_text.lower()

            if not any(kw in text_lower for kw in keywords):
                continue
            if len(content.split()) < 80:
                continue

            # Compute fingerprints
            h_text = hashlib.sha256(f"{title.strip().lower()}|{content.strip()[:200]}".encode("utf-8")).hexdigest()
            title_clean = title.strip().lower()
            url = data.get("url", "")

            if h_text in excluded_fps or title_clean in excluded_fps or (url and url in excluded_fps):
                skipped_excluded += 1
                continue

            pub = data.get("publisher", "Unknown") or "Unknown"
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            pub_counts[pub] = pub_counts.get(pub, 0) + 1
            record = {
                "id": f"prospective_confirmatory_{len(articles)+1:04d}",
                "title": title,
                "text": full_text,
                "publisher": pub,
                "url": url,
                "source_line": idx,
                "fingerprint": h_text,
            }
            articles.append(record)
            if len(articles) >= target_count:
                break

    print(f"[CORPUS] Sampled {len(articles)} articles across {len(pub_counts)} publishers. Skipped {skipped_excluded} excluded.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
    return articles


def extract_paired_logits(
    corpus: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    model_name: str,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    n_articles = len(corpus)
    n_pairs = len(pairs)
    logits_a_list = []
    logits_b_list = []

    with torch.no_grad():
        for i, article in enumerate(corpus):
            text = article.get("text", "")[:2000]

            premises_a = [text] * n_pairs
            hypotheses_a = [p[0] for p in pairs]
            premises_b = [text] * n_pairs
            hypotheses_b = [p[1] for p in pairs]

            inputs_a = tokenizer(premises_a, hypotheses_a, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out_a = model(**inputs_a)
            logits_a_list.append(out_a.logits.cpu().numpy())

            inputs_b = tokenizer(premises_b, hypotheses_b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out_b = model(**inputs_b)
            logits_b_list.append(out_b.logits.cpu().numpy())

    del model
    del tokenizer
    torch.cuda.empty_cache()

    logits_a = np.stack(logits_a_list, axis=0)  # [N, K, 3]
    logits_b = np.stack(logits_b_list, axis=0)  # [N, K, 3]
    return logits_a, logits_b


def compute_paired_contrast(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    def compute_probs(log_arr: np.ndarray) -> np.ndarray:
        e = np.exp(log_arr - np.max(log_arr, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    probs_a = compute_probs(logits_a)
    probs_b = compute_probs(logits_b)
    s_A = (probs_a[..., 2] - probs_a[..., 0]).astype(np.float64)
    s_B = (probs_b[..., 2] - probs_b[..., 0]).astype(np.float64)
    return s_A - s_B


def evaluate_disjoint_partitions(matrix: np.ndarray) -> dict[str, Any]:
    n_axes = matrix.shape[1]
    all_axes = list(range(n_axes))
    k = n_axes // 2
    seen_partitions = set()
    split_corrs = []

    for subset in itertools.combinations(all_axes, k):
        s_set = frozenset(subset)
        comp_set = frozenset(set(all_axes) - s_set)
        part_key = frozenset([s_set, comp_set])
        if part_key in seen_partitions:
            continue
        seen_partitions.add(part_key)

        s_idx = sorted(list(s_set))
        c_idx = sorted(list(comp_set))

        pca_s = PCA(n_components=1).fit(matrix[:, s_idx])
        pca_c = PCA(n_components=1).fit(matrix[:, c_idx])

        proj_s = pca_s.transform(matrix[:, s_idx])[:, 0]
        proj_c = pca_c.transform(matrix[:, c_idx])[:, 0]

        r_val, _ = pearsonr(proj_s, proj_c)
        split_corrs.append(abs(float(r_val)))

    return {
        "n_partitions": len(split_corrs),
        "mean_abs_r": float(np.mean(split_corrs)),
        "std_abs_r": float(np.std(split_corrs)),
        "median_abs_r": float(np.median(split_corrs)),
        "min_abs_r": float(np.min(split_corrs)),
        "max_abs_r": float(np.max(split_corrs)),
        "fraction_ge_0_70": float(np.mean([x >= 0.70 for x in split_corrs])),
    }


def main():
    parser = argparse.ArgumentParser(description="Prospective Confirmatory Preregistered Replication.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--corpus-output", type=Path, default=PROJECT_ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "prospective_confirmatory_replication_results.json")
    args = parser.parse_args()

    model_name = "microsoft/deberta-v2-xlarge-mnli"

    # Step 1: Sample prospective corpus
    corpus = sample_prospective_corpus(args.corpus_output, target_count=120, max_per_pub=8)
    publishers = [a.get("publisher", "Unknown") for a in corpus]

    q0_pairs = load_q0_pairs()
    q1_pairs = load_q1_pairs()

    print("\n" + "=" * 80)
    print("PROSPECTIVE CONFIRMATORY REPLICATION: Q0 & Q1 UNDER AUDITED PAIRED CONTRAST")
    print("=" * 80)

    # Step 2: Extract paired logits for Q0 and Q1
    print("\n[EXTRACT] 1/2 Extracting Q0 paired logits...")
    q0_la, q0_lb = extract_paired_logits(corpus, q0_pairs, model_name, device=args.device)
    q0_delta = compute_paired_contrast(q0_la, q0_lb)

    print("\n[EXTRACT] 2/2 Extracting Q1 paired logits...")
    q1_la, q1_lb = extract_paired_logits(corpus, q1_pairs, model_name, device=args.device)
    q1_delta = compute_paired_contrast(q1_la, q1_lb)

    # Step 3: Dimensionality Gating (10,000 draws)
    print("\n[ANALYSIS] Evaluating Q0 Dimensionality Gate (10,000 draws)...")
    q0_gate = run_dimensionality_null_test(q0_delta, "Q0_prospective_paired", publishers=publishers, draws=args.permutations)
    pca_q0 = PCA().fit(q0_delta)
    proj_q0 = pca_q0.transform(q0_delta)
    q0_c1 = q0_gate["components"][0]
    q0_c2 = q0_gate["components"][1]

    print("\n[ANALYSIS] Evaluating Q1 Dimensionality Gate (10,000 draws)...")
    q1_gate = run_dimensionality_null_test(q1_delta, "Q1_prospective_paired", publishers=publishers, draws=args.permutations)
    pca_q1 = PCA().fit(q1_delta)
    proj_q1 = pca_q1.transform(q1_delta)
    q1_c1 = q1_gate["components"][0]
    q1_c2 = q1_gate["components"][1]

    # Step 4: Cross-Bank Stance Alignment (Q0 vs Q1)
    p0 = proj_q0[:, 0].copy()
    p1 = proj_q1[:, 0].copy()
    comp0 = pca_q0.components_[:1]
    comp1 = pca_q1.components_[:1]

    if np.corrcoef(p0, p1)[0, 1] < 0:
        p1 = -p1
        comp1 = -comp1

    r_cross, p_r_cross = pearsonr(p0, p1)
    rho_cross, p_rho_cross = spearmanr(p0, p1)
    _, _, disp_cross = procrustes(proj_q0[:, :2], proj_q1[:, :2])
    ang_rad = subspace_angles(comp0.T, comp1.T)[0]
    ang_deg = float(np.degrees(ang_rad))

    # Step 5: 35 Disjoint Splits
    print("\n[ANALYSIS] Evaluating 35 Disjoint Partitions for Q0 and Q1...")
    splits_q0 = evaluate_disjoint_partitions(q0_delta)
    splits_q1 = evaluate_disjoint_partitions(q1_delta)

    # Step 6: Descriptive Hidden-State Axis Coherence
    print("\n[ANALYSIS] Extracting CLS embeddings for descriptive axis coherence...")
    config = ExtractionConfig(model_name=model_name, device=args.device, use_cls_tokens=True)
    extractor = UnifiedNLIExtractor(config)

    extractor.hypothesis_pairs = q0_pairs
    extractor.n_bots = len(q0_pairs)
    batch_ideo = extractor.extract_batch(corpus, show_progress=False)

    extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
    extractor.n_bots = len(NEGATIVE_CONTROL_PAIRS)
    batch_ctrl = extractor.extract_batch(corpus, show_progress=False)

    extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
    extractor.n_bots = len(RANDOM_CONTROL_PAIRS)
    batch_rand = extractor.extract_batch(corpus, show_progress=False)

    residuals, pc1_evr = remove_global_pc1([batch_ideo["cls_per_bot"].cpu(), batch_ctrl["cls_per_bot"].cpu(), batch_rand["cls_per_bot"].cpu()])
    pooled_matrix = axis_coherence_matrix(torch.cat(residuals, dim=1))
    obs_coherence = subset_coherence(pooled_matrix, range(8))
    exact_coherence = exhaustive_upper_tail(pooled_matrix, subset_size=8, observed=obs_coherence)

    results = {
        "metadata": {
            "title": "Prospective Confirmatory Preregistered Replication",
            "model_name": model_name,
            "n_articles": len(corpus),
            "n_publishers": len(set(publishers)),
            "permutations": args.permutations,
            "operator": "Audited Paired Contrast: (P_A(E) - P_A(C)) - (P_B(E) - P_B(C))",
        },
        "q0_results": {
            "pc1_evr": float(pca_q0.explained_variance_ratio_[0]),
            "pc2_evr": float(pca_q0.explained_variance_ratio_[1]),
            "observed_pr": q0_gate["observed_participation_ratio"],
            "col_null_mean_pr": q0_gate["column_null_mean_participation_ratio"],
            "detected_rank_col": q0_gate["detected_rank_column_null"],
            "detected_rank_pub": q0_gate["detected_rank_publisher_null"],
            "pc1_col_p": q0_c1["column_null"]["p_value"],
            "pc1_col_q95": q0_c1["column_null"]["q95_variance_ratio"],
            "pc2_col_p": q0_c2["column_null"]["p_value"],
            "pc2_col_q95": q0_c2["column_null"]["q95_variance_ratio"],
            "disjoint_partitions": splits_q0,
        },
        "q1_results": {
            "pc1_evr": float(pca_q1.explained_variance_ratio_[0]),
            "pc2_evr": float(pca_q1.explained_variance_ratio_[1]),
            "observed_pr": q1_gate["observed_participation_ratio"],
            "col_null_mean_pr": q1_gate["column_null_mean_participation_ratio"],
            "detected_rank_col": q1_gate["detected_rank_column_null"],
            "detected_rank_pub": q1_gate["detected_rank_publisher_null"],
            "pc1_col_p": q1_c1["column_null"]["p_value"],
            "pc1_col_q95": q1_c1["column_null"]["q95_variance_ratio"],
            "pc2_col_p": q1_c2["column_null"]["p_value"],
            "pc2_col_q95": q1_c2["column_null"]["q95_variance_ratio"],
            "disjoint_partitions": splits_q1,
        },
        "cross_bank_q0_vs_q1": {
            "pearson_r": float(r_cross),
            "pearson_p": float(p_r_cross),
            "spearman_rho": float(rho_cross),
            "spearman_p": float(p_rho_cross),
            "procrustes_disparity_2pc": float(disp_cross),
            "subspace_angle_pc1_deg": float(ang_deg),
        },
        "descriptive_hidden_state_coherence": {
            "observed_coherence": float(obs_coherence),
            "exact_p_value": float(exact_coherence["exact_upper_tail_probability"]),
            "upper_tail_count": exact_coherence["upper_tail_count_ge_observed"],
            "total_subsets": exact_coherence["total_subsets"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("PROSPECTIVE CONFIRMATORY REPLICATION SUMMARY")
    print("=" * 80)
    print(f"Q0: PC1={results['q0_results']['pc1_evr']*100:.2f}% (p={results['q0_results']['pc1_col_p']:.4f}), Rank=k={results['q0_results']['detected_rank_col']}")
    print(f"Q1: PC1={results['q1_results']['pc1_evr']*100:.2f}% (p={results['q1_results']['pc1_col_p']:.4f}), Rank=k={results['q1_results']['detected_rank_col']}")
    print(f"Q0 <-> Q1: Pearson r={r_cross:.4f} (p={p_r_cross:.2e}), Spearman rho={rho_cross:.4f}, Subspace Angle={ang_deg:.2f}°")
    print(f"Disjoint 35 Splits: Q0 Mean |r|={splits_q0['mean_abs_r']:.4f}, Q1 Mean |r|={splits_q1['mean_abs_r']:.4f}")
    print(f"Descriptive Coherence: Obs={obs_coherence:.5f}, p={exact_coherence['exact_upper_tail_probability']:.4f}")
    print(f"\n[SUCCESS] Saved report to {args.output}")


if __name__ == "__main__":
    main()
