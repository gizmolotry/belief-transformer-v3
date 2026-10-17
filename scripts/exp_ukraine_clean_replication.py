from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


def compute_shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = [w.strip() for w in text.lower().split() if w.strip()]
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


import yaml

def load_ukraine_pairs() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "config" / "ukraine_cross_topic_queries.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [(v["A"], v["B"]) for v in data["queries"].values()]


def sample_clean_ukraine_corpus(
    output_path: Path,
    target_count: int = 60,
    max_per_pub: int = 5,
    max_jaccard: float = 0.35,
) -> list[dict[str, Any]]:
    if output_path.exists():
        print(f"[CORPUS] Loading existing clean Ukraine corpus from {output_path}...")
        articles = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    print(f"[CORPUS] Sampling clean, deduplicated Ukraine corpus from real_corpus.jsonl...")
    real_corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    ukr_primary = ["ukraine", "ukrainian"]
    ukr_secondary = [
        "russia", "russian", "putin", "zelensky", "kyiv", "donbas", "crimea",
        "nato", "frontline", "missile", "artillery", "offensive", "kharkiv",
        "donetsk", "luhansk", "zaporizhzhia", "kherson", "bakhmut", "counteroffensive"
    ]
    junk_filters = [
        "grozny", "azerbaijan", "airliner", "plane crash", "aviation accident",
        "drowning", "shopping mall", "the sun malaysia"
    ]

    articles = []
    accepted_shingles = []
    pub_counts: dict[str, int] = {}
    skipped_junk = 0
    skipped_syndicated = 0

    with open(real_corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
            except Exception:
                continue

            content = data.get("content") or data.get("text") or data.get("body") or ""
            title = data.get("title") or ""
            full_text = f"{title}\n\n{content}".strip()
            full_lower = full_text.lower()

            if len(content.split()) < 100 or len(content) < 600:
                continue

            # Check keyword relevance
            has_primary = any(kw in full_lower for kw in ukr_primary)
            sec_matches = set(kw for kw in ukr_secondary if kw in full_lower)
            if not (has_primary and len(sec_matches) >= 3):
                continue

            # Negative junk filters
            if any(junk in full_lower for junk in junk_filters):
                skipped_junk += 1
                continue

            pub = data.get("publisher") or data.get("source") or data.get("publication") or "Unknown"
            if pub_counts.get(pub, 0) >= max_per_pub:
                continue

            # Content deduplication via 5-token shingle Jaccard
            shingles = compute_shingles(full_text, k=5)
            is_dup = False
            for prev_s in accepted_shingles:
                if jaccard_similarity(shingles, prev_s) >= max_jaccard:
                    is_dup = True
                    break

            if is_dup:
                skipped_syndicated += 1
                continue

            # Accept article
            pub_counts[pub] = pub_counts.get(pub, 0) + 1
            accepted_shingles.append(shingles)
            rec = {
                "id": f"ukraine_clean_{len(articles)+1:03d}",
                "title": title,
                "text": full_text,
                "publisher": pub,
                "url": data.get("url", ""),
                "source_line": idx,
                "secondary_keywords": sorted(list(sec_matches)),
            }
            articles.append(rec)

            if len(articles) >= target_count:
                break

    print(f"[CORPUS] Sampled {len(articles)} clean articles across {len(pub_counts)} publishers.")
    print(f"[CORPUS] Skipped: {skipped_junk} junk/airliner matches, {skipped_syndicated} syndicated near-duplicates.")

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
            if (i + 1) % 15 == 0 or i == 0 or i == n_articles - 1:
                print(f"  [EXTRACT] Processing article {i+1}/{n_articles}...")
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

    return np.stack(logits_a_list, axis=0), np.stack(logits_b_list, axis=0)


def compute_paired_contrast(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    def compute_probs(log_arr: np.ndarray) -> np.ndarray:
        e = np.exp(log_arr - np.max(log_arr, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    probs_a = compute_probs(logits_a)
    probs_b = compute_probs(logits_b)
    s_A = (probs_a[..., 2] - probs_a[..., 0]).astype(np.float64)
    s_B = (probs_b[..., 2] - probs_b[..., 0]).astype(np.float64)
    return s_A - s_B


def main():
    parser = argparse.ArgumentParser(description="Clean Ukraine Replication.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    model_name = "microsoft/deberta-v2-xlarge-mnli"
    corpus_path = PROJECT_ROOT / "data" / "ukraine_clean_N60_corpus.jsonl"
    cache_path = PROJECT_ROOT / "data" / "cache_paired_logits_ukraine_clean_N60.pt"
    results_path = PROJECT_ROOT / "paper" / "ukraine_clean_replication_results.json"

    # Step 1: Sample clean corpus
    corpus = sample_clean_ukraine_corpus(corpus_path, target_count=60, max_per_pub=5, max_jaccard=0.35)
    publishers = [a.get("publisher", "Unknown") for a in corpus]
    pairs = load_ukraine_pairs()

    print("\n" + "=" * 80)
    print("CLEAN UKRAINE REPLICATION UNDER AUDITED PAIRED CONTRAST (s_A - s_B)")
    print("=" * 80)

    # Step 2: Extract or load paired logits
    if cache_path.exists():
        print(f"[CACHE] Loading cached logits from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        la, lb = cached["la"], cached["lb"]
        if isinstance(la, torch.Tensor):
            la = la.numpy()
        if isinstance(lb, torch.Tensor):
            lb = lb.numpy()
    else:
        print("[EXTRACT] Extracting paired logits on GPU...")
        la, lb = extract_paired_logits(corpus, pairs, model_name, device=args.device)
        torch.save({"la": torch.from_numpy(la), "lb": torch.from_numpy(lb)}, cache_path)
        print(f"[CACHE] Saved raw cached logits to {cache_path}")

    # Step 3: Compute paired contrast
    delta = compute_paired_contrast(la, lb)

    # Step 4: Per-axis diagnostics
    per_axis_means = np.mean(delta, axis=0)
    per_axis_stds = np.std(delta, axis=0)
    per_axis_mins = np.min(delta, axis=0)
    per_axis_maxs = np.max(delta, axis=0)
    corr_mat = np.corrcoef(delta.T)
    off_diag = corr_mat[~np.eye(8, dtype=bool)]

    # Step 5: Systematic Dimensionality Gate
    print("\n[GATE] Running 10,000-draw dimensionality gate...")
    gate = run_dimensionality_null_test(delta, "ukraine_clean_paired", publishers=publishers, draws=args.permutations)

    pca = PCA().fit(delta)
    evr = pca.explained_variance_ratio_
    obs_pr = float((np.sum(evr)) ** 2 / np.sum(evr ** 2))

    c1 = gate["components"][0]
    c2 = gate["components"][1]
    pub_c1 = c1.get("publisher_block_null", {})
    pub_c2 = c2.get("publisher_block_null", {})

    print("\n--- RESULTS ON CLEAN UKRAINE CORPUS ---")
    print(f"PC1 EVR: {evr[0]*100:.2f}% (Col Null q95: {c1['column_null']['q95_variance_ratio']*100:.2f}%, p = {c1['column_null']['p_value']:.4f})")
    print(f"PC2 EVR: {evr[1]*100:.2f}% (Col Null q95: {c2['column_null']['q95_variance_ratio']*100:.2f}%, p = {c2['column_null']['p_value']:.4f})")
    print(f"Participation Ratio: {obs_pr:.3f} (Col Null Mean PR: {gate['column_null_mean_participation_ratio']:.3f})")
    print(f"Detected Rank (Col Null): k = {gate['detected_rank_column_null']}")
    print(f"Mean Off-Diagonal Correlation: r = {np.mean(off_diag):+.4f} (range: [{np.min(off_diag):+.4f}, {np.max(off_diag):+.4f}])")

    results = {
        "metadata": {
            "title": "Clean Ukraine Replication under Audited Paired Contrast",
            "model_name": model_name,
            "n_articles": len(corpus),
            "n_publishers": len(set(publishers)),
            "publisher_counts": dict(Counter(publishers)),
            "permutations": args.permutations,
            "operator": "Audited Paired Contrast: (P_A(E) - P_A(C)) - (P_B(E) - P_B(C))",
            "clean_criteria": {
                "max_shingle_jaccard": 0.35,
                "junk_filters": ["grozny", "azerbaijan", "airliner", "plane crash", "drowning", "aviation accident", "shopping mall"],
            },
        },
        "pca": {
            "explained_variance_ratios": [float(x) for x in evr],
            "observed_pr": obs_pr,
            "column_null_mean_pr": gate["column_null_mean_participation_ratio"],
            "detected_rank_col": gate["detected_rank_column_null"],
            "detected_rank_pub": gate.get("detected_rank_publisher_null"),
        },
        "pc1": {
            "evr": float(evr[0]),
            "column_null_q95": c1["column_null"]["q95_variance_ratio"],
            "column_null_p": c1["column_null"]["p_value"],
            "detected": c1["column_null"]["detected"],
        },
        "pc2": {
            "evr": float(evr[1]),
            "column_null_q95": c2["column_null"]["q95_variance_ratio"],
            "column_null_p": c2["column_null"]["p_value"],
            "detected": c2["column_null"]["detected"],
        },
        "axis_diagnostics": {
            "means": [float(x) for x in per_axis_means],
            "stds": [float(x) for x in per_axis_stds],
            "mins": [float(x) for x in per_axis_mins],
            "maxs": [float(x) for x in per_axis_maxs],
            "mean_off_diagonal_r": float(np.mean(off_diag)),
            "min_off_diagonal_r": float(np.min(off_diag)),
            "max_off_diagonal_r": float(np.max(off_diag)),
            "fraction_negative_off_diagonal": float(np.mean(off_diag < 0)),
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Saved full results to {results_path}")


if __name__ == "__main__":
    main()
