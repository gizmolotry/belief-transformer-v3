from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


def audit_label_mappings(models: list[str]) -> dict[str, Any]:
    print("=" * 75)
    print("STEP 1: AUDIT OF LABEL MAPPINGS ACROSS NLI MODELS")
    print("=" * 75)
    mappings = {}
    for m in models:
        cfg = AutoConfig.from_pretrained(m)
        id2label = cfg.id2label
        label2id = cfg.label2id
        print(f"Model: {m}")
        print(f"  id2label: {id2label}")
        print(f"  label2id: {label2id}")
        mappings[m] = {
            "id2label": {int(k): v for k, v in id2label.items()},
            "label2id": {k: int(v) for k, v in label2id.items()},
        }
    return mappings


def extract_both_hypotheses(
    corpus: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    model_name: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cuda",
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw forward-pass logits for BOTH Hypothesis A and Hypothesis B separately.
    Returns:
      logits_a: [N, K, 3]
      logits_b: [N, K, 3]
    """
    print(f"\n[EXTRACT] Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    cfg = model.config
    # Determine entailment and contradiction indices
    id2label = {int(k): v.upper() for k, v in cfg.id2label.items()}
    print(f"[CONFIG] Model label mapping: {id2label}")

    n_articles = len(corpus)
    n_pairs = len(pairs)
    logits_a_list = []
    logits_b_list = []

    print(f"[EXTRACT] Extracting {n_articles} articles across {n_pairs} paired hypotheses...")
    with torch.no_grad():
        for i, article in enumerate(corpus):
            if (i + 1) % 20 == 0 or i == 0 or i == n_articles - 1:
                print(f"  Article {i+1}/{n_articles}: {article.get('title', '')[:50]}...")
            text = article.get("content", "") or article.get("text", "") or article.get("title", "")
            text = text[:2000]  # truncate for consistent length

            # Prepare premise-hypothesis pairs for A and B
            premises_a = [text] * n_pairs
            hypotheses_a = [p[0] for p in pairs]
            premises_b = [text] * n_pairs
            hypotheses_b = [p[1] for p in pairs]

            inputs_a = tokenizer(
                premises_a,
                hypotheses_a,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out_a = model(**inputs_a)
            log_a = out_a.logits.cpu().numpy()  # [K, 3]

            inputs_b = tokenizer(
                premises_b,
                hypotheses_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out_b = model(**inputs_b)
            log_b = out_b.logits.cpu().numpy()  # [K, 3]

            logits_a_list.append(log_a)
            logits_b_list.append(log_b)

    logits_a = np.stack(logits_a_list, axis=0)  # [N, K, 3]
    logits_b = np.stack(logits_b_list, axis=0)  # [N, K, 3]
    return logits_a, logits_b


def main():
    parser = argparse.ArgumentParser(description="Audit of logits_raw and operator comparison.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    # Step 1: Label mapping audit
    models = [
        "microsoft/deberta-v2-xlarge-mnli",
        "facebook/bart-large-mnli",
        "roberta-large-mnli",
    ]
    label_audit = audit_label_mappings(models)

    # Step 2: Load held-out N=120 articles
    corpus_path = PROJECT_ROOT / "data" / "fresh_unseen_120_corpus.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    publishers = [a.get("publisher", "unknown") for a in corpus]

    # Load canonical Q0 pairs
    import yaml
    with open(PROJECT_ROOT / "config" / "framing_queries.yaml", "r", encoding="utf-8") as f:
        pairs = [(v["A"], v["B"]) for v in yaml.safe_load(f)["queries"].values()]

    cache_file = PROJECT_ROOT / "data" / "cache_operator_audit_logits.pt"
    if cache_file.exists():
        print(f"\n[CACHE] Loading dual-hypothesis logits from {cache_file}...")
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        logits_a = cached["logits_a"]
        logits_b = cached["logits_b"]
    else:
        logits_a, logits_b = extract_both_hypotheses(
            corpus=corpus,
            pairs=pairs,
            model_name="microsoft/deberta-v2-xlarge-mnli",
            device=args.device,
        )
        torch.save({"logits_a": logits_a, "logits_b": logits_b}, cache_file)
        print(f"[CACHE] Saved dual logits to {cache_file}")

    # Compute softmax probabilities
    # DeBERTa label mapping: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
    def compute_probs(log_arr: np.ndarray) -> np.ndarray:
        e = np.exp(log_arr - np.max(log_arr, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    probs_a = compute_probs(logits_a)
    probs_b = compute_probs(logits_b)

    # Margins: P(entailment) - P(contradiction)
    s_A = (probs_a[..., 2] - probs_a[..., 0]).astype(np.float64)  # [N, K]
    s_B = (probs_b[..., 2] - probs_b[..., 0]).astype(np.float64)  # [N, K]

    # Operators:
    # 1. Single-side A: s_A
    # 2. Single-side B: s_B
    # 3. Audited Paired Contrast: s_A - s_B
    # 4. Common-Mode Salience: s_A + s_B
    operators = {
        "Single-Side Polarity A (s_A)": s_A,
        "Single-Side Polarity B (s_B)": s_B,
        "Audited Paired Displacement (s_A - s_B)": s_A - s_B,
        "Common-Mode Salience (s_A + s_B)": s_A + s_B,
    }

    print("\n" + "=" * 75)
    print("STEP 2: OPERATOR COMPARISON ON HELD-OUT N=120 CORPUS")
    print("=" * 75)

    operator_results = {}
    for op_name, score_mat in operators.items():
        score_mat = score_mat.astype(np.float64)
        pca = PCA().fit(score_mat)
        evr = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_
        pr = float((np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2))

        null_res = run_dimensionality_null_test(
            score_mat, op_name, publishers=publishers, draws=args.permutations
        )
        c1 = null_res["components"][0]
        c2 = null_res["components"][1]

        print(f"\n--- Operator: {op_name} ---")
        print(f"  PC1 EVR: {c1['observed_variance_ratio']*100:.2f}% (p = {c1['column_null']['p_value']:.4f}, null mean: {c1['column_null']['mean_variance_ratio']*100:.2f}%)")
        print(f"  PC2 EVR: {c2['observed_variance_ratio']*100:.2f}% (p = {c2['column_null']['p_value']:.4f}, null mean: {c2['column_null']['mean_variance_ratio']*100:.2f}%)")
        print(f"  Detected Rank: k = {null_res['detected_rank_column_null']}")
        print(f"  Participation Ratio: {pr:.2f} (null mean: {null_res['column_null_mean_participation_ratio']:.2f})")

        operator_results[op_name] = {
            "pc1_evr": float(c1["observed_variance_ratio"]),
            "pc1_p_value": float(c1["column_null"]["p_value"]),
            "pc2_evr": float(c2["observed_variance_ratio"]),
            "pc2_p_value": float(c2["column_null"]["p_value"]),
            "detected_rank": int(null_res["detected_rank_column_null"]),
            "participation_ratio": float(pr),
            "eigenspectrum": [float(v) for v in evr],
        }

    # Correlation between Single-Side A and Paired Contrast
    pca_sa = PCA(n_components=1).fit(s_A)
    pca_diff = PCA(n_components=1).fit(s_A - s_B)
    z_sa = pca_sa.transform(s_A)[:, 0]
    z_diff = pca_diff.transform(s_A - s_B)[:, 0]
    r_align = float(abs(np.corrcoef(z_sa, z_diff)[0, 1]))
    print(f"\n[CORRELATION] |r(PC1_sA, PC1_diff)| = {r_align:.4f}")

    output_payload = {
        "label_audit": label_audit,
        "n_articles": len(corpus),
        "operator_comparison": operator_results,
        "correlation_sA_vs_paired_diff": r_align,
    }

    out_file = PROJECT_ROOT / "paper" / "operator_audit_and_paired_contrast.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved operator audit report to {out_file}")


if __name__ == "__main__":
    main()
