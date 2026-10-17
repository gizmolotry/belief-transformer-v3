from __future__ import annotations

import argparse
import json
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
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


def load_yaml_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    queries = data.get("queries", {})
    return [(v["A"], v["B"]) for v in queries.values()]


def extract_paired_logits(
    corpus: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 16,
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
    id2label = {int(k): v.upper() for k, v in cfg.id2label.items()}
    # Verify 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
    assert "CONTRADICTION" in id2label[0], f"Expected index 0 to be CONTRADICTION in {model_name}, got {id2label}"
    assert "ENTAILMENT" in id2label[2], f"Expected index 2 to be ENTAILMENT in {model_name}, got {id2label}"

    n_articles = len(corpus)
    n_pairs = len(pairs)
    logits_a_list = []
    logits_b_list = []

    print(f"[EXTRACT] Running {n_articles} articles across {n_pairs} paired hypotheses...")
    with torch.no_grad():
        for i, article in enumerate(corpus):
            text = article.get("content", "") or article.get("text", "") or article.get("title", "")
            text = text[:2000]

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
            log_a = out_a.logits.cpu().numpy()

            inputs_b = tokenizer(
                premises_b,
                hypotheses_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out_b = model(**inputs_b)
            log_b = out_b.logits.cpu().numpy()

            logits_a_list.append(log_a)
            logits_b_list.append(log_b)

    logits_a = np.stack(logits_a_list, axis=0)  # [N, K, 3]
    logits_b = np.stack(logits_b_list, axis=0)  # [N, K, 3]

    # Free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return logits_a, logits_b


def compute_margins(logits_a: np.ndarray, logits_b: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute softmax probabilities and single-side vs paired contrast margins.
    Labels: 0: Contradiction, 1: Neutral, 2: Entailment.
    Margin: P(Entailment) - P(Contradiction).
    """
    def compute_probs(log_arr: np.ndarray) -> np.ndarray:
        e = np.exp(log_arr - np.max(log_arr, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    probs_a = compute_probs(logits_a)
    probs_b = compute_probs(logits_b)

    s_A = (probs_a[..., 2] - probs_a[..., 0]).astype(np.float64)  # [N, K]
    s_B = (probs_b[..., 2] - probs_b[..., 0]).astype(np.float64)  # [N, K]
    delta_s = s_A - s_B                                           # [N, K]
    sum_s = s_A + s_B                                             # [N, K]

    return {
        "s_A": s_A,
        "s_B": s_B,
        "paired_contrast": delta_s,
        "common_salience": sum_s,
    }


def analyze_matrix(
    matrix: np.ndarray,
    label: str,
    publishers: list[str],
    permutations: int = 10_000,
) -> dict[str, Any]:
    pca = PCA()
    proj = pca.fit_transform(matrix)
    evr = pca.explained_variance_ratio_

    gate = run_dimensionality_null_test(
        matrix=matrix,
        label=label,
        publishers=publishers,
        draws=permutations,
    )

    c1 = gate["components"][0]
    c2 = gate["components"][1] if len(gate["components"]) > 1 else None

    # Determine rank
    col_rank = gate["detected_rank_column_null"]
    pub_rank = gate.get("detected_rank_publisher_null")

    return {
        "label": label,
        "evr": [float(x) for x in evr],
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]) if len(evr) > 1 else 0.0,
        "observed_pr": gate["observed_participation_ratio"],
        "column_null_mean_pr": gate["column_null_mean_participation_ratio"],
        "detected_rank_col": col_rank,
        "detected_rank_pub": pub_rank,
        "pc1_col_p": c1["column_null"]["p_value"],
        "pc1_col_q95": c1["column_null"]["q95_variance_ratio"],
        "pc1_pub_p": c1.get("publisher_block_null", {}).get("p_value"),
        "pc1_pub_q95": c1.get("publisher_block_null", {}).get("q95_variance_ratio"),
        "pc2_col_p": c2["column_null"]["p_value"] if c2 else 1.0,
        "pc2_col_q95": c2["column_null"]["q95_variance_ratio"] if c2 else 0.0,
        "pc2_pub_p": c2.get("publisher_block_null", {}).get("p_value") if c2 else 1.0,
        "pc2_pub_q95": c2.get("publisher_block_null", {}).get("q95_variance_ratio") if c2 else 0.0,
        "projections": proj,
        "components": pca.components_,
    }


def main():
    parser = argparse.ArgumentParser(description="Audited Paired Operator (s_A - s_B) Reruns.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "paired_operator_cross_model_cross_topic_rerun.json")
    args = parser.parse_args()

    results: dict[str, Any] = {
        "metadata": {
            "operator": "Audited Paired Contrast: (P_A(E) - P_A(C)) - (P_B(E) - P_B(C))",
            "device": args.device,
            "permutations": args.permutations,
        },
        "cross_model": {},
        "cross_topic": {},
    }

    # =========================================================================
    # PART A: CROSS-MODEL AUDITED PAIRED CONTRAST RERUN
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: CROSS-MODEL AUDITED PAIRED CONTRAST (s_A - s_B) RERUN (N=60)")
    print("=" * 80)

    corpus_n60_path = PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl"
    corpus_n60 = []
    with open(corpus_n60_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus_n60.append(json.loads(line))
    pubs_n60 = [a.get("publisher") or a.get("publication") or "unknown" for a in corpus_n60]

    q0_pairs = load_yaml_pairs(PROJECT_ROOT / "config" / "framing_queries.yaml")

    models = [
        "microsoft/deberta-v2-xlarge-mnli",
        "facebook/bart-large-mnli",
        "roberta-large-mnli",
    ]

    model_matrices = {}
    model_analyses = {}

    for m in models:
        m_slug = m.split("/")[-1]
        cache_file = PROJECT_ROOT / "data" / f"cache_paired_logits_{m_slug}_N60.pt"
        if cache_file.exists():
            print(f"[CACHE] Loading dual logits for {m_slug} from {cache_file}...")
            cached = torch.load(cache_file, map_location="cpu", weights_only=False)
            la, lb = cached["la"], cached["lb"]
        else:
            la, lb = extract_paired_logits(corpus_n60, q0_pairs, m, device=args.device)
            torch.save({"la": la, "lb": lb}, cache_file)
            print(f"[CACHE] Saved dual logits for {m_slug} to {cache_file}")

        margins = compute_margins(la, lb)
        delta_s = margins["paired_contrast"]
        s_A = margins["s_A"]
        s_B = margins["s_B"]

        corr_a_delta = float(np.corrcoef(s_A.flatten(), delta_s.flatten())[0, 1])

        # Analyze paired contrast
        res_paired = analyze_matrix(delta_s, f"{m_slug}_paired_sA_minus_sB", pubs_n60, permutations=args.permutations)
        # Also analyze s_A for direct comparison
        res_single = analyze_matrix(s_A, f"{m_slug}_single_sA", pubs_n60, permutations=args.permutations)

        model_matrices[m] = {
            "delta_s": delta_s,
            "s_A": s_A,
            "res_paired": res_paired,
            "res_single": res_single,
        }

        print(f"\n[{m_slug}] AUDITED PAIRED CONTRAST (s_A - s_B):")
        print(f"  PC1 EVR: {res_paired['pc1_evr']*100:.2f}% (col null q95: {res_paired['pc1_col_q95']*100:.2f}%, p = {res_paired['pc1_col_p']:.4f})")
        print(f"  PC2 EVR: {res_paired['pc2_evr']*100:.2f}% (col null q95: {res_paired['pc2_col_q95']*100:.2f}%, p = {res_paired['pc2_col_p']:.4f})")
        print(f"  Participation Ratio: {res_paired['observed_pr']:.2f} (col null mean: {res_paired['column_null_mean_pr']:.2f})")
        print(f"  Detected Rank (Col Null): k = {res_paired['detected_rank_col']}")
        print(f"  Correlation r(s_A, s_A - s_B): {corr_a_delta:.4f}")

        print(f"[{m_slug}] SINGLE-SIDE MARGIN (s_A):")
        print(f"  PC1 EVR: {res_single['pc1_evr']*100:.2f}%, PC2 EVR: {res_single['pc2_evr']*100:.2f}%, Rank: k = {res_single['detected_rank_col']}")

        model_analyses[m] = {
            "model_name": m,
            "corr_sA_vs_paired": corr_a_delta,
            "paired_contrast": {k: v for k, v in res_paired.items() if k not in ["projections", "components"]},
            "single_side_sA": {k: v for k, v in res_single.items() if k not in ["projections", "components"]},
        }

    # Cross-Model Pairwise Alignment under Paired Contrast
    print("\n" + "-" * 70)
    print("CROSS-MODEL PAIRWISE ALIGNMENT UNDER AUDITED PAIRED CONTRAST (s_A - s_B)")
    print("-" * 70)
    pairwise_model_results = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            key = f"{m1.split('/')[-1]}__vs__{m2.split('/')[-1]}"
            p1 = model_matrices[m1]["res_paired"]["projections"][:, 0].copy()
            p2 = model_matrices[m2]["res_paired"]["projections"][:, 0].copy()
            comp1 = model_matrices[m1]["res_paired"]["components"][:1]
            comp2 = model_matrices[m2]["res_paired"]["components"][:1]

            # Sign alignment
            if np.corrcoef(p1, p2)[0, 1] < 0:
                p2 = -p2
                comp2 = -comp2

            r_val, r_p = pearsonr(p1, p2)
            rho_val, rho_p = spearmanr(p1, p2)
            _, _, disp = procrustes(
                model_matrices[m1]["res_paired"]["projections"][:, :2],
                model_matrices[m2]["res_paired"]["projections"][:, :2],
            )
            ang_rad = subspace_angles(comp1.T, comp2.T)[0]
            ang_deg = float(np.degrees(ang_rad))

            print(f"Comparison: {key}")
            print(f"  PC1 Pearson r    : {r_val:.4f} (p = {r_p:.4e})")
            print(f"  PC1 Spearman rho : {rho_val:.4f} (p = {rho_p:.4e})")
            print(f"  2-PC Procrustes  : {disp:.4f}")
            print(f"  PC1 Subspace Ang : {ang_deg:.2f}°")

            pairwise_model_results[key] = {
                "pearson_r": float(r_val),
                "pearson_p": float(r_p),
                "spearman_rho": float(rho_val),
                "spearman_p": float(rho_p),
                "procrustes_disparity_2pc": float(disp),
                "subspace_angle_pc1_deg": float(ang_deg),
            }

    results["cross_model"] = {
        "per_model": model_analyses,
        "pairwise": pairwise_model_results,
    }

    # =========================================================================
    # PART B: CROSS-TOPIC AUDITED PAIRED CONTRAST RERUN
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: CROSS-TOPIC AUDITED PAIRED CONTRAST (s_A - s_B) RERUN (DeBERTa-v2)")
    print("=" * 80)

    topics = [
        {
            "name": "Russia-Ukraine War",
            "corpus_file": "data/ukraine_N60_corpus.jsonl",
            "query_file": "config/ukraine_cross_topic_queries.yaml",
        },
        {
            "name": "US Immigration",
            "corpus_file": "data/immigration_N60_corpus.jsonl",
            "query_file": "config/immigration_framing_queries.yaml",
        },
        {
            "name": "Climate Politics",
            "corpus_file": "data/climate_N60_corpus.jsonl",
            "query_file": "config/climate_framing_queries.yaml",
        },
        {
            "name": "Abortion Rights",
            "corpus_file": "data/abortion_N60_corpus.jsonl",
            "query_file": "config/abortion_framing_queries.yaml",
        },
    ]

    cross_topic_results = {}

    for t in topics:
        t_name = t["name"]
        print(f"\nEvaluating Topic: {t_name}")
        c_path = PROJECT_ROOT / t["corpus_file"]
        q_path = PROJECT_ROOT / t["query_file"]

        corpus_t = []
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                corpus_t.append(json.loads(line))
        pubs_t = [a.get("publisher", "unknown") for a in corpus_t]
        pairs_t = load_yaml_pairs(q_path)

        t_slug = t_name.lower().replace(" ", "_").replace("-", "_")
        cache_file = PROJECT_ROOT / "data" / f"cache_paired_logits_{t_slug}_N60.pt"

        if cache_file.exists():
            print(f"[CACHE] Loading dual logits for {t_name} from {cache_file}...")
            cached = torch.load(cache_file, map_location="cpu", weights_only=False)
            la, lb = cached["la"], cached["lb"]
        else:
            la, lb = extract_paired_logits(
                corpus=corpus_t,
                pairs=pairs_t,
                model_name="microsoft/deberta-v2-xlarge-mnli",
                device=args.device,
            )
            torch.save({"la": la, "lb": lb}, cache_file)
            print(f"[CACHE] Saved dual logits for {t_name} to {cache_file}")

        margins = compute_margins(la, lb)
        delta_s = margins["paired_contrast"]
        s_A = margins["s_A"]
        s_B = margins["s_B"]

        corr_a_delta = float(np.corrcoef(s_A.flatten(), delta_s.flatten())[0, 1])

        res_paired = analyze_matrix(delta_s, f"{t_slug}_paired_sA_minus_sB", pubs_t, permutations=args.permutations)
        res_single = analyze_matrix(s_A, f"{t_slug}_single_sA", pubs_t, permutations=args.permutations)

        print(f"[{t_name}] AUDITED PAIRED CONTRAST (s_A - s_B):")
        print(f"  PC1 EVR: {res_paired['pc1_evr']*100:.2f}% (col null q95: {res_paired['pc1_col_q95']*100:.2f}%, p = {res_paired['pc1_col_p']:.4f})")
        print(f"  PC2 EVR: {res_paired['pc2_evr']*100:.2f}% (col null q95: {res_paired['pc2_col_q95']*100:.2f}%, p = {res_paired['pc2_col_p']:.4f})")
        print(f"  Participation Ratio: {res_paired['observed_pr']:.2f} (col null mean: {res_paired['column_null_mean_pr']:.2f})")
        print(f"  Detected Rank: k = {res_paired['detected_rank_col']}")
        print(f"  Correlation r(s_A, s_A - s_B): {corr_a_delta:.4f}")

        print(f"[{t_name}] SINGLE-SIDE MARGIN (s_A):")
        print(f"  PC1 EVR: {res_single['pc1_evr']*100:.2f}%, PC2 EVR: {res_single['pc2_evr']*100:.2f}%, Rank: k = {res_single['detected_rank_col']}")

        cross_topic_results[t_name] = {
            "topic_name": t_name,
            "corr_sA_vs_paired": corr_a_delta,
            "paired_contrast": {k: v for k, v in res_paired.items() if k not in ["projections", "components"]},
            "single_side_sA": {k: v for k, v in res_single.items() if k not in ["projections", "components"]},
        }

    results["cross_topic"] = cross_topic_results

    # Save complete report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Completed all paired operator reruns! Saved report to {args.output}")


if __name__ == "__main__":
    main()
