from __future__ import annotations

import argparse
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


def compute_covariance_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Compute sorted eigenvalues of unstandardized sample covariance matrix."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / (matrix.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues


def independent_column_permutation(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each column independently across rows."""
    permuted = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        permuted[:, j] = rng.permutation(matrix[:, j])
    return permuted


def publisher_block_permutation_balanced(
    matrix: np.ndarray, publishers: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Permute balanced publisher blocks per column, matching v2 protocol."""
    labels = np.asarray(publishers)
    groups = np.asarray(sorted(set(labels.tolist())))
    indices = {group: np.flatnonzero(labels == group) for group in groups}
    permuted = np.empty_like(matrix)
    for col in range(matrix.shape[1]):
        source_groups = rng.permutation(groups)
        for target_group, source_group in zip(groups, source_groups):
            source_values = matrix[indices[source_group], col]
            permuted[indices[target_group], col] = rng.permutation(source_values)
    return permuted


def run_dimensionality_null_test(
    matrix: np.ndarray,
    label: str,
    publishers: Sequence[str] | None = None,
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = matrix.shape
    obs_eigs = compute_covariance_eigenvalues(matrix)
    obs_ratios = obs_eigs / obs_eigs.sum()
    obs_pr = float((obs_eigs.sum() ** 2) / (obs_eigs ** 2).sum())

    # 1. Independent Column Permutation Null
    null_eigs_col = np.empty((draws, n_cols), dtype=np.float64)
    null_ratios_col = np.empty((draws, n_cols), dtype=np.float64)
    null_pr_col = np.empty(draws, dtype=np.float64)

    for d in range(draws):
        perm_col = independent_column_permutation(matrix, rng)
        e = compute_covariance_eigenvalues(perm_col)
        null_eigs_col[d] = e
        null_ratios_col[d] = e / e.sum()
        null_pr_col[d] = (e.sum() ** 2) / (e ** 2).sum()

    # 2. Publisher-Block Null (if balanced publishers available)
    has_balanced_pubs = False
    null_eigs_pub = None
    null_ratios_pub = None
    null_pr_pub = None
    if publishers is not None:
        pub_counts = list(np.unique(publishers, return_counts=True)[1])
        if len(set(pub_counts)) == 1 and pub_counts[0] > 1:
            has_balanced_pubs = True
            null_eigs_pub = np.empty((draws, n_cols), dtype=np.float64)
            null_ratios_pub = np.empty((draws, n_cols), dtype=np.float64)
            null_pr_pub = np.empty(draws, dtype=np.float64)
            for d in range(draws):
                perm_pub = publisher_block_permutation_balanced(matrix, publishers, rng)
                e = compute_covariance_eigenvalues(perm_pub)
                null_eigs_pub[d] = e
                null_ratios_pub[d] = e / e.sum()
                null_pr_pub[d] = (e.sum() ** 2) / (e ** 2).sum()

    # Summarize components
    components = []
    for k in range(n_cols):
        obs_val = float(obs_eigs[k])
        obs_rat = float(obs_ratios[k])

        # Column null metrics
        vals_e_col = null_eigs_col[:, k]
        vals_r_col = null_ratios_col[:, k]
        q95_col = float(np.quantile(vals_e_col, 0.95, method="higher"))
        q95_r_col = float(np.quantile(vals_r_col, 0.95, method="higher"))
        exceed_col = int(np.count_nonzero(vals_e_col >= obs_val))
        p_col = float((exceed_col + 1) / (draws + 1))
        detected_col = bool(obs_val > q95_col and p_col <= 0.05)

        comp_dict = {
            "component": k + 1,
            "observed_eigenvalue": obs_val,
            "observed_variance_ratio": obs_rat,
            "column_null": {
                "mean_eigenvalue": float(vals_e_col.mean()),
                "q95_eigenvalue": q95_col,
                "mean_variance_ratio": float(vals_r_col.mean()),
                "q95_variance_ratio": q95_r_col,
                "exceedances": exceed_col,
                "p_value": p_col,
                "detected": detected_col,
            },
        }

        if has_balanced_pubs and null_eigs_pub is not None and null_ratios_pub is not None:
            vals_e_pub = null_eigs_pub[:, k]
            vals_r_pub = null_ratios_pub[:, k]
            q95_pub = float(np.quantile(vals_e_pub, 0.95, method="higher"))
            q95_r_pub = float(np.quantile(vals_r_pub, 0.95, method="higher"))
            exceed_pub = int(np.count_nonzero(vals_e_pub >= obs_val))
            p_pub = float((exceed_pub + 1) / (draws + 1))
            detected_pub = bool(obs_val > q95_pub and p_pub <= 0.05)
            comp_dict["publisher_block_null"] = {
                "mean_eigenvalue": float(vals_e_pub.mean()),
                "q95_eigenvalue": q95_pub,
                "mean_variance_ratio": float(vals_r_pub.mean()),
                "q95_variance_ratio": q95_r_pub,
                "exceedances": exceed_pub,
                "p_value": p_pub,
                "detected": detected_pub,
            }

        components.append(comp_dict)

    # Detected rank
    detected_rank_col = sum(1 for c in components if c["column_null"]["detected"])
    detected_rank_pub = (
        sum(1 for c in components if c.get("publisher_block_null", {}).get("detected", False))
        if has_balanced_pubs
        else None
    )

    return {
        "dataset_label": label,
        "n_samples": n_rows,
        "n_features": n_cols,
        "observed_participation_ratio": obs_pr,
        "column_null_mean_participation_ratio": float(null_pr_col.mean()),
        "detected_rank_column_null": detected_rank_col,
        "detected_rank_publisher_null": detected_rank_pub,
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(description="Systematic Dimensionality Gate Testing.")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "systematic_dimensionality_results.json")
    args = parser.parse_args()

    results: dict[str, Any] = {}

    # =========================================================================
    # Condition 1: Original Israel/Palestine (N=320, 16x20 Balanced Blocks)
    # =========================================================================
    print("\n[EVAL 1] Original Israel/Palestine Corpus (N=320, 16x20 blocks)...")
    cp_path = PROJECT_ROOT / "outputs" / "fresh_scalar_structure_run" / "extraction_checkpoint.jsonl"
    pubs_320 = []
    with open(PROJECT_ROOT / "data" / "controlled_N_dense_corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            pubs_320.append(json.loads(line)["publisher"])

    rows_320 = []
    with open(cp_path, "r", encoding="utf-8") as f:
        for line in f:
            rows_320.append(json.loads(line))

    q0_logits = np.array([r["q0"]["A"] for r in rows_320])
    p = np.exp(q0_logits) / np.exp(q0_logits).sum(axis=-1, keepdims=True)
    q0_polarity = p[..., 2] - p[..., 0]  # [320, 8]

    res_orig = run_dimensionality_null_test(
        q0_polarity,
        "original_israel_palestine_N320",
        publishers=pubs_320,
        draws=args.draws,
    )
    results["original_israel_palestine_N320"] = res_orig

    # =========================================================================
    # Condition 2: Fresh Out-of-Sample Israel/Palestine (N=120, 93 Publishers)
    # =========================================================================
    print("[EVAL 2] Fresh Out-of-Sample Israel/Palestine Corpus (N=120, 93 pubs)...")
    d120 = torch.load(PROJECT_ROOT / "data" / "fresh_unseen_120_embeddings.pt", map_location="cpu", weights_only=False)
    p120_raw = torch.softmax(d120["ideo_logits"].float(), dim=-1).numpy()
    p120_scores = p120_raw[..., 2] - p120_raw[..., 0]  # [120, 8]

    pubs_120 = []
    with open(PROJECT_ROOT / "data" / "fresh_unseen_120_corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pubs_120.append(d.get("publisher") or d.get("publication") or "unknown")

    res_fresh = run_dimensionality_null_test(
        p120_scores,
        "fresh_unseen_israel_palestine_N120",
        publishers=pubs_120,
        draws=args.draws,
    )
    results["fresh_unseen_israel_palestine_N120"] = res_fresh

    # =========================================================================
    # Condition 3: Ukraine War Cross-Topic (N=60, 57 Publishers)
    # =========================================================================
    print("[EVAL 3] Ukraine War Cross-Topic Corpus (N=60, 57 pubs)...")
    dukr = torch.load(PROJECT_ROOT / "data" / "ukraine_N60_embeddings.pt", map_location="cpu", weights_only=False)
    pukr_raw = torch.softmax(dukr["ukr_logits"].float(), dim=-1).numpy()
    pukr_scores = pukr_raw[..., 2] - pukr_raw[..., 0]  # [60, 8]

    pubs_ukr = []
    with open(PROJECT_ROOT / "data" / "ukraine_N60_corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pubs_ukr.append(d.get("publisher") or d.get("publication") or "unknown")

    res_ukr = run_dimensionality_null_test(
        pukr_scores,
        "ukraine_cross_topic_N60",
        publishers=pubs_ukr,
        draws=args.draws,
    )
    results["ukraine_cross_topic_N60"] = res_ukr

    # =========================================================================
    # Condition 4: Hypothesis-Count Scaling (N=60, K in {4, 8, 16, 32})
    # =========================================================================
    print("[EVAL 4] Hypothesis Scaling across K in {4, 8, 16, 32}...")
    dscale = torch.load(PROJECT_ROOT / "data" / "hypothesis_scaling_32_embeddings.pt", map_location="cpu", weights_only=False)
    emb_32 = dscale["emb_32"]  # [60, 32, 1536]

    for k in (4, 8, 16, 32):
        print(f"   -> Testing scaling K={k}...")
        sub_emb = emb_32[:, :k, :]
        flat = sub_emb.reshape(-1, sub_emb.shape[-1])
        centered = flat - flat.mean(dim=0, keepdim=True)
        _, S, V = torch.svd(centered)
        pc1 = V[:, 0]
        proj = torch.matmul(flat, pc1).unsqueeze(1) * pc1.unsqueeze(0)
        residuals = (flat - proj).reshape(sub_emb.shape)

        normed = torch.nn.functional.normalize(residuals, p=2, dim=-1)
        k_matrix = normed.mean(dim=-1).numpy()  # [60, k]

        res_k = run_dimensionality_null_test(
            k_matrix,
            f"scaling_K{k}_N60",
            publishers=None,
            draws=args.draws,
        )
        results[f"scaling_K{k}_N60"] = res_k

    # =========================================================================
    # Summary Table Output
    # =========================================================================
    print("\n" + "=" * 88)
    print("SYSTEMATIC DIMENSIONALITY GATE SUMMARY (OBSERVED SPECTRUM VS NULL ENVELOPE)")
    print("=" * 88)
    print(f"{'Condition':<32} | {'K':<3} | {'Obs PR':<7} | {'Null PR':<7} | {'PC1 (Obs / Null)':<18} | {'PC2 (Obs / Null)':<18} | {'Det. Rank'}")
    print("-" * 88)

    for key, res in results.items():
        k = res["n_features"]
        obs_pr = res["observed_participation_ratio"]
        null_pr = res["column_null_mean_participation_ratio"]
        det_rank = res["detected_rank_column_null"]

        c1 = res["components"][0]
        c1_str = f"{c1['observed_variance_ratio']*100:.1f}% / {c1['column_null']['q95_variance_ratio']*100:.1f}%"
        if k > 1:
            c2 = res["components"][1]
            c2_str = f"{c2['observed_variance_ratio']*100:.1f}% / {c2['column_null']['q95_variance_ratio']*100:.1f}%"
        else:
            c2_str = "N/A"

        print(f"{key:<32} | {k:<3} | {obs_pr:<7.2f} | {null_pr:<7.2f} | {c1_str:<18} | {c2_str:<18} | {det_rank}")

    print("=" * 88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved systematic dimensionality results to {args.output}")


if __name__ == "__main__":
    main()
