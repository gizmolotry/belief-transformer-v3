#!/usr/bin/env python3
"""Cluster Bootstrap Uncertainty Estimation for Operator vs Framework Interaction.

Resolves Item 3 of the Kuhnian Clarification Battery:
Estimates the empirical sampling distribution and 95% confidence interval for
Delta_{O-H} = sigma^2_{AO} - sigma^2_{AH} (Article x Operator vs Article x Framework).

Design:
- Balanced 120 articles x 4 frameworks x 3 operators = 1,440 cells.
- 1,000 cluster bootstrap resamples (resampling articles i in 1..120 with replacement).
- Computes both:
  1. Proportion of Total Sum of Squares (SS)
  2. EMS Variance Components (accounting for df and factor level differences: 4 FW vs 3 Ops)
- Tests whether Delta_{O-H} is statistically distinguishable from zero.

Outputs:
- paper/kuhnian_operator_uncertainty_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).resolve().parents[1]


def build_balanced_factorial_tensor() -> Tuple[np.ndarray, List[str], List[str]]:
    cache_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    fa_cache = np.load(ROOT / "data" / "cache_audit_deberta_embeddings.npz")
    docs_emb = fa_cache["prospective"]
    d_u = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)

    framework_keys = [
        "q0_original_geopolitical",
        "q1_paraphrase_control",
        "q2_humanitarian_rights",
        "q3_realpolitik_statecraft",
    ]
    operator_names = ["paired_nli", "single_nli", "frameaxis"]

    N = 120
    H_count = 4
    O_count = 3

    paired_nli = {}
    single_nli = {}
    frameaxis = {}

    q0_ha = fa_cache["hyp_a"] / (np.linalg.norm(fa_cache["hyp_a"], axis=1, keepdims=True) + 1e-12)
    q0_hb = fa_cache["hyp_b"] / (np.linalg.norm(fa_cache["hyp_b"], axis=1, keepdims=True) + 1e-12)

    for h_key in framework_keys:
        b_val = cache[h_key]
        la = b_val["la"].float()
        lb = b_val["lb"].float()
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        sa = pa[..., 2] - pa[..., 0]
        sb = pb[..., 2] - pb[..., 0]
        paired_nli[h_key] = (sa - sb).mean(axis=1)  # [120]
        single_nli[h_key] = sa.mean(axis=1)         # [120]

    # FrameAxis
    frameaxis["q0_original_geopolitical"] = (d_u @ q0_ha.T - d_u @ q0_hb.T).mean(axis=1)

    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus_path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    texts = []
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    embs_docs_minilm = minilm.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    for h_key in framework_keys:
        if h_key == "q0_original_geopolitical":
            continue
        if h_key == "q1_paraphrase_control":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json").read_text(encoding="utf-8-sig"))
        elif h_key == "q2_humanitarian_rights":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_q2_humanitarian.json").read_text(encoding="utf-8-sig"))
        elif h_key == "q3_realpolitik_statecraft":
            raw = json.loads((ROOT / "config" / "prompt_banks" / "probe_hypotheses_q3_realpolitik.json").read_text(encoding="utf-8-sig"))
        ha, hb = raw[:8], raw[8:16]
        e_ha = minilm.encode(ha, normalize_embeddings=True, show_progress_bar=False)
        e_hb = minilm.encode(hb, normalize_embeddings=True, show_progress_bar=False)
        frameaxis[h_key] = (embs_docs_minilm @ e_ha.T - embs_docs_minilm @ e_hb.T).mean(axis=1)

    # Tensor [N, H, O]
    Y_tensor = np.zeros((N, H_count, O_count), dtype=float)
    for j, h_key in enumerate(framework_keys):
        Y_tensor[:, j, 0] = paired_nli[h_key]
        Y_tensor[:, j, 1] = single_nli[h_key]
        Y_tensor[:, j, 2] = frameaxis[h_key]

    # Standardize each operator slice to standard deviation 1.0 to eliminate scale artifacts
    for k in range(O_count):
        std_k = np.std(Y_tensor[:, :, k])
        if std_k > 1e-8:
            Y_tensor[:, :, k] /= std_k

    return Y_tensor, framework_keys, operator_names


def compute_factorial_decomposition(Y_sub: np.ndarray) -> Dict[str, float]:
    """Compute SS and EMS variance decomposition on [N, H, O] tensor."""
    N, H_count, O_count = Y_sub.shape
    mu = float(np.mean(Y_sub))

    mu_A = np.mean(Y_sub, axis=(1, 2))  # [N]
    mu_H = np.mean(Y_sub, axis=(0, 2))  # [H]
    mu_O = np.mean(Y_sub, axis=(0, 1))  # [O]

    mu_AH = np.mean(Y_sub, axis=2)      # [N, H]
    mu_AO = np.mean(Y_sub, axis=1)      # [N, O]
    mu_HO = np.mean(Y_sub, axis=0)      # [H, O]

    SS_total = float(np.sum((Y_sub - mu) ** 2))
    SS_A = float(H_count * O_count * np.sum((mu_A - mu) ** 2))
    SS_H = float(N * O_count * np.sum((mu_H - mu) ** 2))
    SS_O = float(N * H_count * np.sum((mu_O - mu) ** 2))

    SS_AH = float(O_count * np.sum((mu_AH - mu_A[:, None] - mu_H[None, :] + mu) ** 2))
    SS_AO = float(H_count * np.sum((mu_AO - mu_A[:, None] - mu_O[None, :] + mu) ** 2))
    SS_HO = float(N * np.sum((mu_HO - mu_H[:, None] - mu_O[None, :] + mu) ** 2))
    SS_resid = float(SS_total - (SS_A + SS_H + SS_O + SS_AH + SS_AO + SS_HO))

    # Proportions of SS
    prop_A = SS_A / SS_total
    prop_H = SS_H / SS_total
    prop_O = SS_O / SS_total
    prop_HO = SS_HO / SS_total
    prop_AH = SS_AH / SS_total
    prop_AO = SS_AO / SS_total
    prop_resid = SS_resid / SS_total
    delta_ss_o_minus_h = prop_AO - prop_AH

    # Degrees of freedom
    df_A = N - 1
    df_H = H_count - 1
    df_O = O_count - 1
    df_AH = (N - 1) * (H_count - 1)
    df_AO = (N - 1) * (O_count - 1)
    df_HO = (H_count - 1) * (O_count - 1)
    df_resid = (N - 1) * (H_count - 1) * (O_count - 1)

    # Mean squares
    ms_AH = SS_AH / df_AH
    ms_AO = SS_AO / df_AO
    ms_resid = SS_resid / df_resid

    # EMS estimates
    var_e = max(0.0, ms_resid)
    var_AH = max(0.0, (ms_AH - ms_resid) / O_count)
    var_AO = max(0.0, (ms_AO - ms_resid) / H_count)
    var_total_interactions = var_AH + var_AO + var_e
    pct_ems_AH = (var_AH / var_total_interactions) * 100 if var_total_interactions > 0 else 0.0
    pct_ems_AO = (var_AO / var_total_interactions) * 100 if var_total_interactions > 0 else 0.0
    delta_ems_o_minus_h = pct_ems_AO - pct_ems_AH

    return {
        "prop_AH_ss": prop_AH * 100,
        "prop_AO_ss": prop_AO * 100,
        "delta_ss_o_minus_h": delta_ss_o_minus_h * 100,
        "pct_ems_AH": pct_ems_AH,
        "pct_ems_AO": pct_ems_AO,
        "delta_ems_o_minus_h": delta_ems_o_minus_h,
        "prop_A_ss": prop_A * 100,
        "prop_H_ss": prop_H * 100,
        "prop_O_ss": prop_O * 100,
        "prop_HO_ss": prop_HO * 100,
        "prop_resid_ss": prop_resid * 100,
    }


def main():
    print("=" * 80)
    print("OPERATOR VS FRAMEWORK INTERACTION UNCERTAINTY AUDIT (ITEM 3)")
    print("=" * 80)

    Y_tensor, fw_keys, op_names = build_balanced_factorial_tensor()
    N = Y_tensor.shape[0]
    print(f"Loaded tensor shape: {Y_tensor.shape} (N={N} articles, H=4 frameworks, O=3 operators)")

    point_est = compute_factorial_decomposition(Y_tensor)
    print("\n--- Point Estimates on Full Sample (N=120) ---")
    print(f"  sigma^2_AH (Article x Framework SS): {point_est['prop_AH_ss']:.2f}%")
    print(f"  sigma^2_AO (Article x Operator SS):  {point_est['prop_AO_ss']:.2f}%")
    print(f"  Delta_{{O-H}} (AO - AH SS):          {point_est['delta_ss_o_minus_h']:+.2f}%")
    print(f"  EMS pct_AO:                          {point_est['pct_ems_AO']:.2f}%")
    print(f"  EMS pct_AH:                          {point_est['pct_ems_AH']:.2f}%")
    print(f"  Delta_{{O-H}} (EMS):                 {point_est['delta_ems_o_minus_h']:+.2f}%")

    # 1,000-draw article cluster bootstrap
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_deltas_ss = []
    boot_deltas_ems = []
    boot_ah_ss = []
    boot_ao_ss = []

    print(f"\n[BOOTSTRAP] Running {n_boot} article cluster bootstrap resamples...")
    t0 = time.time()
    for _ in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        Y_b = Y_tensor[idx]
        b_res = compute_factorial_decomposition(Y_b)
        boot_deltas_ss.append(b_res["delta_ss_o_minus_h"])
        boot_deltas_ems.append(b_res["delta_ems_o_minus_h"])
        boot_ah_ss.append(b_res["prop_AH_ss"])
        boot_ao_ss.append(b_res["prop_AO_ss"])

    elapsed = time.time() - t0
    print(f"[BOOTSTRAP] Completed in {elapsed:.2f}s.")

    arr_deltas_ss = np.array(boot_deltas_ss)
    arr_deltas_ems = np.array(boot_deltas_ems)
    arr_ah_ss = np.array(boot_ah_ss)
    arr_ao_ss = np.array(boot_ao_ss)

    ci_ss = np.percentile(arr_deltas_ss, [2.5, 97.5])
    ci_ems = np.percentile(arr_deltas_ems, [2.5, 97.5])
    ci_ah = np.percentile(arr_ah_ss, [2.5, 97.5])
    ci_ao = np.percentile(arr_ao_ss, [2.5, 97.5])

    p_val_ss = float(np.mean(arr_deltas_ss <= 0.0))  # Probability that AO <= AH
    p_val_ems = float(np.mean(arr_deltas_ems <= 0.0))

    print("\n--- 1,000-Draw Cluster Bootstrap Results ---")
    print(f"  Delta_SS (AO - AH):  Mean = {np.mean(arr_deltas_ss):+.2f}%, 95% CI: [{ci_ss[0]:+.2f}%, {ci_ss[1]:+.2f}%], p(Delta <= 0) = {p_val_ss:.4f}")
    print(f"  Delta_EMS (AO - AH): Mean = {np.mean(arr_deltas_ems):+.2f}%, 95% CI: [{ci_ems[0]:+.2f}%, {ci_ems[1]:+.2f}%], p(Delta <= 0) = {p_val_ems:.4f}")
    print(f"  Article x FW (AH):   Mean = {np.mean(arr_ah_ss):.2f}%, 95% CI: [{ci_ah[0]:.2f}%, {ci_ah[1]:.2f}%]")
    print(f"  Article x Op (AO):   Mean = {np.mean(arr_ao_ss):.2f}%, 95% CI: [{ci_ao[0]:.2f}%, {ci_ao[1]:.2f}%]")

    # Scientific calibration
    spans_zero = bool(ci_ss[0] <= 0.0 <= ci_ss[1])
    interpretation = (
        f"Point estimates show Article x Operator interaction explains {point_est['prop_AO_ss']:.2f}% of SS "
        f"vs {point_est['prop_AH_ss']:.2f}% for Article x Framework interaction (Delta = {point_est['delta_ss_o_minus_h']:+.2f}%). "
        f"Under 1,000 cluster bootstrap resamples over articles, the 95% confidence interval for Delta_{{O-H}} is "
        f"[{ci_ss[0]:+.2f}%, {ci_ss[1]:+.2f}%] (p(Delta <= 0) = {p_val_ss:.4f}). "
        f"{'Because the 95% CI spans zero, operator sensitivity is numerically larger in this design but not statistically superior to framework sensitivity; the two interact with articles at comparable magnitudes, and factor level asymmetry (4 frameworks vs 3 operators) must be noted.' if spans_zero else 'Operator interaction is statistically larger than framework interaction in this design.'}"
    )

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": {
            "n_articles": N,
            "n_frameworks": 4,
            "n_operators": 3,
            "total_cells": 1440,
            "frameworks": fw_keys,
            "operators": op_names,
        },
        "point_estimates": point_est,
        "cluster_bootstrap_1000": {
            "delta_ss_o_minus_h": {
                "mean": float(np.mean(arr_deltas_ss)),
                "std": float(np.std(arr_deltas_ss)),
                "ci_95": [float(ci_ss[0]), float(ci_ss[1])],
                "p_value_one_sided_leq_zero": p_val_ss,
                "spans_zero": spans_zero,
            },
            "delta_ems_o_minus_h": {
                "mean": float(np.mean(arr_deltas_ems)),
                "std": float(np.std(arr_deltas_ems)),
                "ci_95": [float(ci_ems[0]), float(ci_ems[1])],
                "p_value_one_sided_leq_zero": p_val_ems,
            },
            "prop_AH_ss": {
                "mean": float(np.mean(arr_ah_ss)),
                "ci_95": [float(ci_ah[0]), float(ci_ah[1])],
            },
            "prop_AO_ss": {
                "mean": float(np.mean(arr_ao_ss)),
                "ci_95": [float(ci_ao[0]), float(ci_ao[1])],
            },
        },
        "epistemic_calibration": interpretation,
    }

    out_file = ROOT / "paper" / "kuhnian_operator_uncertainty_results.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Saved operator uncertainty results to {out_file}")
    print(f"\n[CALIBRATION] {interpretation}")


if __name__ == "__main__":
    main()
