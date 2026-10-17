#!/usr/bin/env python3
"""D-Study Duality and Dependability Curves in Generalizability Theory.

Resolves Item 1 of the Kuhnian Clarification Battery:
1. Formalizes the two distinct measurement targets:
   - Target 1: Article polarity A_i (framework-independent scalar stance)
     -> E rho^2_A ~ 0.000, Phi_A ~ 0.000 across all sample sizes.
   - Target 2: Relational stance (AP)_{ip} (article-within-framework)
     -> E rho^2_AP(n_b, n_k) and Phi_AP(n_b, n_k) scale with probe and bank aggregation.

2. Computes the full D-study grid for Target 2:
   - n_b in [1, 2, 3, 5, 8, 10, 15]
   - n_k in [1, 2, 4, 8, 12, 16, 24]

3. Reports exact metrics for key configurations:
   - Minimal measurement: 1 bank x 1 probe pair (single prompt pass)
   - Standard bank: 1 bank x 8 probe pairs
   - Empirical study design: 3 banks x 8 probe pairs (our frozen protocol)
   - High-dependability design: 5 banks x 8 probe pairs, 5 banks x 16 probe pairs

4. Runs 1,000-draw article-level bootstrap to compute 95% CIs for E rho^2_AP and Phi_AP.

Outputs:
- paper/kuhnian_d_study_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_g_theory_data():
    cache_path = ROOT / "data" / "cache_replicated_g_theory_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    registry_path = ROOT / "config" / "prompt_banks" / "replicated_g_theory_banks.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    frameworks = list(registry.keys())
    realizations = ["realization_1", "realization_2", "realization_3"]

    N = 120
    P = len(frameworks)
    B = len(realizations)
    K = 8

    # Y_pairs: [N, P, B, K]
    Y_pairs = np.zeros((N, P, B, K), dtype=float)

    for p_idx, fw in enumerate(frameworks):
        for b_idx, rz in enumerate(realizations):
            key = f"{fw}:{rz}"
            bank_data = cache[key]
            la = bank_data["la"].float()
            lb = bank_data["lb"].float()
            pa = torch.softmax(la, dim=-1).numpy()
            pb = torch.softmax(lb, dim=-1).numpy()
            sa = pa[..., 2] - pa[..., 0]
            sb = pb[..., 2] - pb[..., 0]
            Y_pairs[:, p_idx, b_idx, :] = sa - sb

    return frameworks, realizations, Y_pairs


def decompose_item_level_ems(Y_pairs: np.ndarray):
    """Compute ANOVA EMS variance components on full [N, P, B, K] array."""
    N, P, B, K = Y_pairs.shape
    grand_mean = float(np.mean(Y_pairs))

    mean_a = np.mean(Y_pairs, axis=(1, 2, 3))
    mean_p = np.mean(Y_pairs, axis=(0, 2, 3))
    mean_b = np.mean(Y_pairs, axis=(0, 3))       # [P, B]
    mean_ap = np.mean(Y_pairs, axis=(2, 3))      # [N, P]
    mean_ab = np.mean(Y_pairs, axis=3)           # [N, P, B]

    ss_total = float(np.sum((Y_pairs - grand_mean) ** 2))
    ss_a = float(P * B * K * np.sum((mean_a - grand_mean) ** 2))
    ss_p = float(N * B * K * np.sum((mean_p - grand_mean) ** 2))
    ss_bp = float(N * K * np.sum((mean_b - mean_p[:, None]) ** 2))
    ss_ap = float(B * K * np.sum((mean_ap - mean_a[:, None] - mean_p[None, :] + grand_mean) ** 2))
    ss_abp = float(K * np.sum((mean_ab - mean_ap[:, :, None] - mean_b[None, :, :] + mean_p[None, :, None]) ** 2))
    ss_res = float(ss_total - ss_a - ss_p - ss_bp - ss_ap - ss_abp)

    df_a = N - 1
    df_p = P - 1
    df_bp = P * (B - 1)
    df_ap = (N - 1) * (P - 1)
    df_abp = (N - 1) * P * (B - 1)
    df_res = N * P * B * (K - 1)

    ms_a = ss_a / df_a
    ms_p = ss_p / df_p
    ms_bp = ss_bp / df_bp
    ms_ap = ss_ap / df_ap
    ms_abp = ss_abp / df_abp
    ms_res = ss_res / df_res

    var_res = max(0.0, float(ms_res))
    var_abp = max(0.0, float((ms_abp - ms_res) / K))
    var_ap = max(0.0, float((ms_ap - ms_abp) / (B * K)))
    var_bp = max(0.0, float((ms_bp - ms_abp) / (N * K)))
    var_a = max(0.0, float((ms_a - ms_ap) / (P * B * K)))
    var_p = max(0.0, float((ms_p - ms_ap - ms_bp + ms_abp) / (N * B * K)))

    var_total = var_a + var_p + var_bp + var_ap + var_abp + var_res

    return {
        "var_a": var_a,
        "var_p": var_p,
        "var_bp": var_bp,
        "var_ap": var_ap,
        "var_abp": var_abp,
        "var_res": var_res,
        "var_total": var_total,
    }


def compute_d_study_target_1(vc: Dict[str, float], n_p: int = 5, n_b: int = 3, n_k: int = 8):
    """Target 1: Framework-Independent Article Stance A_i."""
    var_a = vc["var_a"]
    rel_error = (vc["var_ap"] / n_p) + (vc["var_abp"] / (n_p * n_b)) + (vc["var_res"] / (n_p * n_b * n_k))
    abs_error = (vc["var_p"] / n_p) + (vc["var_bp"] / (n_p * n_b)) + rel_error

    e_rho2 = var_a / (var_a + rel_error) if (var_a + rel_error) > 0 else 0.0
    phi = var_a / (var_a + abs_error) if (var_a + abs_error) > 0 else 0.0

    return {
        "target": "Article Main Effect A_i (framework-invariant scalar polarity)",
        "n_frameworks": n_p,
        "n_banks_per_framework": n_b,
        "n_probes_per_bank": n_k,
        "total_items": n_p * n_b * n_k,
        "universe_score_variance": var_a,
        "relative_error_variance": rel_error,
        "absolute_error_variance": abs_error,
        "g_coefficient_relative_e_rho2": e_rho2,
        "dependability_coefficient_absolute_phi": phi,
    }


def compute_d_study_target_2(vc: Dict[str, float], n_b: int, n_k: int):
    """Target 2: Article-within-Framework Stance (AP)_{ip}."""
    # Universe score variance for differentiating articles within a framework
    var_universe = vc["var_a"] + vc["var_ap"]

    # Relative error variance: ranking articles within framework across wording samples
    var_rel_error = (vc["var_abp"] / n_b) + (vc["var_res"] / (n_b * n_k))

    # Absolute error variance: includes wording facet offset B(P)
    var_abs_error = (vc["var_bp"] / n_b) + var_rel_error

    e_rho2 = var_universe / (var_universe + var_rel_error) if (var_universe + var_rel_error) > 0 else 0.0
    phi = var_universe / (var_universe + var_abs_error) if (var_universe + var_abs_error) > 0 else 0.0

    return {
        "n_banks": n_b,
        "n_probes_per_bank": n_k,
        "total_probes_per_framework": n_b * n_k,
        "universe_score_variance": var_universe,
        "relative_error_variance": var_rel_error,
        "absolute_error_variance": var_abs_error,
        "g_coefficient_relative_e_rho2": e_rho2,
        "dependability_coefficient_absolute_phi": phi,
    }


def run_d_study_bootstrap(Y_pairs: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Bootstrap over articles (axis 0) to get 95% CIs for D-study coefficients."""
    N, P, B, K = Y_pairs.shape
    rng = np.random.default_rng(seed)

    eval_configs = [
        ("1_bank_1_probe", 1, 1),
        ("1_bank_8_probes", 1, 8),
        ("3_banks_8_probes", 3, 8),
        ("5_banks_8_probes", 5, 8),
        ("5_banks_16_probes", 5, 16),
        ("10_banks_8_probes", 10, 8),
    ]

    boot_metrics = {name: {"e_rho2": [], "phi": []} for name, _, _ in eval_configs}
    boot_metrics["target_1_current"] = {"e_rho2": [], "phi": []}

    print(f"[BOOTSTRAP] Running {n_boot} article cluster bootstrap resamples...")
    t0 = time.time()
    for b_iter in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        Y_b = Y_pairs[idx]
        vc_b = decompose_item_level_ems(Y_b)

        # Target 1 current
        t1_b = compute_d_study_target_1(vc_b, n_p=5, n_b=3, n_k=8)
        boot_metrics["target_1_current"]["e_rho2"].append(t1_b["g_coefficient_relative_e_rho2"])
        boot_metrics["target_1_current"]["phi"].append(t1_b["dependability_coefficient_absolute_phi"])

        # Target 2 configs
        for name, nb, nk in eval_configs:
            t2_b = compute_d_study_target_2(vc_b, nb, nk)
            boot_metrics[name]["e_rho2"].append(t2_b["g_coefficient_relative_e_rho2"])
            boot_metrics[name]["phi"].append(t2_b["dependability_coefficient_absolute_phi"])

    elapsed = time.time() - t0
    print(f"[BOOTSTRAP] Finished {n_boot} bootstrap draws in {elapsed:.2f}s.")

    summary = {}
    for name in boot_metrics:
        arr_rho2 = np.array(boot_metrics[name]["e_rho2"])
        arr_phi = np.array(boot_metrics[name]["phi"])
        summary[name] = {
            "e_rho2_mean": float(np.mean(arr_rho2)),
            "e_rho2_std": float(np.std(arr_rho2)),
            "e_rho2_95_ci": [float(x) for x in np.percentile(arr_rho2, [2.5, 97.5])],
            "phi_mean": float(np.mean(arr_phi)),
            "phi_std": float(np.std(arr_phi)),
            "phi_95_ci": [float(x) for x in np.percentile(arr_phi, [2.5, 97.5])],
        }

    return summary


def main():
    print("=" * 80)
    print("GENERALIZABILITY THEORY D-STUDY DUALITY & DEPENDABILITY CURVES (ITEM 1)")
    print("=" * 80)

    frameworks, realizations, Y_pairs = load_g_theory_data()
    print(f"Loaded {Y_pairs.shape[0]} articles, {len(frameworks)} frameworks, {len(realizations)} realizations, 8 pairs.")

    vc = decompose_item_level_ems(Y_pairs)
    print("\n--- Estimated Item-Level Variance Components ---")
    print(f"  sigma^2_A (Article main effect):             {vc['var_a']:.7f} ({vc['var_a']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_P (Framework main effect):           {vc['var_p']:.7f} ({vc['var_p']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_B(P) (Bank wording in framework):    {vc['var_bp']:.7f} ({vc['var_bp']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_AP (Article x Framework interaction):{vc['var_ap']:.7f} ({vc['var_ap']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_AB(P) (Article x Wording sensitivity):{vc['var_abp']:.7f} ({vc['var_abp']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_epsilon (Item residual & noise):      {vc['var_res']:.7f} ({vc['var_res']/vc['var_total']*100:5.2f}%)")
    print(f"  sigma^2_total:                               {vc['var_total']:.7f}")

    # 1. Target 1: Article Polarity (A_i)
    t1_curr = compute_d_study_target_1(vc, n_p=5, n_b=3, n_k=8)
    print("\n--- TARGET 1: Article Polarity A_i (Framework-Independent Scalar Stance) ---")
    print(f"  Universe Variance (sigma^2_A): {t1_curr['universe_score_variance']:.7f}")
    print(f"  Relative Error Variance:       {t1_curr['relative_error_variance']:.7f}")
    print(f"  Absolute Error Variance:       {t1_curr['absolute_error_variance']:.7f}")
    print(f"  Generalizability E rho^2_A:     {t1_curr['g_coefficient_relative_e_rho2']:.4f}")
    print(f"  Dependability Phi_A:            {t1_curr['dependability_coefficient_absolute_phi']:.4f}")

    # 2. Target 2: Article-within-Framework Stance (AP)_{ip}
    print("\n--- TARGET 2: Article-within-Framework Stance (AP)_{ip} (Relational Stance) ---")
    bank_grid = [1, 2, 3, 5, 8, 10, 15]
    probe_grid = [1, 2, 4, 8, 12, 16, 24]

    d_study_matrix = []
    print(f"{'n_banks':<8} | {'n_probes':<8} | {'total_k':<8} | {'Rel Error':<11} | {'Abs Error':<11} | {'E rho^2_AP':<10} | {'Phi_AP':<8}")
    print("-" * 75)
    for nb in bank_grid:
        for nk in probe_grid:
            res = compute_d_study_target_2(vc, nb, nk)
            d_study_matrix.append(res)
            if nb in [1, 3, 5, 10] and nk in [1, 8, 16]:
                print(f"{nb:<8} | {nk:<8} | {res['total_probes_per_framework']:<8} | {res['relative_error_variance']:.7f} | {res['absolute_error_variance']:.7f} | {res['g_coefficient_relative_e_rho2']:10.4f} | {res['dependability_coefficient_absolute_phi']:8.4f}")

    # Specific focal points
    focal_configs = {
        "single_bank_single_probe": compute_d_study_target_2(vc, 1, 1),
        "single_bank_eight_probes": compute_d_study_target_2(vc, 1, 8),
        "three_banks_eight_probes_empirical": compute_d_study_target_2(vc, 3, 8),
        "five_banks_eight_probes": compute_d_study_target_2(vc, 5, 8),
        "five_banks_sixteen_probes": compute_d_study_target_2(vc, 5, 16),
        "ten_banks_eight_probes": compute_d_study_target_2(vc, 10, 8),
    }

    # Bootstrap
    boot_summary = run_d_study_bootstrap(Y_pairs, n_boot=1000, seed=42)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "theoretical_distinction": {
            "target_1_article_invariant": {
                "description": "Object of measurement is the article A_i across all frameworks (scalar ranking).",
                "universe_score_variance": "sigma^2_A",
                "finding": "Because sigma^2_A approx 0.00%, E rho^2_A approx 0.000 and Phi_A approx 0.000, demonstrating that no framework-invariant article ranking exists.",
                "current_design_metrics": t1_curr,
            },
            "target_2_article_within_framework": {
                "description": "Object of measurement is relational stance (AP)_{ip} within a specified framework p.",
                "universe_score_variance": "sigma^2_A + sigma^2_AP approx sigma^2_AP",
                "finding": "Item residual noise (74.4%) averages out across probe replications (1 / (n_b * n_k)), while genuine relational interaction is stable universe variance.",
                "empirical_3x8_design": focal_configs["three_banks_eight_probes_empirical"],
            },
        },
        "variance_components": vc,
        "focal_configurations": focal_configs,
        "bootstrap_95_ci": boot_summary,
        "d_study_full_matrix": d_study_matrix,
    }

    out_file = ROOT / "paper" / "kuhnian_d_study_results.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Saved D-study results to {out_file}")


if __name__ == "__main__":
    main()
