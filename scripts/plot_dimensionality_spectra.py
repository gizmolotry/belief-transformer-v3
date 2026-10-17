from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    json_path = PROJECT_ROOT / "paper" / "systematic_dimensionality_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Output figure
    fig_dir = PROJECT_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "dimensionality_spectra.png"

    conditions = [
        ("original_israel_palestine_N320", "Original (N=320, 16 pubs)"),
        ("fresh_unseen_israel_palestine_N120", "Fresh Unseen (N=120, 93 pubs)"),
        ("ukraine_cross_topic_N60", "Ukraine Cross-Topic (N=60, 57 pubs)"),
        ("scaling_K4_N60", "Scaling K=4 (N=60)"),
        ("scaling_K8_N60", "Scaling K=8 (N=60)"),
        ("scaling_K16_N60", "Scaling K=16 (N=60)"),
        ("scaling_K32_N60", "Scaling K=32 (N=60)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(conditions):
        ax = axes[idx]
        if key not in data:
            continue
        record = data[key]
        comps = record["components"]
        k = len(comps)

        x = np.arange(1, k + 1)
        obs_r = np.array([c["observed_variance_ratio"] * 100 for c in comps])
        null_r = np.array([c["column_null"]["mean_variance_ratio"] * 100 for c in comps])
        q95_r = np.array([c["column_null"]["q95_variance_ratio"] * 100 for c in comps])
        detected = [c["column_null"]["detected"] for c in comps]

        ax.plot(x, obs_r, "o-", color="#1f77b4", label="Observed $r_i$", linewidth=2, markersize=5)
        ax.plot(x, q95_r, "--", color="#d62728", label="Null 95th %ile", linewidth=1.5)
        ax.plot(x, null_r, ":", color="#7f7f7f", label="Null Mean", linewidth=1.2)

        # Highlight detected components
        for i, det in enumerate(detected):
            if det:
                ax.plot(x[i], obs_r[i], "*", color="#2ca02c", markersize=12, zorder=5)

        det_count = sum(detected)
        pr_obs = record["observed_participation_ratio"]
        pr_null = record["column_null_mean_participation_ratio"]

        ax.set_title(f"{title}\nDet. Rank={det_count} | PR: {pr_obs:.1f} vs null {pr_null:.1f}", fontsize=11)
        ax.set_xlabel("Principal Component", fontsize=9)
        ax.set_ylabel("Variance Ratio (%)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # 8th subplot: Synthetic Positive Control (2-Factor vs 1-Factor)
    ax_synth = axes[-1]
    synth_json = PROJECT_ROOT / "paper" / "synthetic_positive_control_results.json"
    if synth_json.exists():
        with open(synth_json, "r", encoding="utf-8") as f:
            synth_data = json.load(f)
        
        c2 = synth_data["two_factor_bank"]["components"]
        c1 = synth_data["one_factor_bank"]["components"]
        x = np.arange(1, 9)
        obs_r2 = np.array([c["observed_variance_ratio"] * 100 for c in c2])
        obs_r1 = np.array([c["observed_variance_ratio"] * 100 for c in c1])
        q95_r2 = np.array([c["column_null"]["q95_variance_ratio"] * 100 for c in c2])

        ax_synth.plot(x, obs_r2, "s-", color="#2ca02c", label="2-Factor Bank (Det k=2)", linewidth=2, markersize=5)
        ax_synth.plot(x, obs_r1, "^--", color="#ff7f0e", label="1-Factor Bank (Det k=1)", linewidth=2, markersize=5)
        ax_synth.plot(x, q95_r2, ":", color="#d62728", label="Null 95th %ile", linewidth=1.5)

        ax_synth.plot(1, obs_r2[0], "*", color="#2ca02c", markersize=12)
        ax_synth.plot(2, obs_r2[1], "*", color="#2ca02c", markersize=12)
        ax_synth.plot(1, obs_r1[0], "*", color="#ff7f0e", markersize=12)

        ax_synth.set_title("Synthetic Positive Control (N=60)\nApparatus Recovers k=2 vs k=1", fontsize=11)
        ax_synth.set_xlabel("Principal Component", fontsize=9)
        ax_synth.set_ylabel("Variance Ratio (%)", fontsize=9)
        ax_synth.grid(True, alpha=0.3)
        ax_synth.legend(fontsize=8, loc="upper right")
    else:
        ax_synth.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[DONE] Saved dimensionality spectra figure to {fig_path}")


if __name__ == "__main__":
    main()
