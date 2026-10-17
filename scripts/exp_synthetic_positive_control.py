from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig
from scripts.exp_systematic_dimensionality_gate import run_dimensionality_null_test


# =========================================================================
# Hypothesis Banks
# =========================================================================

# Bank A: Two-Factor Bank (4 Threat pairs + 4 Humanitarian pairs)
TWO_FACTOR_PAIRS = [
    # Factor 1: Security Threat / Military Urgency (Pairs 1-4)
    (
        "This article portrays the security situation as an imminent military threat requiring defense mobilization.",
        "This article portrays the security situation as peaceful, stable, and free from military threat.",
    ),
    (
        "This article emphasizes dangerous armed escalation and acute risk of military conflict.",
        "This article emphasizes regional diplomatic calm, de-escalation, and international peace.",
    ),
    (
        "This article highlights foreign aggression, enemy troop readiness, and strategic danger.",
        "This article highlights peaceful cooperation, border tranquility, and friendly foreign relations.",
    ),
    (
        "This article reports on critical military alerts, defense emergencies, and hostile attacks.",
        "This article reports on routine peaceful operations with no military alerts or hostilities.",
    ),
    # Factor 2: Humanitarian / Civilian Harm (Pairs 5-8)
    (
        "This article documents severe civilian suffering, acute starvation, and humanitarian catastrophe.",
        "This article documents civilian prosperity, abundant food supplies, and flourishing welfare.",
    ),
    (
        "This article portrays a critical public health crisis with overwhelmed hospitals and medicine shortages.",
        "This article portrays modern, well-equipped hospitals and excellent public healthcare access.",
    ),
    (
        "This article emphasizes widespread civilian homelessness, emergency camps, and mass displacement.",
        "This article emphasizes secure permanent housing, community rebuilding, and high living standards.",
    ),
    (
        "This article describes an unfolding human disaster with desperate civilians pleading for relief.",
        "This article describes content civilians enjoying thriving local markets and economic security.",
    ),
]

# Bank B: One-Factor Bank (8 Threat pairs, 0 Humanitarian pairs)
ONE_FACTOR_PAIRS = [
    (
        "This article portrays the security situation as an imminent military threat requiring defense mobilization.",
        "This article portrays the security situation as peaceful, stable, and free from military threat.",
    ),
    (
        "This article emphasizes dangerous armed escalation and acute risk of military conflict.",
        "This article emphasizes regional diplomatic calm, de-escalation, and international peace.",
    ),
    (
        "This article highlights foreign aggression, enemy troop readiness, and strategic danger.",
        "This article highlights peaceful cooperation, border tranquility, and friendly foreign relations.",
    ),
    (
        "This article reports on critical military alerts, defense emergencies, and hostile attacks.",
        "This article reports on routine peaceful operations with no military alerts or hostilities.",
    ),
    (
        "This article frames national borders as actively threatened by hostile foreign armed forces.",
        "This article frames national borders as completely secure, peaceful, and uncontested.",
    ),
    (
        "This article depicts enemy forces preparing sudden offensive strikes against strategic targets.",
        "This article depicts neighboring forces standing down and strictly adhering to ceasefire pacts.",
    ),
    (
        "This article focuses on expanding defense spending and urgent ballistic missile readiness.",
        "This article focuses on military budget reductions and comprehensive disarmament treaties.",
    ),
    (
        "This article warns of an unavoidable outbreak of regional war in the coming days.",
        "This article reassures that diplomatic peace accords have eliminated the danger of war.",
    ),
]


# =========================================================================
# Synthetic Factorial Corpus Generator (4 Quadrants, 15 articles each = 60)
# =========================================================================

def generate_synthetic_factorial_corpus() -> list[dict[str, Any]]:
    articles = []

    # Quadrant 1: High Threat (+1), High Harm (+1)
    for i in range(15):
        articles.append({
            "id": f"q1_high_threat_high_harm_{i:02d}",
            "quadrant": "high_threat_high_harm",
            "threat_factor": +1,
            "harm_factor": +1,
            "publisher": f"synthetic_wire_{i%5}",
            "title": f"Hostile Forces Advance as Famine and Medical Collapse Stalk Besieged Province {i+1}",
            "content": (
                f"Frontline military garrisons were placed on maximum combat alert yesterday after hostile foreign armored columns "
                f"launched heavy artillery bombardments across the northern frontier sector {i+1}. Defense commanders warned of imminent "
                f"full-scale offensive maneuvers aimed at severing primary logistics corridors. Simultaneously, the United Nations issued "
                f"a catastrophic humanitarian warning as besieged municipal sectors ran out of potable water, surgical anesthetics, and emergency "
                f"food rations. Over three hundred thousand displaced civilians are sheltering in makeshift canvas tents amid freezing winter "
                f"temperatures and rampant cholera outbreaks, with local clinics reporting dozens of infant starvation deaths daily while enemy "
                f"missile strikes continue to batter critical civil infrastructure."
            ),
        })

    # Quadrant 2: High Threat (+1), Low Harm (-1)
    for i in range(15):
        articles.append({
            "id": f"q2_high_threat_low_harm_{i:02d}",
            "quadrant": "high_threat_low_harm",
            "threat_factor": +1,
            "harm_factor": -1,
            "publisher": f"synthetic_wire_{i%5}",
            "title": f"High-Tech Air Defense Standoff Escalates Along Frontier While Civilian Economy Thrives {i+1}",
            "content": (
                f"Strategic defense headquarters deployed cutting-edge hypersonic interceptor batteries and automated early-warning radars "
                f"along the sovereign maritime perimeter {i+1} following provocative supersonic sorties by adversarial stealth bombers. Military "
                f"intelligence warned that regional armed hostilities could ignite instantaneously if foreign warships violate exclusion zones. "
                f"However, the domestic civilian sector remains entirely insulated, prosperous, and secure. Major metropolitan centers reported "
                f"record employment gains, full retail store inventories, and pristine municipal services, with public health indices reaching "
                f"historic highs. Citizens expressed complete confidence in sovereign automated defense shields while enjoying newly opened cultural "
                f"districts and abundant consumer markets completely unmarred by deprivation or physical distress."
            ),
        })

    # Quadrant 3: Low Threat (-1), High Harm (+1)
    for i in range(15):
        articles.append({
            "id": f"q3_low_threat_high_harm_{i:02d}",
            "quadrant": "low_threat_high_harm",
            "threat_factor": -1,
            "harm_factor": +1,
            "publisher": f"synthetic_wire_{i%5}",
            "title": f"Historic Peace Prevails Along Border as Devastating Drought and Famine Ravage Rural Population {i+1}",
            "content": (
                f"Diplomatic delegates signed a permanent bilateral non-aggression and mutual friendship accord yesterday, cementing absolute "
                f"demilitarization and lasting peace across all international boundary sectors {i+1}. Armed forces from both nations have fully "
                f"withdrawn to barracks, with zero military friction, defense emergencies, or security alerts reported across the territory. "
                f"Yet a catastrophic humanitarian emergency has engulfed the central agrarian provinces following a historic multi-year drought. "
                f"Entire agricultural yields have withered to dust, leaving over two million rural villagers facing acute starvation and extreme "
                f"malnutrition. Emergency medical field tents are severely overcrowded with dying children suffering from dysentery and water-borne "
                f"pathogens, while international relief convoys struggle to deliver vital therapeutic nutrition to dying populations."
            ),
        })

    # Quadrant 4: Low Threat (-1), Low Harm (-1)
    for i in range(15):
        articles.append({
            "id": f"q4_low_threat_low_harm_{i:02d}",
            "quadrant": "low_threat_low_harm",
            "threat_factor": -1,
            "harm_factor": -1,
            "publisher": f"synthetic_wire_{i%5}",
            "title": f"Bilateral Disarmament Accords Expand as National Standard of Living Reaches All-Time High {i+1}",
            "content": (
                f"Regional border authorities announced the complete dismantling of legacy surveillance outposts and defense checkpoints "
                f"following the formal ratification of a comprehensive perpetual peace treaty {i+1}. Security analysts confirmed that the entire "
                f"geographic zone is exceptionally tranquil, stable, and completely free from any external military threat or armed tension. "
                f"Concurrently, the civilian population is experiencing unprecedented economic growth, social wellbeing, and physical prosperity. "
                f"Newly inaugurated regional healthcare facilities provide universal state-of-the-art care, modern educational academies have achieved "
                f"record enrollment, and agricultural harvests have generated record food surpluses. Families across the nation celebrated the flourishing "
                f"civic stability, affordable modern housing, and joyful public festivals marking a new era of lasting prosperity."
            ),
        })

    return articles


def main():
    parser = argparse.ArgumentParser(description="Synthetic positive control for multidimensionality.")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "synthetic_positive_control_results.json")
    args = parser.parse_args()

    articles = generate_synthetic_factorial_corpus()
    print(f"[SETUP] Generated synthetic 2x2 factorial corpus of {len(articles)} articles (15 per cell).")

    cache_path = PROJECT_ROOT / "data" / "synthetic_positive_control_embeddings.pt"

    if cache_path.exists():
        print(f"[CACHE] Loading cached embeddings from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        two_factor_logits = cached["two_factor_logits"]
        one_factor_logits = cached["one_factor_logits"]
    else:
        print(f"[EXTRACT] Initializing extractor with {args.model_name} on cuda...")
        config = ExtractionConfig(
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_cls_tokens=True,
        )
        extractor = UnifiedNLIExtractor(config)

        print("\n[EXTRACT] 1. Two-Factor Bank (4 Threat + 4 Humanitarian pairs)...")
        extractor.hypothesis_pairs = TWO_FACTOR_PAIRS
        extractor.n_bots = len(TWO_FACTOR_PAIRS)
        res_two = extractor.extract_batch(articles, show_progress=True)
        two_factor_logits = res_two["logits_raw"].cpu()

        print("\n[EXTRACT] 2. One-Factor Bank (8 Threat pairs)...")
        extractor.hypothesis_pairs = ONE_FACTOR_PAIRS
        extractor.n_bots = len(ONE_FACTOR_PAIRS)
        res_one = extractor.extract_batch(articles, show_progress=True)
        one_factor_logits = res_one["logits_raw"].cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "two_factor_logits": two_factor_logits,
            "one_factor_logits": one_factor_logits,
        }, cache_path)
        print(f"[SAVE] Cached synthetic control embeddings to {cache_path}")

    # Compute Polarity Matrices: P(E) - P(C)
    p_two_raw = torch.softmax(two_factor_logits.float(), dim=-1).numpy()
    p_two = p_two_raw[..., 2] - p_two_raw[..., 0]  # [60, 8]

    p_one_raw = torch.softmax(one_factor_logits.float(), dim=-1).numpy()
    p_one = p_one_raw[..., 2] - p_one_raw[..., 0]  # [60, 8]

    # Run Dimensionality Null Gate
    print("\n[TEST] Evaluating Two-Factor Bank (Expected k=2)...")
    res_two_null = run_dimensionality_null_test(
        p_two,
        "synthetic_two_factor_control",
        publishers=None,
        draws=args.draws,
    )

    print("\n[TEST] Evaluating One-Factor Bank (Expected k=1)...")
    res_one_null = run_dimensionality_null_test(
        p_one,
        "synthetic_one_factor_control",
        publishers=None,
        draws=args.draws,
    )

    print("\n" + "=" * 92)
    print("SYNTHETIC POSITIVE CONTROL DIMENSIONALITY RESULTS")
    print("=" * 92)
    print(f"{'Condition':<30} | {'Exp. k':<6} | {'Obs PR':<7} | {'Null PR':<7} | {'PC1 (Obs / q95)':<18} | {'PC2 (Obs / q95)':<18} | {'Det. k'}")
    print("-" * 92)

    for res, exp_k in [(res_two_null, 2), (res_one_null, 1)]:
        obs_pr = res["observed_participation_ratio"]
        null_pr = res["column_null_mean_participation_ratio"]
        det_k = res["detected_rank_column_null"]
        c1 = res["components"][0]
        c2 = res["components"][1]
        c1_str = f"{c1['observed_variance_ratio']*100:.1f}% / {c1['column_null']['q95_variance_ratio']*100:.1f}%"
        c2_str = f"{c2['observed_variance_ratio']*100:.1f}% / {c2['column_null']['q95_variance_ratio']*100:.1f}%"
        print(f"{res['dataset_label']:<30} | {exp_k:<6} | {obs_pr:<7.2f} | {null_pr:<7.2f} | {c1_str:<18} | {c2_str:<18} | {det_k}")

    print("=" * 92)

    results = {
        "two_factor_bank": res_two_null,
        "one_factor_bank": res_one_null,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved synthetic positive control results to {args.output}")


if __name__ == "__main__":
    main()
