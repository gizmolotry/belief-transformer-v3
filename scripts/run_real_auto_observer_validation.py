import os
import sys
import io
import json
from pathlib import Path
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nli_extraction import UnifiedNLIExtractor, ExtractionConfig

IDEOLOGICAL_PAIRS = [
    ("The text focuses on specific casualties caused by military actions.", "The text focuses on broad international diplomatic relations."),
    ("The text focuses on the fate of individual hostages or soldiers.", "The text focuses on broad institutional assessments of the war."),
    ("The text highlights urgent humanitarian pleas for detained individuals.", "The text highlights overarching geopolitical policies."),
    ("The text focuses on international institutional condemnations.", "The text focuses on localized military operations."),
    ("The text describes physical destruction of civil infrastructure.", "The text describes historical diplomatic efforts."),
    ("The text focuses on personal survival accounts of attacks.", "The text discusses systemic economic impacts of the conflict."),
    ("The text focuses on specific rescue operations and survival.", "The text focuses on systemic institutional policies."),
    ("The text focuses on immediate danger to high-profile officials.", "The text focuses on systemic displacement of patients.")
]

NEGATIVE_CONTROL_PAIRS = [
    ("The text is written in an episodic, hard-news reporting style.", "The text is written in an analytical, op-ed commentary style."),
    ("The text uses short, declarative sentences.", "The text uses complex, multi-clause sentences."),
    ("The text heavily quotes direct speech.", "The text relies entirely on indirect paraphrasing."),
    ("The text uses active voice constructions.", "The text uses passive voice constructions."),
    ("The text is written in the present tense.", "The text is written in the past tense."),
    ("The text employs formal, academic vocabulary.", "The text employs informal, conversational vocabulary."),
    ("The text explicitly names specific geographical locations.", "The text completely omits geographical locations."),
    ("The text uses precise numerical statistics.", "The text uses broad qualitative descriptors.")
]

def main():
    print("=" * 75)
    print(" GENUINE AUTO-OBSERVER & MATCHED CONTROL VALIDATION BATTERY")
    print("=" * 75)

    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")

    # Load original embeddings
    real_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False) # [60, 8, 1536]
    N, B, H = real_embeddings.shape
    print(f"▶ Loaded manual embeddings: N={N}, B={B}, H={H}")

    mean_articles = real_embeddings.mean(dim=1, keepdim=True)
    manual_shear = real_embeddings - mean_articles
    
    axis_directions_manual = torch.zeros((B, H), dtype=torch.float32)
    for k in range(B):
        _, _, V = torch.svd(manual_shear[:, k, :])
        axis_directions_manual[k] = V[:, 0]

    # Load corpus texts
    articles = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N: break
            articles.append(json.loads(line).get("content", ""))

    print("\n▶ Initializing UnifiedNLIExtractor (This may take a few minutes...)")
    config = ExtractionConfig(
        model_name="microsoft/deberta-v2-xlarge-mnli",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True, # Need mean pooling for embedding path
    )
    extractor = UnifiedNLIExtractor(config)

    # 1. Run Auto-Observer (Ideological)
    print("\n▶ Extracting Genuine Auto-Observer Ideological Pairs...")
    extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
    res_auto = extractor.extract_batch(articles, show_progress=True)
    auto_embeddings = res_auto['cls_per_bot'] # [60, 8, 1536]
    auto_shear = auto_embeddings - auto_embeddings.mean(dim=1, keepdim=True)

    axis_cosine_sims = []
    for k in range(B):
        _, _, V = torch.svd(auto_shear[:, k, :])
        auto_dir = V[:, 0]
        # Align sign (SVD direction is arbitrary)
        if torch.dot(auto_dir, axis_directions_manual[k]) < 0:
            auto_dir = -auto_dir
        axis_cosine_sims.append(float(torch.dot(axis_directions_manual[k], auto_dir)))

    # 2. Run Negative Control (Register/Style)
    print("\n▶ Extracting Negative Control Pairs...")
    extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
    res_control = extractor.extract_batch(articles, show_progress=True)
    control_embeddings = res_control['cls_per_bot'] # [60, 8, 1536]
    control_shear = control_embeddings - control_embeddings.mean(dim=1, keepdim=True)

    control_cosine_sims = []
    for k in range(B):
        _, _, V = torch.svd(control_shear[:, k, :])
        control_dir = V[:, 0]
        if torch.dot(control_dir, axis_directions_manual[k]) < 0:
            control_dir = -control_dir
        control_cosine_sims.append(float(torch.dot(axis_directions_manual[k], control_dir)))

    print("\n▶ AXIS-BY-AXIS DIRECTIONAL COSINE SIMILARITY (Manual vs. Genuine Auto-Observer):")
    for k, sim in enumerate(axis_cosine_sims):
        print(f"  ├── Framing Axis Pair #{k+1}: Cosine Similarity = {sim:.4f}")
    mean_auto_sim = float(np.mean(axis_cosine_sims))
    print(f"  └── Mean Auto-Observer Axis Alignment: {mean_auto_sim:.4f}")

    print("\n▶ MATCHED NON-IDEOLOGICAL CONTROL AXIS ALIGNMENT (Register/Tense):")
    for k, sim in enumerate(control_cosine_sims):
        print(f"  ├── Non-Ideological Pair #{k+1}: Cosine Similarity = {sim:.4f}")
    mean_control_sim = float(np.mean(control_cosine_sims))
    print(f"  └── Mean Non-Ideological Control Alignment: {mean_control_sim:.4f}")

    metrics = {
        "mean_auto_observer_axis_similarity": mean_auto_sim,
        "mean_non_ideological_control_similarity": mean_control_sim,
        "per_axis_auto_similarity": axis_cosine_sims,
        "per_axis_control_similarity": control_cosine_sims,
    }
    
    out_path = PROJECT_ROOT / "data" / "genuine_auto_observer_validation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

if __name__ == "__main__":
    main()
