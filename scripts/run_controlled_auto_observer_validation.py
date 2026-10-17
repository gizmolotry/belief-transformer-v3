import json
from pathlib import Path
import torch
import numpy as np
import sys
import io

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

def get_mean_pairwise_sim(tensor_a, tensor_b=None):
    """
    If tensor_b is None, computes mean cosine similarity of all unique pairs in tensor_a.
    If tensor_b is given, computes mean cosine similarity between all pairs (a, b).
    Tensors should be [N, 8, 768].
    """
    # Normalize
    a_norm = torch.nn.functional.normalize(tensor_a, p=2, dim=-1)
    if tensor_b is None:
        sims = []
        for i in range(a_norm.shape[1]):
            for j in range(i + 1, a_norm.shape[1]):
                sim = torch.sum(a_norm[:, i, :] * a_norm[:, j, :], dim=-1).mean().item()
                sims.append(sim)
        return np.mean(sims)
    else:
        b_norm = torch.nn.functional.normalize(tensor_b, p=2, dim=-1)
        sims = []
        for i in range(a_norm.shape[1]):
            for j in range(b_norm.shape[1]):
                sim = torch.sum(a_norm[:, i, :] * b_norm[:, j, :], dim=-1).mean().item()
                sims.append(sim)
        return np.mean(sims)

def main():
    print("=" * 75)
    print(" CONTROLLED PC1-PROJECTED AUTO-OBSERVER VALIDATION")
    print("=" * 75)

    corpus_path = PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl"
    
    # Load corpus
    articles = []
    labels = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # data_utils.py expects a dict
            articles.append(data)
            labels.append(data.get('allsides_score'))

    N = len(articles)
    print(f"▶ Loaded controlled corpus: N={N} articles.")

    print("\n▶ Initializing UnifiedNLIExtractor (Model: cross-encoder/nli-deberta-v3-small)")
    config = ExtractionConfig(
        model_name="cross-encoder/nli-deberta-v3-small",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    # 1. Ideological Extraction
    print("\n▶ Extracting Ideological Pairs...")
    extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
    res_ideo = extractor.extract_batch(articles, show_progress=True)
    ideo_emb = res_ideo['cls_per_bot'] # [60, 8, 768]
    print(f"  Ideological Max Val: {ideo_emb.abs().max().item():.4f}")

    # 2. Structural Control Extraction
    print("\n▶ Extracting Structural Control Pairs...")
    extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
    res_ctrl = extractor.extract_batch(articles, show_progress=True)
    ctrl_emb = res_ctrl['cls_per_bot'] # [60, 8, 768]
    print(f"  Structural Max Val: {ctrl_emb.abs().max().item():.4f}")

    # 3. Save raw embeddings for NMI/ARI evaluation
    torch.save({
        'ideo_emb': ideo_emb,
        'labels': labels
    }, PROJECT_ROOT / "data" / "controlled_ideo_embeddings_v3_small.pt")

    # 4. Compute Anisotropy / PC1 Projection
    print("\n▶ Computing and Regressing PC1 Confound...")
    all_emb = torch.cat([ideo_emb.reshape(-1, 768), ctrl_emb.reshape(-1, 768)], dim=0)
    all_emb = all_emb - all_emb.mean(dim=0, keepdim=True)
    
    # SVD to find PC1
    U, S, V = torch.svd(all_emb)
    pc1 = V[:, 0]
    
    def project_out(tensor, pc):
        # tensor is [N, B, H]
        shape = tensor.shape
        flat = tensor.reshape(-1, shape[-1])
        proj = torch.matmul(flat, pc).unsqueeze(1) * pc.unsqueeze(0)
        resid = flat - proj
        return resid.reshape(shape)

    ideo_resid = project_out(ideo_emb, pc1)
    ctrl_resid = project_out(ctrl_emb, pc1)

    # 5. Compute Direct Cosine Alignments in Residual Space
    # (Do Ideological pairs resemble each other more than they resemble Structural pairs?)
    print("\n▶ RESIDUAL SPACE (PC1-Removed) COSINE ALIGNMENTS:")
    
    ideo_ideo_sim = get_mean_pairwise_sim(ideo_resid)
    ideo_ctrl_sim = get_mean_pairwise_sim(ideo_resid, ctrl_resid)
    ctrl_ctrl_sim = get_mean_pairwise_sim(ctrl_resid)

    print(f"  Ideology vs. Ideology Mean Cosine Sim : {ideo_ideo_sim:.4f}")
    print(f"  Structure vs. Structure Mean Cosine Sim: {ctrl_ctrl_sim:.4f}")
    print(f"  Ideology vs. Structure Mean Cosine Sim : {ideo_ctrl_sim:.4f}")

    metrics = {
        "ideo_ideo_sim": ideo_ideo_sim,
        "ctrl_ctrl_sim": ctrl_ctrl_sim,
        "ideo_ctrl_sim": ideo_ctrl_sim,
        "model_used": "cross-encoder/nli-deberta-v3-small"
    }
    
    with open(PROJECT_ROOT / "data" / "controlled_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
