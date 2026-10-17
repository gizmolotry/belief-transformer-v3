import json
from pathlib import Path
import torch
import numpy as np
import sys
import io
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
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

RANDOM_CONTROL_PAIRS = [
    ("The recipe requires two cups of flour.", "The telescope observed a distant quasar."),
    ("The cat slept on the windowsill.", "The mechanic replaced the brake pads."),
    ("The artist mixed blue and yellow paint.", "The symphony played a piece by Mozart."),
    ("The hiker reached the summit at dawn.", "The baker kneaded the dough for ten minutes."),
    ("The software compiled without any errors.", "The river flowed gently through the valley."),
    ("The chess player sacrificed their queen.", "The gardener planted tomatoes in the spring."),
    ("The historical novel won a major award.", "The physics experiment yielded unexpected data."),
    ("The child learned to ride a bicycle.", "The CEO announced a new strategic vision.")
]

def get_pairwise_sim_list(tensor_a, tensor_b=None):
    """
    Returns the flat list of cosine similarities.
    Tensors should be [N, 8, 1536].
    """
    a_norm = torch.nn.functional.normalize(tensor_a, p=2, dim=-1)
    sims = []
    if tensor_b is None:
        for i in range(a_norm.shape[1]):
            for j in range(i + 1, a_norm.shape[1]):
                sim = torch.sum(a_norm[:, i, :] * a_norm[:, j, :], dim=-1).mean().item()
                sims.append(sim)
    else:
        b_norm = torch.nn.functional.normalize(tensor_b, p=2, dim=-1)
        for i in range(a_norm.shape[1]):
            for j in range(b_norm.shape[1]):
                sim = torch.sum(a_norm[:, i, :] * b_norm[:, j, :], dim=-1).mean().item()
                sims.append(sim)
    return np.array(sims)

def bootstrap_ci(data, n_bootstraps=1000, ci=95):
    bootstrapped_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

def project_out(tensor, pc):
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1])
    proj = torch.matmul(flat, pc).unsqueeze(1) * pc.unsqueeze(0)
    resid = flat - proj
    return resid.reshape(shape)

def main():
    print("=" * 75)
    print(" GPU CONTROLLED PC1-PROJECTED AUTO-OBSERVER VALIDATION")
    print("=" * 75)

    corpus_path = PROJECT_ROOT / "data" / "controlled_N60_corpus.jsonl"
    
    articles = []
    labels = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            articles.append(data)
            labels.append(data.get('allsides_score'))

    N = len(articles)
    print(f"▶ Loaded controlled corpus: N={N} articles.")

    print("\n▶ Initializing UnifiedNLIExtractor (Model: microsoft/deberta-v2-xlarge-mnli)")
    config = ExtractionConfig(
        model_name="microsoft/deberta-v2-xlarge-mnli",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True,
    )
    extractor = UnifiedNLIExtractor(config)

    cache_path = PROJECT_ROOT / "data" / "controlled_ideo_embeddings_xlarge.pt"
    if cache_path.exists():
        print("\n▶ Loading embeddings from cache...")
        cached = torch.load(cache_path)
        ideo_emb = cached['ideo_emb']
        ctrl_emb = cached['ctrl_emb']
        rand_emb = cached.get('rand_emb')
        labels = cached.get('labels', labels)
    
    if 'ideo_emb' not in locals() or ideo_emb is None:
        print("\n▶ 1. Extracting Ideological Pairs...")
        extractor.hypothesis_pairs = IDEOLOGICAL_PAIRS
        ideo_emb = extractor.extract_batch(articles, show_progress=True)['cls_per_bot']

        print("\n▶ 2. Extracting Structural Control Pairs...")
        extractor.hypothesis_pairs = NEGATIVE_CONTROL_PAIRS
        ctrl_emb = extractor.extract_batch(articles, show_progress=True)['cls_per_bot']

    if 'rand_emb' not in locals() or rand_emb is None:
        print("\n▶ Extracting Semantic Random Control Pairs...")
        extractor.hypothesis_pairs = RANDOM_CONTROL_PAIRS
        rand_emb = extractor.extract_batch(articles, show_progress=True)['cls_per_bot']
        torch.save({
            'ideo_emb': ideo_emb,
            'ctrl_emb': ctrl_emb,
            'rand_emb': rand_emb,
            'labels': labels
        }, cache_path)

    # Mathematical Gaussian Noise Baseline
    math_noise = torch.randn_like(ideo_emb)

    # 5. Compute Anisotropy / PC1 Projection
    print("\n▶ Computing PC1 Confound...")
    all_emb = torch.cat([
        ideo_emb.reshape(-1, ideo_emb.shape[-1]), 
        ctrl_emb.reshape(-1, ctrl_emb.shape[-1]),
        rand_emb.reshape(-1, rand_emb.shape[-1])
    ], dim=0)
    all_emb_centered = all_emb - all_emb.mean(dim=0, keepdim=True)
    
    U, S, V = torch.svd(all_emb_centered)
    pc1 = V[:, 0]
    
    # Variance Explained
    total_var = (S ** 2).sum()
    pc1_var = S[0] ** 2
    print(f"  PC1 Explained Variance Ratio: {pc1_var / total_var:.4f}")

    ideo_resid = project_out(ideo_emb, pc1)
    ctrl_resid = project_out(ctrl_emb, pc1)
    rand_resid = project_out(rand_emb, pc1)
    math_resid = project_out(math_noise, pc1)

    print("\n▶ RESIDUAL SPACE (PC1-Removed) COSINE ALIGNMENTS:")
    
    def get_mean_sim(tensor_a, tensor_b=None):
        sims = get_pairwise_sim_list(tensor_a, tensor_b)
        return np.mean(sims)

    # Report true means
    ideo_mean = get_mean_sim(ideo_resid)
    ctrl_mean = get_mean_sim(ctrl_resid)
    rand_mean = get_mean_sim(rand_resid)
    math_mean = get_mean_sim(math_resid)
    
    print(f"  Ideology vs. Ideology       : {ideo_mean:+.4f}")
    print(f"  Structure vs. Structure     : {ctrl_mean:+.4f}")
    print(f"  Random Text vs. Random Text : {rand_mean:+.4f}")
    print(f"  Math Noise vs. Math Noise   : {math_mean:+.4f}")
    print("-" * 75)
    print(f"  Ideology vs. Structure      : {get_mean_sim(ideo_resid, ctrl_resid):+.4f}")
    print(f"  Ideology vs. Random Text    : {get_mean_sim(ideo_resid, rand_resid):+.4f}")
    print(f"  Ideology vs. Math Noise     : {get_mean_sim(ideo_resid, math_resid):+.4f}")

    print("\n▶ PERMUTATION TEST (10,000 resamples over 24 hypothesis axes):")
    # Pool all 24 hypothesis axes [N, 24, 1536]
    pooled_resid = torch.cat([ideo_resid, ctrl_resid, rand_resid], dim=1)
    
    n_perms = 10000
    null_ideo_means = []
    
    for _ in range(n_perms):
        # Shuffle the 24 axes
        perm_indices = torch.randperm(24)
        # Pick 8 as the "fake ideology" axes
        fake_ideo_indices = perm_indices[:8]
        fake_ideo_tensor = pooled_resid[:, fake_ideo_indices, :]
        null_ideo_means.append(get_mean_sim(fake_ideo_tensor))
        
    null_ideo_means = np.array(null_ideo_means)
    p_value = np.mean(null_ideo_means >= ideo_mean)
    lower_null = np.percentile(null_ideo_means, 2.5)
    upper_null = np.percentile(null_ideo_means, 97.5)
    
    print(f"  Null Distribution 95% Interval : [{lower_null:+.4f}, {upper_null:+.4f}]")
    print(f"  Ideology Mean ({ideo_mean:+.4f}) p-value : {p_value:.4f}")
    if p_value < 0.05:
        print("  => SIGNIFICANT: Ideology coherence exceeds random grouping of hypothesis axes.")
    else:
        print("  => NOT SIGNIFICANT: Ideology coherence is indistinguishable from random grouping.")

if __name__ == "__main__":
    main()
