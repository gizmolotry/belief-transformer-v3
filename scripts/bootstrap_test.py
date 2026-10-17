import json, numpy as np, sys
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.exploratory_internal_structure_analysis as expl

def get_logits(filepath):
    q0_A, q0_B, q1_A, q1_B = [], [], [], []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            q0_A.append(d['q0']['A'])
            q0_B.append(d['q0']['B'])
            q1_A.append(d['q1']['A'])
            q1_B.append(d['q1']['B'])
    return np.array(q0_A), np.array(q0_B), np.array(q1_A), np.array(q1_B)

def to_scores(l_A, l_B):
    s_A = []
    C = []
    for i in range(l_A.shape[0]):
        p_A = expl.stable_softmax(np.array([l_A[i]]))[0]
        p_B = expl.stable_softmax(np.array([l_B[i]]))[0]
        sa = p_A[:, 2] - p_A[:, 0]
        sb = p_B[:, 2] - p_B[:, 0]
        s_A.append(sa)
        C.append(sa - sb)
    return np.array(s_A), np.array(C)

q0_lA, q0_lB, q1_lA, q1_lB = get_logits('outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl')
q0_sA, q0_C = to_scores(q0_lA, q0_lB)
q1_sA, q1_C = to_scores(q1_lA, q1_lB)

def compute_spearman(d0, d1, indices):
    pca0 = expl.deterministic_pca(d0[indices])
    pca1 = expl.deterministic_pca(d1[indices])
    return spearmanr(pdist(pca0['scores']), pdist(pca1['scores'])).statistic

rng = np.random.default_rng(42)
B = 1000
N = q0_sA.shape[0]
diffs = []

orig_A = compute_spearman(q0_sA, q1_sA, np.arange(N))
orig_C = compute_spearman(q0_C, q1_C, np.arange(N))

for _ in range(B):
    idx = rng.choice(N, size=N, replace=True)
    if len(np.unique(idx)) < 3: continue
    try:
        diffs.append(compute_spearman(q0_C, q1_C, idx) - compute_spearman(q0_sA, q1_sA, idx))
    except:
        pass

diffs = np.array(diffs)
print(json.dumps({
    'ci_lower': float(np.percentile(diffs, 2.5)),
    'ci_upper': float(np.percentile(diffs, 97.5)),
    'p_value': float(np.mean(diffs <= 0) * 2)
}))
