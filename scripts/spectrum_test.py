import json, numpy as np, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.exploratory_internal_structure_analysis as expl

def get_logits(filepath):
    q0_A, q0_B = [], []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            q0_A.append(d['q0']['A'])
            q0_B.append(d['q0']['B'])
    return np.array(q0_A), np.array(q0_B)

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

q0_lA, q0_lB = get_logits('outputs/fresh_scalar_structure_run/extraction_checkpoint.jsonl')
q0_sA, q0_C = to_scores(q0_lA, q0_lB)

def full_spectrum(mat):
    c = mat - mat.mean(axis=0)
    _, S, _ = np.linalg.svd(c, full_matrices=False)
    ev = S**2 / (mat.shape[0] - 1)
    return ev / ev.sum()

print("A-Only Full:", np.round(full_spectrum(q0_sA), 3).tolist())
print("Contrastive Full:", np.round(full_spectrum(q0_C), 3).tolist())
