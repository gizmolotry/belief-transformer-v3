import torch
import numpy as np
from pathlib import Path

def project_out(tensor, pc):
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1])
    proj = torch.matmul(flat, pc).unsqueeze(1) * pc.unsqueeze(0)
    resid = flat - proj
    return resid.reshape(shape)

def get_mean_sim_vectorized(tensor_a):
    # tensor_a: [N, 8, 1536]
    a_norm = torch.nn.functional.normalize(tensor_a, p=2, dim=-1)
    sim = torch.einsum('nid,njd->nij', a_norm, a_norm).mean(dim=0) # [8, 8]
    idx = torch.triu_indices(8, 8, offset=1)
    return sim[idx[0], idx[1]].mean().item()

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    cache_path = PROJECT_ROOT / "data" / "controlled_ideo_embeddings_xlarge.pt"
    cached = torch.load(cache_path)
    
    ideo_emb = cached['ideo_emb']
    ctrl_emb = cached['ctrl_emb']
    rand_emb = cached['rand_emb']
    
    all_emb = torch.cat([
        ideo_emb.reshape(-1, ideo_emb.shape[-1]), 
        ctrl_emb.reshape(-1, ctrl_emb.shape[-1]),
        rand_emb.reshape(-1, rand_emb.shape[-1])
    ], dim=0)
    all_emb_centered = all_emb - all_emb.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(all_emb_centered)
    pc1 = V[:, 0]
    
    ideo_resid = project_out(ideo_emb, pc1)
    ctrl_resid = project_out(ctrl_emb, pc1)
    rand_resid = project_out(rand_emb, pc1)
    
    ideo_mean = get_mean_sim_vectorized(ideo_resid)
    
    pooled_resid = torch.cat([ideo_resid, ctrl_resid, rand_resid], dim=1) # [60, 24, 1536]
    
    n_perms = 100000
    print(f"Running {n_perms} permutations...")
    
    null_ideo_means = np.zeros(n_perms)
    for i in range(n_perms):
        perm_indices = torch.randperm(24)[:8]
        fake_ideo_tensor = pooled_resid[:, perm_indices, :]
        null_ideo_means[i] = get_mean_sim_vectorized(fake_ideo_tensor)
        
    count_exceeding = np.sum(null_ideo_means >= ideo_mean)
    p_value = (count_exceeding + 1) / (n_perms + 1)
    
    lower_null = np.percentile(null_ideo_means, 2.5)
    upper_null = np.percentile(null_ideo_means, 97.5)
    
    print(f"  Ideology Mean: {ideo_mean:+.4f}")
    print(f"  Null Distribution 95% Interval: [{lower_null:+.4f}, {upper_null:+.4f}]")
    print(f"  Count Exceeding: {count_exceeding}")
    print(f"  p-value: {p_value:.6f}")
    if count_exceeding == 0:
        print(f"  p < {1/n_perms}")

if __name__ == '__main__':
    main()
