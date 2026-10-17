import numpy as np
from scipy import stats

def calculate_fisher_bounds(n):
    df = n - 3
    if df <= 0:
        return None
        
    se = 1.0 / np.sqrt(df)
    
    # Critical value for p=0.05
    z_crit = 1.95996 * se
    r_crit = np.tanh(z_crit)
    
    # Example for r=0.60
    r_ex = 0.60
    z_ex = np.arctanh(r_ex)
    lower_z = z_ex - 1.95996 * se
    upper_z = z_ex + 1.95996 * se
    lower_r = np.tanh(lower_z)
    upper_r = np.tanh(upper_z)
    
    return r_crit, lower_r, upper_r

for n in [6, 8, 10, 13]:
    res = calculate_fisher_bounds(n)
    if res:
        r_crit, l_r, u_r = res
        print(f"N={n:2d} -> Critical r (p=0.05): {r_crit:.3f} | For r=0.60, 95% CI: [{l_r:.3f}, {u_r:.3f}]")
