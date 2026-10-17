import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
observer_file = ROOT_DIR / "outputs" / "non_circular_suite" / "exp1_unstructured" / "observer_42.pt"

data = torch.load(observer_file, map_location="cpu", weights_only=False)
print("Keys in observer_42.pt:")
for k in sorted(data.keys()):
    val = data[k]
    if isinstance(val, (torch.Tensor, list)):
        print(f"  {k}: {type(val)} with size/length {len(val) if isinstance(val, list) else val.shape}")
    elif isinstance(val, dict):
        print(f"  {k}: dict with keys {list(val.keys())}")
        if k == "T3_topology":
            for tk, tv in val.items():
                print(f"    {tk}: {type(tv)} = {tv}")
    else:
        print(f"  {k}: {type(val)} = {val}")
