import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
observer_file = ROOT_DIR / "outputs" / "observer_43.pt"

if not observer_file.exists():
    print(f"File not found: {observer_file}")
else:
    data = torch.load(observer_file, map_location="cpu", weights_only=False)
    print("Keys in outputs/observer_43.pt:")
    for k in sorted(data.keys()):
        val = data[k]
        if isinstance(val, dict):
            print(f"  {k}: dict with keys {list(val.keys())}")
            if k == "T3_topology":
                for tk, tv in val.items():
                    if isinstance(tv, (list, bytes, str)) or tv is None:
                        print(f"    {tk}: {type(tv)} = {tv if tv is None or len(str(tv)) < 100 else len(tv)}")
                    else:
                        print(f"    {tk}: {type(tv)} with shape/length {tv.shape if hasattr(tv, 'shape') else len(tv)}")
        else:
            print(f"  {k}: {type(val)}")
            if k == "bot_rkhs":
                print(f"    shape: {val.shape}")
