import sys
import io
from pathlib import Path
import json
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.auto_observers import select_svd_extremes

def main():
    emb_path = PROJECT_ROOT / "data" / "real_pilot_embeddings.pt"
    corpus_path = PROJECT_ROOT / "data" / "real_corpus.jsonl"

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False).mean(dim=1).numpy()
    
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 60: break
            texts.append(json.loads(line).get("content", ""))

    extremes = select_svd_extremes(embeddings, n_axes=8, extremes_per_side=2)

    for ex in extremes:
        print(f"\n{'='*40}\nAXIS {ex.axis}\n{'='*40}")
        print("POSITIVE EXTREMES:")
        for idx in ex.positive_indices:
            print(f"[{idx}] {texts[idx][:300]}...")
        print("\nNEGATIVE EXTREMES:")
        for idx in ex.negative_indices:
            print(f"[{idx}] {texts[idx][:300]}...")

if __name__ == "__main__":
    main()
