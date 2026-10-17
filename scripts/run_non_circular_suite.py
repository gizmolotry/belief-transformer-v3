"""
Master orchestrator script for the non-circular validation suite.
Generates unstructured and adversarial corpora, runs evaluations for all
three non-circular baseline experiments, and consolidates the results
into data/non_circular_results.json.
"""

import os
import json
import subprocess
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "non_circular_suite"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_corpora():
    print("="*70)
    # Step 1: Unstructured Corpus
    print("[1/2] Generating unstructured non-circular corpus...")
    subprocess.check_call(["python", str(ROOT_DIR / "sythgen" / "generate_unstructured_corpus.py")])
    
    # Step 2: Adversarial Mixed Corpus
    print("[2/2] Generating adversarial mixed noise corpus...")
    subprocess.check_call(["python", str(ROOT_DIR / "sythgen" / "generate_adversarial_corpus.py")])
    print("="*70)

def run_experiment_1():
    print("\n" + "="*70)
    print("EXPERIMENT 1: Blind Generator Test")
    print("="*70)
    
    run_dir = OUTPUT_DIR / "exp1_unstructured"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python",
        str(ROOT_DIR / "run_experiments.py"),
        "--corpus", "data/control_unstructured.jsonl",
        "--mode", "enhanced",
        "--seeds", "42",
        "--output-root", str(run_dir)
    ]
    subprocess.check_call(cmd)
    
    # Load and evaluate
    data = torch.load(run_dir / "observer_42.pt", map_location="cpu", weights_only=False)
    features = data["features"]
    articles = data["article_metadata"]
    
    # Label is 'publication' (e.g. Al Jazeera vs Ynet)
    labels = [a["publication"] for a in articles]
    unique_labels = sorted(set(labels))
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}
    y_true = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
    
    km = KMeans(n_clusters=len(unique_labels), random_state=42, n_init=10)
    y_pred = km.fit_predict(features)
    
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Blind Generator Publisher NMI: {nmi:.4f}")
    print(f"Blind Generator Publisher ARI: {ari:.4f}")
    
    return {"nmi": float(nmi), "ari": float(ari)}

def run_experiment_2():
    print("\n" + "="*70)
    print("EXPERIMENT 2: Held-Out Cross-Validation")
    print("="*70)
    
    subprocess.check_call(["python", str(ROOT_DIR / "scripts" / "run_cross_validation.py")])
    
    with open(DATA_DIR / "cross_val_metrics.json", "r") as f:
        metrics = json.load(f)
        
    return metrics

def run_experiment_3():
    print("\n" + "="*70)
    print("EXPERIMENT 3: Adversarial Noise Injection")
    print("="*70)
    
    run_dir = OUTPUT_DIR / "exp3_adversarial"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python",
        str(ROOT_DIR / "run_experiments.py"),
        "--corpus", "data/control_adversarial.jsonl",
        "--mode", "enhanced",
        "--seeds", "42",
        "--output-root", str(run_dir)
    ]
    subprocess.check_call(cmd)
    
    data = torch.load(run_dir / "observer_42.pt", map_location="cpu", weights_only=False)
    features = data["features"]
    articles = data["article_metadata"]
    
    # Calculate displacement (norm of features from coordinates)
    # We will measure the average feature norm as a proxy for displacement magnitude
    norms = np.linalg.norm(features, axis=1)
    
    structured_norms = []
    ood_norms = []
    
    for idx, art in enumerate(articles):
        # OOD noise articles are tagged with publication="OOD Monitor"
        if art.get("publication") == "OOD Monitor":
            ood_norms.append(norms[idx])
        else:
            structured_norms.append(norms[idx])
            
    mean_structured = float(np.mean(structured_norms)) if structured_norms else 0.0
    mean_ood = float(np.mean(ood_norms)) if ood_norms else 0.0
    
    # Displacement ratio: structured vs noise
    ratio = mean_structured / mean_ood if mean_ood > 0 else 0.0
    
    # Evaluate NMI of structured subset only
    structured_indices = [idx for idx, art in enumerate(articles) if art.get("publication") != "OOD Monitor"]
    struct_features = features[structured_indices]
    struct_labels = [articles[idx]["perspective_tag"] for idx in structured_indices]
    unique_struct = sorted(set(struct_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_struct)}
    y_true = np.array([label_to_idx[l] for l in struct_labels], dtype=np.int32)
    
    km = KMeans(n_clusters=len(unique_struct), random_state=42, n_init=10)
    y_pred = km.fit_predict(struct_features)
    
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Mean Structured Displacement Norm: {mean_structured:.4f}")
    print(f"Mean OOD Noise Displacement Norm: {mean_ood:.4f}")
    print(f"Displacement Ratio (Structured / OOD): {ratio:.4f}")
    print(f"Robustness NMI (Ideological subset under noise): {nmi:.4f}")
    
    return {
        "mean_structured_displacement": mean_structured,
        "mean_ood_displacement": mean_ood,
        "displacement_ratio": ratio,
        "robustness_nmi": float(nmi),
        "robustness_ari": float(ari)
    }

def main():
    generate_corpora()
    
    results = {}
    results["experiment_1_blind_generator"] = run_experiment_1()
    results["experiment_2_cross_val"] = run_experiment_2()
    results["experiment_3_adversarial_noise"] = run_experiment_3()
    
    # Write consolidated outputs
    with open(DATA_DIR / "non_circular_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print("NON-CIRCULAR VALIDATION SUITE COMPLETE")
    print(f"Consolidated results saved to: {DATA_DIR / 'non_circular_results.json'}")
    print("="*70)

if __name__ == "__main__":
    main()
