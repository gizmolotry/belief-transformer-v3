import argparse
from pathlib import Path
import torch
import numpy as np
import sys
import json
import logging

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.topological_diagnostics import SemanticGeometryDiagnostics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_diagnostics(run_dir: Path) -> None:
    logging.info(f"Running topological diagnostics on {run_dir}")
    
    # Load all observer embeddings
    observer_files = list(run_dir.glob("observer_*.pt"))
    if not observer_files:
        # Check subdirectories if it's an enhanced suite root
        observer_files = list(run_dir.rglob("observer_*.pt"))
        if not observer_files:
            logging.warning(f"No observer files found in {run_dir}")
            return
    
    # We will group by the parent directory to process each (kernel, channel, corpus) family
    families = {}
    for f in observer_files:
        family_dir = f.parent
        if family_dir not in families:
            families[family_dir] = []
        families[family_dir].append(f)
        
    for family_dir, files in families.items():
        logging.info(f"Processing family: {family_dir.relative_to(run_dir.parent)}")
        embeddings_dict = {}
        observers = []
        
        for f in files:
            data = torch.load(f, map_location='cpu', weights_only=False)
            emb = None
            for field in ['final_features', 'embeddings', 'features']:
                if field in data:
                    emb = data[field]
                    break
            if emb is not None:
                if torch.is_tensor(emb):
                    emb = emb.cpu().numpy()
                obs_name = f.stem
                observers.append(obs_name)
                embeddings_dict[obs_name] = emb
        
        if not embeddings_dict:
            continue
            
        diagnostics = SemanticGeometryDiagnostics(
            observers=observers,
            embeddings_dict=embeddings_dict,
            article_metadata=None
        )
        
        diagnostics.compute_all_procrustes()
        
        output_dir = family_dir / "topological_diagnostics"
        report = diagnostics.generate_thesis_report(output_dir=str(output_dir))
        
        logging.info(f"Completed diagnostics for {family_dir.name}:")
        logging.info(f"  Naturality: {report.get('mean_naturality', 0):.3f}")
        logging.info(f"  Contestation: {report.get('median_contestation', 0):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topological diagnostics on experiment suite")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to experiment run directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logging.error(f"Directory {run_dir} does not exist.")
        sys.exit(1)
        
    run_diagnostics(run_dir)
