from .contracts import TopologicalManifest
from .naturality import test_procrustes_naturality
from .path_metrics import procrustes_path_length
from .persistence import compute_persistence

import numpy as np
from core.geometry.alignment import fit_procrustes, AlignmentSpec
import json
import os
from typing import Dict, List, Any, Tuple

class SemanticGeometryDiagnostics:
    """
    Complete suite of categorical/topological tests for Belief Transformer.
    """
    
    def __init__(self, observers: List[str], embeddings_dict: Dict[str, np.ndarray], article_metadata: List[Dict] = None):
        self.observers = observers
        self.embeddings = embeddings_dict
        self.metadata = article_metadata or []
        self.procrustes_matrices = {}
        self.persistence = {}
        self.spec = AlignmentSpec(center="none", scale="none", group="orthogonal", rank_tolerance=1e-5)
        
    def compute_all_procrustes(self):
        """Precompute all pairwise Procrustes alignments."""
        for obs_A in self.observers:
            for obs_B in self.observers:
                if obs_A != obs_B:
                    R, _, _ = fit_procrustes(
                        self.embeddings[obs_A],
                        self.embeddings[obs_B],
                        self.spec
                    )
                    self.procrustes_matrices[(obs_A, obs_B)] = R
    
    def test_naturality(self, k: int=10, n_test: int=100) -> Dict[Tuple, Dict]:
        """Test naturality for all observer pairs."""
        results = {}
        for (obs_A, obs_B), R in self.procrustes_matrices.items():
            results[f"{obs_A}_to_{obs_B}"] = test_procrustes_naturality(
                self.embeddings[obs_A],
                self.embeddings[obs_B],
                R, k=k, n_test=n_test
            )
        return results
    
    def compute_path_metrics(self) -> Dict[str, Any]:
        """Compute path lengths for all articles across all observer pairs."""
        path_database = {}
        
        for (obs_A, obs_B), R in self.procrustes_matrices.items():
            lengths = procrustes_path_length(
                self.embeddings[obs_A],
                self.embeddings[obs_B],
                R
            )
            path_database[f"{obs_A}_to_{obs_B}"] = lengths.tolist()
        
        if len(self.observers) > 0 and len(self.embeddings) > 0:
            n_articles = self.embeddings[self.observers[0]].shape[0]
            mean_contestation = np.zeros(n_articles)
            
            for article_idx in range(n_articles):
                paths = [path_database[k][article_idx] for k in path_database.keys()]
                mean_contestation[article_idx] = np.mean(paths) if paths else 0.0
        else:
            mean_contestation = np.array([])
            
        return {
            'pairwise': path_database,
            'mean_contestation': mean_contestation.tolist()
        }
    
    def compute_persistence(self, max_epsilon: float=2.0) -> Dict[str, Dict]:
        """Compute persistent homology for all observers."""
        for obs in self.observers:
            self.persistence[obs] = compute_persistence(
                self.embeddings[obs],
                max_epsilon=max_epsilon
            )
        return self.persistence
    
    def generate_thesis_report(self, output_dir: str='diagnostics') -> Dict[str, Any]:
        """Generate complete diagnostic report with figures and tables."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Naturality analysis
        naturality = self.test_naturality()
        with open(os.path.join(output_dir, 'naturality.json'), 'w') as f:
            json.dump(naturality, f, indent=2)
            
        # 2. Path metrics
        paths = self.compute_path_metrics()
        with open(os.path.join(output_dir, 'path_metrics.json'), 'w') as f:
            json.dump(paths, f, indent=2)
            
        # 3. Persistence
        persistence_res = self.compute_persistence()
        with open(os.path.join(output_dir, 'persistence_summary.json'), 'w') as f:
            json.dump(persistence_res, f, indent=2)
            
        report = {
            'mean_naturality': np.mean([r['mean_naturality'] for r in naturality.values()]) if naturality else 0.0,
            'median_contestation': float(np.median(paths['mean_contestation'])) if paths['mean_contestation'] else 0.0,
        }
        
        with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
