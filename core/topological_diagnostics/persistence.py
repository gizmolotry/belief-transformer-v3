import numpy as np

def compute_persistence(embeddings: np.ndarray, max_epsilon: float=2.0) -> dict:
    """
    Compute persistent homology using Ripser.
    """
    try:
        from ripser import ripser
    except ImportError:
        return {'status': 'UNAVAILABLE_DEPENDENCY'}
        
    result = ripser(
        embeddings,
        maxdim=1,
        thresh=max_epsilon
    )
    
    diagrams = result['dgms']
    
    # Calculate robustness scores for H0
    persistences = []
    for birth, death in diagrams[0]:
        if death == np.inf:
            persistence = max_epsilon - birth
        else:
            persistence = death - birth
        persistences.append(persistence)
    
    persistences = np.array(persistences)
    robust_features = int(np.sum(persistences > 0.3))
    
    return {
        'status': 'OK',
        'robust_features_h0': robust_features,
        'total_features_h0': len(persistences),
        'diagram_summary': {
            'H0_count': len(diagrams[0]),
            'H1_count': len(diagrams[1]) if len(diagrams) > 1 else 0
        }
    }
