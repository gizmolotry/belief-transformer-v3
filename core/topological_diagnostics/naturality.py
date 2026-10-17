import numpy as np
from scipy.spatial.distance import cdist

def test_procrustes_naturality(embeddings_A: np.ndarray, embeddings_B: np.ndarray, R: np.ndarray, k: int=10, n_test: int=100) -> dict:
    """
    Test whether Procrustes alignment R commutes with k-NN operation.
    """
    n_articles = embeddings_A.shape[0]
    n_test = min(n_test, n_articles)
    test_indices = np.random.choice(n_articles, n_test, replace=False)
    
    naturality_scores = []
    
    for idx in test_indices:
        # Path 1: Neighbors in A's space
        dists_A = cdist([embeddings_A[idx]], embeddings_A)[0]
        neighbors_A_transformed = np.argsort(dists_A)[1:k+1]
        
        # Path 2: Neighbors in B's space after transformation
        transformed_A = embeddings_A @ R
        dists_transformed = cdist([transformed_A[idx]], transformed_A)[0]
        neighbors_transformed = np.argsort(dists_transformed)[1:k+1]
        
        # Jaccard similarity
        intersection = len(set(neighbors_A_transformed) & set(neighbors_transformed))
        union = len(set(neighbors_A_transformed) | set(neighbors_transformed))
        jaccard = intersection / union if union > 0 else 0
        
        naturality_scores.append(jaccard)
        
    return {
        'mean_naturality': float(np.mean(naturality_scores)),
        'std_naturality': float(np.std(naturality_scores))
    }
