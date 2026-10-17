import numpy as np

def procrustes_path_length(embeddings_A: np.ndarray, embeddings_B: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    For each article, compute how far Observer B's position is from
    the Procrustes-transformed Observer A position.
    """
    transformed_A = embeddings_A @ R
    residuals = np.linalg.norm(transformed_A - embeddings_B, axis=1)
    return residuals
