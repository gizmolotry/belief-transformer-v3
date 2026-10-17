import numpy as np
from scipy.linalg import orthogonal_procrustes, svd
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

@dataclass(frozen=True)
class AlignmentSpec:
    center: Literal["none", "mean", "median"]
    scale: Literal["none", "fro"]
    group: Literal["orthogonal", "proper_rotation"]
    rank_tolerance: float
    dtype: str = "float64"

def _center_data(data: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return data - np.mean(data, axis=0)
    elif method == "median":
        return data - np.median(data, axis=0)
    return data.copy()

def _scale_data(data: np.ndarray, method: str) -> np.ndarray:
    if method == "fro":
        norm = np.linalg.norm(data, ord='fro')
        if norm > 0:
            return data / norm
    return data.copy()

def fit_procrustes(source: np.ndarray, target: np.ndarray, spec: AlignmentSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits Procrustes transformation from source to target.
    Returns:
        R (np.ndarray): The transformation matrix.
        source_transformed (np.ndarray): The transformed source data.
        target_processed (np.ndarray): The processed target data (centered/scaled).
    """
    # 1. Centering
    source_c = _center_data(source, spec.center)
    target_c = _center_data(target, spec.center)
    
    # 2. Scaling
    source_cs = _scale_data(source_c, spec.scale)
    target_cs = _scale_data(target_c, spec.scale)
    
    # 3. Solve orthogonal Procrustes
    if spec.group == "orthogonal":
        R, _ = orthogonal_procrustes(source_cs, target_cs)
    elif spec.group == "proper_rotation":
        # SVD method ensuring det(R) = 1
        M = target_cs.T @ source_cs
        U, S, Vt = svd(M)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    else:
        raise ValueError(f"Unknown transformation group: {spec.group}")
        
    return R, source_cs @ R, target_cs

def apply_points(source: np.ndarray, R: np.ndarray, spec: AlignmentSpec) -> np.ndarray:
    source_c = _center_data(source, spec.center)
    source_cs = _scale_data(source_c, spec.scale)
    return source_cs @ R

def procrustes_distance(source: np.ndarray, target: np.ndarray, spec: AlignmentSpec) -> float:
    R, transformed_source, processed_target = fit_procrustes(source, target, spec)
    return float(np.linalg.norm(transformed_source - processed_target, ord='fro'))
