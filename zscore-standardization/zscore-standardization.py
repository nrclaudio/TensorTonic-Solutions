import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    z = (X - mu) / std  
    
    mask = std <= eps   
    
    return np.where(mask, 0.0, z)