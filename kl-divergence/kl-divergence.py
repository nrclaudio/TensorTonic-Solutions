import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    mask = p > 0
    p_valid = p[mask]
    q_valid = q[mask]
    q_valid = np.maximum(q_valid, eps)
    return float(np.sum(p_valid * np.log(p_valid/q_valid)))