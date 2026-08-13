import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a,2)
    b_norm = np.linalg.norm(b,2)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a,b)/(a_norm*b_norm)
    