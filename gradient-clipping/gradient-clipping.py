import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g = np.asarray(g)
    g_norm = np.linalg.norm(g)
    if g_norm > max_norm:
        g_clipped = g * (max_norm / g_norm)
        return g_clipped
    return g
    
