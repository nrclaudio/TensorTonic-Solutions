import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x)
    relu_lambda = lambda val: np.maximum(0, val)
    return np.asarray(relu_lambda(x))
