import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        axes = 1
    else:
        axes = 0
    x_max = np.max(x, axis=axes, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    sum_exp_x = np.sum(exp_x, axis=axes, keepdims=True)
    return exp_x/sum_exp_x
    