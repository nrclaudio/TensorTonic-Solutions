import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 2 or y_true.ndim != 1:
        return None
    probs = y_pred[np.arange(len(y_true)), y_true]
    return -np.sum(np.log(probs)) / len(y_true)
    