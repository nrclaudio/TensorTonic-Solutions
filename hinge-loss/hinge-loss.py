import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    margin_v = np.zeros_like(y_true)
    margin_v += margin
    hinge = margin_v - (y_true*y_score)
    result = np.maximum(0, hinge)
    if reduction == "mean":
        return float(np.mean(result))
    elif reduction == "sum":
        return float(np.sum(result))
    else:
        return None
    # loss = max(0, margin - (y_true*y_score))
    # return loss