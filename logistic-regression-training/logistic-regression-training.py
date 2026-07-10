import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _bce(p, y, n):
    """Binary Cross-Entropy"""
    return -1/n*((y * np.log(p)) + (1 - y) * np.log(1-p))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X@w + b)
        loss = _bce(p, y, n)
        g_w = 1/n*(X.T)@(p-y)
        g_b = 1/n*(np.sum(p-y))
        w = w - lr*(g_w)
        b = b - lr*(g_b)
    return (w, b)
