import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return None
    N, D = X.shape
    if N == 1:
        return None
    mean = np.mean(X, axis=0)
    X_c = X - mean
    cov = (1/(N-1))*(X_c.T)@X_c
    return cov