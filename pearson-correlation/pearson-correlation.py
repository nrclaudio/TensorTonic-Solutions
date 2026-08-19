import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        return None
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None
    cov = np.cov(X.T)
    sd = np.std(X, axis=0, ddof=1)
    
    R = cov / np.outer(sd, sd)
    return R
    