import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X)
    samples, features = X.shape
    X_c = X - np.mean(X, axis=0, keepdims=True)
    C = (X_c.T @ X_c) / (len(X) - 1)
    eigenvals = []
    eigenvectors = []
    for _ in range(k):
        v = np.random.rand(features,1)
        for _ in range(10000):
            v = (C@v) / (np.linalg.norm(C@v))
            if np.linalg.norm(C@v) < 1e-12:
                v = np.zeros((features, 1))
                break
    
        _lambda = (v.T@C@v)[0,0]
        eigenvectors.append(v)
        eigenvals.append(_lambda)
        C = C - (_lambda * (v @ v.T))

    W = np.column_stack(eigenvectors)
    X_projected = X_c @ W
    
    return X_projected.tolist()
    
    

    