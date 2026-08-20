import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        # This will fail for jagged lists like [[1, 2, 3], [4, 5]]
        matrix = np.asarray(matrix, dtype=float) 
    except ValueError:
        return None
        
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvals = np.linalg.eigvals(matrix)
    eigenvals = eigenvals[np.lexsort((eigenvals.imag, eigenvals.real))]
    return eigenvals