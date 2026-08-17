import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.asarray(matrix, dtype=float)
    
    # Guard clauses
    if matrix.ndim != 2 or axis not in (0, 1, None):
        return None

    # Calculate the denominator based on the norm type
    if norm_type == 'l2':
        denominator = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
    elif norm_type == 'l1':
        denominator = np.sum(matrix, axis=axis, keepdims=True)
    elif norm_type == 'max':
        denominator = np.max(matrix, axis=axis, keepdims=True)
    else:
        return None

    # Safely divide, leaving 0s where the denominator is 0
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator!=0)