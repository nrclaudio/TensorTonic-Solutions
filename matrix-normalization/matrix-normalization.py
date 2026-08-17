import numpy as np

def l2_norm(vector):
    denominator = np.sqrt(np.sum(vector**2))
    if denominator == 0:
        return vector # Return the zeros as-is
    return vector / denominator

def l1_norm(vector):
    denominator = np.sum(vector)
    if denominator == 0:
        return vector
    return vector / denominator

def max_norm(vector):
    denominator = np.max(vector)
    if denominator == 0:
        return vector
    return vector / denominator
    
def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    
    matrix = np.asarray(matrix)
    # Guard clause: Check if the matrix is not 2D
    if matrix.ndim != 2:
        return None
    if norm_type not in ['l2', 'l1', 'max']:
        return None
    if norm_type == 'l2':
        norm = l2_norm
    elif norm_type == 'l1':
        norm = l1_norm
    elif norm_type == 'max':
        norm = max_norm
    if axis == 0: # col-wise
        new_matrix = []
        for col in zip(*matrix):
           new_matrix.append(norm(np.asarray(col)))
        new_matrix = np.asarray(new_matrix).T
        return new_matrix
    elif axis == 1: #row-wise
        new_matrix = []
        for row in matrix:
            new_matrix.append(norm(np.asarray(row)))
        new_matrix = np.asarray(new_matrix)
        return new_matrix
    elif axis is None:
        return norm(matrix)
        
    
        
    