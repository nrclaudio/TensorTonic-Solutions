import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    
    # Check if shapes match
    if x.shape != p.shape:
        raise ValueError("shape mismatch")
        
    # Check if probabilities sum to 1 (within a 10^-6 tolerance)
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum up to 1")
        
    expected_value = np.sum(x * p)
    return float(expected_value)