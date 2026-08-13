import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    var = np.var(x, ddof=1)
 
    sd = np.sqrt(var)
    return var, sd