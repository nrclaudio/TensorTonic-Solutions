import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """Return the one-sample t-statistic."""
    sd = np.std(x, ddof=1)
    mean = np.mean(x)
    if sd == 0:
        if mean == mu0:
            return 0
        else:
            return np.inf
       
    
    t = (mean - mu0) / (sd/np.sqrt(len(x)))
    return float(t)