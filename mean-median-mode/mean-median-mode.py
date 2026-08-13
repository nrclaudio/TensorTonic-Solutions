import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x)
    mode = float(Counter(x).most_common(1)[0][0])
    median = np.median(x)
    mean = np.mean(x)
    return mean, median, mode