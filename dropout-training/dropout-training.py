import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng:
        r = rng.random(x.shape)
    else:
        r = np.random.random(x.shape)
    scale = 1.0 / (1.0 - p)
    dropout_pattern = (r >= p) * scale
    output = x * dropout_pattern
    return (output, dropout_pattern)