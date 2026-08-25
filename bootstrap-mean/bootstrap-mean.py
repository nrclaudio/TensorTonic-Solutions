import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    rng = np.random.default_rng(seed=seed)
    x = np.asarray(x)
    n = len(x)
    means = []
    for _ in range(n_bootstrap):
        b_x = rng.choice(x, size=n, )
        b_mean = np.mean(b_x)
        means.append(b_mean)
    alpha = (1 - ci) / 2
    upper_q = (1 - alpha) * 100
    lower_q = alpha * 100
    upper, lower = np.percentile(means, [upper_q, lower_q])
    return {"bootstrap_mean": np.mean(means),
           "lower": lower,
            "upper": upper}
    