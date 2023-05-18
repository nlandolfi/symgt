import numpy as np


def dorfman_pool_size(prevalence: float, max_pool_size: int = 100) -> int:
    """
    Compute the optimal pool size according to Dorfman's infinite analysis
    where `prevalence` is the population prevalence rate.

    In other words, minimize `1/m + 1 - (1-prevalence)^m` with respect to
    the pool size `m`.

    This function is a helper for `dorfman_multfn` below.
    """
    if not (0.0 <= prevalence <= 1.0):
        raise ValueError(f"prevalence={prevalence} must be in [0, 1]")

    if not (max_pool_size > 1):
        raise ValueError(f"max_pool_size={max_pool_size} should be > 1")

    costs = [1 / m + 1 - (1 - prevalence) ** m for m in range(1, max_pool_size + 1)]

    m = int(np.argmin(costs)) + 1  # off by one indexing

    if 1 / m + 1 - (1 - prevalence) ** m > 1:
        m = 1  # no pooling
    if m == max_pool_size:
        print("WARNING: m == max_pool_size; might need to increase max_pool_size")

    return m


def dorfman_multfn(n: int, prevalence: float) -> np.ndarray:
    """
    Compute a multiplicity function using Dorfman's infinite analysis,
    adding a pool of irregular size if the indicated pool size does not
    divide evenly into `n`.

    Uses `dorfman_pool_size` above.
    """

    multfn = np.zeros(n + 1, dtype=int)
    m = dorfman_pool_size(prevalence, max_pool_size=n)
    multfn[m] = n // m  # integer division
    if n % m != 0:
        multfn[n % m] = 1  # remainder
    return multfn
