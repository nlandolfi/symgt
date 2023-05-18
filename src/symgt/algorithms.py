import numpy as np

import numpy as np
from typing import List


def dorfman_pool_size(prevalence: float, max_pool_size: int = 100) -> int:
    """
    Compute the optimal pool size according to Dorfman's infinite analysis
    using `prevalence`.

    In other words, minimize `1/m + 1 - (1-prevalence)^m` over the pool size `m`.

    This function is a helper for `dorfman_pattern` below.
    """
    if not (0 <= prevalence <= 1):
        raise ValueError(f"prevalence={prevalence} must be in [0, 1]")

    costs = [1 / m + 1 - (1 - prevalence) ** m for m in range(1, max_pool_size + 1)]

    m = int(np.argmin(costs)) + 1  # off by one indexing

    if 1 / m + 1 - (1 - prevalence) ** m > 1:
        m = 1
    if m == max_pool_size:
        print("WARNING: m == max_pool_size; might need to increase max_pool_size")

    return m


def dorfman_multfn(n: int, prevalence: float) -> List[int]:
    """
    Compute a multiplicity function according to Dorfman's infinite analysis.

    This adds a pool of irregular size if Dorfman's pool size does not
    divide evenly into `n`.

    See `dorfman_pool_size` above.
    """

    multfn = np.zeros(n + 1, dtype=int)
    m = dorfman_pool_size(prevalence, max_pool_size=n)
    multfn[m] = n // m  # integer division
    if n % m != 0:
        multfn[n % m] = 1  # remainder
    return multfn.tolist()


def optimal_multfn(q: np.ndarray, subpopulations=False):
    """
    Compute an optimal multiplicity function for a symmetric distribution with representation `q`.
    The population size `n` is inferred from the length of `q` (i.e., `len(q) - 1`).

    Use the keyword argument `subpopulations=true` to return multiplicity functions and costs
    for all subpopulations.  The multiplicity functions are the rows of the first value returned.

    Examples
    --------
    e.g.,
    ```
        multfns, costs = optimal_patterns(q; subpopulations=true)
    ```
    `multfns[i, :]` is an optimal multiplicty function for a subpopulation of size `i` and `costs[i]` is its cost.
    """
    n = len(q) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    # U[h] is the *expected* number of tests to declare a group of size h = 0, …, n
    U = U_from_q(q)

    # J[m] is the optimal cost to declare a population of size m = 0, …, n
    J = np.zeros(n + 1)  # note, J[0] = 0 by default

    # The cost of declaring one individual is 1 test
    J[1] = 1

    # the n+1 here is for indexing off by one, we don't use i[0] or multfns[0, :]
    i = np.zeros(n + 1, dtype=int)
    multfns = np.zeros((n + 1, n + 1), dtype=int)

    for k in range(1, n + 1):
        # find an optimal i[k]; the +1 here is for off by one indexing
        i[k] = np.argmin([J[k - i] + U[i] for i in range(1, k + 1)]) + 1

        # record the optimal cost
        J[k] = J[k - i[k]] + U[i[k]]

        if k - i[k] > 0:  # if we are using a subproblem
            multfns[k, :] = multfns[k - i[k], :]  # take its pattern
        # otherwise, inherit the zero pattern

        # update the pattern to include a group of size i[k]
        multfns[k, i[k]] += 1

    assert J[0] == 0  # optimal cost of declaring 0 individuals is 0
    assert J[1] == 1  # optimal cost of declaring 1 individual is 1
    assert np.all(np.diff(J) >= 0)  # nondecreasing

    # ith row should be a pattern the number i
    w = np.arange(0, n + 1)
    got, want = multfns @ w, w
    assert np.all(got == want), f"multfns @ np.arange(1, n+1): got {got} want {want}"

    if subpopulations:
        return multfns, J
    else:
        return multfns[n, :], J[n]


def U_from_q(q: np.ndarray) -> np.ndarray:
    """
    Compute the function `U` for the symmetric distribution represented by `q`.

    U : {0,…,n} → R
    U(h) is the expected number of tests used to declare a group of size `h`.


    This is a helper function for `optimal_multfn` below.
    """
    # assert is_plausible_q(q)
    n = len(q) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    U = np.zeros(n + 1)
    U[0] = 1
    U[1] = 1
    for i in range(2, n + 1):
        U[i] = 1 + i * (1 - q[i])
    return U


def integer_partition(multfn: np.ndarray) -> List[int]:
    """
    Convert a multiplicity function to an integer partition.
    An integer partition is a nondescreasing list of (possibly repeating) part sizes.

    `multfn[i]` is the multiplicty of a part of size i
    """
    ss = []
    for i, x in enumerate(multfn):
        for _ in range(x):
            ss.append(i)
    ss.reverse()  # conventionally nonincreasing
    return ss


def ECost(multfn: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the expected cost of a grouping encoded by `multfn` for a
    distribution with representation `q`.

    `multfn[i]` is the number of parts of size `i`.
    `q[i]` is the probability that a group of size i tests negative.

    """
    return np.sum([m * ETests(h, q) for (h, m) in enumerate(multfn)])


def ETests(h: int, q: np.ndarray) -> float:
    """
    Compute the expected number of tests used for a group of size `h`
    for a distribution with representation `q`.

    `h` is the size of the group.
    `q[i]` is the probability that a group of size i tests negative.
    """
    if h == 1:
        return 1
    else:
        return 1 + h * (1 - q[h])
