import numpy as np

from .utils import U_from_q


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


def compute_optimal_multfn(c: np.ndarray, subproblems=False):
    """
    Compute an optimal multiplicity function for cost `c` where `c[i]` is the
    cost of a part of size `i`. The size of the largest part `n` is inferred
    from c (i.e., `n = len(c) - 1`). The value `c[0]` is not used.

    We use dynamic programming. We do not compute all optimal multiplicity
    functions, just a single one.

    Use the keyword argument `subproblems=true` to return multiplicity
    functions and costs for all subproblems. The multiplicity functions
    are the rows of the first value returned.

    Examples
    --------
    To just get the solution for `n = len(c) - 1`:
    ```
        multfn, cost = compute_optimal_multfn(c)
    ```
    Here `multfn` is an nd.array and `cost` is a float.

    To get solutions to all subproblems:
    ```
        multfns, costs = compute_optimal_multfn(c, subpopulations=true)
    ```
    Here `multfns[i, :]` is an optimal multiplicty function for a subpopulation
    of size `i` and `costs[i]` is its cost.
    """
    # c[i] is the cost of a part of size i = 0, …, n
    n = len(c) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    # J[m] is the optimal cost to declare a population of size m = 0, …, n
    J = np.zeros(n + 1)  # note, J[0] = 0 by default

    # The cost of declaring one individual is 1 test
    J[1] = 1

    # the n+1 here is for indexing off by one, we don't use i[0] or multfns[0, :]
    i = np.zeros(n + 1, dtype=int)
    multfns = np.zeros((n + 1, n + 1), dtype=int)

    for k in range(1, n + 1):
        # find an optimal i[k]; the +1 here is for off by one indexing
        i[k] = np.argmin([J[k - i] + c[i] for i in range(1, k + 1)]) + 1

        # record the optimal cost
        J[k] = J[k - i[k]] + c[i[k]]

        if k - i[k] > 0:  # if we are using a subproblem
            multfns[k, :] = multfns[k - i[k], :]  # take its multfn
        # otherwise, inherit the zero pattern

        # update the multfn to include a part of size i[k]
        multfns[k, i[k]] += 1

    # ith row should be a pattern the number i
    w = np.arange(0, n + 1)
    got, want = multfns @ w, w
    assert np.all(got == want), f"multfns @ np.arange(1, n+1): got {got} want {want}"

    if subproblems:
        return multfns, J
    else:
        return multfns[n, :], J[n]


def symmetric_multfn(q: np.ndarray, subproblems=False):
    """
    Compute an optimal multiplicity function for a symmetric distribution
    with representation `q`. The population size `n` is inferred from the
    length of `q` (i.e., `len(q) - 1`).

    Use the keyword argument `subproblems=true` to return multiplicity
    functions and costs for all subpopulations. In that case, the multiplicity
    functions are the rows of the first value returned. The second value is
    an array of the costs.

    Examples
    --------
    e.g.,
    ```
        multfns, costs = optimal_multfn(q, subpopulations=true)
    ```
    Here `multfns[i, :]` is an optimal multiplicty function for a subpopulation
    of size `i` and `costs[i]` is its cost.
    """
    n = len(q) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    # U[h] is the *expected* number of tests used for a group of size h = 0, …, n
    U = U_from_q(q)

    return compute_optimal_multfn(U, subproblems=subproblems)
