import numpy as np

from .utils import U_from_q, U_from_q_orbits, dorfman_pool_size


def dorfman_multfn(n: int, prevalence: float) -> np.ndarray:
    """
    Compute a multiplicity function using Dorfman's infinite analysis,
    adding a pool of irregular size if the indicated pool size does not
    divide evenly into `n`.

    Uses `dorfman_pool_size` from ./utils.py.
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
    are the rows of the first value returned. The second value is the costs.

    See also `compute_optimal_orbit_multfn` for the generalization to
    an arbitrary subgroup of the permutation group.

    Examples
    --------
    To just get the solution for `n = len(c) - 1`:
    ```
        multfn, cost = compute_optimal_multfn(c)
    ```
    Here `multfn` is an nd.array and `cost` is a float.

    To get solutions and costs for all subproblems:
    ```
        multfns, costs = compute_optimal_multfn(c, subpopulations=true)
    ```
    Here `multfns[i, :]` is an optimal multiplicity function for a
    subpopulation of size `i` and `costs[i]` is its cost.
    """
    # c[i] is the cost of a part of size i = 0, …, n
    n = len(c) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    # Mstar[m] is the optimal cost to partition m = 0, …, n (the value function)
    Mstar = np.zeros(n + 1)  # note, Mstar[0] = 0 by default

    # The cost of partitioning 1 is c[1]
    Mstar[1] = c[1]

    # the n+1 here is for indexing convenience, we don't use istar[0] or multfns[0, :]
    istar = np.zeros(n + 1, dtype=int)
    multfns = np.zeros((n + 1, n + 1), dtype=int)

    for k in range(1, n + 1):
        # find an optimal istar[k]
        istar[k] = np.argmin([Mstar[k - i] + c[i] for i in range(1, k + 1)]) + 1

        # record the optimal cost
        Mstar[k] = Mstar[k - istar[k]] + c[istar[k]]

        if k - istar[k] > 0:  # if we are using a subproblem
            multfns[k, :] = multfns[k - istar[k], :]  # take its multfn
        # otherwise, inherit the zero pattern

        # update the multfn to include a part of size istar[k]
        multfns[k, istar[k]] += 1

    # row i should be a be a multfn for i
    w = np.arange(0, n + 1)
    got, want = multfns @ w, w
    assert np.all(got == want), f"multfns @ np.arange(1, n+1): got {got} want {want}"

    if subproblems:
        return multfns, Mstar
    else:
        return multfns[n, :], Mstar[n]


def symmetric_multfn(q: np.ndarray, subproblems=False):
    """
    Compute an optimal multiplicity function for a symmetric distribution
    with representation `q`. The population size `n` is inferred from the
    length of `q` (i.e., `len(q) - 1`).

    The keyword argument `subproblems=true` behaves as in `compute_optimal_multfn`.

    Examples
    --------
    e.g.,
    ```
        multfns, costs = symmetric_multfn(q, subproblems=true)
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


def compute_optimal_orbit_multfn(c: np.ndarray, diffs: dict, subproblems=False):
    """
    Compute an optimal *orbit* multiplicity function for cost c where `c[i]`
    is the cost of *orbit* `i`. Here `diffs` is a dictionary containing the
    orbit differences. In particular, `diffs[(i,j)] = (orbit j) ∖ (orbit i)`.
    This value `diffs[(i,j)]` is defined only when (orbit i) ≼ (orbit j).

    The number of orbits `N` is inferred from `c` (i.e., `N = len(c)`).
    It is assumed that (orbit i) ≺ (orbit j) implies `i < j`.

    We use dynamic programming. We do not compute all optimal orbit
    multiplicity functions, just a single one.

    Use the keyword argument `subproblems=true` to return orbit multiplicity
    functions and costs for all subproblems. The orbit multiplicity functions
    are the rows of the first value returned. The second value is the costs.

    See also `compute_optimal_multfn` for the special case when the group
    is the group of *all* permutations.

    For the subset symmetry case, see the helper functions in `symgt.utils`
    with the prefix `subset_symmetry_*`.

    Examples
    --------
    To just get the solution for orbit `[P]`, identified with index `N-1`:
    ```
        multfn, cost = compute_optimal_orbit_multfn(c, diffs)
    ```
    Here `multfn` is an nd.array and `cost` is a float.

    To get solutions and costs for all subproblems:
    ```
        multfns, costs = compute_optimal_multfn(c, diffs, subpopulations=true)
    ```
    Here `multfns[i, :]` is an optimal orbit multiplicity function for a
    subpopulation in orbit `i` and `costs[i]` is its cost.
    """
    N = len(c)

    if not (N >= 2):
        raise ValueError(f"number of orbits N should be >= 2 ([∅] and [P]), got {N}")

    for val in diffs.values():
        for x in val:
            if type(x) != int or x < 0 or x >= N:
                raise ValueError(
                    f"diffs should only contain elements from 0 to N-1, got {x}"
                )

    Mstar = np.zeros(N)
    istar = np.zeros(N, dtype=int)
    dstar = np.zeros(N, dtype=int)
    multfns = np.zeros((N, N), dtype=int)

    for k in range(1, N):
        candidates = []
        for i in range(1, k + 1):
            if (i, k) in diffs:
                for d in diffs[(i, k)]:
                    candidates.append((i, d))

        # select the optimal subproblem
        istar[k], dstar[k] = candidates[
            np.argmin([Mstar[d] + c[i] for (i, d) in candidates])
        ]

        # record the optimal cost
        Mstar[k] = Mstar[dstar[k]] + c[istar[k]]

        if dstar[k] > 0:  # if we are using a nontrivial subproblem
            multfns[k, :] = multfns[dstar[k], :]  # take its multfn
        # otherwise, inherit the constant zero multiplicity function

        # update the multfn to include a part in orbit istar[k]
        multfns[k, istar[k]] += 1

    if subproblems:
        return multfns, Mstar
    else:
        return multfns[N - 1, :], Mstar[N - 1]


def symmetric_orbit_multfn(
    q: np.ndarray, sizes: np.ndarray, diffs: dict, subproblems=False
):
    """
    Compute an optimal multiplicity function for a symmetric distribution
    with representation `q`. The number of orbits is inferred from the
    length of `q` (i.e., `len(q)`).

    Here `sizes[i]` is the size of subsets in the orbit `i`.

    Here `diffs` is a dictionary containing orbit differences. In particular,
    `diffs[(i, j)] = (orbit j) ∖ (orbit i)`.  This value `diffs[(i, j)]` is
    defined only when (orbit i) ≼ (orbit j).  It is assumed that the orbit
    order obeys the condition (orbit i) ≺ (orbit j) implies `i < j`.

    The keyword argument `subproblems=true` behaves as it does
    in the `compute_optimal_orbit_multfn` function.

    For the fully symmetric case, see the `symmetric_multfn` helper function.

    Examples
    --------
    e.g.,
    ```
        multfns, costs = symmetric_orbit_multfn(q, sizes, diffs, subproblems=true)
    ```
    Here `multfns[i, :]` is an optimal orbit multiplicty function for the
    orbit `i` and `costs[i]` is its cost.
    """
    if len(q) < 2:
        raise ValueError(f"number of orbits N={len(q)} should be >= 2")
    if len(q) != len(sizes):
        raise ValueError("q and sizes must have the same length")

    # U[i] is the *expected* number of tests used for a group of orbit i
    U = U_from_q_orbits(q, sizes)

    return compute_optimal_orbit_multfn(U, diffs, subproblems=subproblems)
