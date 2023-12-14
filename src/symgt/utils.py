import itertools
from typing import Sequence

import numpy as np


def dorfman_cost(prevalence: float, poolsize: int):
    """
    Compute the cost (expected number of tests per individual) according
    to Dorfman's fixed-prevalence and infinite population analysis.

    This function is a helper for `dorfman_pool_size` below.
    """
    if not (0.0 <= prevalence <= 1.0):
        raise ValueError(f"prevalence={prevalence} must be in [0, 1]")

    if not (poolsize > 0):
        raise ValueError(f"poolsize={poolsize} should be > 0")

    p, m = float(prevalence), float(poolsize)
    return 1 / m + 1 - (1 - p) ** m


def dorfman_pool_size(prevalence: float, max_pool_size: int = 100) -> int:
    """
    Compute the optimal pool size according to Dorfman's infinite analysis
    where `prevalence` is the population prevalence rate.

    In other words, minimize `1/m + 1 - (1-prevalence)^m` with respect to
    the pool size `m`.

    This function is a helper for `dorfman_multfn` in ./algorithms.py.
    """
    if not (0.0 <= prevalence <= 1.0):
        raise ValueError(f"prevalence={prevalence} must be in [0, 1]")

    if not (max_pool_size > 1):
        raise ValueError(f"max_pool_size={max_pool_size} should be > 1")

    costs = [dorfman_cost(prevalence, m) for m in range(1, max_pool_size + 1)]

    m = int(np.argmin(costs)) + 1  # off by one indexing

    if 1 / m + 1 - (1 - prevalence) ** m > 1:
        m = 1  # no pooling
    if m == max_pool_size:
        print("WARNING: m == max_pool_size; might need to increase max_pool_size")

    return m


def intpart_from_multfn(multfn: np.ndarray) -> np.ndarray:
    """
    Convert a multiplicity function to an integer partition.
    An integer partition is a nondescreasing list of (possibly repeating) part sizes.

    Parameters
    ----------
    multfn : np.ndarray
        `multfn[i]` is the multiplicty of a part of size `i`.
    """
    ss = []
    for i, x in enumerate(multfn):
        for _ in range(x):
            ss.append(i)
    ss.reverse()  # conventionally nonincreasing
    return np.asarray(ss, dtype=int)


def ETests(q: np.ndarray, h: int) -> float:
    """
    Compute the expected number of tests used for a group of size `h` under the
    symmetric distribution represented by `q`.

    Parameters
    ----------
    q : np.ndarray
        `q[i]` is the probability that a group of size `i` tests negative.
    h : int
        `h` is the size of the group.
    """
    if h == 1:
        return 1
    else:
        return 1 + h * (1 - q[h])


def ECost(q: np.ndarray, multfn: np.ndarray) -> float:
    """
    Compute the expected cost of a grouping encoded by `multfn` under the
    symmetric distribution represented by `q`.

    Parameters
    ----------
    multfn : np.ndarray
        `multfn[i]` is the number of parts of size `i`.
    q : np.ndarray
        `q[i]` is the probability that a group of size i tests negative.
    """
    return np.sum([m * ETests(q, h) for (h, m) in enumerate(multfn)])


def U_from_q(q: np.ndarray) -> np.ndarray:
    """
    Compute the function `U` under the symmetric distribution represented by `q`.
    `U[h]` is the expected number of tests used to declare a group of size `h`.

    Parameters
    ----------
    q : np.ndarray
        `q[i]` is the probability that a group of size `i` tests negative.
    """
    n = len(q) - 1

    if not (n > 0):
        raise ValueError(f"population size n={n} should be > 0")

    U = np.zeros(n + 1)
    U[0] = 1
    U[1] = 1
    for i in range(2, n + 1):
        U[i] = 1 + i * (1 - q[i])  # don't bother calling ETests
    return U


def U_from_q_orbits(q: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """
    Compute the function `U` under the symmetric distribution represented by `q`.
    `U[i]` is the expected number of tests used to declare a group in orbit `i`.

    For the fully symmetric case, see the `U_from_q` helper function.

    Parameters
    ----------
    q : np.ndarray
        `q[i]` is the probability that a group of orbit `i` tests negative.
    sizes : np.ndarray
        `sizes[i]` is the size of orbit `i`.
    """
    N = len(q)
    if N < 2:
        raise ValueError("q must have at least two elements")

    sizes = np.asarray(sizes)
    if len(sizes) != N:
        raise ValueError("q and sizes must have the same length")
    if not np.all(sizes >= 0):
        raise ValueError("sizes must be nonnegative")
    if not np.all(sizes[1:] >= 1):
        raise ValueError("all but first orbit size must be at least 1")

    U = np.zeros(N)
    U[0] = 1  # by convention
    for i in range(1, N):
        if sizes[i] == 1:
            U[i] = 1
        else:
            U[i] = 1 + sizes[i] * (1 - q[i])

    return U


def grouptest_array(multfn: np.ndarray) -> np.ndarray:
    """
    Form a matrix that can be used to compute the number of positives per
    group (and, hence, group statuses) from individual status vectors.

    The matrix is `g` by `n` where `g = np.sum(multfn)` is the number of groups
    and `n = np.dot(np.arange(len(multfn)), multfn)` is the population size.
    The `i, j`th entry of the matrix is 1 if specimen `j` goes to group `i`.

    Large groups first. For example, if there is a group of size 1 and a group of
    size 2, we have [[1, 1, 0], [0, 0, 1]] and *not* [[1, 0, 0], [0, 1, 1]].

    With `multfn` and a vector of outcomes `x`,
    ```
        A = grouptest_array(multfn)
        positives_per_group = A @ x
        group_statuses = (positives_per_group > 0).astype(int)
    ```
    Notice that `positives_per_group` may *not* be a binary vector. For example
    `A = np.array([[1, 1]])` and `x = np.array([1, 1])` will give `np.array([2])`.

    Parameters
    ----------
    multfn : np.ndarray
        `multfn[i]` is the multiplicty of a part of size `i`.

    Returns
    -------
    A : np.ndarray
        `A[i, j]` is 1 if and only if sample `j` is in pool `i`
    """
    assert multfn[0] == 0, f"multfn[0] should be zero, got {multfn[0]}"

    g, n = np.sum(multfn), np.dot(np.arange(len(multfn)), multfn)
    s = intpart_from_multfn(multfn)
    A = np.zeros((g, n))

    cum_s = np.insert(np.cumsum(s), 0, 0)
    for i in range(len(s)):
        A[i, cum_s[i] : cum_s[i + 1]] = 1
    return A


def empirical_tests_used(A: np.ndarray, X: np.ndarray) -> int:
    """
    Compute the number of tests used where `A` is a 2D array of group
    test assignments (see `grouptest_array`) and `X` is a 2D array
    whose rows are individual status vectors (the data matrix).

    Examples
    --------
    With `multfn` and data matrix `X`,
    ```
        A = grouptest_array(multfn)
        empirical_tests_used(A, X)
    ```

    Parameters
    ----------
    A : np.ndarray
        `A[i, j]` is 1 if specimen `j` is included in pool `i`.
    X : np.ndarray
        `X[i, :]` is the `i`th sample of individual status vectors.

    Returns
    -------
    int
        The number of tests used.
    """
    if not (np.all(np.isin(A, [0, 1]))):
        raise ValueError("group test array should be a binary matrix")
    if not (np.all(np.isin(X, [0, 1]))):
        raise ValueError("sample array should be a binary matrix")

    g, n = A.shape
    if not np.all(np.ones(n) == np.sum(A, axis=0)):
        raise ValueError("unexpected: group test array has overlapping pools")

    N, n2 = X.shape
    if n != n2:
        raise ValueError(
            f"differing population size: group test array has {n} \
                    and sample matrix has {n2}"
        )

    # a g-vector
    group_sizes_vector = np.sum(A, axis=1)
    if np.sum(group_sizes_vector) != n:
        raise ValueError(
            f"group sizes should sum to {n}, \
                got {np.sum(group_sizes_vector)}"
        )

    # an N x g matrix
    group_statuses_matrix = (X @ A.T > 0).astype(int)

    # first term is group tests, second is individual retests
    used = (g * N) + np.sum(group_statuses_matrix @ group_sizes_vector)

    assert used == int(used), f"result should be an integer, but got {used}"

    return int(used)


def subset_symmetry_orbits(sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """
    Computes a list of subset symmetry orbits for the subpopulation sizes.

    Examples
    --------
    To get the orbits for a population of size 3, one population of size 1
    and another population of size 2.
    ```
        orbits = subset_symmetry_orbits([1, 2])
    ```
    Here, `orbits` will be [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

    Parameters
    ----------
    sizes : Iterable[int]
        The subpopulation sizes.

    Returns
    -------
    list[tuple[int, ...]]
        The subset symmetry orbits.
    """
    if len(sizes) == 0:
        raise ValueError("sizes cannot be empty")

    out = list(itertools.product(*[range(s + 1) for s in sizes]))
    out.sort(key=sum)

    return out


def subset_symmetry_leq(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """
    Checks whether (orbit a) ≼ (orbit b).

    Examples
    --------
    ```
        subset_symmetry_leq((0,0), (0,1)) # True
        subset_symmetry_leq((1,1), (1,1)) # True
        subset_symmetry_leq((1,3), (1,1)) # False
    ```

    Parameters
    ----------
    a : tuple[int, ...]
        The first orbit.
    b : tuple[int, ...]
        The second orbit.

    Returns
    -------
    bool
        Whether (orbit a) ≼ (orbit b).
    """
    return all(x <= y for x, y in zip(a, b))


def subset_symmetry_lt(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """
    Checks whether (orbit a) ≺ (orbit b).

    Examples
    --------
    ```
        subset_symmetry_lt((0,0), (0,1)) # True
        subset_symmetry_lt((1,1), (1,1)) # False
        subset_symmetry_lt((1,3), (1,1)) # False
    ```

    Parameters
    ----------
    a : tuple[int, ...]
        The first orbit.
    b : tuple[int, ...]
        The second orbit.

    Returns
    -------
    bool
        Whether (orbit a) ≺ (orbit b).
    """
    return subset_symmetry_leq(a, b) and a != b


def subset_symmetry_diff(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compute the difference of the two orbits. In particular b - a.

    Examples
    --------
    ```
        subset_symmetry_diff((0,0), (0,1)) # (0,1)
        subset_symmetry_diff((3,2), (5,5)) # (2,3)
        subset_symmetry_diff((3,2,1,1), (5,5,6,7)) # (2,3,5,6)
    ```
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    if not subset_symmetry_leq(a, b):
        raise ValueError("a must precede or equal b")

    return tuple(b[i] - a[i] for i in range(len(a)))


def subset_symmetry_orbits_order_obeying(orbits: list[tuple[int, ...]]) -> bool:
    """
    Checks whether the `orbits` are ordered as required for the
    dynamic programming algorithm. In other words, whether orbit i
    precedes (but is not equal to) orbit j, implies i < j.

    Examples
    --------
    ```
        subset_symmetry_orbits_order_obeying([(0,0), (0,1), (1,0), (1,1)]) # True
        subset_symmetry_orbits_order_obeying([(0,0), (1,0), (0,1), (1,1)]) # True
        subset_symmetry_orbits_order_obeying([(0,1), (0,0), (1,0), (1,1)]) # False
    ```

    Parameters
    ----------
    x : list[tuple[int, ...]]
        The orbits.

    Returns
    -------
    bool
        Whether the ordering of the orbits is valid.
    """
    for i in range(len(orbits)):
        for j in range(len(orbits)):
            if subset_symmetry_lt(orbits[i], orbits[j]) and not (i < j):
                return False
    return True


def subset_symmetry_orbit_diffs(
    orbits: list[tuple[int, ...]]
) -> dict[tuple[int, int], set[int]]:
    """
    Compute the differences between all orbits in the list.
    Returns the *indexes* of the differences.

    Examples
    --------
    ```
        diffs = subset_symmetry_orbits_ordered([(0,0), (0,1), (1,0), (1,1)])
    ```
    Here `diffs` is a dictionary where `diffs[(i, j)]` is the singleton set of
    differences between orbit at index `j` and orbit at index `i`.

    Parameters
    ----------
    orbits : list[tuple[int, ...]]
        The orbits.

    Returns
    -------
    dict[tuple[int, int], set[int]]
        The differences.
    """
    diffs = {}

    orbit_to_idx = {}
    for i, o in enumerate(orbits):
        orbit_to_idx[o] = i

    for j in range(len(orbits)):
        for i in range(j + 1):
            if subset_symmetry_leq(orbits[i], orbits[j]):
                s = orbit_to_idx[subset_symmetry_diff(orbits[i], orbits[j])]
                diffs[(i, j)] = {s}

    return diffs


def subset_symmetry_multpart_from_multfn(
    orbits: list[tuple[int, ...]], multfn: np.ndarray
) -> np.ndarray:
    """
    Construct the multipartition of `orbits[-1]` given the
    orbit multiplicities `multfn`.

    The number of parts `g` is `sum(multfn)`.
    The number of subpopulations `m` is `len(orbits[0])`

    Examples
    --------
    ```
        multpart = subset_symmetry_multpart_from_multfn(orbits, multfn)
    ```
    Here multpart[i, :] is sizes of part i in the partition.
    The order of the parts returned is reversed from that in `multfn`,
    which matches the behavor of `intpart_from_multfn`.

    For the fully symmetric case, see `intpart_from_multfn`.

    Parameters
    ----------
    orbits : list[tuple[int, ...]]
        The orbits.
    multfn : np.ndarray
        The orbit multiplicities.

    Returns
    -------
    np.ndarray
        The multipartition as a `g` by `m` matrix.
    """
    if multfn[0] != 0:
        raise ValueError("multfn may not include empty part")

    multfn_ints = np.asarray(multfn).astype(int)  # ensure ints

    if not np.allclose(multfn_ints, multfn):
        raise ValueError("multfn must only contain integers")

    g = np.sum(multfn_ints)
    m = len(orbits[0])
    p = np.zeros((g, m))
    o = 0
    for i, c in enumerate(multfn_ints):
        for _ in range(c):
            p[g - 1 - o, :] = orbits[i]
            o += 1

    if not np.allclose(np.sum(p, axis=0), orbits[-1]):
        raise ValueError("multfn parts should sum to orbits[-1]")

    return p


def subset_symmetry_grouptest_array(
    orbits: list[tuple[int, ...]], multfn: np.ndarray
) -> np.ndarray:
    """
    Form a matrix that can be used to compute the number of positives per
    group (and, hence, group statuses) from individual status vectors.

    The matrix is `g` by `n` where `g = np.sum(multfn)` is the number of groups
    and `n = np.sum(orbits[-1])` is the population size.
    The `i, j`th entry of the matrix is 1 if specimen `j` goes to group `i`.

    For the fully symmetric case, see `grouptest_array`.
    """
    multfn_ints = np.asarray(multfn).astype(int)  # ensure ints

    if not np.allclose(multfn_ints, multfn):
        raise ValueError("multfn must only contain integers")

    g = np.sum(multfn_ints)
    m = len(orbits[0])
    n = int(np.sum(orbits[-1]))
    A = np.zeros((g, n), dtype=int)
    offsets = np.insert(np.cumsum(orbits[-1]), 0, 0).astype(int)
    mp = subset_symmetry_multpart_from_multfn(orbits, multfn_ints)
    cum_s = np.insert(
        np.cumsum(mp, axis=0),
        0,
        (0,) * m,
        axis=0,
    ).astype(int)
    for i in range(g):
        for j in range(m):
            A[i, offsets[j] + cum_s[i, j] : offsets[j] + cum_s[i + 1, j]] = 1

    assert np.sum(A) == n, "sanity: A should sum to n"
    assert np.allclose(
        np.sum(A, axis=1), np.sum(mp, axis=1)
    ), "sanity: A rows should sum to multpart rows"

    return A
