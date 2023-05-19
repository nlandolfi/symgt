import numpy as np


def integer_partition_from_multfn(multfn: np.ndarray) -> np.ndarray:
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
    return np.asarray(ss, dtype=int)


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


def ECost(q: np.ndarray, multfn: np.ndarray) -> float:
    """
    Compute the expected cost of a grouping encoded by `multfn` for a
    symmetric distribution with representation `q`.

    `multfn[i]` is the number of parts of size `i`.
    `q[i]` is the probability that a group of size i tests negative.

    """
    return np.sum([m * ETests(q, h) for (h, m) in enumerate(multfn)])


def ETests(q: np.ndarray, h: int) -> float:
    """
    Compute the expected number of tests used for a group of size `h`
    for a symmetric distribution with representation `q`.

    `h` is the size of the group.
    `q[i]` is the probability that a group of size i tests negative.
    """
    if h == 1:
        return 1
    else:
        return 1 + h * (1 - q[h])
