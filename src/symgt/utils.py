import numpy as np


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


def grouptest_array(multfn) -> np.ndarray:
    """
    Form a matrix that can be used to comput group tests statuses from individual
    status vectors.

    The matrix is `g` by `n` where `g = np.sum(multfn)` is the number of groups
    and `n = np.arange(len(multfn)) * multfn` is the population size. The `i, j`th
    entry of the matrix is 1 if specimen `j` goes to group `i`.

    Large groups first. For example, if there is a group of size 1 and a group of
    size 2, we have [[1, 1, 0], [0, 0, 1]] and *not* [[1, 0, 0], [0, 0, 1]].

    With `multfn` and a vector of outcomes `x`,
    ```
        A = grouptest_array(multfn)
        group_statuses = A @ x
    ```

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
