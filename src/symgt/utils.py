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
