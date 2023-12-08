import numpy as np

from symgt.algorithms import (
    compute_optimal_multfn,
    compute_optimal_orbit_multfn,
)


# test compute_optimal_orbit_multfn
# should agree with compute_optimal_multfn

# take a population of size 10, hence 11 orbits; generate orbit differences
N = 11
diffs = {(i, j): {j - i} for j in range(N) for i in range(j + 1)}
assert len(diffs) == N * (N + 1) / 2

for i in range(20):
    c = np.random.rand(11)

    muspecial, outspecial = compute_optimal_multfn(c, subproblems=True)
    mugeneral, outgeneral = compute_optimal_orbit_multfn(c, diffs, subproblems=True)

    assert np.allclose(muspecial, mugeneral)
    assert np.allclose(outspecial, outgeneral)
