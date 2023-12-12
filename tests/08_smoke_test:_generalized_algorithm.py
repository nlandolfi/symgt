import numpy as np

from symgt.algorithms import (
    compute_optimal_multfn,
    compute_optimal_orbit_multfn,
)

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_orbit_diffs,
)

print("THIS IS SMOKE TEST 8: IT TESTS compute_optimal_orbit_multfn")

# test agreement of compute_optimal_orbit_multfn and compute_optimal_multfn
# take a population of size 10, hence 11 orbits; generate orbit differences
N = 11
diffs = {(i, j): {j - i} for j in range(N) for i in range(j + 1)}
assert len(diffs) == N * (N + 1) / 2
# compare on 20 random costs
np.random.seed(0)
for i in range(20):
    c = np.random.rand(11)

    muspecial, outspecial = compute_optimal_multfn(c, subproblems=True)
    mugeneral, outgeneral = compute_optimal_orbit_multfn(c, diffs, subproblems=True)

    assert np.allclose(muspecial, mugeneral)
    assert np.allclose(outspecial, outgeneral)

# now take subset symmetry with two subpopulations of size 5
# set up for some staged and some random tests
orbits = subset_symmetry_orbits((5, 5))
diffs = subset_symmetry_orbit_diffs(orbits)
N = len(orbits)

# stage a clear optimal (5, 5)
c = np.ones(N)
c[N - 1] = 0  # orbit (5, 5)
mu, out = compute_optimal_orbit_multfn(c, diffs)
assert out == 0
assert np.allclose(
    mu,
    np.ones(N) - c,  # all zeros except index N-1
)

# stage a clear optimal split (2, 3) and (3, 2)
c = np.ones(N)
assert orbits[17] == (2, 3)
assert orbits[18] == (3, 2)
c[17] = 0  # orbit (2, 3)
c[18] = 0  # orbit (3, 2)
mu, out = compute_optimal_orbit_multfn(c, diffs)
assert out == 0
assert np.allclose(
    mu,
    np.ones(N) - c,  # all zeros except index 15 and 20
)

# three random tests
np.random.seed(0)

# test once
c = np.random.rand(N)
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(2, 5), (3, 0)}
assert np.allclose(out, 0.5267948062348241)

# test twice
c = np.random.rand(N)
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(1, 2), (4, 3)}
assert np.allclose(out, 0.25680783330932333)

# test thrice
c = np.random.rand(N)
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(0, 4), (5, 1)}
assert np.allclose(out, 0.08425504253627791)

# a test with repeat multiplicities
c = np.ones(N)
c[4] = 0  # (1, 1) part costs 0
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(1, 1)}
assert mu[4] == 5
assert out == 0

# a test with many subpopulations
orbits = subset_symmetry_orbits((5, 5, 5, 5))
N = len(orbits)
diffs = subset_symmetry_orbit_diffs(orbits)
c = np.random.rand(N)
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(2, 0, 3, 1), (2, 3, 1, 4), (1, 2, 1, 0)}
assert np.allclose(out, 0.028811293454162556)
