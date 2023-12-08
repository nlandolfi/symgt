import numpy as np

from symgt.algorithms import (
    compute_optimal_multfn,
    compute_optimal_orbit_multfn,
)

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_lt,
    subset_symmetry_orbits_order_obeying,
    subset_symmetry_orbit_diffs,
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

# population of size 10 with two equivalence classes of size 5
orbits = subset_symmetry_orbits((5, 5))
N = len(orbits)


# assert that the orbit obeys ordering...
for i in range(N):
    for j in range(N):
        if subset_symmetry_lt(orbits[i], orbits[j]):
            assert i < j


print(subset_symmetry_orbits_order_obeying(orbits))
print(subset_symmetry_orbits_order_obeying(subset_symmetry_orbits([3, 3, 2])))

print(N)
diffs = subset_symmetry_orbit_diffs(orbits)
# {}
# for j in range(N):
#     for i in range(j + 1):
#         if subset_symmetry_leq(orbits[i], orbits[j]):
#             s = orbits.index(subset_symmetry_diff(orbits[i], orbits[j]))
#             diffs[(i, j)] = {s}

assert orbits[14] == (2, 2)
assert orbits[35] == (5, 5)
assert orbits[list(diffs[(14, 35)])[0]] == (3, 3)

assert orbits[6] == (1, 0)
assert orbits[24] == (4, 0)
assert orbits[list(diffs[(6, 24)])[0]] == (3, 0)

assert orbits[29] == (4, 5)
assert orbits[35] == (5, 5)
assert orbits[list(diffs[(29, 35)])[0]] == (1, 0)

c = np.ones(N)
c[0] = 0  # empty part costs 0
c[N - 1] = 0  # full set costs 0

mu, out = compute_optimal_orbit_multfn(c, diffs)
assert out == 0
assert np.allclose(
    mu,
    np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ]
    ),
)

# three random tests

# test first time
np.random.seed(0)
c = np.random.rand(N)
c[0] = 0  # empty part costs 0
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(1, 3), (4, 2)}
assert np.allclose(out, 0.5267948062348241)

# test second time
c = np.random.rand(N)
c[0] = 0  # empty part costs 0
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(4, 4), (1, 1)}
assert np.allclose(out, 0.25680783330932333)

# test third time
c = np.random.rand(N)
c[0] = 0  # empty part costs 0
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(4, 1), (1, 4)}
assert np.allclose(out, 0.08425504253627791)

# a test with multiplicities
c = np.ones(N)
c[0] = 0  # empty part costs 0
c[7] = 0  # (1,1) part costs 0
mu, out = compute_optimal_orbit_multfn(c, diffs)
uses = {orbits[i] for i in range(N) if mu[i] > 0}
assert uses == {(1, 1)}
assert mu[7] == 5
assert out == 0
