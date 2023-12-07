import numpy as np

from symgt.algorithms import (
    compute_optimal_multfn,
    compute_optimal_orbit_multfn,
)

print("THIS IS SMOKE TEST 7: IT TESTS compute_optimal_orbit_multfn")

np.random.seed(12032023)

# assuming the group is all permutations
N = 11
diffs = {(i, j): {j - i} for j in range(N) for i in range(j + 1)}
assert len(diffs) == N * (N + 1) / 2

for i in range(10):
    c = np.random.rand(11)
    c[0] = 0  # empty part costs 0

    muwant, outwant = compute_optimal_multfn(c)
    mugot, outgot = compute_optimal_orbit_multfn(c, diffs)

    assert np.allclose(mugot, muwant)
    assert outgot == outwant

# population of size 10 with two equivalence classes of size 5
orbits = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
]
N = len(orbits)


def sub(x, y):
    return (y[0] - x[0], y[1] - x[1])


def leq(x, y):
    return x[0] <= y[0] and x[1] <= y[1]


def lt(x, y):
    return leq(x, y) and x != y


# assert that the orbit obeys ordering...
for i in range(N):
    for j in range(i):
        if lt(orbits[i], orbits[j]):
            assert i < j

diffs = {}
for j in range(N):
    for i in range(j + 1):
        if leq(orbits[i], orbits[j]):
            s = orbits.index(sub(orbits[i], orbits[j]))
            diffs[(i, j)] = {s}

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
