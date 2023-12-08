import numpy as np

from symgt.algorithms import (
    compute_optimal_orbit_multfn,
)

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_leq,
    subset_symmetry_lt,
    subset_symmetry_diff,
    subset_symmetry_orbits_order_obeying,
    subset_symmetry_orbit_diffs,
)


print("THIS IS SMOKE TEST 7: IT TESTS compute_optimal_orbit_multfn")

subset_symmetry_leq_cases = [
    {"a": (0, 0), "b": (0, 0), "expected": True},
    {"a": (0, 0), "b": (0, 1), "expected": True},
    {"a": (1, 1), "b": (1, 1), "expected": True},
    {"a": (1, 3), "b": (1, 1), "expected": False},
]

for case in subset_symmetry_leq_cases:
    a = case["a"]
    b = case["b"]
    want = case["expected"]
    got = subset_symmetry_leq(a, b)
    assert got == want, f"{a} {b}; got {got}, want {want}"

subset_symmetry_lt_cases = [
    {"a": (0, 0), "b": (0, 0), "expected": False},
    {"a": (0, 0), "b": (0, 1), "expected": True},
    {"a": (1, 1), "b": (1, 1), "expected": False},
    {"a": (1, 3), "b": (1, 1), "expected": False},
    {"a": (1, 3, 4), "b": (1, 5, 5), "expected": True},
]

for case in subset_symmetry_lt_cases:
    a = case["a"]
    b = case["b"]
    want = case["expected"]
    got = subset_symmetry_lt(a, b)
    assert got == want, f"{a} {b}; got {got}, want {want}"

subset_symmetry_diff_cases = [
    {"a": (0, 0), "b": (0, 1), "out": (0, 1)},
    {"a": (3, 2), "b": (5, 5), "out": (2, 3)},
    {"a": (3, 2, 1, 1), "b": (5, 5, 6, 7), "out": (2, 3, 5, 6)},
]

for case in subset_symmetry_diff_cases:
    a = case["a"]
    b = case["b"]
    want = case["out"]
    got = subset_symmetry_diff(a, b)
    assert got == want, f"{a} {b}; got {got}, want {want}"

subset_symmetry_orbits_cases = [
    {"in": (1, 2), "out": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]},
    {
        "in": (2, 4, 3),
        "out": [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 0),
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (0, 3, 0),
            (0, 3, 1),
            (0, 3, 2),
            (0, 3, 3),
            (0, 4, 0),
            (0, 4, 1),
            (0, 4, 2),
            (0, 4, 3),
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 0, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
            (1, 1, 3),
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 2),
            (1, 2, 3),
            (1, 3, 0),
            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),
            (1, 4, 0),
            (1, 4, 1),
            (1, 4, 2),
            (1, 4, 3),
            (2, 0, 0),
            (2, 0, 1),
            (2, 0, 2),
            (2, 0, 3),
            (2, 1, 0),
            (2, 1, 1),
            (2, 1, 2),
            (2, 1, 3),
            (2, 2, 0),
            (2, 2, 1),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 0),
            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),
            (2, 4, 0),
            (2, 4, 1),
            (2, 4, 2),
            (2, 4, 3),
        ],
    },
    {
        "in": (5, 5),
        "out": [
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
        ],
    },
]


for case in subset_symmetry_orbits_cases:
    sizes = case["in"]
    want = case["out"]
    got = subset_symmetry_orbits(sizes)
    assert subset_symmetry_orbits_order_obeying(
        got
    ), "output of subset_symmetry_orbits is not ordered"
    assert got == want, "output of subset_symmetry_orbits is not as expected"


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
