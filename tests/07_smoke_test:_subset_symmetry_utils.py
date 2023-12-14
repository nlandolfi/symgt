import numpy as np
import pytest

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_leq,
    subset_symmetry_lt,
    subset_symmetry_diff,
    subset_symmetry_orbits_order_obeying,
    subset_symmetry_orbit_diffs,
    subset_symmetry_multpart_from_multfn,
    subset_symmetry_grouptest_array,
)


print("THIS IS SMOKE TEST 7: IT TESTS subset symmetry orbit utils")

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
    {"in": (1, 2), "out": [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]},
    {
        "in": (2, 4, 3),
        "out": [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, 2),
            (0, 1, 1),
            (0, 2, 0),
            (1, 0, 1),
            (1, 1, 0),
            (2, 0, 0),
            (0, 0, 3),
            (0, 1, 2),
            (0, 2, 1),
            (0, 3, 0),
            (1, 0, 2),
            (1, 1, 1),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (0, 1, 3),
            (0, 2, 2),
            (0, 3, 1),
            (0, 4, 0),
            (1, 0, 3),
            (1, 1, 2),
            (1, 2, 1),
            (1, 3, 0),
            (2, 0, 2),
            (2, 1, 1),
            (2, 2, 0),
            (0, 2, 3),
            (0, 3, 2),
            (0, 4, 1),
            (1, 1, 3),
            (1, 2, 2),
            (1, 3, 1),
            (1, 4, 0),
            (2, 0, 3),
            (2, 1, 2),
            (2, 2, 1),
            (2, 3, 0),
            (0, 3, 3),
            (0, 4, 2),
            (1, 2, 3),
            (1, 3, 2),
            (1, 4, 1),
            (2, 1, 3),
            (2, 2, 2),
            (2, 3, 1),
            (2, 4, 0),
            (0, 4, 3),
            (1, 3, 3),
            (1, 4, 2),
            (2, 2, 3),
            (2, 3, 2),
            (2, 4, 1),
            (1, 4, 3),
            (2, 3, 3),
            (2, 4, 2),
            (2, 4, 3),
        ],
    },
    {
        "in": (5, 5),
        "out": [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 1),
            (4, 0),
            (0, 5),
            (1, 4),
            (2, 3),
            (3, 2),
            (4, 1),
            (5, 0),
            (1, 5),
            (2, 4),
            (3, 3),
            (4, 2),
            (5, 1),
            (2, 5),
            (3, 4),
            (4, 3),
            (5, 2),
            (3, 5),
            (4, 4),
            (5, 3),
            (4, 5),
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


subset_symmetry_orbit_diffs_cases = [
    {
        "in": subset_symmetry_orbits([1, 2]),
        # [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
        "out": {
            (0, 0): {0},  # (0,0) - (0,0) = (0,0)
            (0, 1): {1},  # (0,1) - (0,0) = (0,1)
            (1, 1): {0},  # (0,1) - (0,1) = (0,0)
            (0, 2): {2},  # (1,0) - (0,0) = (1,0)
            (2, 2): {0},  # (1,0) - (1,0) = (0,0)
            (0, 3): {3},  # (0,2) - (0,0) = (0,2)
            (1, 3): {1},  # (0,2) - (0,1) = (0,1)
            (3, 3): {0},  # (0,2) - (0,2) = (0,0)
            (0, 4): {4},  # (1,1) - (0,0) = (1,1)
            (1, 4): {2},  # (1,1) - (0,1) = (1,0)
            (2, 4): {1},  # (1,1) - (1,0) = (0,1)
            (4, 4): {0},  # (1,1) - (1,1) = (0,0)
            (0, 5): {5},  # (1,2) - (0,0) = (1,2)
            (1, 5): {4},  # (1,2) - (0,1) = (1,1)
            (2, 5): {3},  # (1,2) - (1,0) = (0,2)
            (3, 5): {2},  # (1,2) - (0,2) = (1,0)
            (4, 5): {1},  # (1,2) - (1,1) = (0,1)
            (5, 5): {0},  # (1,2) - (1,2) = (0,0)
        },
    }
]

for case in subset_symmetry_orbit_diffs_cases:
    sizes = case["in"]
    want = case["out"]
    got = subset_symmetry_orbit_diffs(sizes)
    assert got == want, "output of subset_symmetry_orbit_diffs is not as expected"


# some additional particular tests
orbits = subset_symmetry_orbits((5, 5))
diffs = subset_symmetry_orbit_diffs(orbits)

assert orbits[12] == (2, 2)
assert orbits[35] == (5, 5)
assert orbits[list(diffs[(12, 35)])[0]] == (3, 3)

assert orbits[2] == (1, 0)
assert orbits[14] == (4, 0)
assert orbits[list(diffs[(2, 14)])[0]] == (3, 0)

assert orbits[33] == (4, 5)
assert orbits[35] == (5, 5)
assert orbits[list(diffs[(33, 35)])[0]] == (1, 0)

# some tests of subset_symmetry_multpart_from_multfn...

orbits = subset_symmetry_orbits((1, 3))
# = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (0, 3), (1, 2), (1, 3)]
mp = subset_symmetry_multpart_from_multfn(
    orbits,
    [0, 0, 0, 0, 0, 0, 0, 1],  # (1,3)
)
assert np.allclose(mp, np.array([[1, 3]]))
mp = subset_symmetry_multpart_from_multfn(
    orbits,
    [0, 1, 0, 0, 0, 0, 1, 0],  # (1,2)+(0,1)
)
assert np.allclose(mp, np.array([[1, 2], [0, 1]]))
mp = subset_symmetry_multpart_from_multfn(
    orbits,
    [0, 2, 0, 0, 1, 0, 0, 0],  # (1,1)+(0,1)+(0,1)
)
assert np.allclose(mp, np.array([[1, 1], [0, 1], [0, 1]]))

# test some invalid invocations of subset_symmetry_multpart_from_multfn...

# includes empty part
with pytest.raises(ValueError):
    subset_symmetry_multpart_from_multfn(
        orbits,
        [1, 0, 0, 0, 0, 0, 0, 0],
    )

# does not sum to (1, 3)
with pytest.raises(ValueError):
    subset_symmetry_multpart_from_multfn(
        orbits,
        [0, 0, 0, 0, 0, 0, 0, 2],
    )

# nonintegral multfn
with pytest.raises(ValueError):
    subset_symmetry_multpart_from_multfn(
        orbits,
        [0, 2.5, 1.5, 0, 0, 0, 0, 0],
    )

# some tests of subset_symmetry_grouptest_array...

orbits = subset_symmetry_orbits((1, 3))
# = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (0, 3), (1, 2), (1, 3)]
A = subset_symmetry_grouptest_array(
    orbits,
    [0, 0, 0, 0, 0, 0, 0, 1],  # (1,3)
)
assert np.allclose(A, np.array([[1, 1, 1, 1]]))
A = subset_symmetry_grouptest_array(
    orbits,
    [0, 1, 0, 0, 0, 0, 1, 0],  # (1,2)+(0,1)
)
assert np.allclose(A, np.array([[1, 1, 1, 0], [0, 0, 0, 1]]))
A = subset_symmetry_grouptest_array(
    orbits,
    [0, 3, 1, 0, 0, 0, 0, 0],  # (1,0)+(0,1)+(0,1)+(0,1)
)
assert np.allclose(A, np.eye(4))  # all individual tests

# test nonintegral multfn...
with pytest.raises(ValueError):
    subset_symmetry_grouptest_array(
        orbits,
        [0, 2.5, 1.5, 0, 0, 0, 0, 0],
    )

# slightly more complex example
orbits = subset_symmetry_orbits((2, 3, 4, 5))  # n = 14
mf = np.zeros(len(orbits))
mf[orbits.index((1, 0, 3, 1))] = 1
mf[orbits.index((0, 3, 0, 2))] = 1
mf[orbits.index((1, 0, 1, 0))] = 1
mf[orbits.index((0, 0, 0, 2))] = 1
A = subset_symmetry_grouptest_array(orbits, mf)
assert np.allclose(
    A,
    np.array(
        [
            [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
    ),
)
