import numpy as np
import pytest

from symgt.algorithms import (
    symmetric_orbit_multfn,
)

from symgt.models import (
    IIDModel,
    IndependentSubpopulationsModel,
)

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_orbit_diffs,
    U_from_q_orbits,
)

print("THIS IS SMOKE TEST 12: IT TESTS symmetric_orbit_multfn")

# A simple example by hand...
orbits = subset_symmetry_orbits((4, 4))
sizes = [sum(o) for o in orbits]
diffs = subset_symmetry_orbit_diffs(orbits)
N = len(orbits)
q = np.zeros(N)  # all groups always test positive
q[0] = 1  # convention

mu, out = symmetric_orbit_multfn(q, sizes, diffs)
assert orbits[1] == (0, 1)
assert mu[1] == 4
assert orbits[2] == (1, 0)
assert mu[2] == 4
assert out == 8  # cost of 8 individual tests

# A second simple example by hand...
m = IndependentSubpopulationsModel(
    (2, 2),
    [IIDModel(2, 0.01), IIDModel(2, 0.05)],
)
orbits = m.orbits
assert orbits == [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (1, 2),
    (2, 1),
    (2, 2),
]
sizes = [sum(o) for o in orbits]
diffs = subset_symmetry_orbit_diffs(orbits)
q = np.exp(m.log_q())
assert np.allclose(
    q,
    [
        1,
        0.95,
        0.99,
        0.95**2,
        0.95 * 0.99,
        0.99**2,
        0.99 * (0.95**2),
        (0.99**2) * 0.95,
        (0.99**2) * (0.95**2),
    ],
)
c = U_from_q_orbits(q, sizes)
assert np.allclose(
    c,
    [
        1,
        1,
        1,
        1 + 2 * (1 - 0.95**2),
        1 + 2 * (1 - 0.99 * 0.95),
        1 + 2 * (1 - 0.99**2),
        1 + 3 * (1 - 0.99 * (0.95**2)),
        1 + 3 * (1 - (0.99**2) * 0.95),
        1 + 4 * (1 - (0.99**2) * (0.95**2)),
    ],
)
mu, out = symmetric_orbit_multfn(q, sizes, diffs)
assert np.all(mu == [0, 0, 0, 0, 0, 0, 0, 0, 1])
assert np.dot(mu, c) == out
options = [  # all partitions of (2, 2)
    [0, 2, 2, 0, 0, 0, 0, 0, 0],  # (0,1)+(0,1)+(1,0)+(1,0)
    [0, 2, 0, 0, 0, 1, 0, 0, 0],  # (0,1)+(0,1)+(2,0)
    [0, 0, 2, 1, 0, 0, 0, 0, 0],  # (1,0)+(1,0)+(0,2)
    [0, 0, 0, 1, 0, 1, 0, 0, 0],  # (0,2)+(2,0)
    [0, 0, 0, 0, 2, 0, 0, 0, 0],  # (1,1)+(1,1)
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # (0,1)+(1,0)+(1,1)
    [0, 1, 0, 0, 0, 0, 0, 1, 0],  # (0,1)+(2,1)
    [0, 0, 1, 0, 0, 0, 1, 0, 0],  # (1,0)+(1,2)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # (2,2)
]
for option in options:
    assert np.dot(c, option) >= out  # assert mu is optimal

# Test some invalid invocations...

# too few orbits
with pytest.raises(ValueError):
    symmetric_orbit_multfn([1], [1], {})

# q and sizes mismatched number
with pytest.raises(ValueError):
    symmetric_orbit_multfn([1, 2, 3], [1, 2, 3, 4], {})
