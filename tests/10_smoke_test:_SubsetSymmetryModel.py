import numpy as np
import pytest

from symgt.models import (
    ExchangeableModel,
    IndependentSubpopulationsModel,
    SubsetSymmetryModel,
)

from symgt.utils import (
    subset_symmetry_orbits,
    subset_symmetry_leq,
)

print("THIS IS SMOKE TEST 10: IT TESTS SubsetSymmetryModel")

# Test a reasonable initialization...

orbits = subset_symmetry_orbits((1, 2))
#      = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
alpha = np.ones((len(orbits))) / len(orbits)
m = SubsetSymmetryModel(orbits, alpha)

# Test some invalid initializations...

# bad number orbits
with pytest.raises(ValueError):
    SubsetSymmetryModel([(0, 0)], [1])

# alpha and orbits size mismatch
with pytest.raises(ValueError):
    SubsetSymmetryModel([(0, 0), (1, 0)], [1])

# invalid alpha (negative)
with pytest.raises(ValueError):
    SubsetSymmetryModel([(0, 0), (1, 0)], [2, -1])

# invalid alpha (does not sum to 1)
with pytest.raises(ValueError):
    SubsetSymmetryModel([(0, 0), (1, 0)], [1, 1])

# negative size
with pytest.raises(ValueError):
    SubsetSymmetryModel([(0, 0), (-1, 0)], [1, 0])

# Test some fitting...

# Test basic fit
m = SubsetSymmetryModel.fit(
    (1, 2, 3),
    np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1],
        ]
    ),
)
assert np.allclose(m.sizes, np.array([1, 2, 3]))
alpha_counts = {
    (0, 0, 0): 1,
    (0, 0, 1): 0,
    (0, 0, 2): 0,
    (0, 0, 3): 0,
    (0, 1, 0): 0,
    (0, 1, 1): 1,
    (0, 1, 2): 1,
    (0, 1, 3): 0,
    (0, 2, 0): 0,
    (0, 2, 1): 0,
    (0, 2, 2): 0,
    (0, 2, 3): 0,
    (1, 0, 0): 0,
    (1, 0, 1): 0,
    (1, 0, 2): 0,
    (1, 0, 3): 0,
    (1, 1, 0): 0,
    (1, 1, 1): 0,
    (1, 1, 2): 0,
    (1, 1, 3): 0,
    (1, 2, 0): 0,
    (1, 2, 1): 0,
    (1, 2, 2): 0,
    (1, 2, 3): 1,
}
assert sum(alpha_counts.values()) == 4  # four samples
assert np.allclose(m.alpha, [alpha_counts[o] / 4 for o in m.orbits])
q = np.exp(m.log_q())
assert np.all(q >= 0)
assert np.all(q <= 1)
# test that q is decreasing in size
for i, o in enumerate(m.orbits):
    for j, o2 in enumerate(m.orbits):
        if subset_symmetry_leq(o, o2):
            assert (
                q[i] >= q[j]
            ), f"q[{i}]={q[i]} >= q[{j}]={q[j]}; orbit i={o}; orbit j={o2}"
# computed once and pasted in; included here as a golden to know if the computation
# of q ever change; for checks on correctness of the log_q function , see below
assert np.allclose(
    q,
    [
        1.0,
        0.5,
        0.5,
        0.75,
        0.33333333,
        0.375,
        0.25,
        0.5,
        0.5,
        0.25,
        0.29166667,
        0.25,
        0.33333333,
        0.375,
        0.25,
        0.25,
        0.25,
        0.25,
        0.29166667,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ],
)
np.random.seed(0)
x = np.hstack([m.sample() for i in range(100000)])
assert np.allclose(np.mean(x), m.prevalence(), atol=0.001)

# Test fit all zeros
m = SubsetSymmetryModel.fit(
    (3, 4, 5),
    np.zeros((10, 12)),
)
assert m.sizes == (3, 4, 5)
assert m.alpha[0] == 1
assert np.allclose(m.alpha[1:], np.zeros(len(m.orbits) - 1))
assert np.allclose(np.exp(m.log_q()), np.ones(len(m.orbits)))
assert np.allclose(m.prevalence(), 0)
assert np.allclose(m.sample(), np.zeros(12))

# Test fit all ones
m = SubsetSymmetryModel.fit(
    (3, 1, 3),
    np.ones((10, 7)),
)
assert m.sizes == (3, 1, 3)
assert m.alpha[-1] == 1
assert np.allclose(m.alpha[:-1], np.zeros(len(m.orbits) - 1))
log_q = np.exp(m.log_q())
assert log_q[0] == 1
assert np.allclose(log_q[1:], np.zeros(len(m.orbits) - 1))
assert np.allclose(m.prevalence(), 1)
assert np.allclose(m.sample(), np.ones(7))

# Test log_q by comparison to independent subpopulations...
# easy case
m1 = IndependentSubpopulationsModel(
    [1, 2],
    [
        ExchangeableModel(1, [0.5, 0.5]),  # half time 0, half time 1
        ExchangeableModel(2, [1, 0, 0]),  # all time time 0
    ],
)
orbits = subset_symmetry_orbits((1, 2))
#      = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
alpha = [0.5, 0, 0.5, 0, 0, 0]
m2 = SubsetSymmetryModel(orbits, alpha)
assert np.allclose(m1.log_q(), m2.log_q())

# trickier case
np.random.seed(0)
m1 = IndependentSubpopulationsModel(
    [1, 2, 3, 4, 5],
    [  # note: dirichlet is just a way of sampling uniformly from simplex
        ExchangeableModel(1, np.random.dirichlet(np.ones(1 + 1))),
        ExchangeableModel(2, np.random.dirichlet(np.ones(2 + 1))),
        ExchangeableModel(3, np.random.dirichlet(np.ones(3 + 1))),
        ExchangeableModel(4, np.random.dirichlet(np.ones(4 + 1))),
        ExchangeableModel(5, np.random.dirichlet(np.ones(5 + 1))),
    ],
)
orbits = subset_symmetry_orbits((1, 2, 3, 4, 5))
alpha = np.exp(m1.log_alpha())
m2 = SubsetSymmetryModel(orbits, alpha)
assert np.allclose(m1.log_q(), m2.log_q())
