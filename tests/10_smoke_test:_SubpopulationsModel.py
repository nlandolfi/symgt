import numpy as np

from symgt.models import (
    ExchangeableModel,
    IndependentSubpopulationsModel,
    SubsetSymmetryModel,
)

from symgt.utils import (
    subset_symmetry_orbits,
)

print("THIS IS SMOKE TEST 10: IT TESTS SubsetSymmetryModel")

orbits = subset_symmetry_orbits((1, 2))
#      = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
alpha = np.ones((len(orbits))) / len(orbits)
print(orbits)
print(alpha)

# TODO: test invalid initializations
# num orbits
# alpha nd orbits size
# sum of alpha
# sizes positivity
# test model 1 size doesnt match size 1
# with pytest.raises(ValueError):
#     IndependentSubpopulationsModel([5, 3], [sm1, sm2])

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
print(sorted(m.orbits, key=lambda x: sum(x)))
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
print(np.exp(m.log_q()))
q = np.exp(m.log_q())
for i, o in enumerate(m.orbits):
    print(f"{o}: {q[i]}")

assert np.all(q >= 0)
assert np.all(q <= 0)

# Test fit all zeros
m = SubsetSymmetryModel.fit(
    (3, 4, 5),
    np.zeros((10, 12)),
)
assert m.sizes == (3, 4, 5)
assert m.alpha[0] == 1

# Test fit all zeros
m = SubsetSymmetryModel.fit(
    (3, 4, 5),
    np.zeros((10, 12)),
)
assert m.sizes == (3, 4, 5)
assert m.alpha[0] == 1

# TODO: comparison to independent submodels...
m1 = IndependentSubpopulationsModel(
    [1, 2, 3],
    [
        ExchangeableModel(1, [0.5, 0.5]),
        ExchangeableModel(2, [1, 0, 0]),
        ExchangeableModel(3, [0, 0.4, 0.6, 0]),
    ],
)
# m2 = SubsetSymmetryModel(m1.sizes
#
# )
