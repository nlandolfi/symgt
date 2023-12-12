import numpy as np
import pytest

from symgt.models import (
    ExchangeableModel,
    IndependentSubpopulationsModel,
)

from symgt.utils import (
    U_from_q_orbits,
)

# Test log_q by comparison to independent subpopulations...
# easy case
m = IndependentSubpopulationsModel(
    [1, 2],
    [
        ExchangeableModel(1, [0.5, 0.5]),  # half time 0, half time 1
        ExchangeableModel(2, [1, 0, 0]),  # all time time 0
    ],
)
assert np.allclose(
    m.orbits,
    [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)],
)
q = np.exp(m.log_q())
assert np.allclose(
    q,
    [1.0, 1.0, 0.5, 1.0, 0.5, 0.5],
)
assert np.allclose(
    U_from_q_orbits(q, np.array([sum(o) for o in m.orbits])),
    [1.0, 1.0, 1.0, 1.0, 2, 2.5],
)

# Test some invalid cases...

# too few orbits
with pytest.raises(ValueError):
    U_from_q_orbits([1], [1])

# mismatched len q and sizes
with pytest.raises(ValueError):
    U_from_q_orbits([1, 0], [1])

# negative sizes
with pytest.raises(ValueError):
    U_from_q_orbits([1, 0], [-1, -1])

# zero size after index 0
with pytest.raises(ValueError):
    U_from_q_orbits([1, 0], [0, 0])
