import numpy as np
import pytest

from symgt.algorithms import (
    dorfman_pool_size,
    dorfman_multfn,
    compute_optimal_multfn,
    symmetric_multfn,
)

print("THIS IS SMOKE TEST 3: IT TESTS algorithms.py")

# Test dorfman_pool_size
assert dorfman_pool_size(0.01) == 11
assert dorfman_pool_size(0.02) == 8
assert dorfman_pool_size(0.05) == 5
assert dorfman_pool_size(0.1) == 4
assert dorfman_pool_size(0.15) == 3

with pytest.raises(ValueError):
    dorfman_pool_size(-0.1)
with pytest.raises(ValueError):
    dorfman_pool_size(0.1, max_pool_size=-1)

# Test dorfman_multfn
assert np.all(dorfman_multfn(5, 0.15) == [0, 0, 1, 1, 0, 0])

# Test compute_optimal_multfn
c = np.ones(11)
c[10] = 0
mu, out = compute_optimal_multfn(c)
assert np.allclose(mu, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
assert out == 0


# Test symmetric_multfn
q = np.array([1.0, 0.5, 0.0])
mus, J = symmetric_multfn(q, subproblems=True)
assert J[0] == 0  # optimal cost of partitioning no individual is 0
assert J[1] == 1  # optimal cost of partitioning 1 individual is 1
assert J[2] == 2  # optimal cost of testing two individuals
assert np.all(np.diff(J) >= 0)  # nondecreasing
assert np.allclose(np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]), mus)
