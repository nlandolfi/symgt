import numpy as np
import pytest

from symgt.algorithms import dorfman_pool_size
from symgt.algorithms import dorfman_multfn

print("THIS IS SMOKE TEST 3: IT TESTS algorithms.py")

assert dorfman_pool_size(0.01) == 11
assert dorfman_pool_size(0.02) == 8
assert dorfman_pool_size(0.05) == 5
assert dorfman_pool_size(0.1) == 4
assert dorfman_pool_size(0.15) == 3

with pytest.raises(ValueError):
    dorfman_pool_size(-0.1)
with pytest.raises(ValueError):
    dorfman_pool_size(0.1, max_pool_size=-1)

assert np.all(dorfman_multfn(5, 0.15) == [0, 0, 1, 1, 0, 0])
