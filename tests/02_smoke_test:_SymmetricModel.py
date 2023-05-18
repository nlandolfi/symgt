import numpy as np
import pytest
import symgt as st

print("THIS IS SMOKE TEST 2: IT TESTS SymmetricModel")

SymmetricModel = st.SymmetricModel

m = SymmetricModel(10, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
lq = m.log_q()
assert np.all(np.exp(lq) == 1)  # this model never has positive specimens

m = SymmetricModel(10, np.ones(11) / 11)
lq = m.log_q()
assert np.all(np.diff(np.exp(lq)) < 0)  # should be decreasing
