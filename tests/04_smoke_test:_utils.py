import numpy as np

from symgt.utils import intpart_from_multfn

print("THIS IS SMOKE TEST 4: IT TESTS utils.py")

assert np.allclose(intpart_from_multfn([0, 0, 1]), [2])
assert np.allclose(intpart_from_multfn([0, 0, 2]), [2, 2])
assert np.allclose(intpart_from_multfn([0, 0, 2, 1, 1]), [4, 3, 2, 2])
assert np.allclose(intpart_from_multfn([0, 0, 0]), [])
