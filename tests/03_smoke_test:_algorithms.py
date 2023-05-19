import numpy as np
import pytest

from symgt.algorithms import dorfman_pool_size
from symgt.algorithms import dorfman_multfn
import symgt as st

# test other things, ECost etc.
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


def test_dorfman_etc():
    assert st.dorfman_pool_size(0.01) == 11
    assert st.dorfman_pool_size(0.02) == 8
    assert st.dorfman_pool_size(0.05) == 5
    assert st.dorfman_pool_size(0.1) == 4
    assert st.dorfman_pool_size(0.15) == 3
    assert np.all(st.dorfman_multfn(5, 0.15) == [0, 0, 1, 1, 0, 0])


test_dorfman_etc()


def test_U_from_q():
    # w_r = np.array([0, 0.5, 0])
    # w_s = np.array([0, 0, 1.0])
    # these give
    q_r = np.array([1.0, 0.5, 0.0])
    q_s = np.array([1.0, 0.0, 0.0])

    U_r = st.U_from_q(q_r)
    U_s = st.U_from_q(q_s)

    want = np.array([1, 1, 3])

    assert np.all(U_r == want), f"U_r is not {want}"
    assert np.all(U_s == want), f"U_r is not {want}"
    assert np.all(U_r == U_s), "U_r is not U_s"
    # they should match (proof that U is not a representation)

    q = np.exp(st.IIDModel(10, 0.1).log_q())
    U = st.U_from_q(q)

    # 1.38 = 1 + 2(1 - .9^2)
    # 1.813 = 1 + 3(1 - .9^3)
    # 2.3756 = 1 + 4(1 - .9^4)
    # 3.04755 = 1 + 5(1 - .9^5)
    # 3.811354 = 1 + 6(1 - .9^6)
    # 4.6519217 = 1 + 7(1 - .9^7)
    # 5.55626232 = 1 + 8(1 - .9^8)
    # 6.513215599 = 1 + 9(1 - .9^9)
    # 7.513215599 = 1 + 10(1 - .9^10)
    assert np.allclose(
        U,
        np.array(
            [
                1,
                1,
                1.38,
                1.813,
                2.3756,
                3.04755,
                3.811354,
                4.6519217,
                5.55626232,
                6.513215599,
                7.513215599,
            ]
        ),
    )


test_U_from_q()
