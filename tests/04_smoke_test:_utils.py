import numpy as np

from symgt.models import IIDModel
from symgt.utils import (
    intpart_from_multfn,
    U_from_q,
    ECost,
    ETests,
    grouptest_array,
    empirical_tests_used,
)

print("THIS IS SMOKE TEST 4: IT TESTS utils.py")

# Test intpart_from_multfn
assert np.allclose(intpart_from_multfn([0, 0, 1]), [2])
assert np.allclose(intpart_from_multfn([0, 0, 2]), [2, 2])
assert np.allclose(intpart_from_multfn([0, 0, 2, 1, 1]), [4, 3, 2, 2])
assert np.allclose(intpart_from_multfn([0, 0, 0]), [])

# Test U_from_q
# the below are w representations: w[i] = p(x) where nnz(x) = i; see paper
# w_r = np.array([0, 0.5, 0])
# w_s = np.array([0, 0, 1.0])
# these give
q_r = np.array([1.0, 0.5, 0.0])
q_s = np.array([1.0, 0.0, 0.0])

U_r = U_from_q(q_r)
U_s = U_from_q(q_s)

want = np.array([1, 1, 3])

assert np.all(U_r == want), f"U_r is not {want}"
assert np.all(U_s == want), f"U_r is not {want}"
assert np.all(U_r == U_s), "U_r is not U_s"
# they should match (proof that U is not a representation)

# Test ETests (and implicitly, IIDModel, U_from_q)
q = np.array([1, 1, 1])
assert ETests(q, 1) == 1
assert ETests(q, 2) == 1
q = np.array([0, 0, 0])
assert ETests([0, 0, 0], 1) == 1
assert ETests([0, 0, 0], 2) == 3

# Test ECost (and implicitly, IIDModel, U_from_q)
q = np.exp(IIDModel(10, 0.1).log_q())
U = U_from_q(q)
assert np.allclose(ECost(q, [0, 0, 0, 0, 0, 2]), 2 * U[5])
assert np.allclose(ECost(q, [0, 0, 0, 0, 0, 2]), 6.0951)
assert np.allclose(ECost(q, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), U[10])
assert np.allclose(ECost(q, [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0]), 2 * U[2] + 1 * U[6])
assert np.allclose(ECost(q, [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0]), 6.571354)

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

# Test grouptest_array
multfn = [0, 1, 2]
assert np.allclose(
    grouptest_array(multfn),
    np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]),
)
multfn = [0, 0, 0, 1]
assert np.allclose(
    grouptest_array(multfn),
    np.array([[1, 1, 1]]),
)
multfn = [0, 0, 0, 1, 1]
assert np.allclose(
    grouptest_array(multfn),
    np.array([[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]),
)
multfn = np.array([0, 0, 1])
x = np.ones(2)
A = grouptest_array(multfn)
assert np.all(A @ x == np.array([2]))
assert np.all((A @ x > 0).astype(int) == np.array([1]))
multfn = np.array([0, 0, 1, 1])
x = np.ones(5)
A = grouptest_array(multfn)
assert np.all(A @ x == np.array([3, 2]))
assert np.all((A @ x > 0).astype(int) == np.array([1, 1]))

# Test empirical_tests_used
multfn = [0, 0, 0, 2, 1]  # n = 10
A = grouptest_array(multfn)
X = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
# should use 3 + (3 + 4) + (3 + 3) = 16 tests
got, want = empirical_tests_used(A, X), 16
assert got == want, f"empirical tests used got {got}, want {want}"
X = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
# should use (3 + 4 + 3 + 3) + (3 + 4) + (3 + 3) = 26 tests
got, want = empirical_tests_used(A, X), 26
assert got == want, f"empirical tests used got {got}, want {want}"
X = np.ones((3, 10))
# should use 3 * (3 + 4 + 3 + 3) = 39 tests
got, want = empirical_tests_used(A, X), 39
assert got == want, f"empirical tests used got {got}, want {want}"
X = np.zeros((3, 10))
# should use 3 * (3) = 9 tests
got, want = empirical_tests_used(A, X), 9
assert got == want, f"empirical tests used got {got}, want {want}"
