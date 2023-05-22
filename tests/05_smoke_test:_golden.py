import numpy as np

from symgt.models import IIDModel, ExchangeableModel
from symgt.utils import (
    intpart_from_multfn as intpart,
    ECost,
    empirical_tests_used,
    grouptest_array,
)
from symgt.algorithms import symmetric_multfn

print("THIS IS SMOKE TEST 5: IT REPRODUCES THE OLD julia code golden.jl")

# this is a shuffled version of the data that comes from batching
# Barak et al's runs, only using pools of size 8, from 4/29/2020 to 6/18/2020
X = np.load("./data/X_shuffled.npy")

m_sym = ExchangeableModel.fit(X[:250, :])
m_iid = IIDModel.fit(X[:250, :])
q_sym = np.exp(m_sym.log_q())
q_iid = np.exp(m_iid.log_q())

mu_sym, cost_sym = symmetric_multfn(q_sym)
mu_iid, cost_iid = symmetric_multfn(q_iid)
l_sym, l_iid = intpart(mu_sym), intpart(mu_iid)
# print(intpart(mu_sym), cost_sym)
assert np.allclose(cost_sym, 7.2579209623910526)
assert np.allclose(l_sym, [27, 27, 26])
# print(intpart(mu_iid), cost_iid)
assert np.allclose(cost_iid, 7.830089342189575)
assert np.allclose(l_iid, [20, 20, 20, 20])

assert np.allclose(ECost(q_iid, mu_sym), 8.067076837069752)
assert np.allclose(ECost(q_sym, mu_iid), 7.3396680473136175)


# new code should use intpart_from_multfn in symgt.utils
def sizes(a):  # for compatability with the old julia code
    b = intpart(a)
    b = b[::-1]  # reverse, cause that code did increasing order
    return b


# new code should use grouptest_array in symgt.utils
def array(multfn):  # for compatibility with the old julia code
    num_groups = np.sum(multfn)
    A = np.zeros(
        (num_groups, len(multfn) - 1),
    )  # - 1 since multfn starts at 0
    j = 0
    for i, s in enumerate(sizes(multfn)):
        for k in range(j, j + s):
            A[i, k] = 1
            j += 1
    return A.T  # switch to rows are samples


# old, has bug that R is not binary; see tests_expended_corrected below
def tests_expended(multfn, samples, complain=True):
    if complain:
        raise Exception("WARN: use tests_expended_corrected")

    # samples is N by n
    A = array(multfn)  # n by g
    R = samples @ A  # N by g [this is the buggy line]
    return np.sum(R @ sizes(multfn)) + A.shape[1] * samples.shape[0]


# corrected the bug; but new code should use utils.empirical_tests_used
def tests_expended_corrected(multfn, samples):
    # samples is N by n
    A = array(multfn)  # n by g
    R = (samples @ A > 0).astype(int)  # N by g [this is the corrected line]
    return np.sum(R @ sizes(multfn)) + A.shape[1] * samples.shape[0]


# Test that we have fixed the tests_expended function above
multfn = np.array([0, 0, 1, 1, 0, 0])
samples = np.ones((1, 5))
# the below gives 15 (from 2 + 3*3 + 2*2), but SHOULD give 2 + 3 + 2 = 7
assert tests_expended(multfn, samples, complain=False) == 15
assert tests_expended_corrected(multfn, samples) == 7  # gives 7
# since `samples` is all ones, the order of turning the parts into rows
# of the group test array should not matter, and so the below should match,
# even though we changed the convention to large pools first in the utils code
assert empirical_tests_used(
    grouptest_array(multfn), samples
) == tests_expended_corrected(multfn, samples)


# print(np.prod(X[250:, :].shape))
# print(tests_expended(mu_iid, X[250:, :]))
# print(tests_expended(mu_sym, X[250:, :]))
assert tests_expended(mu_iid, X[250:, :], complain=False) == 1660
assert tests_expended(mu_sym, X[250:, :], complain=False) == 1630
# print(tests_expended_corrected(mu_iid, X[250:, :]))
# print(tests_expended_corrected(mu_sym, X[250:, :]))
assert tests_expended_corrected(mu_iid, X[250:, :]) == 1600
assert tests_expended_corrected(mu_sym, X[250:, :]) == 1550

iids = []
syms = []
for i in range(1000):
    perm = np.random.permutation(80)
    iids.append(tests_expended_corrected(mu_iid, X[250:, perm]))
    syms.append(tests_expended_corrected(mu_sym, X[250:, perm]))
# print(f"iids; mean={np.mean(iids)} std={np.std(iids)}")
# print(f"syms; mean={np.mean(syms)} std={np.std(syms)}")
assert np.isclose(
    np.mean(iids), 1630, 1
), f"np.means(iids) got {np.means(iids)}, want 1630±1"
assert np.isclose(
    np.mean(syms), 1578, 1
), f"np.means(syms) got {np.means(syms)}, want 1578±1"
assert np.isclose(np.std(iids), 21, 2), f"np.std(iids) got {np.std(iids)}, want 21±2"
assert np.isclose(np.std(syms), 30, 2), f"np.std(syms) got {np.std(syms)}, want 30±2"
