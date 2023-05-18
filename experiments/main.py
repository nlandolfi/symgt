import numpy as np
import pandas as pd
import symgt as st


def matrix(df):
    return np.array(
        [[int(o) for o in sample.split(" ")] for sample in df["sample"]], dtype=int
    )


batches = pd.read_csv("./batches.csv")
X = matrix(batches)

m_sym = st.ExchangeableModel.fit(X)
m_iid = st.IIDModel.fit(X)
q_sym = np.exp(m_sym.log_q())
q_iid = np.exp(m_iid.log_q())

mu_sym, _ = st.optimal_multfn(q_sym)
mu_iid, _ = st.optimal_multfn(q_iid)
mu_drf = st.dorfman_multfn(m_iid.n, m_iid.prevalence())
l_sym, l_iid, l_drf = (
    st.integer_partition(mu_sym),
    st.integer_partition(mu_iid),
    st.integer_partition(mu_drf),
)


def sizes(a):  # for compatability with the old julia code
    b = st.integer_partition(a)
    b.reverse()
    return b


def array(multfn):
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


def tests_expended(multfn, samples):
    # samples is N by n
    A = array(multfn)  # n by g
    R = samples @ A  # N by g
    return np.sum(R @ sizes(multfn)) + A.shape[1] * samples.shape[0]


print(m_sym.alpha)
print(l_drf, st.ECost(mu_drf, q_sym), tests_expended(mu_drf, X))
print(l_iid, st.ECost(mu_iid, q_sym), tests_expended(mu_iid, X))
print(l_sym, st.ECost(mu_sym, q_sym), tests_expended(mu_sym, X))
