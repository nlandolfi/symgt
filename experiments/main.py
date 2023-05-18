import numpy as np
import pandas as pd
import symgt as st


def matrix(df):
    return np.array(
        [[int(o) for o in sample.split(" ")] for sample in df["sample"]], dtype=int
    )


batches = pd.read_csv("./64_batches_first_two_months.csv")
X = matrix(batches)

m_sym = st.ExchangeableModel.fit(X)
m_iid = st.IIDModel.fit(X)
q_sym = np.exp(m_sym.log_q())
q_iid = np.exp(m_iid.log_q())

mu_sym, cost_sym = st.optimal_multiplicity_function(q_sym)
mu_iid, cost_iid = st.optimal_multiplicity_function(q_iid)
l_sym, l_iid = st.integer_partition(mu_sym), st.integer_partition(mu_iid)
print(st.integer_partition(mu_sym), cost_sym)
print(st.integer_partition(mu_iid), cost_iid)
