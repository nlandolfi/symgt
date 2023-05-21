import numpy as np


from symgt import models
from symgt import algorithms
from symgt.utils import intpart_from_multfn, ECost

print("THIS IS SMOKE TEST 6: IT REPRODUCES a different golden")

# this is the data that comes from batching Barak et al' runs,
# only using pools of size 8, from 4/29/2020 to 6/18/2020
X = np.load("./data/X.npy")


m_iid = models.IIDModel.fit(X)
m_sym = models.ExchangeableModel.fit(X)
q_iid = np.exp(m_iid.log_q())
q_sym = np.exp(m_sym.log_q())
# print(f"m_iid n={m_iid.n} p={m_iid.p}")
# print(f"m_sym n={m_sym.n} alpha[:10]={m_sym.alpha[:10]}...")

multfn_iid, cost_iid = algorithms.symmetric_multfn(q_iid)
# print(f"m_iid integer partition={intpart_from_multfn(multfn_iid)}; cost={cost_iid}")
multfn_sym, cost_sym = algorithms.symmetric_multfn(q_sym)
# print(f"m_iid integer partition={intpart_from_multfn(multfn_sym)}; cost={cost_sym}")
assert np.allclose(intpart_from_multfn(multfn_iid), [20, 20, 20, 20])
assert np.allclose(intpart_from_multfn(multfn_sym), [27, 27, 26])

E = ECost
# print(f"ECost(multfn_iid, q_iid)={E(q_iid, multfn_iid)}")
# print(f"ECost(multfn_sym, q_iid)={E(q_iid, multfn_sym)}")
# print(f"ECost(multfn_iid, q_sym)={E(q_sym, multfn_iid)}")
# print(f"ECost(multfn_sym, q_sym)={E(q_sym, multfn_sym)}")
np.allclose(E(q_iid, multfn_iid), 7.216900898670634)
np.allclose(E(q_iid, multfn_sym), 7.261454447408048)
np.allclose(E(q_sym, multfn_iid), 6.932112504669462)
np.allclose(E(q_sym, multfn_sym), 6.785444658410714)

""" from old julia code
sizes(π_iid) = Any[20, 20, 20, 20]
sizes(π_sym) = Any[26, 27, 27]
ECost(π_iid, m_iid) = 7.216900898670634
ECost(π_sym, m_iid) = 7.261454447408048
ECost(π_iid, m_sym) = 6.932112504669462
ECost(π_sym, m_sym) = 6.785444658410714
"""
