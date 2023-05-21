# symgt

A package for group testing against symmetric distributions. Python 3.

```bash
pip install symgt
```

For example, to compute an optimal partition for a symmetric distribution...
```python
import numpy as np
from symgt import models, algorithms, utils

# the representation alpha of symmetric distribution
alpha = np.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0])
m = models.ExchangeableModel(10, alpha)
q = np.exp(m.log_q()) # the representation q of the symmetric distribution
multfn, cost = algorithms.symmetric_multfn(q) # cost is 6
intpart = utils.intpart_from_multfn(multfn) # intpart is [10]

# which differs from that computed using the IID approximation
m_iid = models.IIDModel(10, m.prevalence())
multfn_iid, _ = algorithms.symmetric_multfn(np.exp(m_iid.log_q()))
intpart_iid = utils.intpart_from_multfn(multfn_iid) # is [4, 3, 3]

utils.ECost(q, multfn) # 6
utils.ECost(q, multfn_iid) # 6.63
```
