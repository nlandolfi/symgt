# symgt

A package for group testing against symmetric distributions. Python 3.

```bash
pip install symgt
```

For example, to compute an optimal partition for a symmetric distribution...
```python
import numpy as np
from symgt import models, algorithms, utils

m = models.ExchangeableModel(10, np.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0]))
q = np.exp(m.log_q()) # the representation q
multfn, cost = algorithms.symmetric_multfn(q) # cost is 6
intpart = utils.intpart_from_multfn(multfn) # intpart is [10]
```
