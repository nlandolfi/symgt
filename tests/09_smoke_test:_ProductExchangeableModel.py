import numpy as np
import pytest

from symgt.models import ExchangeableModel, ProductExchangeableModel

print("THIS IS SMOKE TEST 9: IT TESTS ProductExchangeableModel")

# base initialization test...
sm1 = ExchangeableModel(5, [1, 0, 0, 0, 0, 0])
sm2 = ExchangeableModel(5, [1, 0, 0, 0, 0, 0])
m = ProductExchangeableModel([5, 5], [sm1, sm2])

lq = m.log_q()
assert np.all(np.exp(lq) == 1)  # this model never has positive outcomes


# test bad initializations...

# sizes and models length mismatch
with pytest.raises(ValueError):
    ProductExchangeableModel([1, 2, 3], [sm1, sm2])

# test negative integer value
with pytest.raises(ValueError):
    ProductExchangeableModel([-5, -5], [sm1, sm2])

# test model i doesnt match size i
with pytest.raises(ValueError):
    ProductExchangeableModel([5, 3], [sm1, sm2])

# test fit

# all zeros
m = ProductExchangeableModel.fit(
    [3, 2], np.array([np.zeros(5), np.zeros(5), np.zeros(5)])
)
assert np.allclose(m.models[0].prevalence(), 0)
assert np.allclose(m.models[1].prevalence(), 0)
assert np.allclose(m.prevalence(), 0)

# all ones
m = ProductExchangeableModel.fit([1, 4], np.array([np.ones(5), np.ones(5), np.ones(5)]))
assert np.allclose(m.models[0].prevalence(), 1)
assert np.allclose(m.models[1].prevalence(), 1)
assert np.allclose(m.prevalence(), 1)

# mix of zeros and ones
m = ProductExchangeableModel.fit(
    [2, 3],
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
    ),
)
assert np.allclose(m.models[0].prevalence(), 0.5)
assert np.allclose(m.models[1].prevalence(), 0.75)
assert np.allclose(m.prevalence(), 0.4 * 0.5 + 0.6 * 0.75)

# test log_q more extensively...
m = ProductExchangeableModel.fit(
    [2, 2],
    np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    ),
)
assert m.models[0].prevalence() == 0.5
assert m.models[1].prevalence() == 0.75
assert np.allclose(m.prevalence(), 0.5 * 0.5 + 0.5 * 0.75)
assert np.allclose(m.models[0].alpha, [0.5, 0, 0.5])
assert np.allclose(m.models[1].alpha, [0.25, 0, 0.75])
assert np.allclose(np.exp(m.models[0].log_q()), [1, 0.5, 0.5])
assert np.allclose(np.exp(m.models[1].log_q()), [1, 0.25, 0.25])
assert np.allclose(
    m.orbits, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
)
assert np.allclose(
    np.exp(m.log_q()), [1, 0.25, 0.25, 0.5, 0.125, 0.125, 0.5, 0.125, 0.125]
)

# one more test
m = ProductExchangeableModel.fit(
    [1, 3],  # this is different from previous
    np.array(  # this is the same as previous
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    ),
)
assert m.models[0].prevalence() == 0.5
assert np.allclose(m.models[1].prevalence(), 8 / 12.0)
assert np.allclose(m.prevalence(), 0.25 * 0.5 + 0.75 * 8 / 12.0)
assert np.allclose(m.models[0].alpha, [0.5, 0.5])
assert np.allclose(m.models[1].alpha, [0, 0.25, 0.5, 0.25])
assert np.allclose(np.exp(m.models[0].log_q()), [1, 0.5])
assert np.allclose(np.exp(m.models[1].log_q()), [1, 4 / 12.0, 1 / 12.0, 0])
assert np.allclose(
    m.orbits, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
)
assert np.allclose(
    np.exp(m.log_q()), [1, 4 / 12.0, 1 / 12.0, 0.0, 0.5, 4 / 24.0, 1 / 24.0, 0]
)

# test sample a bit
m = ProductExchangeableModel(
    [2, 2], [ExchangeableModel(2, [1, 0, 0]), ExchangeableModel(2, [0, 0, 1])]
)
assert np.allclose(m.sample(), np.array([0, 0, 1, 1]))
assert np.allclose(m.sample(), np.array([0, 0, 1, 1]))

# test that samples empirically matches prevalence
m = ProductExchangeableModel.fit(
    [4, 3],
    np.array(
        [
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0],
        ]
    ),
)
np.random.seed(0)
x = np.hstack([m.sample() for i in range(100000)])
assert np.allclose(np.mean(x), m.prevalence(), atol=0.001)