import numpy as np
import pytest
import symgt as st
from symgt.models import log_comb
from scipy.special import comb

print("THIS IS SMOKE TEST 2: IT TESTS ExchangeableModel")

ExchangeableModel = st.ExchangeableModel

m = ExchangeableModel(10, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
lq = m.log_q()
assert np.all(np.exp(lq) == 1)  # this model never has positive specimens

m = ExchangeableModel(10, np.ones(11) / 11)
lq = m.log_q()
assert np.all(np.diff(np.exp(lq)) < 0)  # should be decreasing

# test exchangeable model init valid parameters
n = 2
alpha = np.array([0.3, 0.4, 0.3])
model = ExchangeableModel(n, alpha)
assert model.n == n
assert np.all(model.alpha == alpha)

# test exchangeable model init invalid n type
n = "2"
alpha = np.array([0.3, 0.4, 0.3])
with pytest.raises(TypeError):
    ExchangeableModel(n, alpha)

# test exchangeable model init invalid n value
n = 0
alpha = np.array([0.3, 0.4, 0.3])
with pytest.raises(ValueError):
    ExchangeableModel(n, alpha)

# test exchangeable model init invalid alpha length
n = 2
alpha = np.array([0.3, 0.4])
with pytest.raises(ValueError):
    ExchangeableModel(n, alpha)

# test exchangeable model init invalid alpha sum
n = 2
alpha = np.array([0.3, 0.4, 0.2])
with pytest.raises(ValueError):
    ExchangeableModel(n, alpha)


def test_fit():
    # Test 1: All zeros
    samples = np.zeros((5, 4))
    model = ExchangeableModel.fit(samples)
    assert isinstance(
        model, ExchangeableModel
    ), "Model is not instance of ExchangeableModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    ), "Incorrect model.alpha."

    # Test 2: All ones
    samples = np.ones((5, 4))
    model = ExchangeableModel.fit(samples)
    assert isinstance(
        model, ExchangeableModel
    ), "Model is not instance of ExchangeableModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    ), "Incorrect model.alpha."
    assert model.prevalence() == 1.0, "Incorrect model.prevalence()"

    # Test 3: Mix of ones and zeros
    samples = np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]]
    )
    model = ExchangeableModel.fit(samples)
    assert isinstance(
        model, ExchangeableModel
    ), "Model is not instance of ExchangeableModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([0.2, 0.0, 0.6, 0.2, 0.0])
    ), "Incorrect model.alpha."


test_fit()


def test_log_comb():
    got, want = comb(10, 5), np.exp(log_comb(10, 5))
    assert np.allclose(
        got, want
    ), f"comb(10,5) !approx= exp(log_comb(10,5): got {got}, want {want}"

    got, want = comb(20, 5), np.exp(log_comb(20, 5))
    assert np.allclose(
        got, want
    ), f"comb(20,5) !approx= exp(log_comb(20,5): got {got}, want {want}"

    got, want = comb(80, 20), np.exp(log_comb(80, 20))
    assert np.allclose(
        got, want
    ), f"comb(80,20) !approx= exp(log_comb(80,20): got {got}, want {want}"


test_log_comb()

# Test sample a bit
alpha = np.zeros(5)
alpha[0] = 1
m = ExchangeableModel(4, alpha)
got, want = m.sample(), np.zeros(4)
assert np.allclose(got, want), f"got {got}, want {want}"
got, want = m.sample(), np.zeros(4)
assert np.allclose(got, want), f"got {got}, want {want}"
alpha[0] = 0
alpha[2] = 1
m = ExchangeableModel(4, alpha)
got, want = np.sum(m.sample()), 2
assert np.allclose(got, want), f"got {got}, want {want}"
got, want = np.sum(m.sample()), 2
assert np.allclose(got, want), f"got {got}, want {want}"
