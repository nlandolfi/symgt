import numpy as np
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


test_log_comb()