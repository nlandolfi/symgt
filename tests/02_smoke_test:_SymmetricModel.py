import numpy as np
import pytest
import symgt as st

print("THIS IS SMOKE TEST 2: IT TESTS SymmetricModel")

SymmetricModel = st.SymmetricModel

m = SymmetricModel(10, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
lq = m.log_q()
assert np.all(np.exp(lq) == 1)  # this model never has positive specimens

m = SymmetricModel(10, np.ones(11) / 11)
lq = m.log_q()
assert np.all(np.diff(np.exp(lq)) < 0)  # should be decreasing


def test_fit():
    # Test 1: All zeros
    samples = np.zeros((5, 4))
    model = SymmetricModel.fit(samples)
    assert isinstance(model, SymmetricModel), "Model is not instance of SymmetricModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    ), "Incorrect model.alpha."

    # Test 2: All ones
    samples = np.ones((5, 4))
    model = SymmetricModel.fit(samples)
    assert isinstance(model, SymmetricModel), "Model is not instance of SymmetricModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    ), "Incorrect model.alpha."

    # Test 3: Mix of ones and zeros
    samples = np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]]
    )
    model = SymmetricModel.fit(samples)
    assert isinstance(model, SymmetricModel), "Model is not instance of SymmetricModel."
    assert model.n == 4, "Incorrect model.n."
    assert np.array_equal(
        model.alpha, np.array([0.2, 0.0, 0.6, 0.2, 0.0])
    ), "Incorrect model.alpha."


test_fit()
