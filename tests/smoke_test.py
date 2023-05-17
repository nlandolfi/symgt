from symgt import models
import pytest

import numpy as np

print("THIS IS SMOKE TEST 1: IT TESTS IIDModel")

m = models.IIDModel(10, 0.1)

got, want = m.prevalence(), 0.1
if got != want:
    raise Exception(f"IIDModel prevalence: got {got}; want {want}")

lms = m.log_marginals()
got, want = np.exp(lms[0]), 1 - m.prevalence()
if not np.isclose(got, want):
    raise Exception(f"IIDModel first log marginal: got {got}; want {want}")

assert np.all(np.diff(np.exp(lms)) < 0)  # should be decreasing


def test_IIDModel_init():
    # Test that ValueError is raised for negative n
    with pytest.raises(ValueError):
        models.IIDModel(-1, 0.5)

    # Test that ValueError is raised for n = 0
    with pytest.raises(ValueError):
        models.IIDModel(0, 0.5)

    # Test that ValueError is raised for p out of range
    with pytest.raises(ValueError):
        models.IIDModel(10, 1.5)

    # Test that ValueError is raised for p out of range
    with pytest.raises(ValueError):
        models.IIDModel(10, -0.5)

    # Test that TypeError is raised for incorrect types
    with pytest.raises(TypeError):
        models.IIDModel("10", 0.5)
    with pytest.raises(TypeError):
        models.IIDModel(10, "0.5")


test_IIDModel_init()
