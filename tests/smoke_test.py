from symgt import models

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
