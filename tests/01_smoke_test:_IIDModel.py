import numpy as np
import pytest
import symgt as st

print("THIS IS SMOKE TEST 1: IT TESTS IIDModel")

IIDModel = st.IIDModel
m = IIDModel(10, 0.1)

got, want = m.prevalence(), 0.1
if got != want:
    raise Exception(f"IIDModel prevalence: got {got}; want {want}")

q = np.exp(m.log_q())
got, want = q[0], 1
if not np.isclose(got, want):
    raise Exception(f"IIDModel q(0): got {got}; want {want}")
got, want = q[1], 1 - m.prevalence()
if not np.isclose(got, want):
    raise Exception(f"IIDModel q(1): got {got}; want {want}")

assert np.all(np.diff(q) < 0)  # should be decreasing


def test_IIDModel_init():
    # Test that ValueError is raised for negative n
    with pytest.raises(ValueError):
        IIDModel(-1, 0.5)

    # Test that ValueError is raised for n = 0
    with pytest.raises(ValueError):
        IIDModel(0, 0.5)

    # Test that ValueError is raised for p out of range
    with pytest.raises(ValueError):
        IIDModel(10, 1.5)

    # Test that ValueError is raised for p out of range
    with pytest.raises(ValueError):
        IIDModel(10, -0.5)

    # Test that TypeError is raised for incorrect types
    with pytest.raises(TypeError):
        IIDModel("10", 0.5)
    with pytest.raises(TypeError):
        IIDModel(10, "0.5")


test_IIDModel_init()


def test_IIDModel_fit():
    # test with single row
    samples = np.array([[1, 0, 1, 1, 0]])
    model = IIDModel.fit(samples)
    got, want = model.n, 5
    if got != want:
        raise Exception(f"IIDModel.fit for n: got {got} want {want}")
    got, want = model.p, 0.6
    if got != want:
        raise Exception(f"IIDModel.fit for p: got {got} want {want}")

    samples = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 1, 1]])
    model = IIDModel.fit(samples)
    got, want = model.n, 5
    if got != want:
        raise Exception(f"IIDModel.fit for n: got {got} want {want}")
    got, want = model.p, 0.7
    if got != want:
        raise Exception(f"IIDModel.fit for p: got {got} want {want}")

    samples = np.array([[0, 0, 0, 0, 0]])
    model = IIDModel.fit(samples)
    got, want = model.n, 5
    if got != want:
        raise Exception(f"IIDModel.fit for n: got {got} want {want}")
    got, want = model.p, 0.0
    if got != want:
        raise Exception(f"IIDModel.fit for p: got {got} want {want}")


test_IIDModel_fit()

# Test sample a bit
m = IIDModel(4, 0.0)
m.sample()
got, want = m.sample(), np.zeros(4)
assert np.allclose(got, want), f"got {got}, want {want}"
got, want = m.sample(), np.zeros(4)
assert np.allclose(got, want), f"got {got}, want {want}"
m = IIDModel(4, 1.0)
got, want = np.sum(m.sample()), 4
assert np.allclose(got, want), f"got {got}, want {want}"
got, want = np.sum(m.sample()), 4
assert np.allclose(got, want), f"got {got}, want {want}"
