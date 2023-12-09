from typing import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp  # type: ignore

from .utils import subset_symmetry_orbits


class IIDModel:
    """
    This class represents a distribution of independent and identically distributed
    (iid) binary outcomes.

    An IIDModel is characterized by a number of specimens (`n`) and a prevalence (`p`).

    Every IIDModel can be represented as an ExchangeableModel, but use this
    class to indicate the additional structure.

    Attributes
    ----------
    n : int
        Population size of model.
    p : float
        Prevalence in the model.
    """

    def __init__(self, n: int, p: float):
        """
        Initializes an IIDModel with a specific number of specimens and a prevalence.

        Parameters
        ----------
        n : int
            Population size of model.
        p : float
            Prevalence in the model.
        """
        if not isinstance(n, int):
            raise TypeError("`n` should be a positive integer.")
        if n <= 0:
            raise ValueError("`n` should be a positive integer.")
        if not isinstance(p, float):
            raise TypeError("`p` should be a float between 0 and 1 inclusive.")
        if not (0 <= p <= 1):
            raise ValueError("`p` should be a float between 0 and 1 inclusive.")

        self.n = n
        self.p = p

    def __str__(self):
        return f"IIDModel(n={self.n}, p={self.p})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fit(cls, samples: np.ndarray) -> "IIDModel":
        """
        Function to fit an independent and identically distributed (IID) model.

        Parameters
        ----------
        samples : np.ndarray
            A 2D numpy array where each row represents a sample of `n` specimens.

        Returns
        -------
        IIDModel
            An `IIDModel` object. The model's parameters are the population
            size (`n`) and the proportion of positive outcomes in the samples.
        """
        N, n = samples.shape
        return cls(n, np.sum(samples) / (n * N))

    def prevalence(self) -> float:
        """
        Returns the prevalence of the model.

        Returns
        -------
        float
            The prevalence of the model.
        """
        return self.p

    def log_q(self) -> np.ndarray:
        """
        Computes the log of the `q` representation of the distribution. See paper.

        The `i`-th entry of the returned array is the log probability that a group of
        size `i` has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the `q` representation.
        """
        # note that by convention q(0) = 1, so log q(0) = 0;
        # handled with multiplication by 0
        return np.log(1 - self.p) * np.arange(0, self.n + 1)

    def sample(self) -> np.ndarray:
        """
        Sample from the model.

        Returns
        -------
        np.ndarray
            An array of length `n` containing the outcomes.
        """
        return np.random.rand(self.n) < self.prevalence()


class ExchangeableModel:
    """
    This class represents a permutation-symmetric distribution of binary
    outcomes. In other words, the specimen statuses are modeled as
    exchangeable random variables.

    An exchangeable model is defined by population size (`n`) and the representation
    `alpha`. `alpha[i]` is the probability of `i` positive statuses.

    Attributes
    ----------
    n : int
        Population size of model.
    alpha : np.ndarray
        Representation of the symmetric distribution.
        `alpha[i]` is the probability that there are `i` ones in a sample.
    """

    def __init__(self, n: int, alpha: np.ndarray):
        """
        Initializes an ExchangeableModel with a specific population size and
        representation.

        Parameters
        ----------
        n : int
            Population size of model.
        alpha : np.ndarray
            Representation of symmetric distribution. See paper.
        """
        if not isinstance(n, int):
            raise TypeError("`n` should be a positive integer.")
        if n <= 0:
            raise ValueError("`n` should be a positive integer.")
        if len(alpha) != n + 1:
            raise ValueError("len of `alpha` should be `n+1`.")
        if not np.allclose(np.sum(alpha), 1.0):
            raise ValueError("`np.sum(alpha)` should be `1`.")

        self.n = n
        self.alpha = np.asarray(alpha).astype(np.float64)

    def __str__(self):
        return f"ExchangeableModel(n={self.n}, alpha=...)"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fit(cls, samples: np.ndarray) -> "ExchangeableModel":
        """
        Function to fit a symmetric distribution model.

        Parameters
        ----------
        samples : np.ndarray
            A 2D numpy array where each row represents a sample and each column
            represents a specimen.
        Returns
        -------
        ExchangeableModel
            An ExchangeableModel object. The model's parameters are the
            population size (`n`) and the normalized histogram of sums of each
            sample.
        """
        N, n = samples.shape
        nnzs = np.sum(samples, axis=1)

        if not np.all(nnzs % 1 == 0):
            raise ValueError("All row sums of `samples` should be integral")

        alpha = np.bincount(nnzs.astype(int), minlength=n + 1)

        return cls(n, alpha / N)

    def prevalence(self) -> float:
        """
        Returns the prevalence of the model.

        Returns
        -------
        float
            The prevalence of the model.
        """
        return 1 - np.exp(self.log_q()[1])

    def log_q(self) -> np.ndarray:
        """
        Computes the log of the `q` representation of the distribution. See paper.

        The `i`-th entry of the returned array is the log probability that a
        group of size `i` has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the `q` representation.
        """
        # note that by convention q(0) = 1, so log q(0) = 0;
        # handled with initialization to 0
        log_q = np.zeros(self.n + 1)

        # by default, np.log also takes log(0) = -np.inf, but throws a warning
        # here we make it explicit and do not print a warning
        log_alpha = np.log(
            self.alpha, where=(self.alpha != 0), out=np.full_like(self.alpha, -np.inf)
        )

        for i in range(1, self.n + 1):
            a = [log_alpha[0]]
            for j in range(1, self.n - i + 1):
                a.append(log_comb(self.n - i, j) - log_comb(self.n, j) + log_alpha[j])
            log_q[i] = logsumexp(a)
        return log_q

    def sample(self) -> np.ndarray:
        """
        Sample from the model.

        Returns
        -------
        np.ndarray
            An array of length `n` containing the outcomes.
        """
        s = np.zeros(self.n)
        s[: np.random.choice(np.arange(self.n + 1), p=self.alpha)] = 1
        np.random.shuffle(s)
        return s


def log_comb(n, k):
    """
    Compute the log of `n` choose `k` using scipy's `gammaln` function.

    Used by `ExchangeableModel`'s `log_q` function.
    """
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class ProductExchangeableModel:
    """
    This class represents a distribution of partially exchangeable random
    variables where the exchangeable subsets are modeled as independent.

    An exchangeable model is defined by subpopulation `sizes` and `models`.

    Attributes
    ----------
    sizes : Sequence[int]
        Subpopulation sizes.

    models : list[ExchangeableModel]
        List of submodels.
    """

    sizes: Sequence[int]
    orbits: list[tuple[int, ...]]
    models: list[ExchangeableModel]

    def __init__(self, sizes: Sequence[int], models: list[ExchangeableModel]):
        """
        Initializes a ProductExchangeableModel with the give subpopulation
        sizes and models.

        Parameters
        ----------
        sizes : Sequence[int]
            Population sizes of each submodel.
        submodels : list[ExchangeableModel]
            List of submodels.
        """
        if len(sizes) != len(models):
            raise ValueError("sizes and models must have the same length")

        for i, x in enumerate(sizes):
            if x <= 0:
                raise ValueError(f"size {i} is not a positive integer")

            if x != models[i].n:
                raise ValueError(f"size {i} does not match model {i}")

        self.sizes = sizes
        self.orbits = subset_symmetry_orbits(sizes)
        self.models = models

    def __str__(self):
        return f"ProductExchangeableModel(sizes={self.sizes}, models=...)"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fit(
        cls, sizes: Sequence[int], samples: np.ndarray
    ) -> "ProductExchangeableModel":
        """
        Function to fit a product of fully symmetric models.

        Parameters
        ----------
        sizes : Sequence[int]
            Population sizes of each submodel.
        samples : np.ndarray
            A 2D numpy array where each row represents a sample and each column
            represents a specimen.

        Returns
        -------
        ProductExchangeableModel
            A ProductExchangeableModel object fit to samples.
        """
        N, n = samples.shape

        if np.sum(np.asarray(sizes)) != n:
            raise ValueError("sum of sizes does not match number of samples")

        models = []
        offset = 0
        for i, size in enumerate(sizes):
            models.append(ExchangeableModel.fit(samples[:, offset : offset + size]))
            offset += size

        return cls(sizes, models)

    def prevalence(self) -> float:
        """
        Returns the prevalence of the model.

        Returns
        -------
        float
            The prevalence of the model.
        """
        total = sum(self.sizes)
        proportions = [float(s) / total for s in self.sizes]
        return np.dot(proportions, np.array([m.prevalence() for m in self.models]))

    def log_q(self) -> np.ndarray:
        """
        Computes the log of the `q` representation of the distribution. See paper.

        The `i`-th entry of the returned array is the log probability that a
        group of orbit `i` has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the `q` representation.
        """
        assert self.orbits[0] == (0,) * len(self.sizes)
        assert self.orbits[len(self.orbits) - 1] == tuple(self.sizes)

        log_qs = [m.log_q() for m in self.models]

        log_q = np.zeros(len(self.orbits))

        for i, orbit in enumerate(self.orbits):
            log_q[i] = np.sum([log_qs[j][o] for (j, o) in enumerate(orbit)])

        return log_q

    def sample(self) -> np.ndarray:
        """
        Sample from the model.

        Returns
        -------
        np.ndarray
            An array of length `n` containing the outcomes.
        """
        return np.concatenate([model.sample() for model in self.models])
