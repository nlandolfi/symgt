from typing import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp  # type: ignore
from scipy.stats import binom  # type: ignore

from .utils import (
    subset_symmetry_orbits,
    subset_symmetry_orbit_diffs,
    subset_symmetry_leq,
)


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

    def log_alpha(self) -> np.ndarray:
        """
        Computes the log of the alpha representation of the distribution.

        The `i`-th entry of the returned array is the log probability to see a
        sample with `i` nonzeros.

        Returns
        -------
        np.ndarray
            An array containing the log of the alpha representation.
        """
        return binom.logpmf(np.arange(0, self.n + 1), self.n, self.p)

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

        log_alpha = self.log_alpha()

        for i in range(1, self.n + 1):
            a = [log_alpha[0]]
            for j in range(1, self.n - i + 1):
                a.append(log_comb(self.n - i, j) - log_comb(self.n, j) + log_alpha[j])
            log_q[i] = logsumexp(a)
        return log_q

    def log_alpha(self) -> np.ndarray:
        """
        Computes the log of the alpha representation of the distribution.

        The `i`-th entry of the returned array is the log probability to see a
        sample with `i` nonzeros.

        Returns
        -------
        np.ndarray
            An array containing the log of the alpha representation.
        """
        # by default, np.log also takes log(0) = -np.inf, but throws a warning
        # here we make it explicit and do not print a warning
        return np.log(
            self.alpha, where=(self.alpha != 0), out=np.full_like(self.alpha, -np.inf)
        )

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


class IndependentSubpopulationsModel:
    """
    This class represents a distribution over independent subpopulations.
    In other words, the members of a given subpopulation are independent of the
    members of all other subpopulations.

    It is defined by subpopulation `sizes` and `models`.

    Every IndependentSubpopulationsModel can be represented as a
    SubsetSymmetryModel, but use this class to indicate that there is
    additional structure.

    Attributes
    ----------
    sizes : Sequence[int]
        Subpopulation sizes.

    models : list[ExchangeableModel]
        List of submodels.
    """

    sizes: Sequence[int]
    orbits: list[tuple[int, ...]]
    models: list

    def __init__(self, sizes: Sequence[int], models: list):
        """
        Initializes an IndependentSubpopulationsModel with the given subpopulation
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
        return f"IndependentSubpopulationsModel(sizes={self.sizes}, models=...)"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fit(
        cls,
        sizes: Sequence[int],
        samples: np.ndarray,
        model_classes,
    ) -> "IndependentSubpopulationsModel":
        """
        Function to fit an IndependentSubpopulationsModel.

        Parameters
        ----------
        sizes : Sequence[int]
            Population sizes of each submodel.
        samples : np.ndarray
            A 2D numpy array where each row represents a sample and each column
            represents a specimen.

        Returns
        -------
        IndependentSubpopulationsModel
            An IndependentSubpopulationsModel object fit to samples.
        """
        N, n = samples.shape

        if np.sum(sizes) != n:
            raise ValueError("sum of sizes does not match number of samples")

        models = []
        offset = 0
        for i, size in enumerate(sizes):
            models.append(model_classes[i].fit(samples[:, offset : offset + size]))
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

    def log_alpha(self) -> np.ndarray:
        """
        Computes the log of the alpha representation of the distribution.
        Useful for comparison with SubsetSymmetryModel whose default
        representation is the alpha representation.

        The `i`-th entry of the returned array is the log probability to see a
        sample from a group of orbit `i`.

        Returns
        -------
        np.ndarray
            An array containing the log of the alpha representation.
        """
        log_alpha = np.zeros(len(self.orbits))

        l_alphas = [m.log_alpha() for m in self.models]

        for i, o in enumerate(self.orbits):
            log_alpha[i] = np.sum([l_alphas[j][s] for (j, s) in enumerate(o)])

        return log_alpha

    def sample(self) -> np.ndarray:
        """
        Sample from the model.

        Returns
        -------
        np.ndarray
            An array of length `n` containing the outcomes.
        """
        return np.concatenate([model.sample() for model in self.models])


class SubsetSymmetryModel:
    """
    This class represents a distribution over fully symmetric subpopulations.

    It is defined by a list of `orbits` and corresponding probabilities `alpha`.
     - Each orbit is a tuple that identifies the number of nonzeros of an element
       of the set {0,1}^P, for each given subpopulation.
     - The last element of the orbits list is a tuple of subpopulation sizes.
     - The elements of `alpha` are the probabilities of each orbit.

    To construct the list of orbits, see `utils.subset_symmetry_orbits`.

    For the case in which the individual members of a given subpopulation are
    independent of the given members of another subpopulation, for all two distinct
    subpopulations, use the `IndependentSubpopulationsModel` class.

    Attributes
    ----------
    sizes : Sequence[int]
        Subpopulation sizes.
    orbits : list[tuple[int, ...]]
        List of orbits.
    orbit_sizes : np.ndarray
        List of orbits sizes (sum of the tuple).
    alpha : np.ndarray
        Representation of the symmetric distribution.
        `alpha[i]` is the probability of obtaining a sample in orbit `i`.
    """

    sizes: Sequence[int]
    orbits: list[tuple[int, ...]]
    orbit_sizes: np.ndarray  # for quick prevalence calculation
    alpha: np.ndarray

    def __init__(self, orbits: list[tuple[int, ...]], alpha: np.ndarray):
        """
        Initializes a SubsetSymmetryModel with the given orbits and representation.

        Parameters
        ----------
        orbits : list[tuple[int, ...]]
            List of orbits. To construct this, see `utils.subset_symmetry_orbits`.
        alpha : np.ndarray
            Representation of the symmetric distribution. Nonnegative and sums to one.
        """
        if len(orbits) < 2:
            raise ValueError("orbits must have at least two elements")

        if len(alpha) != len(orbits):
            raise ValueError("orbits and alpha must have the same length")

        alpha = np.asarray(alpha)
        if not np.all(alpha >= 0):
            raise ValueError("alpha has negative values")
        if not np.allclose(np.sum(alpha), 1.0):
            raise ValueError("`np.sum(alpha)` should be `1`.")

        self.sizes = orbits[-1]
        for i, x in enumerate(self.sizes):
            if x <= 0:
                raise ValueError(f"size {i} is not a positive integer")

        self.orbits = orbits
        self.orbit_sizes = np.array([sum(o) for o in orbits])
        self.alpha = alpha

    def __str__(self):
        return f"SubsetSymmetryModel(sizes={self.sizes}, models=...)"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fit(
        cls,
        sizes: Sequence[int],
        samples: np.ndarray,
    ) -> "SubsetSymmetryModel":
        """
        Function to fit a SubsetSymmetryModel.

        Parameters
        ----------
        sizes : Sequence[int]
            Population sizes of each subpopulation.
        samples : np.ndarray
            A 2D numpy array where each row represents a sample and each column
            represents a specimen.

        Returns
        -------
        SubsetSymmetryModel
            A SubsetSymmetryModel object fit to samples.
        """
        N_samples, n = samples.shape

        if np.sum(sizes) != n:
            raise ValueError("sum of sizes does not match number of samples")

        orbits = subset_symmetry_orbits(sizes)

        segments = []
        offset = 0
        for i, size in enumerate(sizes):
            segments.append(np.arange(offset, offset + size))
            offset += size

        sum_samples = np.array(
            [samples[:, segment].sum(axis=1) for segment in segments]
        ).T

        unique_rows, counts = np.unique(sum_samples, axis=0, return_counts=True)
        counts = {tuple(row): count for row, count in zip(unique_rows, counts)}

        alpha = np.array([counts.get(o, 0) / N_samples for o in orbits])

        return cls(orbits, alpha)

    def prevalence(self) -> float:
        """
        Returns the prevalence of the model.

        Returns
        -------
        float
            The prevalence of the model.
        """
        return np.dot(self.alpha, self.orbit_sizes) / np.sum(self.sizes)

    def log_q(self) -> np.ndarray:
        """
        Computes the log of the `q` representation of the distribution. See paper.

        The `i`-th entry of the returned array is the log probability that a
        group in orbit `i` has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the `q` representation.
        """
        assert self.orbits[0] == (0,) * len(self.sizes)
        assert self.orbits[len(self.orbits) - 1] == tuple(self.sizes)

        # note that by convention q(0) = 1, so log q(0) = 0;
        # handled with initialization to 0
        log_q = np.zeros(self.alpha.shape)

        # by default, np.log also takes log(0) = -np.inf, but throws a warning
        # here we make it explicit and do not print a warning
        log_alpha = np.log(
            self.alpha, where=(self.alpha != 0), out=np.full_like(self.alpha, -np.inf)
        )

        n = sum(self.sizes)
        N = len(self.orbits)
        diffs = subset_symmetry_orbit_diffs(self.orbits)

        # here the orbits identify subsets of P
        for i, o in zip(range(1, N), self.orbits[1:]):
            # imagine x in {0,1}^P so that $x^{-1}(0)$ is in orbit o
            nnzx = n - sum(o)

            # hence, we can only place the ones elsewhere, in diff
            diff = self.orbits[list(diffs[(i, N - 1)])[0]]  # assume singleton
            assert sum(diff) == nnzx

            a = []
            # here the orbits identify elements of {0,1}^P
            # we are to sum up all ways of putting ones in diff
            for j, p in enumerate(self.orbits):
                # check if we can place "shape" p ones in "diff"
                # recall: diff is the shape of places where ones can be allocated
                if not subset_symmetry_leq(p, diff):
                    continue
                # precedence of p implies nnz(p) <= nnzx
                assert sum(p) <= nnzx

                a.append(
                    # number of members of orbit p in R^{-1}(x^{-1}(0), nnz(p))
                    # i.e., number of ways to place "shape" p ones in "shape" diff
                    np.sum([log_comb(n, m) for (n, m) in zip(diff, p)])
                    # log probability of a member of orbit p
                    # i.e., log( alpha[orbit p]/(total # of orbit members) ), where
                    # denominator is # of ways to place "shape" p ones in "shape" sizes
                    + log_alpha[j]
                    - np.sum([log_comb(n, m) for (n, m) in zip(self.sizes, p)])
                )
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
        orbit_index = np.random.choice(np.arange(len(self.orbits)), p=self.alpha)
        orbit = self.orbits[orbit_index]
        samples = []
        for size, nnz in zip(self.sizes, orbit):
            x = np.zeros(size)
            x[:nnz] = 1
            np.random.shuffle(x)
            samples.append(x)
        return np.concatenate(samples)
