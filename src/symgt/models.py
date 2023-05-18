import numpy as np
from scipy.special import gammaln, logsumexp  # type: ignore


class IIDModel:
    """
    This class represents a distribution of independent and identically distributed
    (iid) specimen statuses.

    An IIDModel is characterized by a number of specimens (`n`) and a prevalence (`p`).

    Every IIDModel could be represented as an ExchangeableModel, but use this
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
            An IIDModel object. The model's parameters are the population size (`n`)
            and the proportion of positive statuses in the samples.
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
        Computes the log of the q representation of the distribution. See paper.

        The i-th entry of the returned array is the log probability that a group of
        size i has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the q representation.
        """
        # note that by convention q(0) = 1, so log q(0) = 0; handled with multiplication by 0
        return np.log(1 - self.p) * np.arange(0, self.n + 1)


class ExchangeableModel:
    """
    This class represents a permutation-symmetric distribution. In other
    words, the specimen statuses are modeled as exchangeable random variables.

    An exchangeable model is defined by population size (`n`) and the representation
    `alpha`. `alpha[i]` is the probability of `i` positive statuses.

    Attributes
    ----------
    n : int
        Population size of model.
    alpha : np.ndarray
        Representation of the symmetric distribution.
        alpha[i] is the probability that there are i ones in a sample.
    """

    def __init__(self, n: int, alpha: np.ndarray):
        """
        Initializes a ExchangeableModel with a specific population size and a
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
        if np.sum(alpha) != 1:
            raise ValueError("`np.sum(alpha)` should be `n+1`.")

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
            population size (n) and the normalized histogram of sums of each
            sample.
        """
        N, n = samples.shape
        nnzs = np.sum(samples, axis=1)
        alpha = np.zeros(n + 1)
        for nnz in nnzs:  # TODO: vectorize?
            assert nnz.is_integer()
            alpha[int(nnz)] += 1
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
        Computes the log of the q representation of the distribution. See paper.

        The i-th entry of the returned array is the log probability that a
        group of size i has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log marginal probabilities for each sample.
        """
        # note that by convention q(0) = 1, so log q(0) = 0;
        # handled with initialization to 0
        log_q = np.zeros(self.n + 1)

        # by default, np.log also takes log(0) = -np.inf, but throws a warning
        # here we make it explicit and w/o the warning
        log_alpha = np.log(
            self.alpha, where=(self.alpha != 0), out=np.full_like(self.alpha, -np.inf)
        )

        for i in range(1, self.n + 1):
            a = [log_alpha[0]]
            for j in range(1, self.n - i + 1):
                a.append(log_comb(self.n - i, j) - log_comb(self.n, j) + log_alpha[j])
            log_q[i] = logsumexp(a)
        return log_q


def log_comb(n, k):
    """
    Compute the log of n choose k using scipy's gammaln function.
    """
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
