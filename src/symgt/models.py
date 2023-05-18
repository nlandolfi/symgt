import numpy as np
from scipy.special import gammaln, logsumexp  # type: ignore


class IIDModel:
    """
    This class represents a distribution of independent and identically distributed (iid)
    specimen statuses.

    An IIDModel is characterized by a number of specimens (`n`) and a prevalence (`p`).

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

    @classmethod
    def fit(cls, samples: np.ndarray) -> "IIDModel":
        """
        Function to fit an independent and identically distributed (IID) model.

        Parameters
        ----------
        samples : np.ndarray
            A 2D numpy array where each row represents a sample and of specimens.

        Returns
        -------
        IIDModel
            An IIDModel object. The model's parameters are the population size (n) and the mean of all values in the samples.
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

        The i-th entry of the returned array is the log probability that a group of size i has negative status.

        Returns
        -------
        np.ndarray
            An array containing the log of the q representation.
        """
        # note that by convention q(0) = 1, so log q(0) = 0; handled with multiplication by 0
        return np.log(1 - self.p) * np.arange(0, self.n + 1)
