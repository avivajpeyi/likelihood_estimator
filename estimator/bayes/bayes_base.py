from abc import ABC, abstractmethod

import estimator.bayes.priors as priors
import numpy as np
from scipy.stats import uniform


class BayesToyBase(ABC):
    """Bayes Toy Base

    Abstract base class for Toy Bayesian Inference

    """

    # Initializer / Instance Attributes
    def __init__(self):
        self._ln_post_grid = None

    def __repr__(self):
        return self.__str__()

    # string
    def __str__(self):
        return str(self.__class__.__name__)

    @abstractmethod
    def ln_likelihood(self, theta, *args, **kwargs) -> np.float:
        pass

    def ln_prior(self, theta, *args, **kwargs) -> np.float:
        m, c = theta[0], theta[1]
        mu, sigma = priors.MU_NOISE_TRUE, priors.SIGMA_NOISE_TRUE

        in_m_range = priors.M_MIN <= m <= priors.M_MAX
        in_c_range = priors.C_MIN <= c <= priors.C_MAX
        in_mu_range = priors.MU_MIN <= mu <= priors.MU_MAX
        in_sigma_range = priors.SIGMA_MIN <= sigma <= priors.SIGMA_MAX

        if in_m_range and in_c_range and in_mu_range and in_sigma_range:
            ln_pi_m = uniform.logpdf(m, loc=priors.M_MIN, scale=priors.M_MIN)
            ln_pi_c = uniform.logpdf(c, loc=priors.C_MIN, scale=priors.C_MIN)
            ln_pi = ln_pi_m + ln_pi_c
        else:
            ln_pi = -np.inf

        return ln_pi

    def ln_posterior(self, theta, *args, **kwargs) -> np.float:
        return self.ln_prior(theta) + self.ln_likelihood(theta)

    @np.vectorize
    def ln_posterior_grid(self, mm, cc):
        if not self._ln_post_grid:
            self._ln_post_grid = self.ln_posterior(theta=[mm, cc])
        return self._ln_post_grid
