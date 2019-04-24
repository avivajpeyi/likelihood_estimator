from abc import ABC, abstractmethod

import estimator.bayes.priors as priors
import numpy as np
from scipy.special import logsumexp
from scipy.stats import uniform


class BayesToyBase(ABC):
    """Bayes Toy Base

    Abstract base class for Toy Bayesian Inference

    """

    # Initializer / Instance Attributes
    def __init__(self):
        self._ln_post_grid = None
        self.ln_posterior_grid_vec = np.vectorize(self.ln_posterior_grid)

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
            ln_pi_m = uniform.logpdf(m, loc=priors.M_MIN, scale=priors.M_MAX)
            ln_pi_c = uniform.logpdf(c, loc=priors.C_MIN, scale=priors.C_MAX)
            ln_pi = ln_pi_m + ln_pi_c
        else:
            ln_pi = -np.inf

        assert not np.isnan(ln_pi)

        return ln_pi

    def ln_posterior(self, theta, *args, **kwargs) -> np.float:
        return self.ln_prior(theta) + self.ln_likelihood(theta)

    def ln_posterior_grid(self, mm, cc, *args, **kwargs):
        return self.ln_posterior(theta=[mm, cc])

    def ln_evidence(self):
        ln_post_grid = self.ln_posterior_grid_vec(priors.M_GRID, priors.C_GRID)
        ln_evid = logsumexp(ln_post_grid, b=priors.DC * priors.DM)
        return ln_evid
