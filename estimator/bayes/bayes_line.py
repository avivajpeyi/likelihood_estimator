#!/usr/bin/env python3

import estimator.bayes.priors as priors
import numpy as np
from estimator.bayes.bayes_base import BayesToyBase
from estimator.models.line import LineModel
from estimator.models.toy_dataset import RecordedData
from scipy.stats import norm


class BayesLine(BayesToyBase):
    def __init__(self, data: RecordedData, *args, **kwargs):
        super(BayesLine, self).__init__()
        self.data = data

    def ln_likelihood(self, theta, *args, **kwargs) -> np.float:
        signal_hypothesis = LineModel.model(x=self.data.x, m=theta[0], c=theta[1])
        data_minus_signal = self.data.y - signal_hypothesis

        ln_like_d_i = norm.logpdf(
            x=data_minus_signal, loc=priors.MU_NOISE_TRUE, scale=priors.SIGMA_NOISE_TRUE
        )

        ln_like = np.sum(ln_like_d_i)

        return ln_like
