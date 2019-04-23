#!/usr/bin/env python3
from unittest import TestCase

import estimator.bayes.priors as priors
import numpy as np
from estimator.bayes.bayes_line import BayesLine
from estimator.models.gaussian_noise import GaussianNoiseModel
from estimator.models.line import LineModel
from estimator.models.toy_dataset import RecordedData

__author__ = "Avi"
__version__ = "0.1.0"


class TestDataModels(TestCase):
    @staticmethod
    def make_data():
        n = 10
        time = np.linspace(start=0, stop=10, num=n)
        s = LineModel(size=n, m=priors.M_TRUE, c=priors.C_TRUE, x=time)
        n = GaussianNoiseModel(
            size=n, mu=priors.MU_NOISE_TRUE, sigma=priors.SIGMA_NOISE_TRUE, x=time
        )
        d = RecordedData(noise=n, signal=s)
        return d, n, s

    def test_bayes_line_constructor(self):
        d, n, s = self.make_data()
        b_line = BayesLine(data=d)
        ln_like = b_line.ln_likelihood(theta=[0, 0])
        self.assertTrue(isinstance(ln_like, float))
