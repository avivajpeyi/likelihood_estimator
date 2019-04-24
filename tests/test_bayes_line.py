#!/usr/bin/env python3
from unittest import TestCase

import estimator.bayes.priors as priors
import numpy as np
from estimator.bayes.bayes_line import BayesLine, BayesLineChunks
from estimator.bayes.plotting_utils import plot_contour
from estimator.models.gaussian_noise import GaussianNoiseModel
from estimator.models.line import LineModel
from estimator.models.toy_dataset import RecordedData, split_recorded_data

__author__ = "Avi"
__version__ = "0.1.0"


class TestBayesLine(TestCase):
    @staticmethod
    def make_data():
        n = 10
        time = np.linspace(start=0, stop=10, num=n)
        s = LineModel(size=n, m=priors.M_TRUE, c=priors.C_TRUE, x=time)
        n = GaussianNoiseModel(
            size=n, mu=priors.MU_NOISE_TRUE, sigma=priors.SIGMA_NOISE_TRUE, x=time
        )
        d = RecordedData(noise=n, signal=s)
        return BayesLine(data=d)

    def make_chunked_data(self):
        b_line = self.make_data()
        data_lists = split_recorded_data(recorded_data=b_line.data, splits=5)
        return BayesLineChunks(data_lists)

    def test_bayes_line_constructor(self):
        b_line = self.make_data()
        ln_like = b_line.ln_likelihood(theta=[0, 0])
        self.assertTrue(isinstance(ln_like, float))
        self.assertFalse(np.isnan(ln_like))

    def test_bayes_line_post_pro(self):
        b_line = self.make_data()
        ln_post_prob_grid = b_line.ln_posterior_grid_vec(
            mm=priors.M_GRID, cc=priors.C_GRID
        )
        self.assertFalse(np.isnan(ln_post_prob_grid).any())
        ln_evid = b_line.ln_evidence()
        self.assertTrue(ln_evid is not None)
        b_chunks = self.make_chunked_data()
        b_chunks.ln_posterior_grid_vec(mm=priors.M_GRID, cc=priors.C_GRID)

    def test_plotting(self):
        b_line = self.make_data()
        ln_post_prob_grid = b_line.ln_posterior_grid_vec(
            mm=priors.M_GRID, cc=priors.C_GRID
        )
        print(ln_post_prob_grid)
        plot_contour(twod_z_values=np.exp(ln_post_prob_grid), title="ln_post_prob_test")
