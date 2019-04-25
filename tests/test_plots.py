#!/usr/bin/env python3
from unittest import TestCase

import estimator.bayes.priors as priors
import numpy as np
from estimator.bayes.plotting_utils import plot_contour
from estimator.mcmc.config import SamplerConfig
from estimator.mcmc.plotting_utils import plot_corner
from estimator.mcmc.sampler import Sampler
from estimator.models.plotting_utils import plot_toy_dataset
from estimator.models.toy_dataset import split_recorded_data
from tests.utils import make_chunked_data, make_data


class TestPlots(TestCase):
    def test_contour_plot(self):
        _, _, _, b_line = make_data()
        ln_post_prob_grid = b_line.ln_posterior_grid_vec(
            mm=priors.M_GRID, cc=priors.C_GRID
        )
        plot_contour(
            twod_z_values=np.exp(ln_post_prob_grid),
            title="tests/temp/ln_post_prob_test.html",
        )

    def test_corner_plot(self):
        d, n, s, b = make_data()
        c = SamplerConfig(n_free_parameters=2, start_pos=[0, 4])
        sampler = Sampler(
            lnpostfn=b.ln_likelihood, config=c, samples_df_path="tests/temp/test.csv"
        )
        plot_corner(
            samples=sampler.get_only_samples_array(), fname="tests/temp/corner_test.png"
        )
        b_chunks = make_chunked_data()
        sampler = Sampler(
            lnpostfn=b_chunks.ln_likelihood,
            config=c,
            samples_df_path="tests/temp/test2.csv",
        )
        plot_corner(
            samples=sampler.get_only_samples_array(),
            fname="tests/temp/net_corner_test.png",
        )

    def test_data_plot(self):
        d, n, s, _ = make_data()
        num_splits = 5
        recorded_data_list = split_recorded_data(recorded_data=d, splits=num_splits)
        plot_toy_dataset(recorded_data_list, fname="tests/temp/toy_data_test.html")
