#!/usr/bin/env python3
from unittest import TestCase

from estimator.mcmc.config import SamplerConfig
from estimator.mcmc.sampler import Sampler
from tests.utils import make_data

__author__ = "Avi"
__version__ = "0.1.0"


class TestMCMC(TestCase):
    def test_sampler_constructor(self):
        d, n, s, b = make_data()
        c = SamplerConfig(n_free_parameters=2, start_pos=[0, 4])
        Sampler(lnpostfn=b.ln_likelihood, config=c, samples_df_path="test.csv")
        Sampler(
            lnpostfn=b.ln_likelihood, config=c, samples_df_path="load_in_df_test.csv"
        )
