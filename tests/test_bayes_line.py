#!/usr/bin/env python3
from unittest import TestCase

import estimator.bayes.priors as priors
import numpy as np
from tests.utils import make_chunked_data, make_data

__author__ = "Avi"
__version__ = "0.1.0"


class TestBayesLine(TestCase):
    def test_bayes_line_constructor(self):
        _, _, _, b_line = make_data()
        ln_like = b_line.ln_likelihood(theta=[0, 0])
        self.assertTrue(isinstance(ln_like, float))
        self.assertFalse(np.isnan(ln_like))

    def test_bayes_line_post_pro(self):
        _, _, _, b_line = make_data()
        ln_post_prob_grid = b_line.ln_posterior_grid_vec(
            mm=priors.M_GRID, cc=priors.C_GRID
        )
        self.assertFalse(np.isnan(ln_post_prob_grid).any())
        ln_evid = b_line.ln_evidence()
        self.assertTrue(ln_evid is not None)
        b_chunks = make_chunked_data()
        b_chunks.ln_posterior_grid_vec(mm=priors.M_GRID, cc=priors.C_GRID)
