#!/usr/bin/env python3

import estimator.bayes.priors as priors
import numpy as np
from estimator.bayes.bayes_line import BayesLine, BayesLineChunks
from estimator.models.gaussian_noise import GaussianNoiseModel
from estimator.models.line import LineModel
from estimator.models.toy_dataset import RecordedData, split_recorded_data


def make_data():
    n = 10
    time = np.array(
        [
            1.7808099,
            2.37694209,
            2.8589569,
            3.40190215,
            3.54795612,
            3.59507844,
            5.98858946,
            8.52395088,
            8.84853293,
            9.75006494,
        ]
    )
    s = LineModel(size=n, m=priors.M_TRUE, c=priors.C_TRUE, x=time)
    n = GaussianNoiseModel(
        size=n, mu=priors.MU_NOISE_TRUE, sigma=priors.SIGMA_NOISE_TRUE, x=time
    )
    d = RecordedData(noise=n, signal=s)
    b = BayesLine(data=d)
    return d, n, s, b


def make_chunked_data() -> BayesLineChunks:
    d, n, s, b_line = make_data()
    data_lists = split_recorded_data(recorded_data=b_line.data, splits=5)
    return BayesLineChunks(data_lists)
