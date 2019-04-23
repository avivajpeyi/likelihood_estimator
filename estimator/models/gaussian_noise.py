#!/usr/bin/env python3

from typing import Optional

import numpy as np
from estimator.models.data_container_base import DataContainer


class GaussianNoiseModel(DataContainer):
    def __init__(
        self,
        size: int,
        mu: float,
        sigma: float,
        x: np.array,
        y: Optional[np.array] = None,
    ):
        super(GaussianNoiseModel, self).__init__(size, x, y)
        self.mu = mu
        self.sigma = sigma
        self.y = GaussianNoiseModel.model(mu, sigma, size)

    @staticmethod
    def model(mu: float, sigma: float, size: int) -> np.array:
        return np.random.normal(loc=mu, scale=sigma, size=size)
