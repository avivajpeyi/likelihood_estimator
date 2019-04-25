#!/usr/bin/env python3
from typing import Optional

import numpy as np
from estimator.models.data_container_base import DataContainer


class LineModel(DataContainer):
    def __init__(
        self, size: int, m: float, c: float, x: np.array, y: Optional[np.array] = None
    ):
        super(LineModel, self).__init__(size, x, y)
        self.m = m
        self.c = c
        self.y = LineModel.model(x, m, c)

    @staticmethod
    def model(x: np.array, m: float, c: float) -> np.array:
        return (m * np.array(x)) + c
