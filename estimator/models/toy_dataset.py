#!/usr/bin/env python3

from typing import List

import numpy as np
from estimator.models.data_container_base import DataContainer
from estimator.models.gaussian_noise import GaussianNoiseModel
from estimator.models.line import LineModel


class RecordedData(DataContainer):
    def __init__(self, noise: GaussianNoiseModel, signal: LineModel):
        super(RecordedData, self).__init__(size=noise.size, x=noise.x)
        self.noise = noise
        self.signal = signal
        self.y = RecordedData.model(noise, signal)

    @staticmethod
    def model(noise: GaussianNoiseModel, signal: LineModel) -> np.array:
        return noise.y + signal.y


def split_recorded_data(recorded_data: RecordedData, splits: int) -> List[RecordedData]:
    split_data_sets = []
    split_data_size = recorded_data.size / splits
    if split_data_size.is_integer():
        split_data_size = int(split_data_size)
        sig_split_x = np.split(recorded_data.signal.x, splits)
        sig_split_y = np.split(recorded_data.signal.y, splits)
        noise_split_x = np.split(recorded_data.noise.x, splits)
        noise_split_y = np.split(recorded_data.noise.y, splits)

        for split_idx in range(splits):
            split_data_sets.append(
                RecordedData(
                    signal=LineModel(
                        size=split_data_size,
                        m=recorded_data.signal.m,
                        c=recorded_data.signal.c,
                        x=sig_split_x[split_idx],
                        y=sig_split_y[split_idx],
                    ),
                    noise=GaussianNoiseModel(
                        size=split_data_size,
                        mu=recorded_data.noise.mu,
                        sigma=recorded_data.noise.sigma,
                        x=noise_split_x[split_idx],
                        y=noise_split_y[split_idx],
                    ),
                )
            )

    else:  # if cant split the data into an equal division
        split_data_sets = [recorded_data]

    return split_data_sets
