#!/usr/bin/env python3
"""
Module Docstring
"""
from unittest import TestCase
from unittest.mock import Mock, patch

import estimator.bayes.priors as priors
import numpy as np
from estimator.models.data_container_base import DataContainer
from estimator.models.gaussian_noise import GaussianNoiseModel
from estimator.models.line import LineModel
from estimator.models.plotting_utils import plot_toy_dataset
from estimator.models.toy_dataset import RecordedData, split_recorded_data

__author__ = "Avi"
__version__ = "0.1.0"


class TestDataModels(TestCase):
    @staticmethod
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
        return d, n, s

    def test_constructors(self):
        d, n, s = self.make_data()
        self.assertTrue(np.alltrue(d.x == s.x))
        self.assertTrue(np.alltrue(d.y == s.y + n.y))

    def test_split_recorded_data(self):
        d, n, s = self.make_data()
        num_splits = 5
        recorded_data_list = split_recorded_data(recorded_data=d, splits=num_splits)
        self.assertTrue(len(recorded_data_list) == num_splits)

        num_splits = 100
        recorded_data_list = split_recorded_data(recorded_data=d, splits=num_splits)
        self.assertTrue(len(recorded_data_list) == 1)

    def test_line_model(self):
        sig = LineModel.model([0, 1, 2, 3], m=1, c=0)
        self.assertFalse(np.isnan(sig).any())

    def test_str(self):
        d, n, s = self.make_data()
        self.assertTrue(isinstance(d.__repr__(), str))

    def test_cannot_instantiate(self):
        """showing we normally can't instantiate an abstract class"""
        with self.assertRaises(TypeError):
            DataContainer()

    @patch.multiple(
        DataContainer, __abstractmethods__=set(), model=Mock(return_value=3)
    )
    def test_concrete_method(self):
        """patch abstract class  and its abstract methods for duration of the test"""
        my_abstract = DataContainer(size=10)
        self.assertEqual(my_abstract.model(), 3)

        DataContainer.from_dict({})

    def test_plot(self):
        d, n, s = self.make_data()
        num_splits = 5
        recorded_data_list = split_recorded_data(recorded_data=d, splits=num_splits)
        plot_toy_dataset(recorded_data_list, fname="toy_data_test.html")
