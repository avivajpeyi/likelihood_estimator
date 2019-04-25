#!/usr/bin/env python3
import os
from typing import Callable

import emcee
import numpy as np
import pandas as pd
from estimator.mcmc.config import SamplerConfig

#


class Sampler(object):
    def __init__(self, lnpostfn: Callable, config: SamplerConfig, samples_df_path: str):
        self.lnpostfn = lnpostfn
        self.config = config
        self.emcee_sampler = emcee.EnsembleSampler(
            nwalkers=config.n_walkers, dim=config.n_free_parameters, lnpostfn=lnpostfn
        )
        self._samples_df = None
        self.samples_df_path = samples_df_path
        if os.path.isfile(samples_df_path):
            self.load_samples_df()

    def run(self):
        self.emcee_sampler.run_mcmc(
            pos0=self.config.walkers_starting_positions, N=self.config.n_steps
        )

        # collect MCMC data + clean data
        samples_df = pd.DataFrame(
            {
                "m": self.emcee_sampler.flatchain[:, 0],
                "c": self.emcee_sampler.flatchain[:, 1],
                "likelihood": self.emcee_sampler.flatlnprobability[:],
            },
            columns=["m", "c", "likelihood"],
        )
        samples_df = samples_df.replace([np.inf, -np.inf], np.nan).dropna()

        # drop 2% of lowest likelihood sample values (keep 98% of samples)
        two_percent_val = samples_df.describe(percentiles=[0.02])["likelihood"]["2%"]
        samples_df = samples_df.loc[samples_df["likelihood"] > two_percent_val]

        self._samples_df = samples_df.copy()
        self.save_samples_df()

    def save_samples_df(self):
        self._samples_df.to_csv(self.samples_df_path, index=False, index_label=False)

    def load_samples_df(self):
        self._samples_df = pd.read_csv(self.samples_df_path)

    def get_samples_df(self):
        if self._samples_df is not None:
            return self._samples_df
        else:
            self.run()
            return self._samples_df

    def get_only_samples_array(self):
        df = self.get_samples_df()
        return df[df.columns[:-1]].to_numpy()
