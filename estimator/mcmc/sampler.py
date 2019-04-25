#!/usr/bin/env python3
import os
from typing import Callable, List, Optional

import emcee
import numpy as np
import pandas as pd
from estimator.mcmc import config


def set_up_mcmc_sampler(
    ln_prob_func: Callable,
    func_args: Optional[List] = [],
    ndim: Optional[int] = config.N_DIM,
    nwalkers: Optional[int] = config.N_WALKERS,
):
    """
    Set up for MCMC
    """
    starts = [0, 4]  # IF I DONT USE THIS AS STARTING POINT, I GET WEIRD MCMC
    walker_starting_pos = [
        starts + 1e-2 * np.random.normal(size=ndim) for _ in range(nwalkers)
    ]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_func, args=func_args)
    return walker_starting_pos, sampler


def do_sampling(
    ln_prob_func: Callable,
    func_args: Optional[list] = [],
    ndim: Optional[int] = config.N_DIM,
    nsteps: Optional[int] = config.N_STEPS,
    burnin_amount: Optional[int] = config.BURNIN_AMOUNT,
):
    # do MCMC
    walker_starting_pos, sampler = set_up_mcmc_sampler(ln_prob_func, func_args)
    sampler.run_mcmc(walker_starting_pos, nsteps)

    # collect MCMC data
    samples = sampler.chain[:, burnin_amount:, :].reshape((-1, ndim))
    samples_dataframe = pd.DataFrame(
        {
            "m": sampler.flatchain[burnin_amount:, 0],
            "c": sampler.flatchain[burnin_amount:, 1],
            "likelihood": sampler.flatlnprobability[burnin_amount:],
        },
        columns=["m", "c", "likelihood"],
    )

    # cleaning up data
    samples_dataframe = samples_dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # sanity check of data
    samples_dataframe.describe().transpose()

    return samples, samples_dataframe


# loading and saving data


def save_mcmc_samples_data(samples, samples_dataframe, df_path, np_path):
    # save data
    np.save(np_path, samples)
    samples_dataframe.to_csv(df_path, index=False, index_label=False)
    # making sure successful saving to drive
    exists = os.path.isfile(df_path) and os.path.isfile(np_path)
    if not exists:
        raise Exception("File not properly saved at {}".format(df_path))

    pass


def load_mcmc_samples_data(df_path, np_path):
    # loading data
    samples = np.load(np_path)
    samples_dataframe = pd.read_csv(df_path)
    samples_dataframe = samples_dataframe.replace([np.inf, -np.inf], np.nan).dropna()
    # sanity check of data
    samples_dataframe.describe().transpose()
    return samples, samples_dataframe


def get_mcmc_samples(
    load_previous_data: bool,
    df_path: str,
    np_path: str,
    ln_like_func: Callable,
    func_args: Optional[list] = [],
):
    if load_previous_data:
        samples, samples_df = load_mcmc_samples_data(df_path=df_path, np_path=np_path)
    else:
        samples, samples_df = do_sampling(ln_like_func, func_args=func_args)
        save_mcmc_samples_data(samples, samples_df, df_path=df_path, np_path=np_path)
    return samples, samples_df
