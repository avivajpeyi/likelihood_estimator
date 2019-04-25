#!/usr/bin/env python3
from typing import List, Optional

import numpy as np

N_WALKERS = 100
N_STEPS = 2000
BURNIN_AMOUNT = 0


class SamplerConfig(object):
    def __init__(
        self,
        n_free_parameters: int,
        start_pos: List[int],
        n_walkers: Optional[int] = N_WALKERS,
        n_steps: Optional[int] = N_STEPS,
        burnin_amount: Optional[int] = BURNIN_AMOUNT,
    ):
        self.n_free_parameters = n_free_parameters
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.burnin_amount = burnin_amount
        self.walkers_starting_positions = [
            start_pos + 1e-2 * np.random.normal(size=n_free_parameters)
            for _ in range(n_walkers)
        ]
