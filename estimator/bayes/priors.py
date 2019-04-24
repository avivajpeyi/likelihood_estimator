#!/usr/bin/env python3
import numpy as np

# SIGNAL PARAMETERS
M_TRUE = -0.9594  # line slope of straight-line component
C_TRUE = 4.3  # size of the power-law component
M_MIN, M_MAX = -2, 2
C_MIN, C_MAX = 2, 20
M_VEC, C_VEC = np.linspace(M_MIN, M_MAX, 100), np.linspace(C_MIN, C_MAX, 100)
M_GRID, C_GRID = np.meshgrid(M_VEC, C_VEC)
DM, DC = M_VEC[1] - M_VEC[0], C_VEC[1] - C_VEC[0]


# NOISE PARAMETERS
SIGMA_NOISE_TRUE = 0.534  # std. dev. on noise (noise is Gaussian)
MU_NOISE_TRUE = 9.5  # mean of noise (noise is Gaussian)
MU_MIN, MU_MAX = 6, 12
SIGMA_MIN, SIGMA_MAX = 0.1, 2
