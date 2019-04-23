#!/usr/bin/env python3

M_TRUE = -0.9594  # line slope of straight-line component
C_TRUE = 4.3  # size of the power-law component
SIGMA_NOISE_TRUE = 0.534  # std. dev. on noise (noise is Gaussian)
MU_NOISE_TRUE = 9.5  # mean of noise (noise is Gaussian)

# Ranges of the parameters
M_MIN, M_MAX = -2, 2
C_MIN, C_MAX = 2, 20
MU_MIN, MU_MAX = 6, 12
SIGMA_MIN, SIGMA_MAX = 0.1, 2
