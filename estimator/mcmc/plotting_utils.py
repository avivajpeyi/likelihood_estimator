#!/usr/bin/env python3
import matplotlib
from estimator.bayes import priors

matplotlib.use("PS")


def plot_corner(samples, fname):
    import corner
    import matplotlib.pyplot as plt

    corner.corner(samples, labels=["$m$", "$c$"], truths=[priors.M_TRUE, priors.C_TRUE])
    plt.savefig(fname)
