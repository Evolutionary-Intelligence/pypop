"""Repeat Figure 1 of the JMLR-2014-NES paper:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


class CanonicalSearchGradientAlgorithm(object):
    """Canonical Search Gradient Algorithm Supporting Only One-Dimensional Gaussian Distribution.
        See Algorithm 1, Algorithm 2, Section 2.2, and Figure 1 of the JMLR-2014-NES paper for more details.
    """
    def __init__(self, f, ndim_problem=1, max_generations=1e3, eta=0.01, n_individuals=10, seed_rng=0):
        self.f = f  # fitness function (problem) to be minimized
        self.ndim_problem = ndim_problem  # number of dimensionality of problem
        self.max_generations = max_generations  # maximum of generations
        self.eta = eta  # learning rate
        self.n_individuals = n_individuals  # number of individuals (population size, lambda)
        self.seed_rng = seed_rng  # seed of random number generator (rng)

    def initialize(self, mu=1.0, sigma=1.0):
        z = np.empty((self.n_individuals, self.ndim_problem))  # population
        f_z = np.empty((self.n_individuals,))  # fitness of population
        g_mu = np.empty((self.n_individuals, self.ndim_problem))  # log-derivatives of mean of Gaussian distribution
        g_sigma = np.empty((self.n_individuals, self.ndim_problem))  # log-derivatives of std of Gaussian distribution
        history = {"mu": [mu], "sigma": [sigma]}
        return z, f_z, g_mu, g_sigma, mu, sigma, history

    def iterate(self, z, f_z, g_mu, g_sigma, mu, sigma):
        inv_sigma = np.linalg.inv(np.array([[sigma]]))
        for k in range(self.n_individuals):
            # draw sample
            z[k] = mu + sigma * default_rng(self.seed_rng).standard_normal((self.ndim_problem,))
            # evaluate fitness
            f_z[k] = self.f(z[k])
            # calculate fitness-weighted log-derivatives
            zk_minus_mu = z[k] - mu
            g_mu[k] = (inv_sigma * zk_minus_mu) * f_z[k]
            g_sigma[k] = (-0.5 * inv_sigma + 0.5 * inv_sigma * zk_minus_mu * zk_minus_mu * inv_sigma) * f_z[k]
        return g_mu, g_sigma

    def optimize(self):
        z, f_z, g_mu, g_sigma, mu, sigma, history = self.initialize()
        for _ in range(int(self.max_generations)):
            g_mu, g_sigma = self.iterate(z, f_z, g_mu, g_sigma, mu, sigma)
            mu -= float(self.eta * np.mean(g_mu, 0))
            sigma -= float(self.eta * np.mean(g_sigma, 0))
            history["mu"].append(mu)
            history["sigma"].append(sigma)
        return history


if __name__ == "__main__":
    def sphere(x):
        return np.power(x, 2)
    n_trials = 3
    stat_mean, stat_std = np.empty((n_trials, 1001)), np.empty((n_trials, 1001))
    for t in range(n_trials):
        sg = CanonicalSearchGradientAlgorithm(sphere, seed_rng=t)
        history_mean_std = sg.optimize()
        stat_mean[t] = np.array(history_mean_std["mu"])
        stat_std[t] = np.array(history_mean_std["sigma"])
    plt.plot(np.median(stat_mean, 0), "black")
    plt.plot(np.median(stat_std, 0), "red")
    plt.yscale("log")
    plt.show()
