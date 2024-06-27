import numpy as np  # engine for numerical computing
from nevergrad.functions.photonics.core import Photonics


def benchmark_photonics(optimizer, ndim=10, max_function_evaluations=50000, seed=20221001):
    """Benchmark on the photonics model from the `NeverGrad
       <https://github.com/facebookresearch/nevergrad>`_ platform.

    Parameters
    ----------
    optimizer                : class
                               any black-box optimizer.
    ndim                     : int
                               number of dimensionality of the fitness function to be minimized.
    max_function_evaluations : int
                               maximum of function evalutations.
    seed                     : int
                               seed for random number generation (RNG).

    Returns
    -------
    results : dict
              final optimization results.
    """
    half = int(ndim / 2)
    # define problem arguments
    problem = {'fitness_function': Photonics('bragg', ndim),
               'ndim_problem': ndim,
               'lower_boundary': np.hstack((2.0 * np.ones(half), 30.0 * np.ones(half))),
               'upper_boundary': np.hstack((3.0 * np.ones(half), 180.0 * np.ones(half)))}
    # set algorithm options
    options = {'max_function_evaluations': max_function_evaluations,
               'is_bound': True,
               'seed_rng': seed,
               'saving_fitness': 1,
               'verbose': 100}
    results = optimizer(problem, options).optimize()
    return results
