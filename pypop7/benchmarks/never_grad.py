import numpy as np
import matplotlib.pyplot as plt
from nevergrad.functions.photonics.core import Photonics


def benchmark_nevergrad(optimizer, ndim=10, max_function_evaluations=50000, seed=20221001):
    """Test the application from `nevergrad` platform for Large-Scale Black-Box Optimization (LBO).

    Parameters
    ----------
    optimizer                : class
                               any black-box optimizer.
    ndim                     : int
                               number of dimensionality.
    max_function_evaluations : int
                               maximum of function evalutations.
    seed                     : int
                               seed for random number generation (RNG).

    Returns
    -------
    results                  : dict
                               final optimization results.
    """
    half = int(ndim/2)
    func = Photonics("bragg", ndim)
    problem = {'fitness_function': func,
               'ndim_problem': ndim,
               'lower_boundary': np.hstack((2*np.ones(half), 30*np.ones(half))),
               'upper_boundary': np.hstack((3*np.ones(half), 180*np.ones(half)))}
    options = {'max_function_evaluations': max_function_evaluations,
               'n_individuals': 200,
               'is_bound': True,
               'seed_rng': seed,
               'saving_fitness': 1,
               'verbose': 200}
    solver = optimizer(problem, options)
    results = solver.optimize()
    res = results['fitness']

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'
    plt.figure(figsize=(7, 7))
    plt.grid(True)
    plt.plot(res[:, 0], res[:, 1], linewidth=2.0, linestyle='-', label=optimizer.__name__)
    plt.legend()
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Fitness (to be Minimized)')
    plt.title('Bragg Mirrors Structure')
    plt.show()

    return results
