"""Repeat the official implementation for `MDE`:
    Karkinli, A.E., 2023.
    Detection of object boundary from point cloud by using multi-population based differential evolution algorithm.
    Neural Computing and Applications, 35(7), pp.5193-5206.
    https://doi.org/10.1007/s00521-022-07969-w

    The author's official Python implementation (statistically checked against the original MATLAB code
    `algo_MDE.m`) is available at:
    https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm
    and can be installed via:
    $ pip install git+https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm.git

    When both random streams are aligned (since the official code uses one single random generator for both
    initialization and optimization), our Python code returns **bit-identical** results to the official code.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import numpy as np

from mde_optimizer import minimize  # official Python implementation from the author

from pypop7.benchmarks.base_functions import sphere, rastrigin, rosenbrock
from pypop7.optimizers.de.mde import MDE


if __name__ == '__main__':
    ndim_problem, n_sub = 10, 30
    max_function_evaluations = 3*n_sub + 1000*n_sub  # 30090 (1000 generations)
    for f, b in [(sphere, 100.0), (rastrigin, 5.12), (rosenbrock, 30.0)]:
        for seed in range(5):
            res_official = minimize(f, -b, b, dim=ndim_problem, pop_size=n_sub, max_iter=1000, seed=seed)
            problem = {'fitness_function': f,
                       'ndim_problem': ndim_problem,
                       'lower_boundary': -b*np.ones((ndim_problem,)),
                       'upper_boundary': b*np.ones((ndim_problem,))}
            options = {'max_function_evaluations': max_function_evaluations,
                       'n_individuals': 3*n_sub,
                       'seed_initialization': seed,
                       'saving_fitness': 1,
                       'verbose': False}
            mde = MDE(problem, options)
            # to align with the single random stream of the official code
            mde.rng_optimization = np.random.default_rng(seed)
            mde.rng_optimization.random((3*n_sub, ndim_problem))
            results = mde.optimize()
            # best-so-far fitness at the end of each generation (convergence curve)
            curve = results['fitness'][3*n_sub + n_sub*np.arange(1, 1001) - 1, 1]
            is_identical = (res_official.fun == results['best_so_far_y']) and np.array_equal(
                res_official.history, curve)
            print(f.__name__, seed, res_official.nfev, results['n_function_evaluations'],
                  res_official.fun, results['best_so_far_y'], is_identical)
    # all 15 runs: `True` (bit-identical final best-so-far fitness and whole convergence curve)
    # note that on plateaus (e.g., several different solutions all with 0.0 on rastrigin), the
    #   returned `best_so_far_x` may differ, since PyPop7 keeps the *first* solution reaching it
