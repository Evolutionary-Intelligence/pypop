"""Repeat the following paper for `JADE`:
    Ahrari, A., Deb, K. and Preuss, M., 2017.
    Multimodal optimization by covariance matrix self-adaptation evolution strategy with repelling subpopulations.
    Evolutionary computation, 25(3), pp.439-471.
    https://doi.org/10.1162/evco_a_00182

    Very close performance can be obtained by our code. Therefore, we argue that
    the repeatability of `JADE` can be well-documented (*at least partly*).
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere, step, rosenbrock, rastrigin, ackley
from pypop7.optimizers.es.rscmsa import RSCMSA


if __name__ == '__main__':
    ndim_problem = 30

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'fitness_threshold': 1e-5,
               'seed_rng': 0,  # undefined in the original paper
               }
    rscmsa = RSCMSA(problem, options)
    results = rscmsa.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 7.09e-05
    # vs (from the original paper)

    problem = {'fitness_function': rosenbrock,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'fitness_threshold': 1e-5,
               'seed_rng': 0,  # undefined in the original paper
               }
    rscmsa = RSCMSA(problem, options)
    results = rscmsa.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 22.80
    # vs (from the original paper)

    problem = {'fitness_function': rastrigin,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10 * np.ones((ndim_problem,)),
               'upper_boundary': 10 * np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 1500 * 100,
               'fitness_threshold': 1e-5,
               'seed_rng': 0,  # undefined in the original paper
               }
    rscmsa = RSCMSA(problem, options)
    results = rscmsa.optimize()
    print(results)
    print(results['best_so_far_y'])
    # 64.68
    # vs (from the original paper)
