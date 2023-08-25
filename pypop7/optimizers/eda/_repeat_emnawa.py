"""Repeat the following paper for `EMNAWA`:
    Teytaud, F. and Teytaud, O., 2009, July.
    Why one must use reweighting in estimation of distribution algorithms.
    In Proceedings of ACM Annual Conference on Genetic and Evolutionary Computation (pp. 453-460)
    https://doi.org/10.1145/1569901.1569964

    There seems to be vey small performance gaps between the original paper and our Python code, which
    may be ignored for benchmarking. Such gaps may attribute to a slight implementation difference
    (where our Python code does not use *ad-hoc elitist selection* employed in the original paper).
    We argue that its repeatability can be **well-documented**.
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.eda.emnawa import EMNAWA


if __name__ == '__main__':
    ndim_problem = 3

    problem = {'fitness_function': sphere,
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'max_function_evaluations': 10*np.power(ndim_problem, 3)*25*int(np.power(ndim_problem, 3/2)),
               'n_individuals': 10*np.power(ndim_problem, 3),
               'seed_rng': 0}  # undefined in the original paper
    emnawa = EMNAWA(problem, options)
    results = emnawa.optimize()
    print(results)
    print(results['best_so_far_y'])
