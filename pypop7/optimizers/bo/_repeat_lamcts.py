"""Repeat the following paper for `LAMCTS`:
    Wang, L., Fonseca, R. and Tian, Y., 2020.
    Learning search space partition for black-box optimization using monte carlo tree search.
    Advances in Neural Information Processing Systems, 33, pp.19511-19522.
    https://arxiv.org/abs/2007.00708 (an updated version)
    https://proceedings.neurips.cc/paper/2020/hash/e2ce14e81dba66dbff9cbc35ecfdb704-Abstract.html
    (the original version)

    Luckily our Python code could still repeat the data reported in the updated Pyton code *well*.
    Therefore, we argue that its repeatability could be **well-documented**.
"""
import time

import numpy as np

from pypop7.optimizers.bo.lamcts import LAMCTS as Solver


class Levy(object):
    def __init__(self, ndim):
        self.ndim = ndim

    def __call__(self, x):
        w = 1.0 + (x - 1.0)/4.0
        y = np.sin(np.pi*w[0])**2 + np.sum((w[1:self.ndim - 1] - 1.0)**2 * (
                1.0 + 10.0*np.sin(np.pi*w[1:self.ndim - 1] + 1.0)**2)) + (
                w[self.ndim - 1] - 1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[self.ndim - 1])**2)
        return y


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 100
    problem = {'fitness_function': Levy(ndim_problem),
               'ndim_problem': ndim_problem,
               'lower_boundary': -10*np.ones((ndim_problem,)),
               'upper_boundary': 10*np.ones((ndim_problem,))}
    options = {'seed_rng': 3,
               'verbose': 1,
               'saving_fitness': 10,
               'max_function_evaluations': 15000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)
    print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
    # n_function_evaluations 15000: best_so_far_y 45.292
