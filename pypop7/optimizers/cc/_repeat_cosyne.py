"""Repeat the following paper for `COSYNE`:
    Gomez, F., Schmidhuber, J. and Miikkulainen, R., 2008.
    Accelerated neural evolution through cooperatively coevolved synapses.
    Journal of Machine Learning Research, 9(31), pp.937-965.
    https://jmlr.org/papers/v9/gomez08a.html

    We notice that the EvoTorch library provides an open-source implementation for it:
    https://docs.evotorch.ai/v0.3.0/reference/evotorch/algorithms/ga/#evotorch.algorithms.ga.Cosyne
    However, we found that it (without explicit decomposition) cannot match the original paper perfectly.
    In our implementation, we refer to this open-source implementation rather than the original paper,
    but with a slight simplification for the *permutation* operator.

    Luckily our code could repeat the data reported in the reference library *well*.
    Therefore, we argue that the repeatability of `COSYNE` could be **well-documented**.

    The reference code based on EvoTorch is given below:

    import torch
    from evotorch import Problem
    from evotorch.algorithms import ga
    from evotorch.logging import StdOutLogger

    def norm(x: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.norm(x, dim=-1)).pow(2)

    problem = Problem('min', norm, initial_bounds=(-5.0, 5.0), solution_length=1000)
    searcher = ga.Cosyne(problem, popsize=100, tournament_size=10, mutation_probability=1.0, mutation_stdev=1.0)
    logger = StdOutLogger(searcher)
    searcher.run(num_generations=3000)
"""
import time

import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.cc.cosyne import COSYNE as Solver


if __name__ == '__main__':
    start_run = time.time()
    ndim_problem = 1000
    for f in [sphere]:
        print('*' * 7 + ' ' + f.__name__ + ' ' + '*' * 7)
        problem = {'fitness_function': f,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 150 * 3000 + 100,
                   'sigma': 1.0,
                   'seed_rng': 0}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)  # 7613.71992419764 vs 7388.05517578125  (from the EvoTorch library)
        # 10 dimension: 0.5348416948073422 vs 0.03605746477842331 (from the EvoTorch library)
        print('*** Runtime: {:7.5e}'.format(time.time() - start_run))
