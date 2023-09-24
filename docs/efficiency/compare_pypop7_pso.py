# Please first install pypop7 (see https://pypop.rtfd.io/ for details):
#    $ pip install pypop7
import pickle
import argparse

import numpy

from pypop7.benchmarks.base_functions import sphere  # function to be minimized
from pypop7.optimizers.pso.spso import SPSO  # Standard Particle Swarm Optimizer with a global topology


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', '-i', type=int)  # index of the experiment
    args = parser.parse_args()
    params = vars(args)
    ndim_problem = 2000
    problem = {'fitness_function': sphere,  # define problem arguments
               'ndim_problem': ndim_problem,
               'lower_boundary': -10.0*numpy.ones((ndim_problem,)),
               'upper_boundary': 10.0*numpy.ones((ndim_problem,))}
    options = {'max_runtime': 60*60*3,  # set optimizer options
               'verbose': False,
               'saving_fitness': 2000,
               'seed_rng': params['index']}
    spso = SPSO(problem, options)  # initialize the optimizer class
    results = spso.optimize()  # run the optimization process
    # return the number of function evaluations and best-so-far fitness
    print(f"SPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
    with open('PYPOP7-PSO.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
