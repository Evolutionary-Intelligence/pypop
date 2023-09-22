# Please first install pypop7 (see https://pypop.rtfd.io/ for details):
#    $ pip install pypop7
import pickle

import numpy

from pypop7.benchmarks.base_functions import sphere  # function to be minimized
from pypop7.optimizers.de.cde import CDE


if __name__ == "__main__":
    ndim_problem = 2000
    problem = {'fitness_function': sphere,  # define problem arguments
               'ndim_problem': ndim_problem,
               'lower_boundary': -10.0*numpy.ones((ndim_problem,)),
               'upper_boundary': 10.0*numpy.ones((ndim_problem,))}
    options = {'max_runtime': 60*60*3,  # set optimizer options
               'verbose': False,
               'seed_rng': 0}
    cde = CDE(problem, options)  # initialize the optimizer class
    results = cde.optimize()  # run the optimization process
    # return the number of function evaluations and best-so-far fitness
    print(f"CDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
    with open('PYPOP7-DE.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
