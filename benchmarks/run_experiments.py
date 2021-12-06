import time
import os
import pickle
import argparse
import numpy as np

import continuous_functions as cf


class Experiment(object):
    def __init__(self, index, function, seed):
        self.index = index
        self.function = function
        self.seed = seed
        self.ndim_problem = 1000
        self._folder = 'pypop_benchmarks_lso'
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

    def run(self, optimizer):
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0 * np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0 * np.ones((self.ndim_problem,))}
        if self.function.__name__ == 'exponential':
            fitness_threshold = -1
        else:
            fitness_threshold = 1e-10
        options = {'max_function_evaluations': 1e5 * self.ndim_problem,
                   'max_runtime': 3600 * 2,  # seconds
                   'fitness_threshold': fitness_threshold,
                   'seed_rng': self.seed,
                   'record_fitness': True,
                   'record_fitness_frequency': 2000,
                   'verbose': False}
        solver = optimizer(problem, options)
        results = solver.optimize()
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function.__name__,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.indices = range(self.start, self.end + 1)
        self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus,
                          cf.ellipsoid, cf.different_powers, cf.schwefel221, cf.step,
                          cf.schwefel222, cf.rosenbrock, cf.exponential, cf.schwefel12]
        rng = np.random.default_rng(2021)
        self.seeds = rng.integers(np.iinfo(np.int64).max, size=(2001, len(self.functions)))

    def run(self, optimizer):
        for index in self.indices:
            for f in self.functions:
                start_time = time.time()
                print('* function: {:s}'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[index, f])
                experiment.run(optimizer)
                print('    runtime: {:7.5e}'.format(time.time() - start_time))


if __name__ == '__main__':
    start_run = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--optimizer', '-o', type=str)
    args = parser.parse_args()
    params = vars(args)
    if params['optimizer'] == 'PRS':
        from optimizers.rs.prs import PRS as Optimizer
    experiments = Experiments(params['start'], params['end'])
    experiments.run(Optimizer)
    print('* Total runtime: {:7.5e}).'.format(time.time() - start_run))
