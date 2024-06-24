# Written/Checked by Chang Shao, Mingyang Feng, and *Qiqi Duan*
import os
import time
import pickle

import numpy as np

import pypop7.benchmarks.continuous_functions as cf
from pypop7.optimizers.rs.prs import PRS as Optimizer


class Experiment(object):
    def __init__(self, index, function, seed, ndim_problem):
        self.index, self.seed = index, seed
        self.function, self.ndim_problem = function, ndim_problem
        self._folder = 'pypop7_benchmarks_lso'  # to save all local data generated during optimization
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')  # file format

    def run(self, optimizer):
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0*np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
        options = {'max_function_evaluations': 100000*self.ndim_problem,
                   'max_runtime': 3600*3,  # seconds (=3 hours)
                   'fitness_threshold': 1e-10,
                   'seed_rng': self.seed,
                   'sigma': 20.0/3.0,
                   'saving_fitness': 2000,
                   'verbose': 0}
        options['temperature'] = 100.0  # for simulated annealing (SA)
        solver = optimizer(problem, options)
        results = solver.optimize()
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function.__name__,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:  # data format (pickle)
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    def __init__(self, start, end, ndim_problem):
        self.start, self.end = start, end
        self.ndim_problem = ndim_problem
        self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                          cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
        self.seeds = np.random.default_rng(2022).integers(  # for repeatability
            np.iinfo(np.int64).max, size=(len(self.functions), 50))

    def run(self, optimizer):
        for index in range(self.start, self.end + 1):
            print('* experiment: {:d} ***:'.format(index))
            for i, f in enumerate(self.functions):
                start_time = time.time()
                print('  * function: {:s}:'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[i, index], self.ndim_problem)
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


def test_local_search_for_lso(optimizer):
    experiments = Experiments(1, 14, 2000)
    experiments.run(optimizer)


if __name__ == '__main__':
    start_runtime = time.time()
    test_local_search_for_lso(optimizer=Optimizer)
    print('Total runtime: {:7.5e}.'.format(time.time() - start_runtime))
