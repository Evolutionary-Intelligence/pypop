"""Before running this script, please first run the following script to generate necessary data:
    https://github.com/Evolutionary-Intelligence/pypop/blob/main/tutorials/benchmarking_lsbbo_1.py
"""
import os
import time
import pickle
import argparse

import numpy as np

import pypop7.benchmarks.continuous_functions as cf


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
                   'saving_fitness': 2000,
                   'verbose': 0}
        if optimizer.__name__ in ['PRS', 'SRS', 'GS', 'BES', 'HJ', 'NM', 'POWELL', 'FEP', 'GENITOR', 'G3PCX',
                                  'GL25', 'COCMA', 'HCC', 'SPSO', 'SPSOL', 'CLPSO', 'CCPSO2']:
            options['sigma'] = 20.0/3.0
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
        # for testing the local search ability
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


if __name__ == '__main__':
    start_runtime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)  # starting index of experiments (from 0 to 49)
    parser.add_argument('--end', '-e', type=int)  # ending index of experiments (from 0 to 49)
    parser.add_argument('--optimizer', '-o', type=str)  # any optimizer from PyPop7
    parser.add_argument('--ndim_problem', '-d', type=int, default=2000)  # dimension of fitness function
    args = parser.parse_args()
    params = vars(args)
    assert isinstance(params['start'], int) and 0 <= params['start'] < 50  # from 0 to 49
    assert isinstance(params['end'], int) and 0 <= params['end'] < 50  # from 0 to 49
    assert isinstance(params['optimizer'], str)
    assert isinstance(params['ndim_problem'], int) and params['ndim_problem'] > 0
    if params['optimizer'] == 'PRS':
        from pypop7.optimizers.rs.prs import PRS as Optimizer
    elif params['optimizer'] == 'SRS':
        from pypop7.optimizers.rs.srs import SRS as Optimizer
    elif params['optimizer'] == 'GS':
        from pypop7.optimizers.rs.gs import GS as Optimizer
    elif params['optimizer'] == 'BES':
        from pypop7.optimizers.rs.bes import BES as Optimizer
    elif params['optimizer'] == 'HJ':
        from pypop7.optimizers.ds.hj import HJ as Optimizer
    elif params['optimizer'] == 'NM':
        from pypop7.optimizers.ds.nm import NM as Optimizer
    elif params['optimizer'] == 'POWELL':
        from pypop7.optimizers.ds.powell import POWELL as Optimizer
    elif params['optimizer'] == 'FEP':
        from pypop7.optimizers.ep.fep import FEP as Optimizer
    elif params['optimizer'] == 'GENITOR':
        from pypop7.optimizers.ga.genitor import GENITOR as Optimizer
    elif params['optimizer'] == 'G3PCX':
        from pypop7.optimizers.ga.g3pcx import G3PCX as Optimizer
    elif params['optimizer'] == 'GL25':
        from pypop7.optimizers.ga.gl25 import GL25 as Optimizer
    elif params['optimizer'] == 'COCMA':
        from pypop7.optimizers.cc.cocma import COCMA as Optimizer
    elif params['optimizer'] == 'HCC':
        from pypop7.optimizers.cc.hcc import HCC as Optimizer
    elif params['optimizer'] == 'SPSO':
        from pypop7.optimizers.pso.spso import SPSO as Optimizer
    elif params['optimizer'] == 'SPSOL':
        from pypop7.optimizers.pso.spsol import SPSOL as Optimizer
    elif params['optimizer'] == 'CLPSO':
        from pypop7.optimizers.pso.clpso import CLPSO as Optimizer
    elif params['optimizer'] == 'CCPSO2':
        from pypop7.optimizers.pso.ccpso2 import CCPSO2 as Optimizer
    else:
        raise ValueError(f"Cannot find optimizer class {params['optimizer']} in PyPop7!")
    experiments = Experiments(params['start'], params['end'], params['ndim_problem'])
    experiments.run(Optimizer)
    print('Total runtime: {:7.5e}.'.format(time.time() - start_runtime))
