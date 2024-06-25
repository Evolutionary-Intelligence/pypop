import time

import numpy as np  # engine for numerical computing

from pypop7.benchmarks.utils import save_optimization
from pypop7.benchmarks.shifted_functions import generate_shift_vector
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix
import pypop7.benchmarks.continuous_functions as cf


class Experiment(object):
    def __init__(self, index, function, seed, ndim_problem, max_runtime):
        self.index, self.seed = index, seed
        self.function, self.ndim_problem = function, ndim_problem
        self.max_runtime = max_runtime

    def run(self, optimizer):
        # define problem arguments
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0 * np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0 * np.ones((self.ndim_problem,))}
        # set algorithm options
        if self.function != cf.michalewicz:  # special case
            fitness_threshold = 1e-10
        else:
            fitness_threshold = -np.inf
        options = {'max_function_evaluations': 100000 * self.ndim_problem,
                   'max_runtime': self.max_runtime,  # seconds
                   'fitness_threshold': fitness_threshold,
                   'seed_rng': self.seed,
                   'sigma': 20.0 / 3.0,
                   'saving_fitness': 2000,
                   'verbose': 0,
                   'temperature': 100.0,  # for simulated annealing (SA)
                   }
        # run black-box optimizer
        solver = optimizer(problem, options)
        results = solver.optimize()
        save_optimization(results,
                          solver.__class__.__name__,
                          solver.fitness_function.__name__,
                          solver.ndim_problem,
                          self.index)


class Experiments(object):
    def __init__(self, start, end, ndim_problem, max_runtime, is_local=True):
        """

        Parameters
        ----------
        start        : int
                       starting index of experiment.
        end          : int
                       ending index of experiment.
        ndim_problem : int
                       number of dimensionality.
        max_runtime  : int
                       maximum of runtime to be allowed
        is_local     : bool
                       flag to determine if local or global optimization functions should be tested

        """
        self.start, self.end = start, end
        self.ndim_problem = ndim_problem  # number of dimensionality
        self.max_runtime = max_runtime  # maximum of runtime to be allowed
        if is_local:  # to test local search abilities on 10 cases
            self.functions = [cf.sphere, cf.cigar,
                              cf.discus, cf.cigar_discus,
                              cf.ellipsoid, cf.different_powers,
                              cf.schwefel221, cf.step,
                              cf.rosenbrock, cf.schwefel12]
        else:  # to test global search abilities on 10 cases
            self.functions = [cf.griewank, cf.bohachevsky,
                              cf.ackley, cf.rastrigin,
                              cf.scaled_rastrigin, cf.skew_rastrigin,
                              cf.levy_montalvo, cf.michalewicz,
                              cf.salomon, cf.schaffer]
        self.seeds = np.random.default_rng(2022).integers(  # for repeatability
            np.iinfo(np.int64).max, size=(len(self.functions), 50))

    def run(self, optimizer):
        for index in range(self.start, self.end + 1):
            print('*** experiment: {:d} ***:'.format(index))
            for i, f in enumerate(self.functions):
                start_time = time.time()
                print('  ** function: {:s}:'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[i, index],
                                        self.ndim_problem, self.max_runtime)
                experiment.run(optimizer)
                print('    * [runtime]: {:7.5e}.'.format(time.time() - start_time))


def benchmark_local_search(optimizer, ndim=2000, max_runtime=3600*3,
                           start_index=1, end_index=14,
                           seed=20221001):
    """Test **Local Search** Abilities for Large-Scale Black-Box Optimization (LBO).

    Parameters
    ----------
    optimizer   : class
                  any black-box optimizer.
    ndim        : int
                  number of dimensionality.
    max_runtime : float
                  maximum of runtime to be allowed (seconds).
    start_index : int
                  starting index of independent experiments.
    end_index   : int
                  ending index of independent experiments.
    seed        : int
                  seed for random number generation (RNG).

    Returns
    -------
    A set of data files from independent experiments in the working space (`pwd()`).
    """
    # use the following 10 benchmarking function common in the
    # black-box optimization community
    functions = [cf.sphere.__name__, cf.cigar.__name__,
                 cf.discus.__name__, cf.cigar_discus.__name__,
                 cf.ellipsoid.__name__, cf.different_powers.__name__,
                 cf.schwefel221.__name__, cf.step.__name__,
                 cf.rosenbrock.__name__, cf.schwefel12.__name__]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions),))
    print('[Preprocessing] - First generate shift vectors for all (10) test functions.')
    for i, f in enumerate(functions):
        generate_shift_vector(f, ndim, -9.5, 9.5, seeds[i])
    print('[Preprocessing] - Second generate rotation matrices for all (10) test functions.')
    start_run = time.time()
    for i, f in enumerate(functions):
        start_time = time.time()
        generate_rotation_matrix(f, ndim, seeds[i])
        print(' * {:d}-d {:s}: runtime {:7.5e}'.format(
            ndim, f, time.time() - start_time))
    print('[Preprocessing] - Total runtime: {:7.5e}.'.format(time.time() - start_run))
    experiments = Experiments(start_index, end_index, ndim, max_runtime)
    experiments.run(optimizer)


def benchmark_global_search(optimizer, ndim=2000, max_runtime=3600*3,
                            start_index=1, end_index=14,
                            seed=20221001):
    """Test **Global Search** Abilities for Large-Scale Black-Box Optimization (LBO).

    Parameters
    ----------
    optimizer   : class
                  any black-box optimizer.
    ndim        : int
                  number of dimensionality.
    max_runtime : float
                  maximum of runtime to be allowed (seconds).
    start_index : int
                  starting index of independent experiments.
    end_index   : int
                  ending index of independent experiments.
    seed        : int
                  seed for random number generation (RNG).

    Returns
    -------
    A set of data files from independent experiments in the working space (`pwd()`).
    """
    # use the following 10 benchmarking function common in the
    # black-box optimization community
    functions = [cf.griewank.__name__, cf.bohachevsky.__name__,
                 cf.ackley.__name__, cf.rastrigin.__name__,
                 cf.scaled_rastrigin.__name__, cf.skew_rastrigin.__name__,
                 cf.levy_montalvo.__name__, cf.michalewicz.__name__,
                 cf.salomon.__name__, cf.schaffer.__name__]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions),))
    print('[Preprocessing] - First generate shift vectors for all (10) test functions.')
    for i, f in enumerate(functions):
        generate_shift_vector(f, ndim, -9.5, 9.5, seeds[i])
    print('[Preprocessing] - Second generate rotation matrices for all (10) test functions.')
    start_run = time.time()
    for i, f in enumerate(functions):
        start_time = time.time()
        generate_rotation_matrix(f, ndim, seeds[i])
        print(' * {:d}-d {:s}: runtime {:7.5e}'.format(
            ndim, f, time.time() - start_time))
    print('[Preprocessing] - Total runtime: {:7.5e}.'.format(time.time() - start_run))
    experiments = Experiments(start_index, end_index, ndim, max_runtime, is_local=False)
    experiments.run(optimizer)
