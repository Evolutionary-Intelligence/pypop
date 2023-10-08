# Taken directly from https://github.com/DEAP/deap/blob/master/examples/es/cma_minfct.py
#    with slight modifications for comparisons
#
# Please first install deap (http://deap.readthedocs.org/):
#    $ pip install deap
import time
import pickle
import argparse

import numpy as np
from deap import base, cma, creator
import pypop7.benchmarks.continuous_functions as cf  # for rotated and shifted benchmarking functions


def sphere(x):
    return cf.sphere(x),


def cigar(x):
    return cf.cigar(x),


def discus(x):
    return cf.discus(x),


def cigar_discus(x):
    return cf.cigar_discus(x),


def ellipsoid(x):
    return cf.ellipsoid(x),


def different_powers(x):
    return cf.different_powers(x),


def schwefel221(x):
    return cf.schwefel221(x),


def step(x):
    return cf.step(x),


def rosenbrock(x):
    return cf.rosenbrock(x),


def schwefel12(x):
    return cf.schwefel12(x),


def ea_generate_update(toolbox, start_time, ii):
    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    while (time.time() - start_time) < (60*60*3):  # 3 hours
        population = toolbox.generate()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            n_fe += 1  # current number of function evaluations
            fe.append(n_fe)
            if len(fitness) == 0 or fitness[-1] > fit[0]:
                fitness.append(fit[0])
            else:
                fitness.append(fitness[-1])
        toolbox.update(population)

    fitness = np.vstack((fe, fitness)).T
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    filename = 'Algo-DEAPCMAES_Func-{}_Dim-2000_Exp-{}.pickle'.format(f.__name__, ii)
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(ff, ii):
    start_time = time.time()

    toolbox = base.Toolbox()
    toolbox.register("evaluate", ff)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    centroid = np.random.uniform(-10.0, 10.0, size=(2000,))
    lambda_ = 4 + int(3*np.log(2000))
    strategy = cma.Strategy(centroid=centroid, sigma=20.0/3.0, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    ea_generate_update(toolbox, start_time, ii)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', '-i', type=int)  # experiment index
    args = parser.parse_args()
    params = vars(args)
    for f in [sphere, cigar, discus, cigar_discus, ellipsoid,
              different_powers, schwefel221, step, rosenbrock, schwefel12]:
        main(f, params['index'])
