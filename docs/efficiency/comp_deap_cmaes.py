# Taken directly from https://github.com/DEAP/deap/blob/master/examples/es/cma_minfct.py
#    with slight modifications for comparisons
#
# Please first install deap (http://deap.readthedocs.org/):
#    $ pip install deap
import time
import pickle

import numpy as np
from deap import base, cma, creator
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return _sphere(x),


def eaGenerateUpdate(toolbox, start_time):
    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours 
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
    with open('DEAP-CMAES.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


def main(f):
    start_time = time.time()

    toolbox = base.Toolbox()
    toolbox.register("evaluate", f)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    centroid = np.random.uniform(-10.0, 10.0, size=(2000,))
    lambda_ = 4 + int(3*np.log(2000))
    strategy = cma.Strategy(centroid=centroid, sigma=20.0/3.0, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    eaGenerateUpdate(toolbox, start_time)


if __name__ == "__main__":
    main(sphere)
