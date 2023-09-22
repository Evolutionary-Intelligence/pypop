# Taken directly from https://github.com/DEAP/deap/blob/master/examples/es/onefifth.py
#    with slight modifications for comparisons
#
# Please first install deap (see http://deap.readthedocs.org/ for details):
#    $ pip install deap
import time
import array
import random
import pickle

import numpy as np
from deap import base, creator
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return _sphere(x),


def update(ind, mu, std):
    for i, mu_i in enumerate(mu):
        ind[i] = random.gauss(mu_i, std)


def main(f):
    start_time = time.time()

    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("update", update)
    toolbox.register("evaluate", f)

    IND_SIZE = 2000
    interval = (-10.0, 10.0)
    mu = (random.uniform(interval[0], interval[1]) for _ in range(IND_SIZE))
    sigma = (interval[1] - interval[0])/3.0
    alpha = 2.0**(1.0/IND_SIZE)

    best = creator.Individual(mu)
    best.fitness.values = toolbox.evaluate(best)
    n_fe += 1  # current number of function evaluations
    fe.append(n_fe)
    if len(fitness) == 0 or fitness[-1] > best.fitness.values[0]:
        fitness.append(best.fitness.values[0])
    else:
        fitness.append(fitness[-1])
    worst = creator.Individual((0.0,)*IND_SIZE)

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours
        toolbox.update(worst, best, sigma) 
        worst.fitness.values = toolbox.evaluate(worst)
        n_fe += 1  # current number of function evaluations
        fe.append(n_fe)
        if len(fitness) == 0 or fitness[-1] > worst.fitness.values[0]:
            fitness.append(worst.fitness.values[0])
        else:
            fitness.append(fitness[-1])
        if best.fitness <= worst.fitness:
            sigma *= alpha
            best, worst = worst, best
        else:
            sigma *= alpha**(-0.25)

    fitness = np.vstack((fe, fitness)).T
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    with open('DEAP-RES.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


if __name__ == "__main__":
    main(sphere)
