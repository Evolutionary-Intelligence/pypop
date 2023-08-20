# Taken directly from https://github.com/DEAP/deap/blob/master/examples/de/basic.py
#    with slight modifications for comparisons
#
# Please first install deap (http://deap.readthedocs.org/):
#    $ pip install deap
import time
import array
import random
import pickle

import numpy as np
from deap import base, creator, tools
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return _sphere(x),


def main(f):
    start_time = time.time()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -10.0, 10.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 2000)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", f)

    CR, F = 0.9, 0.5

    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    pop = toolbox.population(n=100)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        n_fe += 1  # current number of function evaluations
        fe.append(n_fe)
        if len(fitness) == 0 or fitness[-1] > fit[0]:
            fitness.append(fit[0])
        else:
            fitness.append(fitness[-1])

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(2000)
            for i, _ in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F*(b[i] - c[i])
            y.fitness.values = toolbox.evaluate(y)
            n_fe += 1  # current number of function evaluations
            fe.append(n_fe)
            if len(fitness) == 0 or fitness[-1] > y.fitness.values[0]:
                fitness.append(y.fitness.values[0])
            else:
                fitness.append(fitness[-1])
            if y.fitness > agent.fitness:
                pop[k] = y

    fitness = np.vstack((fe, fitness)).T
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    with open('DEAP_DE.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


if __name__ == "__main__":
    main(sphere)
