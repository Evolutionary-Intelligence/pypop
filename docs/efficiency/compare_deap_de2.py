# Taken directly from https://github.com/DEAP/deap/blob/master/examples/de/sphere.py
#    with slight modifications for comparisons
#
# Please first install deap (http://deap.readthedocs.org/):
#    $ pip install deap
import time
import array
import random
import pickle

import numpy as np
from deap import base, tools, creator
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return _sphere(x),


def mutDE(y, a, b, c, f):
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i] - c[i])
    return y


def cxBinomial(x, y, cr):
    index = random.randrange(len(x))
    for i in range(len(x)):
        if i == index or random.random() < cr:
            x[i] = y[i]
    return x


def main(f):
    start_time = time.time()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -10.0, 10.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 2000)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutDE, f=0.5)
    toolbox.register("mate", cxBinomial, cr=0.9)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", f)

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
        children = []
        for agent in pop:
            a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
            x = toolbox.clone(agent)
            y = toolbox.clone(agent)
            y = toolbox.mutate(y, a, b, c)
            z = toolbox.mate(x, y)
            del z.fitness.values
            children.append(z)

        fitnesses = toolbox.map(toolbox.evaluate, children)
        for (i, ind), fit in zip(enumerate(children), fitnesses):
            ind.fitness.values = fit
            n_fe += 1  # current number of function evaluations
            fe.append(n_fe)
            if len(fitness) == 0 or fitness[-1] > fit[0]:
                fitness.append(fit[0])
            else:
                fitness.append(fitness[-1])
            if ind.fitness > pop[i].fitness:
                pop[i] = ind

    fitness = np.vstack((fe, fitness)).T
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    with open('DEAP_DE2.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


if __name__ == "__main__":
    main(sphere)
