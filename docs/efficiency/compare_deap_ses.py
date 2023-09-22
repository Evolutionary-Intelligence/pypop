# Taken directly from https://github.com/DEAP/deap/blob/master/examples/es/fctmin.py
#    with slight modifications for comparisons
#
# Please first install deap (see http://deap.readthedocs.org/ for details):
#    $ pip install deap
import time
import array
import random
import pickle

import numpy as np
from deap import base, creator, tools, algorithms
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return _sphere(x),


def generateES(icls, scls):
    ind = icls(random.uniform(-10.0, 10.0) for _ in range(2000))
    ind.strategy = scls(random.uniform(0.5, 3.0) for _ in range(2000))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


def eaMuCommaLambda(population, toolbox, cxpb, mutpb):
    start_time = time.time()

    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        n_fe += 1  # current number of function evaluations
        fe.append(n_fe)
        if len(fitness) == 0 or fitness[-1] > ind.fitness.values[0]:
            fitness.append(ind.fitness.values[0])
        else:
            fitness.append(fitness[-1])

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours 
        offspring = algorithms.varOr(population, toolbox, 100, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            n_fe += 1  # current number of function evaluations
            fe.append(n_fe)
            if len(fitness) == 0 or fitness[-1] > ind.fitness.values[0]:
                fitness.append(ind.fitness.values[0])
            else:
                fitness.append(fitness[-1])
        population[:] = toolbox.select(offspring, 10)

    fitness = np.vstack((fe, fitness)).T
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    with open('DEAP-SES.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


def main(f):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", f)
    toolbox.decorate("mate", checkStrategy(0.5))
    toolbox.decorate("mutate", checkStrategy(0.5))
    pop = toolbox.population(n=10)
    eaMuCommaLambda(pop, toolbox, cxpb=0.6, mutpb=0.3)


if __name__ == "__main__":
    main(sphere)
