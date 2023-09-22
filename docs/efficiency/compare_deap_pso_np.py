# Taken directly from https://github.com/DEAP/deap/blob/master/examples/pso/basic_numpy.py
#    with slight modifications for comparisons
#
# Please first install deap (see http://deap.readthedocs.org/ for details):
#    $ pip install deap
import math
import time
import random
import pickle
import operator

import numpy as np
from deap import base, tools, creator
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return -_sphere(x),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # for maximization
creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax, size)) 
    part.speed = np.random.uniform(smin, smax, size)
    part.smin, part.smax = smin, smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = np.random.uniform(0, phi1, len(part))
    u2 = np.random.uniform(0, phi2, len(part))
    v_u1, v_u2 = u1*(part.best - part), u2*(best - part)
    part.speed += v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part += part.speed


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2000, pmin=-10.0, pmax=10.0,
                 smin=-4.0, smax=4.0)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", sphere)


def main():
    start_time = time.time()

    pop = toolbox.population(n=20)  # initial population
    best = None  # globally best position

    n_fe = 0  # number of function evaluations
    # to store a list of sampled function evaluations and best-so-far fitness
    fe, fitness = [], []

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            n_fe += 1  # current number of function evaluations
            fe.append(n_fe)
            if len(fitness) == 0 or fitness[-1] < part.fitness.values[0]:
                fitness.append(part.fitness.values[0])
            else:
                fitness.append(fitness[-1])
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

    fitness = np.vstack((fe, fitness)).T
    fitness[:, -1] *= -1  # for minimization
    results = {'best_so_far_y': fitness[-1],
               'n_function_evaluations': n_fe,
               'runtime': time.time() - start_time,
               'fitness': fitness}
    with open('DEAP_PSO_NP.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('*** runtime (seconds) ***:', time.time() - start_time)


if __name__ == "__main__":
    main()
