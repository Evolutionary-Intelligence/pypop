# Taken directly from https://github.com/DEAP/deap/blob/master/examples/pso/basic.py
#    with slight modifications for comparisons
#
# Please first install deap (http://deap.readthedocs.org/):
#    $ pip install deap
import math
import time
import random
import operator

import numpy
from deap import base, tools, creator
from pypop7.benchmarks.base_functions import sphere as _sphere


def sphere(x):
    return -_sphere(x),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # for maximization
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin, part.smax = smin, smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2000, pmin=-10.0, pmax=10.0,
                 smin=-4.0, smax=4.0)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", sphere)


def main():
    start_time = time.time()

    pop = toolbox.population(n=20)  # initial population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    g = 0
    best = None  # globally best position

    while (time.time() - start_time) < (60 * 60 * 3):  # 3 hours
        g += 1
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    print('*** runtime (seconds) ***:', time.time() - start_time)


if __name__ == "__main__":
    main()
