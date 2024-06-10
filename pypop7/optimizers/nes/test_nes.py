import numpy
from pypop7.optimizers.nes import NES


def test_initialize():
    ndim_problem = 777
    problem = {'fitness_function': None,  # to define problem arguments
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.0 * numpy.ones((ndim_problem,)),
               'upper_boundary': 5.0 * numpy.ones((ndim_problem,))}
    options = {'max_function_evaluations': 5000,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((ndim_problem,)),
               'sigma': 3.0}
    nes = NES(problem, options)
    nes.initialize()
    print(sum(nes._u))
    assert numpy.all(0.0 <= nes._u)
    assert numpy.all(nes._u <= 1.0)
