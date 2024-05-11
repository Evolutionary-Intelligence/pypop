from pypop7.optimizers.es.es import ES


class TES(ES):
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)

    def initialize(self):
        pass

    def iterate(self):
        pass

    def optimize(self, fitness_function=None):
        pass


def test_initialize():
    import numpy as np
    from pypop7.benchmarks.base_functions import rosenbrock
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * np.ones((2,)),
               'upper_boundary': 5.0 * np.ones((2,))}
    options = {'max_function_evaluations': 5000,  # to set optimizer options
               'sigma': 3.0}
    test_es = TES(problem, options)
    assert test_es.fitness_function == rosenbrock
    assert test_es.problem_name == 'rosenbrock'
    assert test_es.ndim_problem == 2
    assert np.all(test_es.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_es.upper_boundary == 5.0 * np.ones((2,)))
    assert test_es.max_function_evaluations == 5000
    assert test_es.sigma == 3.0
