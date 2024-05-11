from pypop7.optimizers.ep.ep import EP


class TEP(EP):
    def __init__(self, problem, options):
        EP.__init__(self, problem, options)

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
               'n_individuals': 100}
    test_ep = TEP(problem, options)
    assert test_ep.fitness_function == rosenbrock
    assert test_ep.problem_name == 'rosenbrock'
    assert test_ep.ndim_problem == 2
    assert np.all(test_ep.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_ep.upper_boundary == 5.0 * np.ones((2,)))
    assert test_ep.max_function_evaluations == 5000
    assert test_ep.n_individuals == 100
