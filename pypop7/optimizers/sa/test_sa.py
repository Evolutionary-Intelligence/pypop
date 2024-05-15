from pypop7.optimizers.sa.sa import SA


class TSA(SA):
    def __init__(self, problem, options):
        SA.__init__(self, problem, options)

    def initialize(self):
        pass

    def iterate(self):
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
    test_sa = TSA(problem, options)
    assert test_sa.fitness_function == rosenbrock
    assert test_sa.problem_name == 'rosenbrock'
    assert test_sa.ndim_problem == 2
    assert np.all(test_sa.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_sa.upper_boundary == 5.0 * np.ones((2,)))
    assert test_sa.max_function_evaluations == 5000
    assert test_sa.temperature is None
