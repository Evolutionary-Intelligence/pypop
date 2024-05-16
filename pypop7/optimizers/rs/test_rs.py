from pypop7.optimizers.rs.rs import RS


class TRS(RS):
    def __init__(self, problem, options):
        RS.__init__(self, problem, options)

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
    test_rs = RS(problem, options)
    assert test_rs.fitness_function == rosenbrock
    assert test_rs.problem_name == 'rosenbrock'
    assert test_rs.ndim_problem == 2
    assert np.all(test_rs.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_rs.upper_boundary == 5.0 * np.ones((2,)))
    assert test_rs.max_function_evaluations == 5000
