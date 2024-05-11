from pypop7.optimizers.bo.bo import BO


class TBO(BO):
    def __init__(self, problem, options):
        BO.__init__(self, problem, options)

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
               'n_individuals': 100}
    test_bo = TBO(problem, options)
    assert test_bo.fitness_function == rosenbrock
    assert test_bo.problem_name == 'rosenbrock'
    assert test_bo.ndim_problem == 2
    assert np.all(test_bo.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_bo.upper_boundary == 5.0 * np.ones((2,)))
    assert test_bo.max_function_evaluations == 5000
    assert test_bo.n_individuals == 100
