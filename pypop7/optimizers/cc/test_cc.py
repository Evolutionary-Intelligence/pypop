from pypop7.optimizers.cc.cc import CC


class TCC(CC):
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)

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
    options = {'max_function_evaluations': 5000}  # to set optimizer options
    test_cc = TCC(problem, options)
    assert test_cc.fitness_function == rosenbrock
    assert test_cc.problem_name == 'rosenbrock'
    assert test_cc.ndim_problem == 2
    assert np.all(test_cc.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_cc.upper_boundary == 5.0 * np.ones((2,)))
    assert test_cc.max_function_evaluations == 5000
    assert test_cc.n_individuals == 100
