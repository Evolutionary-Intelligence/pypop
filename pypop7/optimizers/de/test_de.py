from pypop7.optimizers.de.de import DE


class TDE(DE):
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)

    def initialize(self):
        pass

    def mutate(self):
        pass

    def crossover(self):
        pass

    def select(self):
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
               'n_individuals': 200}
    test_de = TDE(problem, options)
    assert test_de.fitness_function == rosenbrock
    assert test_de.problem_name == 'rosenbrock'
    assert test_de.ndim_problem == 2
    assert np.all(test_de.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_de.upper_boundary == 5.0 * np.ones((2,)))
    assert test_de.max_function_evaluations == 5000
    assert test_de.n_individuals == 200
