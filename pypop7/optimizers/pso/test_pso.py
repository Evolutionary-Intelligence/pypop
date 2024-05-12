from pypop7.optimizers.pso.pso import PSO


class TPSO(PSO):
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)


def test_initialize():
    import numpy as np
    from pypop7.benchmarks.base_functions import rosenbrock
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * np.ones((2,)),
               'upper_boundary': 5.0 * np.ones((2,))}
    options = {'max_function_evaluations': 5000}  # to set optimizer options
    test_pso = TPSO(problem, options)
    assert test_pso.fitness_function == rosenbrock
    assert test_pso.problem_name == 'rosenbrock'
    assert test_pso.ndim_problem == 2
    assert np.all(test_pso.lower_boundary == -5.0 * np.ones((2,)))
    assert np.all(test_pso.upper_boundary == 5.0 * np.ones((2,)))
    assert test_pso.max_function_evaluations == 5000
    assert test_pso.n_individuals == 20
