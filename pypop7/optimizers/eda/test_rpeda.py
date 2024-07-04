def test_optimize():
    import numpy  # engine for numerical computing
    from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
    from pypop7.optimizers.eda.rpeda import RPEDA
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 4,
               'lower_boundary': -5.0 * numpy.ones((4,)),
               'upper_boundary': 5.0 * numpy.ones((4,))}
    options = {'max_function_evaluations': 5000,  # to set optimizer options
               'seed_rng': 2022}
    rpeda = RPEDA(problem, options)  # to initialize the black-box optimizer class
    results = rpeda.optimize()  # to run its optimization/evolution process
    assert results['n_function_evaluations'] == 5000
    assert results['best_so_far_y'] < 3.0
