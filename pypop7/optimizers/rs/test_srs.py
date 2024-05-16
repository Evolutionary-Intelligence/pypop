def test_optimize():
    import numpy  # engine for numerical computing
    from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
    from pypop7.optimizers.rs.srs import SRS
    problem = {'fitness_function': rosenbrock,  # define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * numpy.ones((2,)),
               'upper_boundary': 5.0 * numpy.ones((2,))}
    options = {'max_function_evaluations': 5000,  # set optimizer options
               'seed_rng': 2022,
               'x': 3*numpy.ones((2,)),
               'sigma': 3.0}
    srs = SRS(problem, options)  # initialize the optimizer class
    results = srs.optimize()  # run the optimization process
    # return the number of function evaluations and best-so-far fitness
    print(f"SRS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
