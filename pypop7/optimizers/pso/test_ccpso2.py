def test_optimize():
    import numpy  # engine for numerical computing
    from pypop7.benchmarks.base_functions import sphere  # function to be minimized
    from pypop7.optimizers.pso.ccpso2 import CCPSO2
    ndim_problem = 250
    problem = {'fitness_function': sphere,  # to define problem arguments
               'ndim_problem': ndim_problem,
               'lower_boundary': -5.0 * numpy.ones((ndim_problem,)),
               'upper_boundary': 5.0 * numpy.ones((ndim_problem,))}
    options = {'max_function_evaluations': 100000,  # to set optimizer options
               'seed_rng': 2022}  # global step-size may need to be tuned for optimality
    ccpso2 = CCPSO2(problem, options)  # to initialize the black-box optimizer class
    results = ccpso2.optimize()  # to run its optimization/evolution process
    assert results['n_function_evaluations'] == 100000
    assert results['best_so_far_y'] < 50.0
