def test_optimize():
    import numpy  # engine for numerical computing
    from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
    from pypop7.optimizers.de.mde import MDE
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * numpy.ones((2,)),
               'upper_boundary': 5.0 * numpy.ones((2,))}
    options = {'max_function_evaluations': 5000,  # to set optimizer options
               'seed_rng': 2022}
    mde = MDE(problem, options)  # to initialize the black-box optimizer class
    results = mde.optimize()  # to run its optimization/evolution process
    assert results['n_function_evaluations'] == 5000
    assert results['best_so_far_y'] < 1.0


def test_repeatability_and_boundary():
    import numpy  # engine for numerical computing
    from pypop7.benchmarks.base_functions import sphere, rastrigin
    from pypop7.optimizers.de.mde import MDE
    for f in [sphere, rastrigin]:
        problem = {'fitness_function': f,
                   'ndim_problem': 10,
                   'lower_boundary': -5.12 * numpy.ones((10,)),
                   'upper_boundary': 5.12 * numpy.ones((10,))}
        options = {'max_function_evaluations': 3000,
                   'n_individuals': 60,
                   'seed_rng': 0,
                   'saving_fitness': 1,
                   'verbose': False}
        results = MDE(problem, options).optimize()
        results_repeated = MDE(problem, options).optimize()
        assert results['best_so_far_y'] == results_repeated['best_so_far_y']
        assert numpy.array_equal(results['best_so_far_x'], results_repeated['best_so_far_x'])
        assert results['n_function_evaluations'] == 3000
        assert len(results['fitness']) == 3000
        assert results['success']  # inside the search range


def test_n_individuals():
    import numpy  # engine for numerical computing
    import pytest
    from pypop7.benchmarks.base_functions import sphere
    from pypop7.optimizers.de.mde import MDE
    problem = {'fitness_function': sphere,
               'ndim_problem': 2,
               'lower_boundary': -5.0 * numpy.ones((2,)),
               'upper_boundary': 5.0 * numpy.ones((2,))}
    mde = MDE(problem, {'max_function_evaluations': 100})
    assert mde.n_individuals == 90 and mde.n_sub == 30
    mde = MDE(problem, {'max_function_evaluations': 100, 'n_individuals': 10})
    assert mde.n_individuals == 10 and mde.n_sub == 3
    with pytest.raises(AssertionError):
        MDE(problem, {'max_function_evaluations': 100, 'n_individuals': 8})
