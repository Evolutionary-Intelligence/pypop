# PyGMO: https://esa.github.io/pygmo2/install.html
import pygmo as pg


# https://esa.github.io/pagmo2/docs/cpp/problems/lennard_jones.html
prob = pg.problem(pg.lennard_jones(150))


def lennard_jones(optimizer, max_function_evaluations=700000):
    """444-dimensional Lennard-Jones cluster optimization problem from **PyGMO**.

    Parameters
    ----------
    optimizer                : class
                               any black-box optimizer.
    max_function_evaluations : int
                               maximum of function evaluations.

    Returns
    -------
    results : dict
              final optimization results
    """
    # print(prob)  # 444-dimensional
    problem = {'fitness_function': float(prob.fitness(x)),
               'ndim_problem': 444,
               'upper_boundary': prob.get_bounds()[1],
               'lower_boundary': prob.get_bounds()[0]}
    options = {'max_function_evaluations': max_function_evaluations,
               'seed_rng': 2022,  # RNG seed for repeatability
               # to save all fitness generated during entire optimization
               'saving_fitness': 1,
               'is_bound': True}
    results = optimizer(problem, options).optimize()
    return results
