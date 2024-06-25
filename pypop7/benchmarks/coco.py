"""To install `COCO` successfully, please read the following open link carefully:
    https://github.com/numbbo/coco
"""
import os
import webbrowser  # for post-processing in browser

import numpy as np  # engine for numerical computing
import cocoex  # experimentation module of `COCO`
import cocopp  # post-processing module of `COCO`


def coco(optimizer, seed_rng=2022, budget_multiplier=100000):
    """Test on the well-designed **COCO** platform.

       .. note:: To install `COCO` successfully, please read the following open link carefully:
          https://github.com/numbbo/coco.

          The maximum of function evaluations is set to `budget_multiplier * dimension`.

    Parameters
    ----------
    optimizer         : class
                        any black-box optimizer.
    seed_rng          : int
                        seed for random number generation (RNG).
    budget_multiplier : int
                        budget multiplier of function evaluations (`budget_multiplier*dimension`).

    Returns
    -------
    results : dict
              final optimization results.
    """
    # choose the specific test suite and output format
    suite, output, results = 'bbob', 'COCO-PyPop7-MAES', None
    observer = cocoex.Observer(suite, 'result_folder: ' + output)
    cocoex.utilities.MiniPrint()
    for function in cocoex.Suite(suite, '', ''):
        # generate data for `cocopp` post-processing
        function.observe_with(observer)
        # define problem arguments
        problem = {'fitness_function': function,
                   'ndim_problem': function.dimension,
                   'lower_boundary': function.lower_bounds,
                   'upper_boundary': function.upper_bounds}
        # set algorithm options
        options = {'max_function_evaluations': function.dimension * budget_multiplier,
                   'seed_rng': seed_rng,
                   'x': function.initial_solution,
                   'sigma': np.min(function.upper_bounds - function.lower_bounds) / 3.0}
        # run black-box optimizer
        results = optimizer(problem, options).optimize()
    cocopp.main(observer.result_folder)
    # open browser to show final optimization results
    webbrowser.open('file://' + os.getcwd() + '/ppdata/index.html')
    return results
