"""A simple example for `COCO` Benchmarking using `PyPop7`:
    https://github.com/numbbo/coco
    
    To install `COCO` successfully, please read the above link carefully.
"""
import os
import webbrowser  # for post-processing in the browser

import numpy as np
import cocoex  # experimentation module of `COCO`
import cocopp  # post-processing module of `COCO`

from pypop7.optimizers.es.maes import MAES


if __name__ == '__main__':
    suite, output = 'bbob', 'coco-maes'
    budget_multiplier = 1e3  # or 1e4, 1e5, ...
    observer = cocoex.Observer(suite, 'result_folder: ' + output)
    minimal_print = cocoex.utilities.MiniPrint()
    for function in cocoex.Suite(suite, '', ''):
        function.observe_with(observer)  # generate data for `cocopp` post-processing
        sigma = np.min(function.upper_bounds - function.lower_bounds)/3.0
        problem = {'fitness_function': function,
                   'ndim_problem': function.dimension,
                   'lower_boundary': function.lower_bounds,
                   'upper_boundary': function.upper_bounds}
        options = {'max_function_evaluations': function.dimension*budget_multiplier,
                   'seed_rng': 2022,
                   'x': function.initial_solution,
                   'sigma': sigma}
        solver = MAES(problem, options)
        print(solver.optimize())
    cocopp.main(observer.result_folder)
    webbrowser.open('file://' + os.getcwd() + '/ppdata/index.html')
