"""A short and simple example experiment of coco benchmark

This code is referenced from following library:
https://github.com/numbbo/coco

In this program we have implement our maes algorithm to the coco library

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
from numpy.random import rand  # for randomised restarts
from pypop7.optimizers.es.maes import MAES as Solver
import os, webbrowser  # to show post-processed results in the browser

### input
suite_name = "bbob"
output_folder = "coco-benchmarking-maes-fmin"
budget_multiplier = 1  # increase to 10, 100, ...
def fmin(fun, x0, disp=False):
    problem = {
        'fitness_function': fun,
        'ndim_problem': fun.dimension,
        'lower_boundary': fun.lower_bounds,
        'upper_boundary': fun.upper_bounds
    }
    options = {
        'fitness_threshold': 1e-10,
        'max_function_evaluations': 1e1,
        'seed_rng': 2022,
        'x': x0,
        'sigma': 1.0,
        'verbose': disp
    }
    solver = Solver(problem, options)
    results = solver.optimize()
    return results['best_so_far_x']

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while (problem.evaluations < problem.dimension * budget_multiplier
           and not problem.final_target_hit):
        result = fmin(problem, x0, disp=False)  # here we assume that `fmin` evaluates the final/returned solution
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")