"""
To run this part of code, you should install bbob coco library first.
The way to install bbob lib, can be seen in the following website:
https://github.com/numbbo/coco
You should follow the Getting Started part to install the lib in order to use python3

Reference:
----------
Nikolaus Hansen, Dimo Brockhoff, Olaf Mersmann, Tea Tusar,
Dejan Tusar, Ouassim Ait ElHara, Phillipe R. Sampaio, Asma Atamna, Konstantinos Varelas,
Umut Batu, Duc Manh Nguyen, Filip Matzner, Anne Auger.
COmparing Continuous Optimizers: numbbo/COCO on Github.
Zenodo, DOI:10.5281/zenodo.2594848, March 2019.
"""
import cocoex
import numpy as np
from pypop7.optimizers.ds.nm import NM as Solver


suite_name = "bbob"  # see cocoex.known_suite_names
suite_year_option = ""
suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                        # "dimensions: 2,3,5,10,20 "  # skip dimension 40
                        # "instance_indices: 1-5 "  # relative to suite instances
                       )
suite = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)
for current_problem in suite:
    print(current_problem)
    d = current_problem.dimension
    problem = {'fitness_function': current_problem,
               'ndim_problem': d,
               'lower_boundary': -10 * np.ones((d,)),
               'upper_boundary': 10 * np.ones((d,))}
    options = {'fitness_threshold': 1e-10,
               'max_function_evaluations': 1e4,
               'seed_rng': 2022,  # not given in the original paper
               'x': np.ones((d,)),
               'sigma': 1.0,
               'stagnation': np.Inf,
               'verbose': 1000,
               'is_resart': False,
               'saving_fitness': 1000}
    solver = Solver(problem, options)
    results = solver.optimize()
    print(results)

