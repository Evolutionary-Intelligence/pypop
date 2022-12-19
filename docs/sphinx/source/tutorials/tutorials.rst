Tutorials
=========

Here we provide several *interesting* tutorials to help better use this library `PyPop7
<https://pypop.readthedocs.io/en/latest/installation.html>`_, as shown below:

* Lens Shape Optimization (15-dimensional)
* Lennard-Jones Cluster Optimization (444-dimensional)
* Benchmarking for Large-Scale Black-Box Optimization (up to 2000-dimensional)
* Benchmarking on the Well-Designed `COCO <https://github.com/numbbo/coco>`_ Platform (up to 640-dimensional)
* Benchmarking on the Famous `NeverGrad <https://github.com/facebookresearch/nevergrad>`_ Platform (developed
  recently by FacebookResearch)

For all optimizers from this library, we also provide a *toy* example on their corresponding
`API <https://pypop.readthedocs.io/_/downloads/en/latest/pdf/>`_ documentations.

Lens Shape Optimization
-----------------------

.. image:: images/lens_optimization.gif
   :width: 321px
   :align: center

This figure shows the (interesting) evolution process of lens shape, optimized by `MAES
<https://pypop.readthedocs.io/en/latest/es/maes.html>`_, a *simplified* modern version of
the well-established `CMA-ES` nearly without significant performance loss.

The objective of Lens Shape Optimization is to find the optimal shape of glass body such that parallel incident light
rays are concentrated in a given point on a plane while using a minimum of glass material possible.
Refer to `Beyer, 2020, GECCO <https://dl.acm.org/doi/abs/10.1145/3377929.3389870>`_ for more mathematical details
about the 15-dimensional objective function used here. To repeat this above figure, please run the following code:
https://github.com/Evolutionary-Intelligence/pypop/blob/main/tutorials/lens_optimization.py.

Lennard-Jones Cluster Optimization
----------------------------------

.. image:: images/Lennard-Jones-cluster-optimization.gif
   :width: 321px
   :align: center

Note that the above figure (i.e., three clusters of atoms) is directly from
http://doye.chem.ox.ac.uk/jon/structures/LJ/pictures/LJ.new.gif.

In chemistry, `Lennard-Jones Cluster Optimization <https://tinyurl.com/4ukrspc9>`_ is a popular single-objective
real-parameter (black-box) optimization problem, which is to minimize the energy of a cluster of atoms assuming a
`Lennard-Jones <http://doye.chem.ox.ac.uk/jon/structures/LJ.html>`_ potential between each pair.

    .. code-block:: python

        import numpy as np
        import pygmo as pg  # need to be installed: https://esa.github.io/pygmo2/install.html
        from pypop7.optimizers.de.cde import CDE  # https://pypop.readthedocs.io/en/latest/de/cde.html
        from pypop7.optimizers.de.jade import JADE  # https://pypop.readthedocs.io/en/latest/de/jade.html
        import seaborn as sns
        import matplotlib.pyplot as plt

        prob = pg.problem(pg.lennard_jones(150))
        print(prob)  # 444-dimensional

        def energy_func(x):  # wrapper
            return float(prob.fitness(x))

        results = []  # to save all optimization results from different optimizers
        for DE in [CDE, JADE]:
            problem = {'fitness_function': energy_func,
                               'ndim_problem': 444,
                               'upper_boundary': 3*np.ones((444,)),
                               'lower_boundary': -3*np.ones((444,))}
            options = {'max_function_evaluations': 300000,
                              'seed_rng': 2022,  # for repeatability
                              'saving_fitness': 1,  # to save all fitness generated during optimization
                              'boundary': True}  # for JADE (not for CDE)
            solver = DE(problem, options)  # without boundary constraints
            results.append(solver.optimize())
            print(results[-1])

        sns.set_theme(style='darkgrid')
        plt.figure()
        labels = ['CDE', 'JADE']
        for i, res in enumerate(results):
            # starting 1000 can avoid excessively high values generated during the early stage to disrupt convergence curve
            plt.plot(res['fitness'][1000:, 0], res['fitness'][1000:, 1], label=labels[i])

        plt.legend()
        plt.show()

The generated convergence curves for both `CDE` (without box constraints) and `JADE` (with box constraints) are
presented in the following image:

.. image:: images/CDE_vs_JADE.png
   :align: center

From the above figure, different `DE` versions show different search performance: `CDE` does not limit samples into
the given search boundaries during optimization and generate a out-of-box solution very fast, while `JADE` limits
all samples into the given search boundaries during optimization and generate an inside-of-box solution relatively
slow. In other words, open-source implementations play an important role for repeatability, since *slightly different*
implementation details could sometimes even result in *totally different* search behaviors.

For more interesting applications of `DE` on challenging real-world problems, refer to e.g.,
`[An et al., 2020, PNAS] <https://www.pnas.org/doi/suppl/10.1073/pnas.1920338117>`_;
`[Gagnon et al., 2017, PRL] <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.053203>`_;
`[Laganowsky et al., 2014, Nature] <https://www.nature.com/articles/nature13419>`_;
`[Lovett et al., 2013, PRL] <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.220501>`_,
just to name a few.

Benchmarking for Large-Scale Black-Box Optimization (LSBBO)
-----------------------------------------------------------

Benchmarking of optimization algorithms plays a very crucial role on understanding their search dynamics, comparative
performance, analyzing their advantages and limitations, and also choosing state-of-the-art (SOTA) versions, usually
before applying them to more challenging real-world problems.

.. note:: *“A biased benchmark, excluding large parts of the real-world needs, leads to biased conclusions, no matter
   how many experiments we perform.”* ---`[Meunier et al., 2022, IEEE-TEVC]
   <https://ieeexplore.ieee.org/abstract/document/9524335>`_

Here we show how to benchmark multiple black-box optimizers on a *relatively large* (10) collection of
LSBBO test functions, in order to mainly compare their *local search* capability:

First, generate shift vectors and rotation matrices needed in the experiments:

    .. code-block:: python

        import time
        import numpy as np

        from pypop7.benchmarks.shifted_functions import generate_shift_vector
        from pypop7.benchmarks.rotated_functions import generate_rotation_matrix


        def generate_sv_and_rm(functions=None, ndims=None, seed=None):
            if functions is None:
                functions = ['sphere', 'cigar', 'discus', 'cigar_discus', 'ellipsoid',
                    'different_powers', 'schwefel221', 'step', 'rosenbrock', 'schwefel12']
            if ndims is None:
                ndims = [2, 10, 100, 200, 1000, 2000]
            if seed is None:
                seed = 20221001

            rng = np.random.default_rng(seed)
            seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions), len(ndims)))

            for i, f in enumerate(functions):
                for j, d in enumerate(ndims):
                    generate_shift_vector(f, d, -9.5, 9.5, seeds[i, j])

            start_run = time.time()
            for i, f in enumerate(functions):
                for j, d in enumerate(ndims):
                    start_time = time.time()
                    generate_rotation_matrix(f, d, seeds[i, j])
                    print('* {:d}-d {:s}: runtime {:7.5e}'.format(
                        d, f, time.time() - start_time))
            print('*** Total runtime: {:7.5e}.'.format(time.time() - start_run))


        if __name__ == '__main__':
            generate_sv_and_rm()


Then, invoke different optimizers on these (rotated and shifted) test functions:

    .. code-block:: python

        import os
        import time
        import pickle
        import argparse

        import numpy as np

        import pypop7.benchmarks.continuous_functions as cf


        class Experiment(object):
            def __init__(self, index, function, seed, ndim_problem):
                self.index = index
                self.function = function
                self.seed = seed
                self.ndim_problem = ndim_problem
                self._folder = 'pypop7_benchmarks_lso'
                if not os.path.exists(self._folder):
                    os.makedirs(self._folder)
                self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

            def run(self, optimizer):
                problem = {'fitness_function': self.function,
                           'ndim_problem': self.ndim_problem,
                           'upper_boundary': 10.0*np.ones((self.ndim_problem,)),
                           'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
                options = {'max_function_evaluations': 100000 * self.ndim_problem,
                           'max_runtime': 3600*3,  # seconds
                           'fitness_threshold': 1e-10,
                           'seed_rng': self.seed,
                           'saving_fitness': 2000,
                           'verbose': 0}
                if optimizer.__name__ in ['SRS', 'RHC', 'ARHC', 'CSA',
                    'RES', 'DSAES', 'CSAES',
                    'OPOC2006', 'OPOC2009', 'SEPCMAES', 'OPOA2010', 'OPOA2015',
                    'CCMAES2009', 'MAES', 'LMCMA', 'LMMAES', 'MMES',
                    'SCEM', 'DSCEM', 'DCEM']:
                    options['sigma'] = 20.0/3.0
                solver = optimizer(problem, options)
                results = solver.optimize()
                file = self._file.format(solver.__class__.__name__,
                                         solver.fitness_function.__name__,
                                         solver.ndim_problem,
                                         self.index)
                with open(file, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        class Experiments(object):
            def __init__(self, start, end, ndim_problem):
                self.start = start
                self.end = end
                self.ndim_problem = ndim_problem
                self.indices = range(self.start, self.end + 1)
                self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                                  cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
                self.seeds = np.random.default_rng(2022).integers(
                    np.iinfo(np.int64).max, size=(len(self.functions), 50))

            def run(self, optimizer):
                for index in self.indices:
                    print('* experiment: {:d} ***:'.format(index))
                    for d, f in enumerate(self.functions):
                        start_time = time.time()
                        print('  * function: {:s}:'.format(f.__name__))
                        experiment = Experiment(index, f, self.seeds[d, index], self.ndim_problem)
                        experiment.run(optimizer)
                        print('    runtime: {:7.5e}.'.format(time.time() - start_time))


        if __name__ == '__main__':
            start_runtime = time.time()
            parser = argparse.ArgumentParser()
            parser.add_argument('--start', '-s', type=int)  # starting index of experiments (from 0 to 49)
            parser.add_argument('--end', '-e', type=int)  # ending index of experiments (from 0 to 49)
            parser.add_argument('--optimizer', '-o', type=str)
            parser.add_argument('--ndim_problem', '-d', type=int, default=2000)
            args = parser.parse_args()
            params = vars(args)
            if params['optimizer'] == 'MAES':  # 2017
                from pypop7.optimizers.es.maes import MAES as Optimizer
            elif params['optimizer'] == 'FMAES':  # 2017
                from pypop7.optimizers.es.fmaes import FMAES as Optimizer
            elif params['optimizer'] == 'LMCMA':  # 2017
                from pypop7.optimizers.es.lmcma import LMCMA as Optimizer
            elif params['optimizer'] == 'LMMAES':  # 2019
                from pypop7.optimizers.es.lmmaes import LMMAES as Optimizer
            elif params['optimizer'] == 'MMES':  # 2021
                from pypop7.optimizers.es.mmes import MMES as Optimizer
            elif params['optimizer'] == 'BES':  # 2022
                from pypop7.optimizers.rs.bes import BES as Optimizer
            experiments = Experiments(params['start'], params['end'], params['ndim_problem'])
            experiments.run(Optimizer)
            print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_runtime))


Please run the above code (named as `run_experiments.py`) in the background, since it needs very long runtime for LSBBO:

    .. code-block:: bash

        $ nohup python run_experiments.py -s=1 -e=2 -o=LMCMA >LMCMA_1_2.out 2>&1 &  # on Linux

Benchmarking on the Well-Designed COCO Platform
-----------------------------------------------

From the `evolutionary computation <https://www.nature.com/articles/nature14544>`_ community,
`COCO <https://github.com/numbbo/coco>`_ is a *well-designed* platform for comparing continuous optimizers
in a black-box setting.

    .. code-block:: python

        import cocoex
        import numpy as np

        from pypop7.optimizers.ds.nm import NM as Solver


        if __name__ == '__main__':
            print(cocoex.known_suite_names)
            suite = cocoex.Suite('bbob', '', '')
            for current_problem in suite:
                print(current_problem)
                d = current_problem.dimension
                problem = {'fitness_function': current_problem,
                           'ndim_problem': d,
                           'lower_boundary': -10 * np.ones((d,)),
                           'upper_boundary': 10 * np.ones((d,))}
                options = {'max_function_evaluations': 1e3 * d,
                           'seed_rng': 2022,
                           'sigma': 1.0,
                           'verbose': False,
                           'saving_fitness': 2000}
                solver = Solver(problem, options)
                results = solver.optimize()
                print('  best-so-far fitness:', results['best_so_far_y'])

Benchmarking on the Famous NeverGrad Platform
---------------------------------------------

As pointed out in the recent paper `[Meunier et al., 2022, IEEE-TEVC]
<https://ieeexplore.ieee.org/abstract/document/9524335>`_, *"Existing studies in black-box optimization suffer from
low generalizability, caused by a typically selective choice of problem instances used for training and testing of
different optimization algorithms. Among other issues, this practice promotes overfitting and poor-performing user
guidelines."*
