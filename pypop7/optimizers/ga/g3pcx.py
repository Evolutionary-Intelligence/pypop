import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.ga.ga import GA


class G3PCX(GA):
    """Generalized Generation Gap with Parent-Centric Recombination (G3PCX).

    .. note:: Originally `G3PCX` was proposed to scale up the efficiency of `GA` mainly by Deb,
       `the recipient of IEEE Evolutionary Computation Pioneer Award 2018 <https://tinyurl.com/456as566>`_.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'n_parents'     - parent size (`int`, default: `3`),
                * 'n_offsprings'  - offspring size (`int`, default: `2`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.g3pcx import G3PCX
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> g3pcx = G3PCX(problem, options)  # initialize the optimizer class
       >>> results = g3pcx.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"G3PCX: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       G3PCX: 5000, 0.0

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/yc39x6d5>`_ for more details.

    Attributes
    ----------
    n_individuals : `int`
                    population size.
    n_offsprings  : `int`
                    offspring size.
    n_parents     : `int`
                    parent size.

    References
    ----------
    https://www.egr.msu.edu/~kdeb/codes/g3pcx/g3pcx.tar    (See the original C source code.)

    https://pymoo.org/algorithms/soo/g3pcx.html

    Deb, K., Anand, A. and Joshi, D., 2002.
    A computationally efficient evolutionary algorithm for real-parameter optimization.
    Evolutionary Computation, 10(4), pp.371-395.
    https://direct.mit.edu/evco/article-abstract/10/4/371/1136/A-Computationally-Efficient-Evolutionary-Algorithm
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.n_offsprings = options.get('n_offsprings', 2)
        assert self.n_offsprings > 0
        self.n_parents = options.get('n_parents', 3)
        assert self.n_parents > 0
        self._std_pcx_1 = options.get('_std_pcx_1', 0.1)
        self._std_pcx_2 = options.get('_std_pcx_2', 0.1)
        self._elitist = None  # index of elitist

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, args=None):
        self._elitist, fitness = np.argmin(y), []
        # (Step 1:) select the best and `self.n_parents - 1` other parents randomly from the population
        parents = self.rng_optimization.choice(self.n_individuals, size=self.n_parents, replace=False)
        if self._elitist not in parents:  # to ensure that elitist is always included
            parents[0] = self._elitist
        # (Step 2:) generate offspring from the chosen parents using a recombination scheme
        xx, yy = np.empty((self.n_offsprings, self.ndim_problem)), np.empty((self.n_offsprings,))
        g = np.mean(x[parents], axis=0)  # mean vector of the chosen parents
        for i in range(self.n_offsprings):
            if self._check_terminations():
                break
            p = self._elitist  # for faster local convergence
            d = g - x[p]
            d_norm = np.linalg.norm(d)
            d_mean = np.empty((self.n_parents - 1,))
            diff = np.empty((self.n_parents - 1, self.ndim_problem))  # for distance computation
            for ii, j in enumerate(parents[1:]):
                diff[ii] = x[j] - x[p]  # distance from one parent
            for ii in range(self.n_parents - 1):
                d_mean[ii] = np.linalg.norm(diff[ii])*np.sqrt(
                    1.0 - np.power(np.dot(diff[ii], d)/(np.linalg.norm(diff[ii])*d_norm), 2))
            d_mean = np.mean(d_mean)  # average of perpendicular distances
            orth = self._std_pcx_2*d_mean*self.rng_optimization.standard_normal((self.ndim_problem,))
            orth = orth - (np.dot(orth, d)*d)/np.power(d_norm, 2)
            xx[i] = x[p] + self._std_pcx_1*self.rng_optimization.standard_normal()*d + orth
            yy[i] = self._evaluate_fitness(xx[i], args)
            fitness.append(yy[i])
        # (Step 3:) choose two parents at random from the population
        offsprings = self.rng_optimization.choice(self.n_individuals, size=2, replace=False)
        # (Step 4:) from a combined subpopulation of two chosen parents and created offspring, choose
        #   the best two solutions and replace the chosen two parents (in Step 3) with these solutions
        xx, yy = np.vstack((xx, x[offsprings])), np.hstack((yy, y[offsprings]))
        x[offsprings], y[offsprings] = xx[np.argsort(yy)[:2]], yy[np.argsort(yy)[:2]]
        self._n_generations += 1
        return fitness

    def _collect(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        for i in range(1, len(fitness)):  # to avoid `np.nan`
            if np.isnan(fitness[i]):
                fitness[i] = fitness[i - 1]
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y = self.initialize(args)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            yy = self.iterate(x, y, args)
        return self._collect(fitness, yy)
