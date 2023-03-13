import numpy as np

from pypop7.optimizers.ga.ga import GA


class GENITOR(GA):
    """GENetic ImplemenTOR (GENITOR).

    .. note:: `"Selective pressure and population diversity should be controlled as directly as possible."---[Whitley,
       1989] <https://dl.acm.org/doi/10.5555/93126.93169>`_

       This is a *slightly modified* version of `GENITOR` for continuous optimization. Originally `GENITOR` was proposed
       to solve challenging `neuroevolution <https://www.nature.com/articles/s42256-018-0006-z>`_ problems by Whitley,
       `recipient of IEEE Evolutionary Computation Pioneer Award 2022 <https://tinyurl.com/456as566>`_.

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
              and with the following particular setting (`key`):
                * 'n_individuals'  - population size (`int`, default: `100`),
                * 'cv_prob'        - crossover probability (`float`, default: `0.5`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.genitor import GENITOR
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> genitor = GENITOR(problem, options)  # initialize the optimizer class
       >>> results = genitor.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"GENITOR: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       GENITOR: 5000, 0.004382445279905116

    For its correctness checking of coding, the code-based repeatability report cannot be provided owing to
    the lack of its simulation environment.

    Attributes
    ----------
    cv_prob       : `float`
                    crossover probability.
    n_individuals : `int`
                    population size.

    References
    ----------
    https://www.cs.colostate.edu/~genitor/

    Whitley, D., Dominic, S., Das, R. and Anderson, C.W., 1993.
    Genetic reinforcement learning for neurocontrol problems.
    Machine Learning, 13, pp.259-284.
    https://link.springer.com/article/10.1023/A:1022674030396

    Whitley, D., 1989, December.
    The GENITOR algorithm and selection pressure: Why rank-based allocation of reproductive trials is best.
    In Proceedings of International Conference on Genetic Algorithms (pp. 116-121).
    https://dl.acm.org/doi/10.5555/93126.93169
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.cv_prob = options.get('cv_prob', 0.5)  # crossover probability
        assert 0.0 <= self.cv_prob <= 1.0
        _rank_prob = np.arange(self.n_individuals, 0, -1) - 1.0
        self.rank_prob = _rank_prob/np.sum(_rank_prob)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        cv_probs = self.cv_prob*np.ones((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y, cv_probs

    def iterate(self, x=None, y=None, cv_probs=None, args=None):
        order, xx, yy = np.argsort(y), None, None
        # use rank-based selection for two parents

        offspring = self.rng_optimization.choice(order, size=2, replace=False, p=self.rank_prob)
        if self.rng_optimization.random() < cv_probs[offspring[0]]:  # crossover
            # use intermediate crossover (not one-point crossover proposed in the original paper)
            xx = (x[offspring[0]] + x[offspring[1]])/2.0
            yy = self._evaluate_fitness(xx, args)
            if yy < y[order[-1]]:  # to replace the worst individual
                x[order[-1]], y[order[-1]] = xx, yy
                cv_probs[offspring[0]] += 0.1
            else:
                cv_probs[offspring[0]] -= 0.1
            cv_probs[offspring[0]] = np.maximum(0.05, np.minimum(0.95, cv_probs[offspring[0]]))
        else:  # mutation
            xx = np.copy(x[offspring[0]])  # offspring
            xx += self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)/10.0
            yy = self._evaluate_fitness(xx, args)
            if yy < y[order[-1]]:  # to replace the worst individual
                x[order[-1]], y[order[-1]] = xx, yy
        self._n_generations += 1
        return x, yy, cv_probs

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, cv_probs = self.initialize(args)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            x, yy, cv_probs = self.iterate(x, y, cv_probs, args)
        return self._collect(fitness, yy)
