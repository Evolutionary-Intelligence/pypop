import numpy as np

from pypop7.optimizers.ga.ga import GA


class GENITOR(GA):
    """GENITOR.

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
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),

    Examples
    --------
    Use the GA optimizer `GENITOR` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.genitor import GENITOR
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5 * numpy.ones((100,)),
       ...            'upper_boundary': 5 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'n_individuals': 100,
       ...            'seed_rng': 2022}
       >>> genitor = GENITOR(problem, options)  # initialize the optimizer class
       >>> results = genitor.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"GENITOR: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       GENITOR: 200000, 1330.4144752335092

    Attributes
    ----------
    n_individuals : `int`
                    population size.

    References
    ----------
    Whitley, D., Dominic, S., Das, R. and Anderson, C.W., 1993.
    Genetic reinforcement learning for neurocontrol problems.
    Machine Learning, 13, pp.259-284.
    https://link.springer.com/article/10.1023/A:1022602019183
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.crossover_prob = options.get('crossover_prob', 0.5)  # crossover probability
        self._rank_prob = np.arange(self.n_individuals, 0, -1) - 1.0

    def initialize(self):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        crossover_probs = self.crossover_prob*np.ones((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i])
        self._n_generations += 1
        return x, y, crossover_probs

    def iterate(self, x=None, y=None, crossover_probs=None):
        order, xx, yy = np.argsort(y), None, None
        # use rank-based selection for two parents
        rank_prob = self._rank_prob/np.sum(self._rank_prob)
        offspring = self.rng_optimization.choice(order, size=2, replace=False, p=rank_prob)
        if self.rng_optimization.random() < crossover_probs[offspring[0]]:  # crossover
            # use intermediate crossover (not one-point crossover proposed in the original paper)
            xx = (x[offspring[0]] + x[offspring[1]])/2.0
            yy = self._evaluate_fitness(xx)
            if yy < y[order[-1]]:  # to replace the worst individual
                x[order[-1]], y[order[-1]] = xx, yy
                crossover_probs[offspring[0]] += 0.1
            else:
                crossover_probs[offspring[0]] -= 0.1
            crossover_probs[offspring[0]] = np.maximum(0.05, np.minimum(0.95, crossover_probs[offspring[0]]))
        else:  # mutation
            xx = np.copy(x[offspring[0]])  # offspring
            xx += self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)/10.0
            yy = self._evaluate_fitness(xx)
            if yy < y[order[-1]]:  # to replace the worst individual
                x[order[-1]], y[order[-1]] = xx, yy
        self._n_generations += 1
        return x, y, crossover_probs

    def optimize(self, fitness_function=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, crossover_probs = self.initialize()
        fitness.extend(y)
        while True:
            x, y, crossover_probs = self.iterate(x, y, crossover_probs)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
        return self._collect_results(fitness)
