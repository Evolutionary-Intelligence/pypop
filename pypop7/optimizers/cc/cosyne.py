import numpy as np

from pypop7.optimizers.cc.cc import CC


class COSYNE(CC):
    """Cooperative Synapse Neuroevolution(CoSyNE)

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
              and with the following particular settings (`keys`):
                * 'n_individuals'            - number of individuals/samples (`int`),
                * 'n_combine'                - number of combine (`int`, default: `2`),
                * 'crossover_type'           - type of crossover operation (`string`, default: `one_point`),
                * 'prob_mutate'              - probability to mutate (`float`, default: `0.3`).

    Examples
    --------
    Use the Cooperative Evolution optimizer `COSYNE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:
    .. code-block:: python
       :linenos:
       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import ellipsoid # function to be minimized
       >>> from pypop7.optimizers.cc.cosyne import COSYNE
       >>> problem = {'fitness_function': ellipsoid,  # define problem arguments
       ...            'ndim_problem': 10,
       ...            'lower_boundary': -5 * numpy.ones((10,)),
       ...            'upper_boundary': 5 * numpy.ones((10,))}
       >>> options = {'max_function_evaluations': 3e5,  # set optimizer options
       ...            'n_individuals': 20,
       ...            'n_combine': 2,
       ...            'prob_mutate': 0.3,
       ...            'crossover_type': "one_point"}
       >>> cosyne = COSYNE(problem, options)  # initialize the optimizer class
       >>> results = cosyne.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"COSYNE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       COSYNE: 300061, 0.561405461371189(1.0746113061904907 in Evotorch)

    Attributes
    ----------
    prob_mutate      : `float`
                        probability to mutate.
    crossover_type   : `string`
                        type of crossover.
    n_combine        : `int`
                        number of combine individuals.
    n_individuals    : `int`
                        number of individuals/samples.

    Reference
    ---------
    F. Gomez, J. Schmidhuber, R. Miikkulainen
    Accelerated Neural Evolution through Cooperatively Coevolved Synapses
    https://jmlr.org/papers/v9/gomez08a.html
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)
        self.n_combine = options.get('n_combine', 2)
        self.crossover_type = options.get('crossover_type', "one_point")
        self.prob_mutate = options.get('prob_mutate', 0.3)

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x()
        return x, y

    def mutate(self, x):
        for i in range(2):
            for j in range(self.ndim_problem):
                rand = np.random.random()
                if rand < self.prob_mutate:
                    x[i][j] = x[i][j] + 0.3 * self.rng_optimization.standard_cauchy(1)
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
        return x

    def permute(self, log):
        temp = []
        length = len(log)
        for k in range(length):
            index = np.random.randint(0, len(log)) % len(log)
            temp.append(log[index])
            log.pop(index)
        return temp

    def iterate(self, x, y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[i] = self._evaluate_fitness(x[i])
        order = np.argsort(y)
        o = self.crossover(x[order[0]], x[order[1]], self.crossover_type)
        o = self.mutate(o)
        for i in range(self.ndim_problem):
            order_1 = np.argsort(y)
            for k in range(2):
                x[order_1[self.n_individuals - k - 1]][i] = o[k][i]
            mark_weight, log = [], []
            for j in range(self.n_individuals):
                temp = (y[j] - y[order[0]])/(y[order[-1]] - y[order[0]])
                prob = 1 - np.power(temp, 1.0 / self.ndim_problem)
                rand = np.random.random()
                if rand < prob:
                    mark_weight.append(x[j][i])
                    log.append(j)
            log = self.permute(log)
            for j in range(len(log)):
                x[log[j]][i] = mark_weight[j]
                y[log[j]] = self._evaluate_fitness(x[log[j]])
        return x, y

    def optimize(self, fitness_function=None):
        fitness = CC.optimize(self, fitness_function)
        x, y = self.initialize()
        while True:
            x, y = self.iterate(x, y)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
