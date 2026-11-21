import numpy as np  # engine for numerical computing

# abstract class of all Differential Evolution (DE) classes
from pypop7.optimizers.de.de import DE


class MDE(DE):
    """Multi-population based Differential Evolution (MDE).

    .. note:: `MDE` is a *parameter-free* (in practice) version of `DE`: there is no mutation factor
       and no crossover probability to be set, only the population size. It keeps a *main* population
       of `n_individuals` (i.e., `3N` in the original paper) vectors. At each generation, `N` distinct
       vectors are randomly drawn from it to form a *sub-population*, which is evolved by a
       **partial-elitist** mutation (each variable of the difference vector is randomly taken either
       from the global best vector or from a random member) with a *power-law* random scale factor
       and a *parameter-free* random crossover map, and then written back into the main population
       after greedy (one-to-one) selection. Therefore, each generation costs only `N` function
       evaluations. Out-of-bounds variables are re-sampled uniformly within the search range (Eq. 15
       of the original paper).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective/cost function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`,
                  default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`,
                  default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed
                  to be *explicitly* set (`int`);
              and with the following particular setting (`key`):
                * 'n_individuals' - size of the main population, i.e., `3N` in the original
                  paper (`int`, default: `90`, at least `9`). The sub-population evolved at
                  each generation has `n_individuals // 3` members.

    Examples
    --------
    Use the optimizer `MDE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.mde import MDE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> mde = MDE(problem, options)  # initialize the optimizer class
       >>> results = mde.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"MDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MDE: 5000, 0.0006479034925996662

    For its correctness checking of Python coding, refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/_repeat_mde.py>`_
    for more details.

    Attributes
    ----------
    n_individuals : `int`
                    size of the main population (`3N` in the original paper).

    References
    ----------
    Karkinli, A.E., 2023.
    `Detection of object boundary from point cloud by using multi-population
    based differential evolution algorithm.
    <https://doi.org/10.1007/s00521-022-07969-w>`_
    Neural Computing and Applications, 35(7), pp.5193-5206.

    https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm
    (official MATLAB and Python implementations from the author)
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 90)  # size of the main population (3N)
        assert self.n_individuals >= 9, 'n_individuals (3N) should be >= 9.'
        self.n_sub = int(self.n_individuals // 3)  # size of the sub-population (N)
        self._lower_boundary = np.asarray(self.lower_boundary, dtype=float)
        self._upper_boundary = np.asarray(self.upper_boundary, dtype=float)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # main population
        y = np.full((self.n_individuals,), np.inf)  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        noise = np.zeros((self.n_sub, self.ndim_problem))  # noise term of the sub-population
        return x, y, noise

    def _scale(self):
        """Power-law random scale factor: |a - u^b| * z^d, a~U{0,1}, u~U(0,1), b~U{1..10},
           z~N(0,1), d~U{1..5}.
        """
        rng = self.rng_optimization
        return np.abs(rng.integers(0, 2) - rng.random()**rng.integers(1, 11))*rng.standard_normal()**rng.integers(1, 6)

    def mutate(self, x_sub=None, best_x=None):
        """Partial-elitist mutation."""
        n, rng = self.n_sub, self.rng_optimization
        v = np.empty((n, self.ndim_problem))
        for i in range(n):
            scale = self._scale()
            while True:  # two distinct indices, both different from `i`
                r1, r2 = rng.choice(n, 2, replace=False)
                if r1 != i and r2 != i:
                    break
            dx = np.where(rng.random(self.ndim_problem) < 0.5, x_sub[r1], best_x)
            dy = np.where(rng.random(self.ndim_problem) < 0.5, x_sub[i], x_sub[r2])
            v[i] = x_sub[r2] + scale*(dx - dy)
        return v

    def crossover(self, v=None, x_sub=None, noise=None):
        """Parameter-free crossover via a random binary map."""
        n, rng = self.n_sub, self.rng_optimization
        c = 1 if rng.random()**rng.integers(1, 6) < 0.5 else self.ndim_problem
        cm = np.abs(rng.integers(0, 2, (n, c)) - rng.random((n, self.ndim_problem))**rng.integers(1, 6, (n, c))) < 0.5
        return x_sub + cm*(v + noise - x_sub)

    def boundary(self, v=None):
        """Re-sample out-of-bounds variables uniformly within the search range (Eq. 15)."""
        out = (v < self._lower_boundary) | (v > self._upper_boundary)
        if out.any():
            v = np.where(out, self._lower_boundary + self.rng_optimization.random(v.shape)*(
                self._upper_boundary - self._lower_boundary), v)
        return v

    def select(self, v=None, x_sub=None, y_sub=None, args=None):
        yy = np.full((self.n_sub,), np.inf)  # unevaluated trial vectors are never selected
        n_evaluated = 0
        for i in range(self.n_sub):
            if self._check_terminations():
                break
            yy[i] = self._evaluate_fitness(v[i], args)
            n_evaluated += 1
        better = yy < y_sub
        x_sub[better], y_sub[better] = v[better], yy[better]
        return x_sub, y_sub, yy[:n_evaluated]

    def iterate(self, x=None, y=None, noise=None, args=None):
        best_x = x[np.argmin(y)]  # global best vector of the main population
        index = self.rng_optimization.permutation(self.n_individuals)[:self.n_sub]
        x_sub, y_sub = x[index], y[index]  # sub-population (copied via fancy indexing)
        v = self.boundary(self.crossover(self.mutate(x_sub, best_x), x_sub, noise))
        x_sub, y_sub, yy = self.select(v, x_sub, y_sub, args)
        x[index], y[index] = x_sub, y_sub  # write back into the main population
        # noise term (Eq. 13) of the updated sub-population, used at the next generation
        noise = x_sub*10.0**self.rng_optimization.integers(-12, -8, (self.n_sub, self.ndim_problem))*(
            self.rng_optimization.random((self.n_sub, 1)) - 0.5)
        self._n_generations += 1
        return x, y, noise, yy

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, noise = self.initialize(args)
        yy = y  # fitness evaluated most recently
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            x, y, noise, yy = self.iterate(x, y, noise, args)
        # rare case: no evaluation at the last generation (e.g., stopped by `max_runtime`)
        return self._collect(fitness, yy if len(yy) > 0 else y)
