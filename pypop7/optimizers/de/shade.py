import numpy as np
from scipy.stats import cauchy

from pypop7.optimizers.de.jade import JADE


class SHADE(JADE):
    """Success-History based Adaptive Differential Evolution (SHADE).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `100`),
                * 'mu'            - mean of normal distribution for adaptation of crossover probability (`float`,
                  default: `0.5`),
                * 'median'        - median of Cauchy distribution for adaptation of mutation factor (`float`,
                  default: `0.5`).
                *  'h'            - length of historical memory (`int`, default: `100`)

    Examples
    --------
    Use the Differential Evolution optimizer `SHADE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.de.shade import SHADE
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 0}
       >>> shade = SHADE(problem, options)  # initialize the optimizer class
       >>> results = shade.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"SHADE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SHADE: 5000, 6.231767087107073e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <>`_ for more details.

    Attributes
    ----------
    h             : `int`
                    length of historical memory.
    median        : `float`
                    median of Cauchy distribution for adaptation of mutation factor.
    mu            : `float`
                    mean of normal distribution for adaptation of crossover probability.
    n_individuals : `int`
                    number of offspring, offspring population size.

    References
    ----------
    Tanabe, R. and Fukunaga, A., 2013, June.
    Success-history based parameter adaptation for differential evolution.
    In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
    https://ieeexplore.ieee.org/document/6557555
    """
    def __init__(self, problem, options):
        JADE.__init__(self, problem, options)
        self.h = options.get('h', 100)  # length of historical memory
        assert 0 < self.h
        self.m_mu = np.ones(self.h)*self.mu  # means of normal distribution
        self.m_median = np.ones(self.h)*self.median  # medians of Cauchy distribution
        self._k = 0  # index to update
        self.p_min = 2.0/self.n_individuals

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))  # mutated population
        f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
        x_un = np.vstack((np.copy(x), a))  # union of the population x and the archive a
        r = self.rng_optimization.choice(self.h, (self.n_individuals,))

        order = np.argsort(y)[:]
        p = (0.2 - self.p_min)*self.rng_optimization.random((self.n_individuals,)) + self.p_min
        idx = [order[self.rng_optimization.choice(int(i))] for i in np.ceil(p*self.n_individuals)]
        for k in range(self.n_individuals):
            f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k))
            r2 = self.rng_optimization.choice(np.setdiff1d(np.arange(len(x_un)), np.union1d(k, r1)))
            x_mu[k] = x[k] + f_mu[k] * (x[idx[k]] - x[k]) + f_mu[k] * (x[r1] - x_un[r2])
        return x_mu, f_mu, r

    def crossover(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x)
        p_cr = np.empty((self.n_individuals,))  # crossover probabilities
        for k in range(self.n_individuals):
            p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
            p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)  # truncate to [0, 1]
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f = np.empty((0,))  # set of all successful mutation factors
        p = np.empty((0,))  # set of all successful crossover probabilities
        d = np.empty((0,))  # set of all successful fitness differences
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive of the inferior solution
                f = np.hstack((f, f_mu[k]))  # archive of the successful mutation factor
                p = np.hstack((p, p_cr[k]))  # archive of the successful crossover probability
                d = np.hstack((d, y[k] - yy))  # archive of the successful fitness differences
                x[k] = x_cr[k]
                y[k] = yy
        if (len(p)!= 0) and (len(f) != 0):
            w = d/np.sum(d)
            self.m_mu[self._k] = np.sum(w*p)  # for mean update of normal distribution
            self.m_median[self._k] = np.sum(w*np.power(f, 2))/np.sum(w*f)  # for location update of Cauchy distribution
            self._k = (self._k + 1)%self.h
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu, r = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x, r)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        # randomly remove solutions to keep the archive size fixed
        if len(a) > self.n_individuals:
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        self._n_generations += 1
        return x, y, a
