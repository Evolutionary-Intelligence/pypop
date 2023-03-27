import numpy as np

from pypop7.optimizers.nes.nes import NES


class R1NES(NES):
    """Rank-One Natural Evolution Strategies (R1NES).

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
                * 'n_individuals' - number of offspring/descendants, aka offspring population size (`int`),
                * 'n_parents'     - number of parents/ancestors, aka parental population size (`int`),
                * 'mean'          - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'sigma'         - initial global step-size, aka mutation strength (`float`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.nes.r1nes import R1NES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> r1nes = R1NES(problem, options)  # initialize the optimizer class
       >>> results = r1nes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"R1NES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       R1NES: 5000, 0.005172532562628031

    References
    ----------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html

    Schaul, T., 2011.
    Studies in continuous black-box optimization.
    Doctoral Dissertation, Technische Universität München.
    https://people.idsia.ch/~schaul/publications/thesis.pdf

    Schaul, T., Glasmachers, T. and Schmidhuber, J., 2011, July.
    High dimensions and heavy tails for natural evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 845-852). ACM.
    https://dl.acm.org/doi/abs/10.1145/2001576.2001692

    See the official Python source code from PyBrain:
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/rank1.py
    """
    def __init__(self, problem, options):
        options['sigma'] = np.Inf  # not used for `R1NES`
        NES.__init__(self, problem, options)
        self.n_individuals = int(max(5, max(4*np.log2(self.ndim_problem), 0.2*self.ndim_problem)))
        self.lr_sigma = 0.1
        self.lr_cv = 0.1

    def initialize(self, is_restart=False):
        s = np.empty((self.n_individuals, self.ndim_problem))  # noise of offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_v = self.rng_initialization.standard_normal((self.ndim_problem,))
        p_v /= np.sqrt(np.dot(p_v, p_v))
        l_d = np.log(1.0)/2.0
        self._w = np.maximum(0.0, np.log(self.n_individuals/2.0 + 1.0) - np.log(
            self.n_individuals - np.arange(self.n_individuals)))
        return s, y, mean, p_v, l_d

    def iterate(self, s=None, y=None, mean=None, p_v=None, l_d=None, args=None):
        for k in range(self.n_individuals):
            if self._check_terminations():
                return s, y
            s[k] = (self.rng_optimization.standard_normal((self.ndim_problem,)) +
                    p_v*self.rng_optimization.standard_normal())
            y[k] = self._evaluate_fitness(mean + np.exp(l_d)*s[k], args)
        return s, y

    def _update_distribution(self, s=None, y=None, mean=None, p_v=None, l_d=None):
        order = np.argsort(-y)
        u = np.empty((self.n_individuals,))
        for i, o in enumerate(order):
            u[o] = self._w[i]
        u = u/np.sum(u)
        ww = [w for i, w in enumerate(s) if u[i] != 0]
        u = [k for k in u if k != 0]
        r = np.sqrt(np.dot(p_v, p_v))
        v, c = p_v/r, np.log(r)
        w_2 = np.array([np.dot(w, w) for w in ww])
        v_w = np.array([np.dot(v, w) for w in ww])
        wv_2 = np.array([np.square(vw) for vw in v_w])
        mean += np.exp(l_d)*np.dot(u, ww)
        k = ((np.square(r) - self.ndim_problem + 2.0)*wv_2 - (np.square(r) + 1.0)*w_2)/(
                2.0*r*(self.ndim_problem - 1.0))
        d_u = np.dot(k, u)*v + np.dot(v_w/r*u, ww)
        d_c = np.dot(d_u, v)/r
        e = min(self.lr_cv, 2.0*np.sqrt(np.square(r)/np.dot(d_u, d_u)))
        if d_c > 0.0:
            p_v += e*d_u
        else:
            c += e*d_c
            v += e*(d_u/r - d_c*v)
            v /= np.sqrt(np.dot(v, v))
            p_v = np.exp(c)*v
        l_d += self.lr_sigma*(1.0/(2.0*(self.ndim_problem - 1.0))*np.dot(
            (w_2 - self.ndim_problem) - (wv_2 - 1.0), u))
        return mean, p_v, l_d

    def restart_reinitialize(self, s=None, y=None, mean=None, p_v=None, l_d=None):
        if self.is_restart and NES.restart_reinitialize(self, y):
            s, y, mean, p_v, l_d = self.initialize(True)
        return s, y, mean, p_v, l_d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = NES.optimize(self, fitness_function)
        s, y, mean, p_v, l_d = self.initialize()
        while True:
            s, y = self.iterate(s, y, mean, p_v, l_d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, p_v, l_d = self._update_distribution(s, y, mean, p_v, l_d)
            self._n_generations += 1
            s, y, mean, p_v, l_d = self.restart_reinitialize(s, y, mean, p_v, l_d)
        return self._collect(fitness, y, mean)
