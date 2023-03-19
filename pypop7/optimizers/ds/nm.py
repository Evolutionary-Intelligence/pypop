import numpy as np

from pypop7.optimizers.ds.ds import DS


class NM(DS):
    """Nelder-Mead simplex method (NM).

    .. note:: `NM` is perhaps the best-known and most-cited Direct (Pattern) Search method from 1965, till now.
       As pointed out by `Wright <https://tinyurl.com/mrmemn34>`_ (`Member of National Academy of Engineering
       1997 <https://www.nae.edu/MembersSection/MemberDirectory/30068.aspx>`_), *"In addition to concerns about
       the lack of theory, mainstream optimization researchers were not impressed by the Nelder-Mead method's
       practical performance, which can be appallingly poor."* However, today `NM` is still widely used to optimize
       *relatively low-dimensional* objective functions. It is **highly recommended** to first attempt other more
       advanced methods for large-scale black-box optimization.

       AKA `downhill simplex method <https://www.jmlr.org/papers/v3/strens02a.html>`_, `polytope algorithm
       <https://www.jstor.org/stable/3182874>`_.

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
                * 'sigma'     - initial global step-size (`float`, default: `1.0`),
                * 'x'         - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'alpha'     - reflection factor (`float`, default: `1.0`),
                * 'beta'      - contraction factor (`float`, default: `0.5`),
                * 'gamma'     - expansion factor (`float`, default: `2.0`),
                * 'shrinkage' - shrinkage factor (`float`, default: `0.5`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.nm import NM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3*numpy.ones((2,)),
       ...            'sigma': 0.1,
       ...            'verbose': 500}
       >>> nm = NM(problem, options)  # initialize the optimizer class
       >>> results = nm.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"NM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       NM: 5000, 1.3337953711044745e-13

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/2hv3yk7e>`_ for more details.

    Attributes
    ----------
    alpha     : `float`
                reflection factor.
    beta      : `float`
                contraction factor.
    gamma     : `float`
                expansion factor.
    shrinkage : `float`
                shrinkage factor.
    sigma     : `float`
                initial global step-size.
    x         : `array_like`
                initial (starting) point.

    References
    ----------
    Singer, S. and Nelder, J., 2009.
    Nelder-mead algorithm.
    Scholarpedia, 4(7), p.2928.
    http://var.scholarpedia.org/article/Nelder-Mead_algorithm

    Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 2007.
    Numerical recipes: The art of scientific computing.
    Cambridge University Press.
    http://numerical.recipes/
    
    Senn, S. and Nelder, J., 2003.
    A conversation with John Nelder.
    Statistical Science, pp.118-131.
    https://www.jstor.org/stable/3182874

    Wright, M.H., 1996.
    Direct search methods: Once scorned, now respectable.
    Pitman Research Notes in Mathematics Series, pp.191-208.
    https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable

    Nelder, J.A. and Mead, R., 1965.
    A simplex method for function minimization.
    The Computer Journal, 7(4), pp.308-313.
    https://academic.oup.com/comjnl/article-abstract/7/4/308/354237
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.alpha = options.get('alpha', 1.0)  # reflection factor
        assert self.alpha > 0.0
        self.beta = options.get('beta', 0.5)  # contraction factor
        assert self.beta > 0.0
        self.gamma = options.get('gamma', 2.0)  # expansion factor
        assert self.gamma > 0.0
        self.shrinkage = options.get('shrinkage', 0.5)  # shrinkage factor
        assert self.shrinkage > 0.0
        self.n_individuals = self.ndim_problem + 1

    def initialize(self, args=None, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # simplex
        y = np.empty((self.n_individuals,))  # fitness
        x[0] = self._initialize_x(is_restart)  # as suggested in [Wright, 1996]
        y[0] = self._evaluate_fitness(x[0], args)
        for i in range(1, self.n_individuals):
            if self._check_terminations():
                return x, y, y
            x[i] = x[0]
            x[i, i - 1] += self.sigma*self.rng_initialization.uniform(-1, 1)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y, y

    def iterate(self, x=None, y=None, args=None):
        order, fitness = np.argsort(y), []
        l, h = order[0], order[-1]  # index of lowest and highest points
        p_mean = np.mean(x[order[:-1]], axis=0)  # centroid of all vertices except the worst
        p_star = (1 + self.alpha)*p_mean - self.alpha*x[h]  # reflection
        y_star = self._evaluate_fitness(p_star, args)
        fitness.append(y_star)
        if self._check_terminations():
            return x, y, fitness
        if y_star < y[l]:
            p_star_star = self.gamma*p_star + (1 - self.gamma)*p_mean  # expansion
            y_star_star = self._evaluate_fitness(p_star_star, args)
            fitness.append(y_star_star)
            if self._check_terminations():
                return x, y, fitness
            if y_star_star < y_star:  # as suggested in [Wright, 1996]
                x[h], y[h] = p_star_star, y_star_star
            else:
                x[h], y[h] = p_star, y_star
        else:
            if np.all(y_star > y[order[:-1]]):
                if y_star <= y[h]:
                    x[h], y[h] = p_star, y_star
                p_star_star = self.beta*x[h] + (1 - self.beta)*p_mean
                y_star_star = self._evaluate_fitness(p_star_star, args)
                fitness.append(y_star_star)
                if self._check_terminations():
                    return x, y, fitness
                if y_star_star > y[h]:
                    for i in range(1, self.n_individuals):  # shrinkage
                        x[order[i]] = x[l] + self.shrinkage*(x[order[i]] - x[l])
                        y[order[i]] = self._evaluate_fitness(x[order[i]], args)
                        fitness.append(y[order[i]])
                        if self._check_terminations():
                            return x, y, fitness
                else:
                    x[h], y[h] = p_star_star, y_star_star
            else:
                x[h], y[h] = p_star, y_star
        return x, y, fitness

    def restart_reinitialize(self, args=None, x=None, y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info(fitness, y)
            x, y, y = self.initialize(args, is_restart)
            self._fitness_list = [self.best_so_far_y]
            self._n_generations = 0
            self._n_restart += 1
            if self.verbose:
                print(' ....... *** restart *** .......')
        return x, y, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, yy = self.initialize(args)
        while True:
            self._print_verbose_info(fitness, yy)
            x, y, yy = self.iterate(x, y, args)
            if self._check_terminations():
                break
            self._n_generations += 1
            if self.is_restart:
                x, y, yy = self.restart_reinitialize(args, x, y, fitness)
        return self._collect(fitness, yy)
