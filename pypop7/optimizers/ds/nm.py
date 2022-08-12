import numpy as np

from pypop7.optimizers.ds.ds import DS


class NM(DS):
    """Nelder-Mead simplex method (NM).

    .. note:: `NM` is perhaps the best-known and most-cited Direct (Pattern) Search method (citations > 35000).
       As pointed out by `Wright <https://tinyurl.com/mrmemn34>`_ (`Member of National Academy of Engineering
       1997 <https://www.nae.edu/MembersSection/MemberDirectory/30068.aspx>`_), *"In addition to concerns about
       the lack of theory, mainstream optimization researchers were not impressed by the Nelder-Mead method's
       practical performance, which can be appallingly poor."* However, today `NM` is still widely used to optimize
       *relatively low-dimensional* objective functions.

       It is **highly recommended** to first attempt other more advanced methods for large-scale black-box
       optimization (LSBBO).

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
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `10`);
              and with six particular settings (`keys`):
                * 'x'         - initial (starting) point (`array_like`),
                * 'sigma'     - initial (global) step-size (`float`),
                * 'alpha'     - reflection factor (`float`, default: `1.0`),
                * 'gamma'     - expansion factor (`float`, default: `2.0`),
                * 'beta'      - contraction factor (`float`, default: `0.5`),
                * 'shrinkage' - shrinkage factor (`float`, default: `0.5`).
    Examples
    --------
    Use the Pattern Search optimizer `NM` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ds.nm import NM
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1,
       ...            'verbose_frequency': 500}
       >>> nelder_mead = NM(problem, options)  # initialize the optimizer class
       >>> results = nelder_mead.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"Nelder-Mead: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 500: best_so_far_y 1.12608e-19, min(y) 6.94232e-01 & Evaluations 975
         * Generation 1000: best_so_far_y 1.12608e-19, min(y) 1.63768e+01 & Evaluations 1946
         * Generation 1500: best_so_far_y 1.12608e-19, min(y) 6.24550e-02 & Evaluations 2934
         * Generation 2000: best_so_far_y 1.12608e-19, min(y) 3.18114e+01 & Evaluations 3920
         * Generation 2500: best_so_far_y 1.12608e-19, min(y) 1.84040e+04 & Evaluations 4910
       Nelder-Mead: 5000, 1.1260780227758508e-19

    Furthermore, an interesting visualization of `NM`'s search trajectory on a 2-dimensional test function is shown in
    `this GitHub link <https://github.com/Evolutionary-Intelligence/pypop/blob/main/docs/demo/demo_nm.gif>`_.

    Attributes
    ----------
    x         : `array_like`
                initial (starting) point.
    sigma     : `float`
                (global) step-size.
    alpha     : `float`
                reflection factor
    gamma     : `float`
                expansion factor
    beta      : `float`
                contraction factor
    shrinkage : `float`
                shrinkage factor

    References
    ----------
    Singer, S. and Nelder, J., 2009.
    Nelder-mead algorithm.
    Scholarpedia, 4(7), p.2928.
    http://var.scholarpedia.org/article/Nelder-Mead_algorithm

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
        self.gamma = options.get('gamma', 2.0)  # expansion factor
        self.beta = options.get('beta', 0.5)  # contraction factor
        self.shrinkage = options.get('shrinkage', 0.5)  # shrinkage factor
        self.n_individuals = self.ndim_problem + 1

    def initialize(self, args=None, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # simplex
        y = np.empty((self.n_individuals,))  # fitness
        x[0] = self._initialize_x(is_restart)  # as suggested in [Wright, 1996]
        y[0] = self._evaluate_fitness(x[0], args)
        for i in range(1, self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = x[0]
            x[i, i - 1] += self.sigma*self.rng_initialization.uniform(-1, 1)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, args=None, fitness=None):
        order = np.argsort(y)
        l, h = order[0], order[-1]  # index of lowest and highest points
        p_mean = np.mean(x[order[:-1]], axis=0)  # centroid of all vertices except the worst
        p_star = (1 + self.alpha)*p_mean - self.alpha*x[h]  # reflection
        y_star = self._evaluate_fitness(p_star, args)
        if self.record_fitness:
            fitness.append(y_star)
        if self._check_terminations():
            return x, y
        if y_star < y[l]:
            p_star_star = self.gamma*p_star + (1 - self.gamma)*p_mean  # expansion
            y_star_star = self._evaluate_fitness(p_star_star, args)
            if self.record_fitness:
                fitness.append(y_star_star)
            if self._check_terminations():
                return x, y
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
                if self.record_fitness:
                    fitness.append(y_star_star)
                if self._check_terminations():
                    return x, y
                if y_star_star > y[h]:
                    for i in range(1, self.n_individuals):  # shrinkage
                        x[order[i]] = x[l] + self.shrinkage*(x[order[i]] - x[l])
                        y[order[i]] = self._evaluate_fitness(x[order[i]], args)
                        if self.record_fitness:
                            fitness.append(y[order[i]])
                        if self._check_terminations():
                            return x, y
                else:
                    x[h], y[h] = p_star_star, y_star_star
            else:
                x[h], y[h] = p_star, y_star
        return x, y

    def restart_initialize(self, args=None, x=None, y=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self.n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            x, y = self.initialize(args, is_restart)
            fitness.extend(y)
            self._fitness_list = [self.best_so_far_y]
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y = self.iterate(x, y, args, fitness)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y = self.restart_initialize(args, x, y, fitness)
        results = self._collect_results(fitness)
        return results
