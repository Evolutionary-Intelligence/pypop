import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class DS(Optimizer):
    """Direct Search (DS).

    Reference
    ---------
    Torczon, V., 1997.
    On the convergence of pattern search algorithms.
    SIAM Journal on Optimization, 7(1), pp.1-25.
    https://epubs.siam.org/doi/abs/10.1137/S1052623493250780

    Wright, M.H., 1996.
    Direct search methods: Once scorned, now respectable.
    Pitman Research Notes in Mathematics Series, pp.191-208.
    https://nyuscholars.nyu.edu/en/publications/direct-search-methods-once-scorned-now-respectable

    Nelder, J.A. and Mead, R., 1965.
    A simplex method for function minimization.
    The Computer Journal, 7(4), pp.308-313.
    https://academic.oup.com/comjnl/article-abstract/7/4/308/354237

    Hooke, R. and Jeeves, T.A., 1961.
    “Direct search” solution of numerical and statistical problems.
    Journal of the ACM, 8(2), pp.212-229.
    https://dl.acm.org/doi/10.1145/321062.321069

    Fermi, E. and Metropolis N., 1952.
    Numerical solution of a minimum problem.
    Los Alamos Scientific Lab., Los Alamos, NM.
    https://www.osti.gov/servlets/purl/4377177
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.x = options.get('x')  # initial point
        self.sigma = options.get('sigma')  # global step-size
        self._n_generations = 0
        # for restart
        self.n_restart = 0
        self._sigma_bak = np.copy(self.sigma)
        self.sigma_threshold = options.get('sigma_threshold', 1e-10)
        self._fitness_list = [self.best_so_far_y]  # store `best_so_far_y` generated in each generation
        self.stagnation = options.get('stagnation', np.maximum(32, self.ndim_problem))  # number of generations
        self.fitness_diff = options.get('fitness_diff', 1e-10)  # threshold of fitness difference

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        else:
            x = np.copy(self.x)
        return x

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness):
        results = Optimizer._collect_results(self, fitness)
        results['sigma'] = self.sigma
        results['_n_generations'] = self._n_generations
        results['n_restart'] = self.n_restart
        return results
